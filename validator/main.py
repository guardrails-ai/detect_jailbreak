import importlib
import os
from typing import Callable, Optional, Union

import torch
from torch.nn import functional as F
from transformers import pipeline, AutoTokenizer, AutoModel

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)
from .resources import KNOWN_ATTACKS
from .models import PromptSaturationDetectorV3


@register_validator(name="guardrails/detect-jailbreak", data_type="string")
class DetectJailbreak(Validator):
    """Validates that a prompt does not attempt to circumvent restrictions on behavior.
    An example would be convincing the model via prompt to provide instructions that 
    could cause harm to one or more people.

    **Key Properties**

    | Property                      | Description                       |
    | ----------------------------- | --------------------------------- |
    | Name for `format` attribute   | `guardrails/detect-jailbreak`     |
    | Supported data types          | `string`                          |
    | Programmatic fix              | `None` |

    Args:
        threshold (float): Defaults to 0.9. A float between 0 and 1, with lower being 
        more sensitive. A high value means the model will be fairly permissive and 
        unlikely to flag any but the most flagrant jailbreak attempts. A low value will 
        be pessimistic and will possibly flag legitimate inquiries.
        
        device (str): Defaults to 'cpu'. The device on which the model will be run.
        Accepts 'mps' for hardware acceleration on MacOS and 'cuda' for GPU acceleration
        on supported hardware. A device ID can also be specified, e.g., "cuda:0".
    """  # noqa

    TEXT_CLASSIFIER_NAME = "jackhhao/jailbreak-classifier"
    TEXT_CLASSIFIER_PASS_LABEL = "benign"
    TEXT_CLASSIFIER_FAIL_LABEL = "jailbreak"
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_KNOWN_PROMPT_MATCH_THRESHOLD = 0.9
    MALICIOUS_EMBEDDINGS = KNOWN_ATTACKS
    SATURATION_CLASSIFIER_NAME = "prompt_saturation_detector_v3_1_final.pth"
    SATURATION_CLASSIFIER_PASS_LABEL = "safe"
    SATURATION_CLASSIFIER_FAIL_LABEL = "jailbreak"

    def __init__(
        self,
        threshold: float = 0.515,
        device: str = "cpu",
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(on_fail=on_fail)
        self.device = device
        self.threshold = threshold
        self.saturation_attack_detector = PromptSaturationDetectorV3()
        self.text_classifier = pipeline(
            "text-classification",
            DetectJailbreak.TEXT_CLASSIFIER_NAME,
            max_length=512,  # HACK: Fix classifier size.
            truncation=True,
            device=device,
        )
        # There are a large number of fairly low-effort prompts people will use.
        # The embedding detectors do checks to roughly match those.
        self.embedding_tokenizer = AutoTokenizer.from_pretrained(
            DetectJailbreak.EMBEDDING_MODEL_NAME
        )
        self.embedding_model = AutoModel.from_pretrained(
            DetectJailbreak.EMBEDDING_MODEL_NAME
        ).to(device)
        self.known_malicious_embeddings = self._embed(KNOWN_ATTACKS)

    @staticmethod
    def _mean_pool(model_output, attention_mask):
        """Taken from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2."""
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(
            token_embeddings.size()
        ).float()
        return torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def _embed(self, prompts: list[str]):
        """Taken from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        We use the long-form to avoid a dependency on sentence transformers.
        This method returns the maximum of the matches against all known attacks.
        """
        encoded_input = self.embedding_tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512,  # This may be too small to adequately capture the info.
        )
        with torch.no_grad():
            model_outputs = self.embedding_model(**encoded_input)
        embeddings = DetectJailbreak._mean_pool(
            model_outputs, attention_mask=encoded_input['attention_mask'])
        return F.normalize(embeddings, p=2, dim=1)

    def _match_known_malicious_prompts(
            self,
            prompts: list[str] | torch.Tensor,
    ) -> list[float]:
        """Returns an array of floats, one per prompt, with the max match to known
        attacks.  If prompts is a list of strings, embeddings will be generated.  If
        embeddings are passed, they will be used."""
        if isinstance(prompts, list):
            prompt_embeddings = self._embed(prompts)
        else:
            prompt_embeddings = prompts
        # These are already normalized. We don't need to divide by magnitudes again.
        distances = prompt_embeddings @ self.known_malicious_embeddings.T
        return torch.max(distances, axis=1).values.tolist()

    def _predict_and_remap(
            self,
            model,
            prompts: list[str],
            label_field: str,
            score_field: str,
            safe_case: str,
            fail_case: str,
    ):
        predictions = model(prompts)
        scores = list()  # We want to remap so 0 is 'safe' and 1 is 'unsafe'.
        for pred in predictions:
            old_score = pred[score_field]
            is_safe = pred[label_field] == safe_case
            assert pred[label_field] in {safe_case, fail_case} \
                   and 0.0 <= old_score <= 1.0
            if is_safe:
                scores.append(0.5 - (old_score * 0.5))
            else:
                scores.append(0.5 + (old_score * 0.5))
        return scores

    def _predict_jailbreak(self, prompts: list[str]) -> list[float]:
        return self._predict_and_remap(
            self.text_classifier,
            prompts,
            "label",
            "score",
            self.TEXT_CLASSIFIER_PASS_LABEL,
            self.TEXT_CLASSIFIER_FAIL_LABEL,
        )

    def _predict_saturation(self, prompts: list[str]) -> list[float]:
        return self._predict_and_remap(
            self.saturation_attack_detector,
            prompts,
            "label",
            "score",
            self.SATURATION_CLASSIFIER_PASS_LABEL,
            self.SATURATION_CLASSIFIER_FAIL_LABEL,
        )

    def predict_jailbreak(self, prompts: list[str]) -> list[float]:
        known_attack_scores = self._match_known_malicious_prompts(prompts)
        saturation_scores = self._predict_saturation(prompts)
        predicted_scores = self._predict_jailbreak(prompts)
        return [
            max(subscores)
            for subscores in
            zip(known_attack_scores, saturation_scores, predicted_scores)
        ]

    def validate(
            self,
            value: Union[str, list[str]],
            metadata: Optional[dict] = None,
    ) -> ValidationResult:
        """Validates that will return a failure if the value is a jailbreak attempt.
        If the provided value is a list of strings the validation result will be based
        on the maximum injection likelihood.  A single validation result will be
        returned for all.
        """
        if metadata:
            pass  # Log that this model supports no metadata?

        # In the case of a single string, make a one-element list -> one codepath.
        if isinstance(value, str):
            prompts = [value, ]

        scores = self.predict_jailbreak(value)

        failed_prompts = list()
        failed_scores = list()  # To help people calibrate their thresholds.

        for p, score in zip(value, scores):
            if score > self.threshold:
                failed_prompts.append(p)
                failed_scores.append(score)

        if failed_prompts:
            failure_message = f"{len(failed_prompts)} detected as potential jailbreaks:"
            for txt, score in zip(failed_prompts, failed_scores):
                failure_message += f"\n\"{txt}\" (Score: {score})"
            return FailResult(
                error_message=failure_message
            )
        return PassResult()
