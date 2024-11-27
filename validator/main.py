import json
import math
from typing import Callable, List, Optional, Union, Any

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
from .resources import KNOWN_ATTACKS, get_tokenizer_and_model_by_path, get_pipeline_by_path
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
        threshold (float): Defaults to 0.81. A float between 0 and 1, with lower being 
        more sensitive. A high value means the model will be fairly permissive and 
        unlikely to flag any but the most flagrant jailbreak attempts. A low value will 
        be pessimistic and will possibly flag legitimate inquiries.
        
        device (str): Defaults to 'cpu'. The device on which the model will be run.
        Accepts 'mps' for hardware acceleration on MacOS and 'cuda' for GPU acceleration
        on supported hardware. A device ID can also be specified, e.g., "cuda:0".
        
        model_path_override (str): A pointer to an ensemble tar file in S3 or on disk.
    """  # noqa

    TEXT_CLASSIFIER_NAME = "zhx123/ftrobertallm"
    TEXT_CLASSIFIER_PASS_LABEL = 0
    TEXT_CLASSIFIER_FAIL_LABEL = 1

    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_KNOWN_PROMPT_MATCH_THRESHOLD = 0.9
    MALICIOUS_EMBEDDINGS = KNOWN_ATTACKS

    SATURATION_CLASSIFIER_PASS_LABEL = "safe"
    SATURATION_CLASSIFIER_FAIL_LABEL = "jailbreak"

    # These were found with a basic low-granularity beam search.
    DEFAULT_KNOWN_ATTACK_SCALE_FACTORS = (0.5, 2.0)
    DEFAULT_SATURATION_ATTACK_SCALE_FACTORS = (3.5, 2.5)
    DEFAULT_TEXT_CLASSIFIER_SCALE_FACTORS = (3.0, 2.5)

    def __init__(
            self,
            threshold: float = 0.81,
            device: str = "cpu",
            on_fail: Optional[Callable] = None,
            model_path_override: str = "",
            **kwargs,
    ):
        super().__init__(on_fail=on_fail, **kwargs)
        self.device = device
        self.threshold = threshold
        self.saturation_attack_detector = None
        self.text_classifier = None
        self.embedding_tokenizer = None
        self.embedding_model = None
        self.known_malicious_embeddings = []

        # It's possible for self.use_local to be unset and in some indeterminate state.
        # First take use_local as a kwarg as the truth.
        # If that's not present, try self.use_local.
        # If that's not present, default to true.
        if "use_local" in kwargs:
            self.use_local = kwargs["use_local"]
        elif self.use_local is None:
            self.use_local = True

        if self.use_local:
            if not model_path_override:
                self.saturation_attack_detector = PromptSaturationDetectorV3(
                    device=torch.device(device),
                )
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
            else:
                # Saturation:
                self.saturation_attack_detector = PromptSaturationDetectorV3(
                    device=torch.device(device),
                    model_path_override=model_path_override
                )
                # Known attacks:
                embedding_tokenizer, embedding_model = get_tokenizer_and_model_by_path(
                    model_path_override,
                    "embedding",
                    AutoTokenizer,
                    AutoModel
                )
                self.embedding_tokenizer = embedding_tokenizer
                self.embedding_model = embedding_model.to(device)
                # Other text attacks:
                self.text_classifier = get_pipeline_by_path(
                    model_path_override,
                    "text-classifier",
                    "text-classification",
                    max_length=512,
                    truncation=True,
                    device=device
                )

            # Quick compute on startup:
            self.known_malicious_embeddings = self._embed(KNOWN_ATTACKS)

        # These _are_ modifyable, but not explicitly advertised.
        self.known_attack_scales = DetectJailbreak.DEFAULT_KNOWN_ATTACK_SCALE_FACTORS
        self.saturation_attack_scales = DetectJailbreak.DEFAULT_SATURATION_ATTACK_SCALE_FACTORS
        self.text_attack_scales = DetectJailbreak.DEFAULT_TEXT_CLASSIFIER_SCALE_FACTORS

    @staticmethod
    def _rescale(x: float, a: float = 1.0, b: float = 1.0):
        return 1.0 / (1.0 + (a*math.exp(-b*x)))

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

    def _embed(self, prompts: List[str]):
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
        ).to(self.device)
        with torch.no_grad():
            model_outputs = self.embedding_model(**encoded_input)
        embeddings = DetectJailbreak._mean_pool(
            model_outputs, attention_mask=encoded_input['attention_mask'])
        return F.normalize(embeddings, p=2, dim=1)

    def _match_known_malicious_prompts(
            self,
            prompts: Union[List[str], torch.Tensor],
    ) -> List[float]:
        """Returns an array of floats, one per prompt, with the max match to known
        attacks.  If prompts is a list of strings, embeddings will be generated.  If
        embeddings are passed, they will be used."""
        if isinstance(prompts, list):
            prompt_embeddings = self._embed(prompts)
        else:
            prompt_embeddings = prompts
        # These are already normalized. We don't need to divide by magnitudes again.
        distances = prompt_embeddings @ self.known_malicious_embeddings.T
        return [
            DetectJailbreak._rescale(s, *self.known_attack_scales)
            for s in (torch.max(distances, axis=1).values).tolist()
        ]

    def _predict_and_remap(
            self,
            model,
            prompts: List[str],
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
                new_score = 0.5 - (old_score * 0.5)
            else:
                new_score = 0.5 + (old_score * 0.5)
            scores.append(new_score)
        return scores

    def _predict_jailbreak(self, prompts: List[str]) -> List[float]:
        return [
            DetectJailbreak._rescale(s, *self.text_attack_scales)
            for s in self._predict_and_remap(
                self.text_classifier,
                prompts,
                "label",
                "score",
                self.TEXT_CLASSIFIER_PASS_LABEL,
                self.TEXT_CLASSIFIER_FAIL_LABEL,
            )
        ]

    def _predict_saturation(self, prompts: List[str]) -> List[float]:
        return [
            DetectJailbreak._rescale(
                s,
                self.saturation_attack_scales[0],
                self.saturation_attack_scales[1],
            ) for s in self._predict_and_remap(
                self.saturation_attack_detector,
                prompts,
                "label",
                "score",
                self.SATURATION_CLASSIFIER_PASS_LABEL,
                self.SATURATION_CLASSIFIER_FAIL_LABEL,
            )
        ]

    def predict_jailbreak(
            self,
            prompts: List[str],
            reduction_function: Optional[Callable] = max,
    ) -> Union[List[float], List[dict]]:
        """predict_jailbreak will return an array of floats by default, one per prompt.
        If reduction_function is set to 'none' it will return a dict with the different
        sub-validators and their scores. Useful for debugging and tuning."""
        if isinstance(prompts, str):
            print("WARN: predict_jailbreak should be called with a list of strings.")
            prompts = [prompts, ]
        known_attack_scores = self._match_known_malicious_prompts(prompts)
        saturation_scores = self._predict_saturation(prompts)
        predicted_scores = self._predict_jailbreak(prompts)
        if reduction_function is None:
            return [{
                "known_attack": known,
                "saturation_attack": sat,
                "other_attack": pred
            } for known, sat, pred in zip(
                known_attack_scores, saturation_scores, predicted_scores
            )]
        else:
            return [
                reduction_function(subscores)
                for subscores in
                zip(known_attack_scores, saturation_scores, predicted_scores)
            ]

    def validate(
            self,
            value: Union[str, List[str]],
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
            value = [value, ]

        # _inference is to support local/remote. It is equivalent to this:
        # scores = self.predict_jailbreak(value)
        scores = self._inference(value)

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

    # The rest of these methods are made for validator compatibility and may have some
    # strange properties,

    def _inference_local(self, model_input: List[str]) -> Any:
        return self.predict_jailbreak(model_input)

    def _inference_remote(self, model_input: List[str]) -> Any:
        # This needs to be kept in-sync with app_inference_spec.
        request_body = {"prompts": model_input}
        response = self._hub_inference_request(
            json.dumps(request_body),
            self.validation_endpoint
        )
        if not response or "scores" not in response:
            raise ValueError("Invalid response from remote inference", response)

        return response["scores"]
