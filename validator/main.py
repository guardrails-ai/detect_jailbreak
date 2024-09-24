from typing import Callable, Optional, Tuple

from transformers import pipeline

from guardrails.validator_base import (
    FailResult,
    PassResult,
    ValidationResult,
    Validator,
    register_validator,
)


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
        be pessimistic and will possible flag legitimate inquiries.
        
        deviec (str): Defaults to 'cpu'. The device on which the model will be run.
        Accepts 'mps' for hardware acceleration on MacOS and 'cuda' for GPU acceleration
        on supported hardware. A device ID can also be specified, e.g., "cuda:0".
    """  # noqa

    MODEL_NAME = "jackhhao/jailbreak-classifier"
    MODEL_PASS_LABEL = "benign"
    MODEL_FAIL_LABEL = "jailbreak"

    # If you don't have any init args, you can omit the __init__ method.
    def __init__(
        self,
        threshold: float = 0.9,
        device: str = "cpu",
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(on_fail=on_fail)
        self.threshold = threshold
        self.model = pipeline(
            "text-classification",
            DetectJailbreak.MODEL_NAME,
            device_map=device
        )

    @staticmethod
    def _remap_score(score: float, safe: bool) -> float:
        """
        We want the model to output '0' for safe and '1' for unsafe.
        The model has two outputs, both from 0 to 1.
        Remap 0-1 in safe to 0.5-0.0 and 0-1 unsafe to 0.5-1.0.
        """
        if not (0.0 < score < 1.0):
            # Log a sanity problem.
            score = max(0.0, score)
            score = min(1.0, score)

        if safe:
            return (1.0 - score)*0.5
        else:
            return 0.5*score + 0.5

    def validate(
            self,
            value: Tuple[str, list[str]],
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
            value = [value,]

        failed_prompts = list()
        failed_scores = list()  # To help people calibrate their thresholds.
        predictions = self.model(value)

        # Zip and evaluate predictions to return any failures.
        for text_input, pred in zip(value, predictions):
            score = DetectJailbreak._remap_score(
                score=pred['score'],
                safe=pred['label'] == self.MODEL_PASS_LABEL
            )
            if score > self.threshold:
                failed_prompts.append(text_input)
                failed_scores.append(score)

        if failed_prompts:
            failure_message = f"{len(failed_prompts)} detected as potential jailbreaks:"
            for txt, score in zip(failed_prompts, failed_scores):
                failure_message += f"\n\"{txt}\" (Score: {score})"
            return FailResult(
                error_message=failure_message
            )
        return PassResult()
