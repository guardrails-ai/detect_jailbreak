from typing import Any, Callable, Dict, Optional

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
        threshold (float): A float between 0 and 1, with lower being more sensitive. A high value means the model will be fairly permissive and unlikely to flag any but the most flagrant jailbreak attempts. A low value will be pessimistic and will possible flag legitimate inquiries.
    """  # noqa

    # If you don't have any init args, you can omit the __init__ method.
    def __init__(
        self,
        threshold: float = 0.5,
        on_fail: Optional[Callable] = None,
    ):
        super().__init__(on_fail=on_fail)
        self.threshold = threshold
        self.model = load_model()

    def validate(self, value: str, metadata: Optional[dict] = None) -> ValidationResult:
        """Validates that will return a failure if the value is a jailbreak attempt."""
        if value != "pass": # FIXME
            return FailResult(
                error_message="{A descriptive but concise error message about why validation failed}",
                fix_value="{The programmtic fix if applicable, otherwise remove this kwarg.}",
            )
        return PassResult()
