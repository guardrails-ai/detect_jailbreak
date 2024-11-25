# app_inference_spec.py
# Forked from spec:
# github.com/guardrails-ai/models-host/tree/main/ray#adding-new-inference-endpoints
import os
from typing import Optional

from fastapi import HTTPException
from pydantic import BaseModel
from models_host.base_inference_spec import BaseInferenceSpec

from validator import DetectJailbreak


class InputRequest(BaseModel):
    message: str
    threshold: Optional[float] = None


class OutputResponse(BaseModel):
    classification: str
    score: float
    is_jailbreak: bool


# Using same nomenclature as in Sagemaker classes
class InferenceSpec(BaseInferenceSpec):
    def __init__(self):
        self.model = None

    @property
    def device_name(self):
        env = os.environ.get("env", "dev")
        # JC: Legacy usage of 'env' as a device.
        torch_device = "cuda" if env == "prod" else "cpu"
        return torch_device

    def load(self):
        print(f"Loading model DetectJailbreak and moving to {self.device_name}...")
        self.model = DetectJailbreak(device=self.device_name)

    def process_request(self, input_request: InputRequest):
        message = input_request.message
        # If needed, sanity check.
        # raise HTTPException(status_code=400, detail="Invalid input format")
        args = (message,)
        kwargs = {}
        if input_request.threshold is not None:
            kwargs["threshold"] = input_request.threshold
            if not 0.0 <= input_request.threshold <= 1.0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Threshold must be between 0.0 and 1.0. "
                           f"Got {input_request.threshold}"
                )
        return args, kwargs

    def infer(self, message: str, threshold: Optional[float] = None) -> OutputResponse:
        if threshold is None:
            threshold = 0.81

        score = self.model.predict_jailbreak([message,])[0]
        if score > threshold:
            classification = "jailbreak"
            is_jailbreak = True
        else:
            classification = "safe"
            is_jailbreak = False

        return OutputResponse(
            classification=classification,
            score=score,
            is_jailbreak=is_jailbreak,
        )
