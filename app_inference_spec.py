# app_inference_spec.py
# Forked from spec:
# github.com/guardrails-ai/models-host/tree/main/ray#adding-new-inference-endpoints
import os
from logging import getLogger
from typing import List

from pydantic import BaseModel
from models_host.base_inference_spec import BaseInferenceSpec

from validator import DetectJailbreak


# Environment variables:
# MODEL_PATH - "s3" (default) read model from MODEL_S3_PATH, "hf" read from huggingface.
# MODEL_S3_PATH - Defaults to
#   s3://guardrails-ai-public-read-only/detect-jailbreak-v0/detect-jailbreak-v0.tar.gz
# HF_TOKEN - if set, will read model from HF.

logger = getLogger(__name__)


class InputRequest(BaseModel):
    prompts: List[str]


class OutputResponse(BaseModel):
    scores: List[float]


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
        kwargs = {
            "device": self.device_name
        }
        read_from = os.environ.get("MODEL_PATH", "s3").lower()
        if read_from == "s3":
            print("Reading model from S3.")
            kwargs["model_path_override"] = os.environ.get(
                "MODEL_S3_PATH",
                "s3://guardrails-ai-public-read-only/detect-jailbreak-v0/detect-jailbreak-v0.tar.gz"
            )
        elif read_from == "hf":
            print("Reading model from HF.")
            pass  # Auto read from HF by default.
        else:
            logger.warning(f"MODEL_PATH is not set to 's3' or 'hf': '{read_from}'")
        print(f"Loading model DetectJailbreak and moving to {self.device_name}...")
        self.model = DetectJailbreak(**kwargs)

    def process_request(self, input_request: InputRequest):
        prompts = input_request.prompts
        # If needed, sanity check.
        # raise HTTPException(status_code=400, detail="Invalid input format")
        args = (prompts,)
        kwargs = {}
        return args, kwargs

    def infer(self, prompts: List[str]) -> OutputResponse:
        return OutputResponse(
            scores=self.model.predict_jailbreak(prompts),
        )
