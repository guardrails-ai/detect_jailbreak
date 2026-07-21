from transformers import pipeline, AutoTokenizer, AutoModel, AutoConfig

print("post-install starting...")
# TODO: It's not clear if the DetectJailbreak will be on the path yet.
# If we can import Detect Jailbreak, it will be safer to read the names of the models
# from the composite model as exposed by DetectJailbreak.XYZ.
print("Fetching model 1 of 3 (Saturation)")
AutoModel.from_pretrained("GuardrailsAI/prompt-saturation-attack-detector")
AutoTokenizer.from_pretrained("google-bert/bert-base-cased")
print("Fetching model 2 of 3 (Embedding)")
AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
print("Fetching model 3 of 3 (Detection)")
# Workaround: the zhx123/ftrobertallm model has integer id2label values
# (e.g. {'0': 0, '1': 1}) which fail strict validation in huggingface_hub >= 0.23.
# Load the config, coerce id2label values to strings, then pass it to the pipeline.
config = AutoConfig.from_pretrained("zhx123/ftrobertallm")
if hasattr(config, "id2label") and config.id2label is not None:
    config.id2label = {str(k): str(v) for k, v in config.id2label.items()}
pipeline("text-classification", "zhx123/ftrobertallm", config=config)
print("post-install complete!")
