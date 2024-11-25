from transformers import pipeline, AutoTokenizer, AutoModel

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
pipeline("text-classification", "zhx123/ftrobertallm")
print("post-install complete!")