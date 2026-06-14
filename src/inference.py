import os
from transformers import pipeline

model_name = os.environ.get("HF_MODEL_NAME", "AravindSreedharan/mlops-distilbert-imdb")
input_text = os.environ.get("INPUT_TEXT", "This movie was absolutely fantastic!")
hf_token   = os.environ.get("HF_TOKEN", None)

print(f"Loading model: {model_name}")
classifier = pipeline("text-classification", model=model_name, token=hf_token)

result = classifier(input_text)
print(f"\nInput:      {input_text}")
print(f"Prediction: {result[0]['label']} (confidence: {result[0]['score']:.2%})")