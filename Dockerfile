FROM python:3.11-slim

WORKDIR /app

ARG HF_MODEL_NAME=AravindSreedharan/mlops-distilbert-imdb
ENV HF_MODEL_NAME=${HF_MODEL_NAME}

COPY requirements.txt .
RUN pip install --no-cache-dir transformers torch huggingface_hub

COPY src/inference.py .

CMD ["python", "inference.py"]