FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download Whisper-small weights so the container starts without internet
RUN python -c "from transformers import WhisperModel, WhisperFeatureExtractor; WhisperModel.from_pretrained('openai/whisper-small'); WhisperFeatureExtractor.from_pretrained('openai/whisper-small')"

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
