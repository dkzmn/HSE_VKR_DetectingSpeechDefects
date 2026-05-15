import json
import logging
import os
import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, File, Form, UploadFile
from transformers import WhisperFeatureExtractor, WhisperModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000
MAX_SAMPLES = 10 * SAMPLE_RATE
ENCODER_DIM = 768
EXTRA_DIM = 10
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models/whisper_small_finetuned"))


def _resolve_device() -> torch.device:
    spec = os.getenv("DEVICE", "auto")
    if spec != "auto":
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _resolve_device()

if DEVICE.type == "cuda":
    torch.backends.cudnn.benchmark = True
    logger.info("GPU: %s (cuda %s)", torch.cuda.get_device_name(DEVICE), torch.version.cuda)
else:
    logger.info("Device: %s", DEVICE)


class SpeechClassifier(nn.Module):
    def __init__(self, encoder):
        # энкодер Whisper + голова для бинарной классификации
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Linear(ENCODER_DIM + EXTRA_DIM, 2)

    def forward(self, mel, extra):
        # mean-pool скрытых состояний, объединить с доп. признаками
        hidden = self.encoder(mel).last_hidden_state
        pooled = self.dropout(hidden.mean(dim=1))
        return self.head(torch.cat([pooled, extra], dim=-1))


def _load_model() -> SpeechClassifier:
    # загрузить чекпоинт дообученной модели на нужное устройство
    base = WhisperModel.from_pretrained("openai/whisper-small")
    clf = SpeechClassifier(base.encoder)
    state = torch.load(MODEL_DIR / "best_ckpt.pt", map_location=DEVICE)
    clf.load_state_dict(state)
    clf.to(DEVICE)
    clf.eval()
    return clf


_feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")
_model = _load_model()

with open(MODEL_DIR / "threshold.json") as _f:
    THRESHOLD = float(json.load(_f)["threshold"])

app = FastAPI(title="Speech Scoring API")


def _decode_audio(audio_bytes: bytes) -> np.ndarray:
    # декодировать аудио в сырой float32 PCM 16 кГц моно через ffmpeg
    cmd = [
        "ffmpeg", "-i", "pipe:0",
        "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "1",
        "pipe:1", "-loglevel", "quiet",
    ]
    result = subprocess.run(cmd, input=audio_bytes, capture_output=True)
    if not result.stdout:
        raise RuntimeError(f"ffmpeg decode failed: {result.stderr.decode()[:200]}")
    wav = np.frombuffer(result.stdout, dtype=np.float32).copy()
    if len(wav) < MAX_SAMPLES:
        wav = np.pad(wav, (0, MAX_SAMPLES - len(wav)))
    else:
        wav = wav[:MAX_SAMPLES]
    return wav


def _quality_score(proba: float, threshold: float) -> int:
    # линейная шкала 0–100: порог делит на «хорошо» (50–100) и «плохо» (0–50)
    if proba < threshold:
        return round(100 - 50 * proba / threshold)
    return round(50 * (1 - proba) / (1 - threshold))


@app.get("/health")
def health():
    info: dict = {"status": "ok", "device": DEVICE.type}
    if DEVICE.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(DEVICE)
        info["gpu_memory_allocated_mb"] = round(torch.cuda.memory_allocated(DEVICE) / 1e6, 1)
        info["gpu_memory_reserved_mb"] = round(torch.cuda.memory_reserved(DEVICE) / 1e6, 1)
    return info


@app.post("/score")
async def score(
    audio: UploadFile = File(...),
    twister_id: int = Form(...),
    letters: str = Form(default="[0,0,0,0,0,0,0,0]"),
    duration: float = Form(default=0.0),
    n_speakers: int = Form(default=1),
):
    # принять аудио, извлечь мел-спектрограмму, запустить инференс
    audio_bytes = await audio.read()
    wav = _decode_audio(audio_bytes)

    inputs = _feature_extractor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    mel = inputs.input_features.to(DEVICE)

    extra = torch.tensor(
        [json.loads(letters) + [duration, float(n_speakers)]],
        dtype=torch.float32,
    ).to(DEVICE)

    with torch.no_grad():
        try:
            logits = _model(mel, extra)
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and DEVICE.type == "cuda":
                torch.cuda.empty_cache()
                logits = _model(mel, extra)
            else:
                raise
    proba_bad = torch.softmax(logits, dim=-1)[0, 1].item()

    label = "bad" if proba_bad >= THRESHOLD else "good"

    return {
        "score": _quality_score(proba_bad, THRESHOLD),
        "label": label,
        "proba": round(proba_bad, 4),
        "threshold": THRESHOLD,
        "model_version": "whisper-small-finetune-v1",
        "twister_id": twister_id,
    }
