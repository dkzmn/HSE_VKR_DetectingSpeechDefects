# -*- coding: utf-8 -*-
"""
Классификатор на эмбеддингах wav2vec 2.0: замороженный encoder → mean pool → FC.
Источник: arXiv:2309.14107, IEEE 2024. Вход: сырой waveform 16 kHz.
Сначала пробуем torchaudio; при ошибке импорта — локальный 1D-CNN энкодер (без transformers).
"""
import torch
import torch.nn as nn
from collections import namedtuple
import torchaudio

EncoderOutput = namedtuple("EncoderOutput", ["last_hidden_state"])


class FallbackWaveformEncoder(nn.Module):
    """Лёгкий 1D-CNN по waveform (768-dim на выход), если wav2vec2 недоступен."""

    def __init__(self, sample_rate=16000, max_sec=10, out_dim=768):
        super().__init__()
        n = sample_rate * max_sec
        self.conv = nn.Sequential(
            nn.Conv1d(1, 64, 80, stride=4),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, stride=2),
            nn.ReLU(),
            nn.Conv1d(128, 256, 3, stride=2),
            nn.ReLU(),
            nn.Conv1d(256, 512, 3, stride=2),
            nn.ReLU(),
        )
        with torch.no_grad():
            x = torch.zeros(1, 1, n)
            for m in self.conv:
                x = m(x)
            t = x.size(2)
        self.fc = nn.Linear(512 * t, out_dim)

    def forward(self, x, attention_mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.flatten(1)
        x = self.fc(x)
        return EncoderOutput(last_hidden_state=x.unsqueeze(1))

    def extract_features(self, x):
        out = self.forward(x)
        return out.last_hidden_state, None


class Wav2Vec2Classifier(nn.Module):
    def __init__(self, encoder, hidden_size, num_classes=2, n_letters=0, dropout=0.3, freeze_encoder=True, use_torchaudio=True):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.n_letters = n_letters
        self.use_torchaudio = use_torchaudio
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size + n_letters, num_classes)

    def forward(self, input_values, attention_mask=None, letters=None):
        if self.use_torchaudio:
            out = self.encoder.extract_features(input_values)
            if isinstance(out, (list, tuple)) and len(out) >= 2:
                features = out[0]
            elif isinstance(out, (list, tuple)) and len(out) == 1:
                features = out[0]
            else:
                features = out
            if isinstance(features, (list, tuple)):
                features = features[-1]
            hidden = features.mean(dim=1)
        else:
            out = self.encoder(input_values, attention_mask=attention_mask)
            hidden = out.last_hidden_state
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                hidden = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            else:
                hidden = hidden.mean(dim=1)
        hidden = self.dropout(hidden)
        if self.n_letters > 0 and letters is not None:
            hidden = torch.cat([hidden, letters], dim=1)
        return self.fc(hidden)


def get_model(num_classes=2, n_letters=0, dropout=0.3, freeze_encoder=True, model_id="facebook/wav2vec2-base"):
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    encoder = bundle.get_model()
    hidden_size = getattr(bundle, "_params", {}).get("encoder_embed_dim", 768)
    return Wav2Vec2Classifier(
        encoder=encoder,
        hidden_size=hidden_size,
        num_classes=num_classes,
        n_letters=n_letters,
        dropout=dropout,
        freeze_encoder=freeze_encoder,
        use_torchaudio=True,
    )
