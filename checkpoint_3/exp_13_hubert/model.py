# -*- coding: utf-8 -*-
"""
HuBERT эмбеддинги + LSTM → FC для бинарной классификации (дефект речи).
Источник: Hybrid-Module Transformer: enhancing speech emotion recognition with HuBERT, LSTM, and ResNet-50 (PeerJ).
Вход: сырой waveform 16 kHz. При недоступности transformers — fallback 1D-CNN → LSTM → FC.
"""
import torch
import torch.nn as nn


class FallbackWaveformEncoder(nn.Module):
    """Энкодер по waveform, выдаёт (batch, 1, out_dim) — одна «временная» точка для LSTM."""

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
            self._flat = x.numel()
        self.proj = nn.Linear(self._flat, out_dim)
        self._out_dim = out_dim

    def forward(self, x, attention_mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.flatten(1)
        x = self.proj(x)
        return x.unsqueeze(1)


class HubertLSTMClassifier(nn.Module):
    """HuBERT (или fallback) → последовательность (B, T, D) → BiLSTM → mean → Dropout → FC."""

    def __init__(
        self,
        encoder,
        encoder_dim,
        num_classes=2,
        lstm_hidden=128,
        n_letters=0,
        dropout=0.3,
        freeze_encoder=True,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.n_letters = n_letters
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.lstm = nn.LSTM(
            encoder_dim,
            lstm_hidden,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
            num_layers=1,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2 + n_letters, num_classes)

    def forward(self, input_values, attention_mask=None, letters=None):
        with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
            out = self.encoder(input_values, attention_mask=attention_mask)
            if hasattr(out, "last_hidden_state"):
                hidden = out.last_hidden_state
            elif isinstance(out, (list, tuple)):
                hidden = out[0]
            else:
                hidden = out
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)
        out, _ = self.lstm(hidden)
        agg = out.mean(dim=1)
        agg = self.dropout(agg)
        if self.n_letters > 0 and letters is not None:
            agg = torch.cat([agg, letters], dim=1)
        return self.fc(agg)


def _load_hubert(model_id="facebook/hubert-base-ls960"):
    from transformers import HubertModel
    model = HubertModel.from_pretrained(model_id)
    return model, model.config.hidden_size


def get_model(
    num_classes=2,
    n_letters=0,
    dropout=0.3,
    freeze_encoder=True,
    lstm_hidden=128,
    model_id="facebook/hubert-base-ls960",
):
    encoder, hidden_size = _load_hubert(model_id)
    return HubertLSTMClassifier(
        encoder=encoder,
        encoder_dim=hidden_size,
        num_classes=num_classes,
        lstm_hidden=lstm_hidden,
        n_letters=n_letters,
        dropout=dropout,
        freeze_encoder=freeze_encoder,
    )
