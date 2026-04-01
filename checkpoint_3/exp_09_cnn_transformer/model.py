# -*- coding: utf-8 -*-
"""
CNN + Transformer в стиле AutoDEAP: mel-спектрограмма → CNN (патчи по времени) → Transformer encoder → mean pool → FC.
Вход: (batch, 1, n_mels, T). Один поток по mel.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (B, T, D)
        return x + self.pe[:, : x.size(1), :]


class CNNTransformerClassifier(nn.Module):
    def __init__(
        self,
        n_mels=80,
        n_frames=320,
        cnn_channels=(32, 64, 128, 256),
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        dim_feedforward=512,
        num_classes=2,
        n_letters=0,
        dropout=0.3,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_letters = n_letters
        self.n_frames = n_frames
        layers = []
        in_c = 1
        for c in cnn_channels:
            layers += [
                nn.Conv2d(in_c, c, kernel_size=3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_c = c
        self.cnn = nn.Sequential(*layers)
        with torch.no_grad():
            x = torch.zeros(1, 1, n_mels, n_frames)
            for m in self.cnn:
                x = m(x)
            self._cnn_H = x.size(2)
            self._cnn_W = x.size(3)
            self._seq_len = self._cnn_W
            self._cnn_out = in_c * self._cnn_H
        self.proj = nn.Linear(self._cnn_out, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=self._seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model + n_letters, num_classes)

    def forward(self, x, letters=None):
        # x: (B, 1, n_mels, T)
        x = self.cnn(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, W, C * H)
        x = self.proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        if self.n_letters > 0 and letters is not None:
            x = torch.cat([x, letters], dim=1)
        return self.fc(x)


def get_model(n_mels=80, n_frames=320, num_classes=2, n_letters=0, **kwargs):
    return CNNTransformerClassifier(n_mels=n_mels, n_frames=n_frames, num_classes=num_classes, n_letters=n_letters, **kwargs)
