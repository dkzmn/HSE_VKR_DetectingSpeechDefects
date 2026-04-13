# -*- coding: utf-8 -*-
"""
CNN-BiLSTM-Transformer с cross-attention для классификации степени/типа дефекта речи.
Источник: «A Hybrid Cross-Attentive CNN-BiLSTM-Transformer Network for Dysarthria Severity Classification».
Nature Scientific Reports, 2025.
Вход: мел-спектрограмма (batch, 1, n_mels, T).
Архитектура: CNN → последовательность → BiLSTM (K, V) → обучаемый Query → cross-attention → FC.
"""
import torch
import torch.nn as nn


class CrossAttnCNNBiLSTM(nn.Module):
    def __init__(
        self,
        n_mels=80,
        n_frames=320,
        cnn_channels=(32, 64, 128, 256),
        d_model=128,
        lstm_hidden=64,
        nhead=4,
        num_classes=2,
        n_letters=0,
        dropout=0.4,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_letters = n_letters
        self.lstm_hidden = lstm_hidden
        self.kv_dim = 2 * lstm_hidden  # BiLSTM output

        # CNN: (B, 1, n_mels, T) -> (B, C, H', T')
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
            self._cnn_out = in_c * self._cnn_H
        self.proj = nn.Linear(self._cnn_out, d_model)

        # BiLSTM: (B, T', d_model) -> (B, T', 2*lstm_hidden)
        self.lstm = nn.LSTM(
            d_model,
            lstm_hidden,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
            num_layers=1,
        )

        # Обучаемый вектор запроса для cross-attention: (1, 1, kv_dim)
        self.query = nn.Parameter(torch.zeros(1, 1, self.kv_dim))
        nn.init.xavier_uniform_(self.query)

        # Cross-attention: Query (B, 1, kv_dim), Key/Value (B, T', kv_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.kv_dim,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.kv_dim + n_letters, num_classes)

    def forward(self, x, letters=None):
        # x: (B, 1, n_mels, T)
        x = self.cnn(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, W, C * H)
        x = self.proj(x)

        lstm_out, _ = self.lstm(x)
        kv = lstm_out

        q = self.query.expand(B, -1, -1)
        attn_out, _ = self.cross_attn(q, kv, kv)
        out = attn_out.squeeze(1)
        out = self.dropout(out)
        if self.n_letters > 0 and letters is not None:
            out = torch.cat([out, letters], dim=1)
        return self.fc(out)


def get_model(n_mels=80, n_frames=320, num_classes=2, n_letters=0, **kwargs):
    return CrossAttnCNNBiLSTM(
        n_mels=n_mels,
        n_frames=n_frames,
        num_classes=num_classes,
        n_letters=n_letters,
        **kwargs,
    )
