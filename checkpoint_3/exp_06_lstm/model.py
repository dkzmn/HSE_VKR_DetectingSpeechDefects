# -*- coding: utf-8 -*-
"""
BiLSTM по последовательности кадров (MFCC) для бинарной классификации good/bad.
Агрегация: усреднение выхода по времени → FC → num_classes.
"""
import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=2, n_letters=0, dropout=0.3):
        super().__init__()
        self.n_letters = n_letters
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2 + n_letters, num_classes)

    def forward(self, x, letters=None):
        # x: (B, T, F)
        out, _ = self.lstm(x)
        out = out.mean(dim=1)
        out = self.dropout(out)
        if self.n_letters > 0 and letters is not None:
            out = torch.cat([out, letters], dim=1)
        return self.fc(out)


def get_model(input_size, num_classes=2, hidden_size=128, num_layers=2, n_letters=0, dropout=0.3, **kwargs):
    return BiLSTMClassifier(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=num_classes,
        n_letters=n_letters,
        dropout=dropout,
    )
