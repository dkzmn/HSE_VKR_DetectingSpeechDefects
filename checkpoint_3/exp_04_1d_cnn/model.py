# -*- coding: utf-8 -*-
"""
1D CNN по сырому waveform для бинарной классификации (дефект речи).
Несколько блоков Conv1d → BatchNorm → ReLU → MaxPool1d, затем GlobalAvgPool1d и FC.
Источник: обзор представлений аудио (Habr); Hershey et al., arXiv:1609.09430.
Вход: сырой waveform фиксированной длины (например, 10 с при 16 kHz).
"""
import torch
import torch.nn as nn


class WaveformCNN1d(nn.Module):
    def __init__(
        self,
        in_length,
        num_classes=2,
        channels=(32, 64, 128, 256),
        kernel_sizes=(80, 3, 3, 3),
        strides=(4, 2, 2, 2),
        pool_sizes=(4, 2, 2, 2),
        n_letters=0,
        dropout=0.5,
    ):
        super().__init__()
        self.n_letters = n_letters
        assert len(channels) == len(kernel_sizes) == len(strides) == len(pool_sizes)
        layers = []
        in_c = 1
        for c, k, s, p in zip(channels, kernel_sizes, strides, pool_sizes):
            layers += [
                nn.Conv1d(in_c, c, kernel_size=k, stride=s),
                nn.BatchNorm1d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(p),
            ]
            in_c = c
        self.backbone = nn.Sequential(*layers)
        with torch.no_grad():
            x = torch.zeros(1, 1, in_length)
            for m in self.backbone:
                x = m(x)
            self._out_channels = x.size(1)
            self._out_len = x.size(2)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self._out_channels + n_letters, num_classes),
        )

    def forward(self, x, letters=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        if self.n_letters > 0 and letters is not None:
            x = torch.cat([x, letters], dim=1)
        return self.classifier(x)


def get_model(
    in_length=160000,
    num_classes=2,
    channels=(32, 64, 128, 256),
    n_letters=0,
    dropout=0.5,
    **kwargs,
):
    return WaveformCNN1d(
        in_length=in_length,
        num_classes=num_classes,
        channels=channels,
        n_letters=n_letters,
        dropout=dropout,
        **kwargs,
    )
