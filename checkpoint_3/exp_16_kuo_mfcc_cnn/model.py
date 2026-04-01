# -*- coding: utf-8 -*-
"""
2D CNN на MFCC «изображении» (Kuo-style, exp_15).
Вход: (B, 2, n_mfcc, T) — каналы: MFCC и delta.
"""
import torch
import torch.nn as nn


class MFCCCNN2d(nn.Module):
    def __init__(self, n_mfcc=20, n_frames=320, in_channels=2, num_classes=2, n_letters=0, channels=(32, 64, 128, 256), dropout=0.5):
        super().__init__()
        self.n_letters = n_letters
        layers = []
        c_in = in_channels
        for c in channels:
            layers += [
                nn.Conv2d(c_in, c, kernel_size=3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            c_in = c
        self.backbone = nn.Sequential(*layers)
        with torch.no_grad():
            x = torch.zeros(1, in_channels, n_mfcc, n_frames)
            for m in self.backbone:
                x = m(x)
            self._flat = x.numel()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self._flat + n_letters, num_classes),
        )

    def forward(self, x, letters=None):
        x = self.backbone(x)
        x = x.flatten(1)
        if self.n_letters > 0 and letters is not None:
            x = torch.cat([x, letters], dim=1)
        return self.classifier(x)


def get_model(n_mfcc=20, n_frames=320, in_channels=2, num_classes=2, n_letters=0, **kwargs):
    return MFCCCNN2d(n_mfcc=n_mfcc, n_frames=n_frames, in_channels=in_channels, num_classes=num_classes, n_letters=n_letters, **kwargs)
