# -*- coding: utf-8 -*-
"""
2D CNN на мел-спектрограмме для бинарной классификации (good/bad).
Лёгкая архитектура: 4 блока Conv2d+BN+ReLU+MaxPool, Global Average Pooling, FC.
Поддержка признаков букв из CSV: конкатенация с выходом backbone перед классификатором.
"""
import torch
import torch.nn as nn


class MelCNN2d(nn.Module):
    def __init__(self, n_mels=80, n_frames=320, num_classes=2, n_letters=0, channels=(32, 64, 128, 256), dropout=0.5):
        super().__init__()
        self.n_mels = n_mels
        self.n_frames = n_frames
        self.n_letters = n_letters
        layers = []
        in_c = 1
        for c in channels:
            layers += [
                nn.Conv2d(in_c, c, kernel_size=3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            ]
            in_c = c
        self.backbone = nn.Sequential(*layers)
        with torch.no_grad():
            x = torch.zeros(1, 1, n_mels, n_frames)
            for m in self.backbone:
                x = m(x)
            self._flat = x.numel()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self._flat + n_letters, num_classes),
        )

    def forward(self, x, letters=None):
        # x: (B, 1, n_mels, n_frames)
        x = self.backbone(x)
        x = x.flatten(1)
        if self.n_letters > 0 and letters is not None:
            x = torch.cat([x, letters], dim=1)
        return self.classifier(x)


def get_model(n_mels=80, n_frames=320, num_classes=2, n_letters=0, **kwargs):
    return MelCNN2d(n_mels=n_mels, n_frames=n_frames, num_classes=num_classes, n_letters=n_letters, **kwargs)
