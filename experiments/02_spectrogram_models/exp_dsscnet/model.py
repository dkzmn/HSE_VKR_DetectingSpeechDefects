# -*- coding: utf-8 -*-
"""
DSSCNet: Residual блоки + Squeeze-Excitation на мел-спектрограмме.
Источник: arXiv:2509.13442 (Speaker-Independent Dysarthric Speech Severity).
Вход: (batch, 1, n_mels, T).

Модель не изменена по сравнению с checkpoint_3/exp_10_dsscnet/model.py.
"""
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    """Squeeze-and-Excitation: GAP -> FC -> ReLU -> FC -> sigmoid -> scale по каналам."""

    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 8), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        s = self.pool(x).view(b, c)
        s = self.fc(s).view(b, c, 1, 1)
        return x * s


class SEResBlock(nn.Module):
    """Residual блок с двумя 3x3 свёртками и SE после второго."""

    def __init__(self, in_c, out_c, reduction=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = SEBlock(out_c, reduction=reduction)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv2d(in_c, out_c, 1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.relu(out + identity)
        return out


class DSSCNet(nn.Module):
    def __init__(self, n_mels=80, n_frames=320, num_classes=2,
                 base_c=64, num_blocks=4, reduction=8, n_letters=0, dropout=0.3):
        super().__init__()
        self.n_letters = n_letters
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_c, 7, stride=2, padding=3),
            nn.BatchNorm2d(base_c),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        layers = []
        in_c = base_c
        for _ in range(num_blocks):
            layers.append(SEResBlock(in_c, base_c, reduction=reduction))
            in_c = base_c
        self.blocks = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_c + n_letters, num_classes)

    def forward(self, x, letters=None):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        if self.n_letters > 0 and letters is not None:
            x = torch.cat([x, letters], dim=1)
        return self.fc(x)


def get_model(n_mels=80, n_frames=320, num_classes=2, n_letters=0, **kwargs):
    return DSSCNet(n_mels=n_mels, n_frames=n_frames,
                   num_classes=num_classes, n_letters=n_letters, **kwargs)
