# -*- coding: utf-8 -*-
"""
FluentNet-style: две ветки — raw waveform (1D CNN) и mel-спектрограмма (2D CNN), конкатенация → FC.
Источник: Kourkounakis et al. «FluentNet: End-to-End Detection of Speech Disfluency with Deep Learning». arXiv:2009.11394.
"""
import torch
import torch.nn as nn


class Conv1DBranch(nn.Module):
    """Ветка по сырому waveform: 1D свёртки + downsampling → глобальная агрегация."""

    def __init__(self, in_len, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 80, stride=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, 3, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, stride=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        with torch.no_grad():
            x = torch.zeros(1, 1, in_len)
            for m in self.conv:
                x = m(x)
            self._out_dim = x.numel()
        self.proj = nn.Linear(self._out_dim, out_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.flatten(1)
        return self.proj(x)


class Conv2DMelBranch(nn.Module):
    """Ветка по mel-спектрограмме: 2D CNN → GAP → проекция."""

    def __init__(self, n_mels, n_frames, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        with torch.no_grad():
            x = torch.zeros(1, 1, n_mels, n_frames)
            for m in self.conv:
                x = m(x)
            self._flat = x.numel()
        self.proj = nn.Linear(self._flat, out_dim)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.proj(x)


class FluentNetClassifier(nn.Module):
    def __init__(self, waveform_len, n_mels, n_frames, branch_dim=128, num_classes=2, n_letters=0, dropout=0.3):
        super().__init__()
        self.n_letters = n_letters
        self.branch_wav = Conv1DBranch(waveform_len, out_dim=branch_dim)
        self.branch_mel = Conv2DMelBranch(n_mels, n_frames, out_dim=branch_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(branch_dim * 2 + n_letters, num_classes)

    def forward(self, waveform, mel, letters=None):
        h_wav = self.branch_wav(waveform)
        h_mel = self.branch_mel(mel)
        h = torch.cat([h_wav, h_mel], dim=1)
        if self.n_letters > 0 and letters is not None:
            h = torch.cat([h, letters], dim=1)
        h = self.dropout(h)
        return self.fc(h)


def get_model(waveform_len=160000, n_mels=80, n_frames=320, num_classes=2, branch_dim=128, n_letters=0, dropout=0.3, **kwargs):
    return FluentNetClassifier(
        waveform_len=waveform_len,
        n_mels=n_mels,
        n_frames=n_frames,
        branch_dim=branch_dim,
        num_classes=num_classes,
        n_letters=n_letters,
        dropout=dropout,
    )
