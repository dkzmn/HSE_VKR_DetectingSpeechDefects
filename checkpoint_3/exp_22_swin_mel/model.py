# -*- coding: utf-8 -*-
"""
Swin Transformer на мел-спектрограмме (exp_22).
Вход: (B, 3, 224, 224) — mel, ресайзнутый и повторённый в 3 канала для timm.
"""
import torch
import torch.nn as nn
import timm


class SwinMelClassifier(nn.Module):
    def __init__(self, num_classes=2, n_letters=0, pretrained=True, model_name="swin_tiny_patch4_window7_224"):
        super().__init__()
        self.n_letters = n_letters
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=3,
        )
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            self._feat_dim = self.backbone(dummy).shape[1]
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self._feat_dim + n_letters, num_classes),
        )

    def forward(self, x, letters=None):
        # x: (B, 3, 224, 224)
        feat = self.backbone(x)
        if self.n_letters > 0 and letters is not None:
            feat = torch.cat([feat, letters], dim=1)
        return self.classifier(feat)


def get_model(num_classes=2, n_letters=0, pretrained=True, **kwargs):
    return SwinMelClassifier(num_classes=num_classes, n_letters=n_letters, pretrained=pretrained, **kwargs)
