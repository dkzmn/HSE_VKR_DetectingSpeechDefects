# -*- coding: utf-8 -*-
"""
Полная модель из статьи Huang et al. PeerJ 2024:
Hybrid-Module Transformer: HuBERT + LSTM + ResNet-50 + Transformer.

Ветка 1: Log-Mel спектрограмма → ResNet-50 → 2048-dim.
Ветка 2: Сырой waveform → HuBERT → LSTM → 768-dim (mean).
Слияние: concat 2048+768=2816 → проекция → 4-слойный Transformer encoder → FC → num_classes.
"""
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision.models import resnet50, ResNet50_Weights


def _load_hubert(model_id="facebook/hubert-base-ls960"):
    from transformers import HubertModel
    model = HubertModel.from_pretrained(model_id)
    return model, model.config.hidden_size


class ResNet50Branch(nn.Module):
    """Log-Mel (1, H, W) → 3 канала, интерполяция до 224×224 → ResNet-50 без FC → 2048."""

    def __init__(self, target_size=224, pretrained=True):
        super().__init__()
        try:
            if pretrained and ResNet50_Weights is not None:
                resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                resnet = resnet50(weights=None)
        except (TypeError, AttributeError):
            resnet = resnet50(pretrained=pretrained)
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        self._out_dim = 2048
        self.target_size = target_size

    @property
    def out_dim(self):
        return self._out_dim

    def forward(self, mel):
        # mel: (B, 1, H, W) → (B, 3, target_size, target_size)
        if mel.size(1) == 1:
            mel = mel.repeat(1, 3, 1, 1)
        if mel.shape[2] != self.target_size or mel.shape[3] != self.target_size:
            mel = nn.functional.interpolate(
                mel, size=(self.target_size, self.target_size),
                mode="bilinear", align_corners=False
            )
        out = self.features(mel)
        return out.flatten(1)


class HubertLSTMBranch(nn.Module):
    """Waveform → HuBERT → last_hidden_state → LSTM → mean → 768-dim."""

    def __init__(self, encoder, encoder_dim, lstm_hidden=128, freeze_encoder=True):
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = encoder_dim
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
        self._out_dim = lstm_hidden * 2

    @property
    def out_dim(self):
        return self._out_dim

    def forward(self, input_values, attention_mask=None):
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
        return out.mean(dim=1)


class HybridModuleTransformer(nn.Module):
    """
    HuBERT-LSTM (768) + ResNet-50 (2048) → concat 2816 → проекция → 4-layer Transformer → FC.
    Статья: Huang et al. PeerJ CS 2024.
    """

    def __init__(
        self,
        resnet_branch,
        hubert_lstm_branch,
        num_classes=2,
        n_letters=0,
        fusion_dim=2816,
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=1024,
        dropout=0.3,
    ):
        super().__init__()
        self.resnet_branch = resnet_branch
        self.hubert_lstm_branch = hubert_lstm_branch
        self.n_letters = n_letters
        self.d_model = d_model

        self.fusion_proj = nn.Linear(fusion_dim, d_model)
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model + n_letters, num_classes)

    def forward(self, input_values, mel, attention_mask=None, letters=None):
        # ResNet: mel (B, 1, H, W)
        resnet_out = self.resnet_branch(mel)  # (B, 2048)
        # HuBERT-LSTM: waveform
        hubert_out = self.hubert_lstm_branch(input_values, attention_mask=attention_mask)  # (B, 768)
        fused = torch.cat([resnet_out, hubert_out], dim=1)  # (B, 2816)
        # Проекция и один токен для Transformer
        x = self.fusion_proj(fused).unsqueeze(1)  # (B, 1, d_model)
        x = self.transformer(x)  # (B, 1, d_model)
        x = x.squeeze(1)  # (B, d_model)
        x = self.dropout(x)
        if self.n_letters > 0 and letters is not None:
            x = torch.cat([x, letters], dim=1)
        return self.fc(x)


def get_model(
    num_classes=2,
    n_letters=0,
    dropout=0.3,
    freeze_hubert=True,
    lstm_hidden=128,
    hubert_model_id="facebook/hubert-base-ls960",
    resnet_target_size=224,
    resnet_pretrained=True,
    d_model=256,
    nhead=8,
    num_encoder_layers=4,
):
    """Собирает полную модель: ResNet-50 + HuBERT-LSTM + Transformer."""
    # ResNet-50 ветка
    resnet_branch = ResNet50Branch(target_size=resnet_target_size, pretrained=resnet_pretrained)
    resnet_out = resnet_branch.out_dim  # 2048

    # HuBERT-LSTM ветка
    encoder, hidden_size = _load_hubert(hubert_model_id)
    hubert_branch = HubertLSTMBranch(
        encoder=encoder,
        encoder_dim=hidden_size,
        lstm_hidden=lstm_hidden,
        freeze_encoder=freeze_hubert,
    )
    hubert_out = hubert_branch.out_dim  # lstm_hidden*2

    fusion_dim = resnet_out + hubert_out  # 2048 + 768 = 2816
    model = HybridModuleTransformer(
        resnet_branch=resnet_branch,
        hubert_lstm_branch=hubert_branch,
        num_classes=num_classes,
        n_letters=n_letters,
        fusion_dim=fusion_dim,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=1024,
        dropout=dropout,
    )
    return model
