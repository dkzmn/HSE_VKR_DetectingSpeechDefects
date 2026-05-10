# -*- coding: utf-8 -*-
"""
HuBERT (замороженный) + BiLSTM → FC для бинарной классификации.
Источник: PeerJ 2024 (Hybrid-Module Transformer).

Модель не изменена по сравнению с checkpoint_3/exp_13_hubert/model.py.
"""
import torch
import torch.nn as nn


class HubertLSTMClassifier(nn.Module):
    """HuBERT (frozen) → последовательность (B, T, D) → BiLSTM → mean → Dropout → FC."""

    def __init__(
        self,
        encoder,
        encoder_dim,
        num_classes=2,
        lstm_hidden=128,
        n_letters=0,
        dropout=0.3,
        freeze_encoder=True,
    ):
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.n_letters = n_letters
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
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2 + n_letters, num_classes)

    def forward(self, input_values, attention_mask=None, letters=None):
        ctx = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
        with ctx:
            out = self.encoder(input_values, attention_mask=attention_mask)
            hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        if hidden.dim() == 2:
            hidden = hidden.unsqueeze(1)
        out, _ = self.lstm(hidden)
        agg = out.mean(dim=1)
        agg = self.dropout(agg)
        if self.n_letters > 0 and letters is not None:
            agg = torch.cat([agg, letters], dim=1)
        return self.fc(agg)


def get_model(num_classes=2, n_letters=0, dropout=0.3, freeze_encoder=True,
              lstm_hidden=128, model_id="facebook/hubert-base-ls960"):
    from transformers import HubertModel
    encoder = HubertModel.from_pretrained(model_id)
    hidden_size = encoder.config.hidden_size
    return HubertLSTMClassifier(
        encoder=encoder,
        encoder_dim=hidden_size,
        num_classes=num_classes,
        lstm_hidden=lstm_hidden,
        n_letters=n_letters,
        dropout=dropout,
        freeze_encoder=freeze_encoder,
    )
