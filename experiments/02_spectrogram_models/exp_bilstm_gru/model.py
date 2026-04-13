# -*- coding: utf-8 -*-
"""
BiLSTM + GRU на log-mel спектрограмме.
Вход: (batch, time, n_mels). BiLSTM → GRU → усреднение по времени → FC.

Модель не изменена по сравнению с checkpoint_3/exp_07_bilstm_gru/model.py.
"""
import torch
import torch.nn as nn


class BiLSTMGRUClassifier(nn.Module):
    def __init__(
        self,
        input_size,
        lstm_hidden=128,
        gru_hidden=64,
        num_layers_lstm=1,
        num_layers_gru=1,
        num_classes=2,
        n_letters=0,
        dropout=0.3,
    ):
        super().__init__()
        self.n_letters = n_letters
        self.lstm = nn.LSTM(
            input_size,
            lstm_hidden,
            num_layers=num_layers_lstm,
            batch_first=True,
            bidirectional=True,
            dropout=0,
        )
        self.gru = nn.GRU(
            lstm_hidden * 2,
            gru_hidden,
            num_layers=num_layers_gru,
            batch_first=True,
            dropout=dropout if num_layers_gru > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(gru_hidden + n_letters, num_classes)

    def forward(self, x, letters=None):
        out, _ = self.lstm(x)
        out, _ = self.gru(out)
        out = out.mean(dim=1)
        out = self.dropout(out)
        if self.n_letters > 0 and letters is not None:
            out = torch.cat([out, letters], dim=1)
        return self.fc(out)


def get_model(input_size, num_classes=2, lstm_hidden=128, gru_hidden=64,
              num_layers_lstm=1, num_layers_gru=1, n_letters=0, dropout=0.3, **kwargs):
    return BiLSTMGRUClassifier(
        input_size=input_size,
        lstm_hidden=lstm_hidden,
        gru_hidden=gru_hidden,
        num_layers_lstm=num_layers_lstm,
        num_layers_gru=num_layers_gru,
        num_classes=num_classes,
        n_letters=n_letters,
        dropout=dropout,
    )
