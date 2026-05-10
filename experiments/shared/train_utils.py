# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import torch

from . import config


# ---------------------------------------------------------------------------
# Воспроизводимость
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Early Stopping и LR scheduler
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Ранняя остановка по val_f1_macro (mode='max') с patience эпох."""
    def __init__(self, patience: int = 10, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_value = -np.inf if mode == "max" else np.inf

    def step(self, value: float) -> bool:
        improved = value > self.best_value if self.mode == "max" else value < self.best_value
        if improved:
            self.best_value = value
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def should_stop(self) -> bool:
        return self.counter >= self.patience


def get_lr_scheduler(optimizer, mode: str = "max", factor: float = 0.5, patience: int = 5):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)


def save_best_checkpoint(model, path) -> None:
    torch.save(model.state_dict(), path)


def load_best_checkpoint(model, path, device=None) -> None:
    if device is None:
        device = next(model.parameters()).device
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)