# -*- coding: utf-8 -*-
"""
Утилиты для обучения DL-моделей: early stopping, сохранение лучшего чекпоинта по val_f1,
планировщик lr, градиентный клип.
"""
import numpy as np
import torch


# Рекомендуемый max_norm для clip_grad_norm_
DEFAULT_GRAD_CLIP = 1.0


def set_seed(seed=42):
    """Фиксирует семена для воспроизводимости."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class EarlyStopping:
    """Ранняя остановка по val_f1_macro (mode='max') с patience эпох."""
    def __init__(self, patience=10, mode="max"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_value = -np.inf if mode == "max" else np.inf

    def step(self, value):
        """Возвращает True, если нужно остановить обучение (нет улучшения patience эпох)."""
        if self.mode == "max":
            improved = value > self.best_value
        else:
            improved = value < self.best_value
        if improved:
            self.best_value = value
            self.counter = 0
            return False  # не останавливаем
        self.counter += 1
        return self.counter >= self.patience  # остановиться только после patience эпох без улучшения

    def should_stop(self):
        return self.counter >= self.patience


def get_lr_scheduler(optimizer, mode="max", factor=0.5, patience=5, verbose=True):
    """
    ReduceLROnPlateau по val_f1_macro (mode='max'): при отсутствии улучшения
    patience эпох умножает lr на factor. После каждой эпохи вызывать scheduler.step(val_f1).
    """
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=mode, factor=factor, patience=patience
    )


def save_best_checkpoint(model, path):
    """Сохраняет state_dict модели в path (лучшая эпоха по val_f1)."""
    torch.save(model.state_dict(), path)


def load_best_checkpoint(model, path, device=None):
    """Загружает state_dict в модель перед оценкой на тесте."""
    if device is None:
        device = next(model.parameters()).device
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
