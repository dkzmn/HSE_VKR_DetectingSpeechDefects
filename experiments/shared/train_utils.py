# -*- coding: utf-8 -*-
"""
Утилиты обучения DL-моделей.

Изменения по сравнению с checkpoint_3/shared/train_utils.py:
  - Добавлена run_ml_cv() — кросс-валидация sklearn-пайплайна с оптимизацией порога
    на каждом фолде. Исправляет selection bias от единственного сплита.
  - EarlyStopping и чекпойнтинг без изменений (уже корректны).
"""
from __future__ import annotations

import time
import numpy as np
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV

from . import config
from .data_utils import get_cv_folds, build_feature_matrix
from .evaluate import find_optimal_threshold, evaluate, evaluate_cv_folds, print_cv_summary


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


# ---------------------------------------------------------------------------
# Кросс-валидация для ML-пайплайнов (GridSearchCV внутри каждого фолда)
# ---------------------------------------------------------------------------

def run_ml_cv(
    paths_trainval: np.ndarray,
    labels_trainval: np.ndarray,
    letters_trainval: np.ndarray,
    extractor,
    pipeline,
    param_grid: dict,
    n_splits: int = None,
    n_jobs: int = -1,
    verbose: bool = True,
) -> tuple[list[dict], dict]:
    """
    Полная кросс-валидация ML-пайплайна с оптимизацией порога.

    Алгоритм на каждом фолде:
      1. Извлечь признаки на train и val.
      2. Нормировать по train-статистике (если есть StandardScaler в pipeline).
      3. GridSearchCV по train-части фолда (scoring='f1_macro').
      4. Найти оптимальный порог по F1-bad на VAL-части фолда.
      5. Вычислить метрики на VAL-части с оптимальным порогом.

    Parameters
    ----------
    paths_trainval, labels_trainval, letters_trainval
        Данные, исключая holdout-тест (из get_holdout_split).
    extractor : callable
        Функция path -> np.ndarray признаков (без букв контроля).
    pipeline : sklearn Pipeline
        Должен принимать X (признаки + буквы) и предсказывать probability.
    param_grid : dict
        Параметры для GridSearchCV.
    n_splits : int
        Количество фолдов (по умолчанию config.CV_N_SPLITS).

    Returns
    -------
    fold_results : list[dict]   метрики каждого фолда
    cv_agg : dict               mean ± std по всем фолдам
    """
    n_splits = n_splits or config.CV_N_SPLITS
    fold_results = []
    t_total = time.perf_counter()

    for (
        paths_tr, paths_val,
        labels_tr, labels_val,
        letters_tr, letters_val,
        fold,
    ) in get_cv_folds(paths_trainval, labels_trainval, letters_trainval, n_splits):

        if verbose:
            print(f"\n{'='*50}")
            print(f"Фолд {fold + 1}/{n_splits}  (train={len(paths_tr)}, val={len(paths_val)})")

        # --- Признаки ---
        X_tr = build_feature_matrix(paths_tr, extractor, n_jobs=n_jobs)
        X_val = build_feature_matrix(paths_val, extractor, n_jobs=n_jobs)

        X_tr = np.hstack([X_tr, letters_tr])
        X_val = np.hstack([X_val, letters_val])

        # --- GridSearchCV на train-части фолда ---
        grid = GridSearchCV(
            pipeline, param_grid, cv=3, scoring="f1_macro",
            refit=True, n_jobs=n_jobs, verbose=0,
        )
        grid.fit(X_tr, labels_tr)
        clf = grid.best_estimator_

        if verbose:
            print(f"  Лучшие параметры: {grid.best_params_}")

        # --- Оптимизация порога по val ---
        y_proba_val = clf.predict_proba(X_val)[:, config.CLASS_BAD]
        thr = find_optimal_threshold(labels_val, y_proba_val)

        # --- Метрики на val с оптимальным порогом ---
        metrics = evaluate(labels_val, y_proba_val, threshold=thr, verbose=verbose)
        fold_results.append(metrics)

    cv_agg = evaluate_cv_folds(fold_results)

    if verbose:
        print(f"\n{'='*50}")
        print(f"CV сводка ({n_splits} фолдов):")
        print_cv_summary(cv_agg)
        print(f"Общее время: {time.perf_counter() - t_total:.1f} с")

    return fold_results, cv_agg


# ---------------------------------------------------------------------------
# Финальное обучение после CV (на всём train+val, оценка на holdout test)
# ---------------------------------------------------------------------------

def fit_final_ml(
    paths_trainval: np.ndarray,
    labels_trainval: np.ndarray,
    letters_trainval: np.ndarray,
    paths_test: np.ndarray,
    labels_test: np.ndarray,
    letters_test: np.ndarray,
    extractor,
    pipeline,
    best_params: dict,
    cv_threshold: float = 0.5,
    n_jobs: int = -1,
    verbose: bool = True,
) -> dict:
    """
    После CV обучает модель на полном train+val с лучшими params,
    оценивает на holdout test с порогом из CV.

    Returns
    -------
    dict с метриками теста
    """
    X_trainval = build_feature_matrix(paths_trainval, extractor, n_jobs=n_jobs)
    X_test = build_feature_matrix(paths_test, extractor, n_jobs=n_jobs)
    X_trainval = np.hstack([X_trainval, letters_trainval])
    X_test = np.hstack([X_test, letters_test])

    pipeline.set_params(**best_params)
    pipeline.fit(X_trainval, labels_trainval)

    y_proba_test = pipeline.predict_proba(X_test)[:, config.CLASS_BAD]
    metrics = evaluate(labels_test, y_proba_test, threshold=cv_threshold, verbose=verbose)

    if verbose:
        print(f"\nФинальные метрики на holdout test (threshold={cv_threshold:.2f}):")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")

    return metrics
