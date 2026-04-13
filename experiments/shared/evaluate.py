# -*- coding: utf-8 -*-
"""
Единая функция оценки для всех экспериментов.

Ключевые исправления по сравнению с checkpoint_3:
  1. Оптимизация порога классификации по F1-bad на валидационной выборке
     (вместо фиксированного threshold=0.5 для всех моделей).
  2. Все метрики считаются одной функцией — нет расхождений между экспериментами.
  3. Поддержка CV: evaluate_cv_folds принимает список результатов фолдов
     и возвращает mean ± std по каждой метрике.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    classification_report,
)

from . import config


# ---------------------------------------------------------------------------
# Оптимизация порога
# ---------------------------------------------------------------------------

def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Ищет порог в config.THRESHOLD_GRID, максимизирующий F1-bad на переданной выборке.
    Вызывать только на ВАЛИДАЦИОННОМ множестве, а потом применять порог на ТЕСТОВОМ.

    Parameters
    ----------
    y_true : массив истинных меток (0/1)
    y_proba : массив вероятностей класса BAD (0..1)

    Returns
    -------
    Оптимальный порог (float)
    """
    best_thr, best_f1 = 0.5, -1.0
    for thr in config.THRESHOLD_GRID:
        preds = (y_proba >= thr).astype(int)
        f1 = f1_score(y_true, preds, pos_label=config.CLASS_BAD, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr


# ---------------------------------------------------------------------------
# Основная функция оценки
# ---------------------------------------------------------------------------

def evaluate(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    Вычисляет полный набор метрик для одного прогона.

    Parameters
    ----------
    y_true     : истинные метки
    y_proba    : вероятности класса BAD
    threshold  : порог бинаризации (по умолчанию 0.5; для лучших результатов
                 передавать порог, найденный find_optimal_threshold на val-выборке)
    verbose    : печатать ли classification_report

    Returns
    -------
    dict с ключами: accuracy, f1_macro, f1_bad, roc_auc,
                    precision_bad, recall_bad, threshold
    """
    y_pred = (y_proba >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_bad = f1_score(y_true, y_pred, pos_label=config.CLASS_BAD, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except ValueError:
        auc = float("nan")
    prec_bad = precision_score(y_true, y_pred, pos_label=config.CLASS_BAD, zero_division=0)
    rec_bad = recall_score(y_true, y_pred, pos_label=config.CLASS_BAD, zero_division=0)

    if verbose:
        print(classification_report(y_true, y_pred, target_names=config.CLASS_NAMES))
        print(f"Threshold : {threshold:.2f}")
        print(f"Accuracy  : {acc:.4f}")
        print(f"F1-macro  : {f1_macro:.4f}")
        print(f"F1-bad    : {f1_bad:.4f}")
        print(f"ROC-AUC   : {auc:.4f}")

    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_bad": f1_bad,
        "roc_auc": auc,
        "precision_bad": prec_bad,
        "recall_bad": rec_bad,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Агрегация результатов кросс-валидации
# ---------------------------------------------------------------------------

def evaluate_cv_folds(fold_results: list[dict]) -> dict:
    """
    Принимает список dict (один на фолд, из evaluate()) и возвращает
    mean и std по каждой метрике.

    Returns
    -------
    dict вида {
        "accuracy_mean": ..., "accuracy_std": ...,
        "f1_macro_mean": ..., "f1_macro_std": ...,
        ...
    }
    """
    metrics = ["accuracy", "f1_macro", "f1_bad", "roc_auc",
               "precision_bad", "recall_bad", "threshold"]
    agg = {}
    for m in metrics:
        vals = np.array([r[m] for r in fold_results if not np.isnan(r.get(m, np.nan))])
        agg[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")
        agg[f"{m}_std"] = float(np.std(vals)) if len(vals) else float("nan")
    return agg


def print_cv_summary(agg: dict) -> None:
    """Печатает сводную таблицу кросс-валидации."""
    metrics = ["accuracy", "f1_macro", "f1_bad", "roc_auc", "precision_bad", "recall_bad"]
    print(f"{'Метрика':<20} {'mean':>8} {'± std':>8}")
    print("-" * 40)
    for m in metrics:
        mean = agg.get(f"{m}_mean", float("nan"))
        std = agg.get(f"{m}_std", float("nan"))
        print(f"{m:<20} {mean:>8.4f} {std:>8.4f}")
    print(f"\nСредний оптимальный порог: {agg.get('threshold_mean', 0.5):.2f}")
