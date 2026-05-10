# -*- coding: utf-8 -*-
import csv
from pathlib import Path

from . import config


_COLUMNS = [
    "experiment_id",
    "experiment_name",
    "accuracy",
    "f1_macro",
    "f1_bad",
    "roc_auc",
    "precision_bad",
    "recall_bad",
    "threshold",
    "embed_dim",
    "embed_dim_note",
    "notes",
    "num_params",
    "train_time_sec",
]


def save_result_csv(
    exp_dir: Path,
    experiment_id: str,
    experiment_name: str,
    accuracy: float,
    f1_macro: float = None,
    f1_bad: float = None,
    roc_auc: float = None,
    precision_bad: float = None,
    recall_bad: float = None,
    threshold: float = None,
    embed_dim: int = None,
    embed_dim_note: str = None,
    notes: str = "",
    num_params: int = None,
    train_time_sec: float = None,
    append: bool = False,
) -> Path:
    """
    Сохраняет результат одного эксперимента в <exp_dir>/result.csv.
    append=False (по умолчанию): перезаписывает файл.
    append=True: добавляет строку без заголовка (для мульти-модельных ноутбуков).
    """
    result_path = exp_dir / "result.csv"
    row = {
        "experiment_id":   experiment_id,
        "experiment_name": experiment_name,
        "accuracy":        accuracy,
        "f1_macro":        f1_macro,
        "f1_bad":          f1_bad,
        "roc_auc":         roc_auc,
        "precision_bad":   precision_bad,
        "recall_bad":      recall_bad,
        "threshold":       threshold,
        "embed_dim":       embed_dim,
        "embed_dim_note":  embed_dim_note,
        "notes":           notes,
        "num_params":      num_params,
        "train_time_sec":  train_time_sec,
    }
    mode = "a" if append else "w"
    with open(result_path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        if not append:
            w.writeheader()
        w.writerow(row)
    return result_path


def save_cv_result(
    exp_dir: Path,
    experiment_id: str,
    experiment_name: str,
    cv_agg: dict,
    notes: str = "",
    num_params: int = None,
    train_time_sec: float = None,
) -> Path:
    """Обёртка: сохраняет средние значения CV-метрик."""
    return save_result_csv(
        exp_dir=exp_dir,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        accuracy=cv_agg.get("accuracy_mean"),
        f1_macro=cv_agg.get("f1_macro_mean"),
        f1_bad=cv_agg.get("f1_bad_mean"),
        roc_auc=cv_agg.get("roc_auc_mean"),
        precision_bad=cv_agg.get("precision_bad_mean"),
        recall_bad=cv_agg.get("recall_bad_mean"),
        threshold=cv_agg.get("threshold_mean"),
        notes=notes,
        num_params=num_params,
        train_time_sec=train_time_sec,
    )