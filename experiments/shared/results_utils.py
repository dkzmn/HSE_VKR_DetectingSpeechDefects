# -*- coding: utf-8 -*-
"""
Утилиты сохранения результатов экспериментов.

Изменения по сравнению с checkpoint_3/shared/results_utils.py:
  - Добавлены поля threshold (оптимальный порог) и cv_std_* (дисперсия по фолдам).
  - save_cv_result() сохраняет результаты кросс-валидации (mean ± std).
  - save_result_csv() автоматически логирует в активный MLflow run (если есть).
"""
import csv
from pathlib import Path

from . import config

try:
    import mlflow as _mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


# ---------------------------------------------------------------------------
# Колонки результата
# ---------------------------------------------------------------------------
_COLUMNS = [
    "experiment_id",
    "experiment_name",
    "model",
    "accuracy",
    "f1_macro",
    "f1_bad",
    "roc_auc",
    "precision_bad",
    "recall_bad",
    "threshold",          # оптимальный порог (nan если не оптимизировался)
    "cv_f1_bad_std",      # std F1-bad по фолдам (nan для одиночного прогона)
    "cv_f1_macro_std",    # std F1-macro по фолдам
    "notes",
    "num_params",
    "train_time_sec",
]


def save_result_csv(
    exp_dir: Path,
    experiment_id: str,
    experiment_name: str,
    model: str,
    accuracy: float,
    f1_macro: float = None,
    f1_bad: float = None,
    roc_auc: float = None,
    precision_bad: float = None,
    recall_bad: float = None,
    threshold: float = None,
    cv_f1_bad_std: float = None,
    cv_f1_macro_std: float = None,
    notes: str = "",
    num_params: int = None,
    train_time_sec: float = None,
) -> Path:
    """
    Сохраняет результат одного эксперимента в <exp_dir>/result.csv.
    Перезаписывает файл при каждом вызове (не append).
    """
    result_path = exp_dir / "result.csv"
    row = {
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "model": model,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_bad": f1_bad,
        "roc_auc": roc_auc,
        "precision_bad": precision_bad,
        "recall_bad": recall_bad,
        "threshold": threshold,
        "cv_f1_bad_std": cv_f1_bad_std,
        "cv_f1_macro_std": cv_f1_macro_std,
        "notes": notes,
        "num_params": num_params,
        "train_time_sec": train_time_sec,
    }
    with open(result_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_COLUMNS)
        w.writeheader()
        w.writerow(row)

    # --- MLflow: логируем если есть активный run ---
    if _MLFLOW_AVAILABLE:
        try:
            if _mlflow.active_run() is not None:
                # Метрики
                mlflow_metrics = {}
                for key in ("accuracy", "f1_macro", "f1_bad", "roc_auc",
                            "precision_bad", "recall_bad"):
                    val = row.get(key)
                    if val is not None and val == val:  # исключаем None и nan
                        mlflow_metrics[f"test_{key}"] = float(val)
                if mlflow_metrics:
                    _mlflow.log_metrics(mlflow_metrics)

                # Параметры (threshold, cv_std и т.п.)
                mlflow_params = {}
                for key in ("threshold", "cv_f1_bad_std", "cv_f1_macro_std",
                            "num_params", "train_time_sec"):
                    val = row.get(key)
                    if val is not None and val == val:
                        mlflow_params[key] = val
                if mlflow_params:
                    _mlflow.log_params(mlflow_params)

                # Артефакт result.csv
                _mlflow.log_artifact(str(result_path))
        except Exception:
            pass  # не ломаем сохранение если MLflow недоступен

    return result_path


def save_cv_result(
    exp_dir: Path,
    experiment_id: str,
    experiment_name: str,
    model: str,
    cv_agg: dict,
    notes: str = "",
    num_params: int = None,
    train_time_sec: float = None,
) -> Path:
    """
    Удобная обёртка для сохранения результатов кросс-валидации.
    cv_agg — dict, возвращённый evaluate.evaluate_cv_folds().
    """
    return save_result_csv(
        exp_dir=exp_dir,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        model=model,
        accuracy=cv_agg.get("accuracy_mean"),
        f1_macro=cv_agg.get("f1_macro_mean"),
        f1_bad=cv_agg.get("f1_bad_mean"),
        roc_auc=cv_agg.get("roc_auc_mean"),
        precision_bad=cv_agg.get("precision_bad_mean"),
        recall_bad=cv_agg.get("recall_bad_mean"),
        threshold=cv_agg.get("threshold_mean"),
        cv_f1_bad_std=cv_agg.get("f1_bad_std"),
        cv_f1_macro_std=cv_agg.get("f1_macro_std"),
        notes=notes,
        num_params=num_params,
        train_time_sec=train_time_sec,
    )
