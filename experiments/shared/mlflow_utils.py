# -*- coding: utf-8 -*-
"""
Обёртки над MLflow для экспериментов с детекцией дефектов речи.

Использование в ноутбуке:
    from shared.mlflow_utils import start_run, log_epoch, log_cv_fold

    with start_run("exp_mfcc_svm", group="01_baselines"):
        mlflow.log_params({...})
        for epoch in ...:
            log_epoch(epoch, train_loss=..., val_f1=...)
        # save_result_csv() автоматически залогирует метрики если run активен
"""
from __future__ import annotations

import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Any

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False

from . import config


# ---------------------------------------------------------------------------
# Инициализация
# ---------------------------------------------------------------------------

def setup(tracking_uri: str | None = None) -> None:
    """
    Устанавливает tracking URI и создаёт эксперименты для каждой группы.
    Вызывать один раз, например в начале results_summary.ipynb.
    """
    if not _MLFLOW_AVAILABLE:
        return
    uri = tracking_uri or config.MLFLOW_TRACKING_URI
    mlflow.set_tracking_uri(uri)


def _get_git_info() -> dict[str, str]:
    """Возвращает {'git_commit': '...', 'git_branch': '...'} или пустой dict."""
    info: dict[str, str] = {}
    try:
        info["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        info["git_branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# Контекстный менеджер запуска
# ---------------------------------------------------------------------------

@contextmanager
def start_run(run_name: str, group: str):
    """
    Контекстный менеджер для одного эксперимента.

    - Создаёт (или находит) MLflow-эксперимент с именем group.
    - Стартует run с именем run_name.
    - Автоматически логирует git commit + branch.
    - При выходе (в том числе при исключении) корректно закрывает run.

    Использование:
        with start_run("exp_whisper_svm", group="03_pretrained_frozen"):
            mlflow.log_params(...)
            # ... обучение ...
    """
    if not _MLFLOW_AVAILABLE:
        # MLflow не установлен — просто выполняем блок без логирования
        yield None
        return

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(group)

    with mlflow.start_run(run_name=run_name) as run:
        # Автоматически логируем git-состояние
        git_info = _get_git_info()
        if git_info:
            mlflow.set_tags(git_info)
        mlflow.set_tag("group", group)
        yield run


# ---------------------------------------------------------------------------
# Логирование в процессе обучения DL
# ---------------------------------------------------------------------------

def log_epoch(
    epoch: int,
    train_loss: float | None = None,
    val_f1_macro: float | None = None,
    val_f1_bad: float | None = None,
    lr: float | None = None,
    **extra_metrics,
) -> None:
    """
    Логирует метрики одной эпохи в активный MLflow run.
    step = epoch (начиная с 0).

    Пример вызова внутри цикла обучения:
        log_epoch(epoch, train_loss=loss, val_f1_macro=val_f1, val_f1_bad=f1_bad)
    """
    if not _MLFLOW_AVAILABLE:
        return
    try:
        run = mlflow.active_run()
        if run is None:
            return
        metrics: dict[str, float] = {}
        if train_loss is not None:
            metrics["train_loss"] = float(train_loss)
        if val_f1_macro is not None:
            metrics["val_f1_macro"] = float(val_f1_macro)
        if val_f1_bad is not None:
            metrics["val_f1_bad"] = float(val_f1_bad)
        if lr is not None:
            metrics["lr"] = float(lr)
        metrics.update({k: float(v) for k, v in extra_metrics.items()})
        if metrics:
            mlflow.log_metrics(metrics, step=epoch)
    except Exception:
        pass  # не ломаем обучение если MLflow недоступен


def log_cv_fold(
    fold: int,
    f1_bad: float,
    f1_macro: float,
    roc_auc: float | None = None,
    threshold: float | None = None,
) -> None:
    """
    Логирует метрики одного фолда CV. step=fold.
    """
    if not _MLFLOW_AVAILABLE:
        return
    try:
        if mlflow.active_run() is None:
            return
        metrics: dict[str, float] = {
            "cv_f1_bad":   float(f1_bad),
            "cv_f1_macro": float(f1_macro),
        }
        if roc_auc is not None:
            metrics["cv_roc_auc"] = float(roc_auc)
        if threshold is not None:
            metrics["cv_threshold"] = float(threshold)
        mlflow.log_metrics(metrics, step=fold)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Логирование финальных метрик и артефактов
# ---------------------------------------------------------------------------

def log_final_metrics(metrics: dict[str, Any]) -> None:
    """
    Логирует финальные метрики теста в активный run.
    metrics — dict из evaluate() или evaluate_cv_folds().
    """
    if not _MLFLOW_AVAILABLE:
        return
    try:
        if mlflow.active_run() is None:
            return
        float_metrics = {
            k: float(v) for k, v in metrics.items()
            if isinstance(v, (int, float)) and v == v  # исключаем nan
        }
        # Добавляем префикс test_ если ключи без него
        prefixed = {
            (k if k.startswith("test_") else f"test_{k}"): v
            for k, v in float_metrics.items()
        }
        mlflow.log_metrics(prefixed)
    except Exception:
        pass


def log_artifact_if_exists(path: Path | str) -> None:
    """Логирует файл-артефакт если он существует и run активен."""
    if not _MLFLOW_AVAILABLE:
        return
    try:
        if mlflow.active_run() is None:
            return
        p = Path(path)
        if p.exists():
            mlflow.log_artifact(str(p))
    except Exception:
        pass
