# -*- coding: utf-8 -*-
import csv


def save_result_csv(
    exp_dir,
    experiment_id,
    experiment_name,
    model,
    accuracy,
    f1_macro=None,
    f1_bad=None,
    roc_auc=None,
    precision_bad=None,
    recall_bad=None,
    notes="",
    num_params=None,
    train_time_sec=None,
):
    result_path = exp_dir / "result.csv"
    columns = [
        "experiment_id",
        "experiment_name",
        "model",
        "accuracy",
        "f1_macro",
        "f1_bad",
        "roc_auc",
        "precision_bad",
        "recall_bad",
        "notes",
        "num_params",
        "train_time_sec",
    ]
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
        "notes": notes,
        "num_params": num_params,
        "train_time_sec": train_time_sec,
    }
    with open(result_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerow(row)
    return result_path
