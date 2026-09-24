"""Metrics, the cross-model comparison table, and threshold selection by
F1 — so a submission's cutoff is chosen deliberately and then actually
used, unlike the original notebook, where the cutoff analysis and the
cutoff applied to the final submission didn't match."""
import numpy as np
import pandas as pd
from sklearn import metrics


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "accuracy": metrics.accuracy_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred, zero_division=0),
        "recall": metrics.recall_score(y_true, y_pred, zero_division=0),
        "f1": metrics.f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": metrics.roc_auc_score(y_true, y_proba),
        "pr_auc": metrics.average_precision_score(y_true, y_proba),
    }


def comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    return pd.DataFrame(results).T[["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]]


def select_threshold(y_true, y_proba, thresholds=None) -> tuple[float, pd.DataFrame]:
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]

    rows = []
    for t in thresholds:
        y_pred = (y_proba > t).astype(int)
        rows.append({
            "threshold": t,
            "precision": metrics.precision_score(y_true, y_pred, zero_division=0),
            "recall": metrics.recall_score(y_true, y_pred, zero_division=0),
            "f1": metrics.f1_score(y_true, y_pred, zero_division=0),
        })

    table = pd.DataFrame(rows)
    best_threshold = table.loc[table["f1"].idxmax(), "threshold"]
    return best_threshold, table
