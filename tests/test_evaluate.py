import numpy as np
import pytest
from churn import evaluate


def test_compute_metrics_matches_hand_computed_values():
    # 4 samples: TP=1, TN=1, FP=1, FN=1 -> accuracy=0.5, precision=0.5, recall=0.5
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.6, 0.4, 0.1])

    result = evaluate.compute_metrics(y_true, y_pred, y_proba)

    assert result["accuracy"] == 0.5
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert 0.0 <= result["roc_auc"] <= 1.0
    assert 0.0 <= result["pr_auc"] <= 1.0


def test_comparison_table_has_one_row_per_model_and_expected_columns():
    results = {
        "Model A": {"accuracy": 0.9, "precision": 0.8, "recall": 0.7, "f1": 0.75, "roc_auc": 0.95, "pr_auc": 0.85},
        "Model B": {"accuracy": 0.8, "precision": 0.7, "recall": 0.6, "f1": 0.65, "roc_auc": 0.90, "pr_auc": 0.80},
    }
    table = evaluate.comparison_table(results)
    assert list(table.index) == ["Model A", "Model B"]
    assert list(table.columns) == ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def test_select_threshold_picks_the_threshold_with_highest_f1():
    # Overlapping classes (not perfectly separable) so F1 varies across the
    # default threshold grid instead of plateauing at 1.0. One negative
    # (index 5, proba=0.85) and one positive (index 6, proba=0.40) sit on
    # the "wrong side" of a clean split, which makes the F1-maximizing
    # threshold a unique peak at 0.35 (grid index 6 of 18) -- not the
    # lowest (0.05, index 0), middle (0.5, index 9), or highest (0.95,
    # index 18) threshold in the grid. This was verified by actually
    # running evaluate.select_threshold against this fixture and checking
    # that a "always return the lowest/middle/highest threshold" stub
    # fails the assertions below (see task-8-report.md, Fix Round 1).
    # Note: this fixture also pins the implementation's strict "> t" cutoff
    # (proba=0.35 sits exactly on the grid's t=0.35 threshold) -- using
    # ">=" instead shifts the optimum to threshold=0.40, so this test
    # would also catch a >/>= regression, not just a wrong-pick regression.
    y_true = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_proba = np.array([0.05, 0.15, 0.25, 0.35, 0.45, 0.85, 0.40, 0.55, 0.65, 0.75, 0.90])

    best_threshold, table = evaluate.select_threshold(y_true, y_proba)

    best_f1 = table.loc[table["threshold"] == best_threshold, "f1"].iloc[0]
    assert best_f1 == table["f1"].max()
    # Pin down the exact expected threshold (not just "achieves the max"),
    # so a stub that returns a fixed threshold unrelated to the data fails.
    assert best_threshold == 0.35
    assert best_f1 == pytest.approx(5 / 6)
