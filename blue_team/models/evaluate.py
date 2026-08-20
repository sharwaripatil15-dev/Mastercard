"""
blue_team/models/evaluate.py

AI Defense Lab — Model Evaluation & Metrics Utility
Mastercard Innovation Challenge 2026

Calculates and formats comprehensive evaluation metrics for payment fraud classifiers:
Precision, Recall, F1 Score, ROC AUC, PR AUC, Confusion Matrix, and Financial Cost Analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    cost_missed_fraud: float = 500.0,
    cost_false_positive: float = 25.0,
) -> Dict[str, Any]:
    """
    Computes precision, recall, F1, ROC AUC, PR AUC, confusion matrix, and cost analysis.
    """
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.0

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = 0.0

    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    total_cost = (fn * cost_missed_fraud) + (fp * cost_false_positive)

    metrics = {
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "total_cost": float(total_cost),
    }

    return metrics


def print_evaluation_report(metrics: Dict[str, Any], title: str = "Fraud Detection Model Evaluation") -> None:
    """Prints formatted metrics report to console."""
    print(f"\n==================================================")
    print(f"  {title}")
    print(f"==================================================")
    print(f"Decision Threshold:    {metrics['threshold']:.2f}")
    print(f"--------------------------------------------------")
    print(f"Precision:             {metrics['precision'] * 100:.2f}%")
    print(f"Recall (Detection):    {metrics['recall'] * 100:.2f}%")
    print(f"F1 Score:              {metrics['f1'] * 100:.2f}%")
    print(f"ROC AUC:               {metrics['roc_auc']:.4f}")
    print(f"PR AUC:                {metrics['pr_auc']:.4f}")
    print(f"--------------------------------------------------")
    print(f"Confusion Matrix:")
    print(f"  True Negatives (Genuine Approved): {metrics['tn']:,}")
    print(f"  False Positives (Genuine Blocked): {metrics['fp']:,}")
    print(f"  False Negatives (Missed Fraud):    {metrics['fn']:,}")
    print(f"  True Positives  (Fraud Caught):    {metrics['tp']:,}")
    print(f"--------------------------------------------------")
    print(f"Estimated Financial Loss: ${metrics['total_cost']:,.2f}")
    print(f"==================================================\n")
