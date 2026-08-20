"""
blue_team/thresholding/cost_aware_threshold.py

AI Defense Lab — Cost-Aware Threshold Optimization
Mastercard Innovation Challenge 2026

Calculates optimal decision threshold based on business financial trade-off:
Cost of missed fraud (FN) vs Cost of false positive (FP customer friction).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_missed_fraud: float = 500.0,
    cost_false_positive: float = 25.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Finds decision threshold tau in [0.01, 0.99] that minimizes financial loss.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    best_cost = float("inf")
    best_threshold = 0.50
    best_metrics = {}

    for tau in thresholds:
        y_pred = (y_prob >= tau).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        cost = (fn * cost_missed_fraud) + (fp * cost_false_positive)

        if cost < best_cost:
            best_cost = cost
            best_threshold = tau
            best_metrics = {
                "threshold": float(tau),
                "total_cost": float(cost),
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }

    return float(best_threshold), best_metrics
