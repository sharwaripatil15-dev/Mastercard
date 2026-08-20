"""
loop/metrics.py

AI Defense Lab — Closed-Loop Metrics Engine
Mastercard Innovation Challenge 2026

Calculates, logs, and persists performance metrics across adversarial sparring rounds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

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

ROUND_HISTORY_DIR = Path(__file__).resolve().parent.parent / "data" / "round_history"


class MetricsEngine:
    """
    Tracks and persists performance metrics per round for the adversarial loop.
    """

    def __init__(self, history_dir: Path = ROUND_HISTORY_DIR):
        self.history_dir = history_dir
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.round_history: List[Dict[str, Any]] = []

    def evaluate_round(
        self,
        round_num: int,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        attack_types: pd.Series,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Computes complete round metrics and saves history snapshot."""
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        try:
            roc_auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            roc_auc = 0.0

        try:
            pr_auc = float(average_precision_score(y_true, y_prob))
        except Exception:
            pr_auc = 0.0

        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        fp_rate = float(fp / max(tn + fp, 1))

        # Per attack-type breakdown
        per_attack_metrics = {}
        eval_df = pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "attack_type": attack_types.values if hasattr(attack_types, "values") else attack_types
        })

        fraud_df = eval_df[eval_df["y_true"] == 1]
        for atype, group in fraud_df.groupby("attack_type"):
            if pd.notna(atype):
                caught = (group["y_pred"] == 1).sum()
                total = len(group)
                per_attack_metrics[str(atype)] = {
                    "total": int(total),
                    "caught": int(caught),
                    "missed": int(total - caught),
                    "detection_rate": float(caught / max(total, 1)),
                }

        round_data = {
            "round": round_num,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "fp_rate": fp_rate,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "per_attack_metrics": per_attack_metrics,
        }

        self.round_history.append(round_data)

        # Save snapshot
        snapshot_file = self.history_dir / f"round_{round_num}_metrics.json"
        with open(snapshot_file, "w", encoding="utf-8") as f:
            json.dump(round_data, f, indent=2)

        return round_data

    def print_history_summary(self) -> None:
        """Prints formatted summary table across all executed rounds."""
        print("\n" + "=" * 70)
        print("    CLOSED-LOOP ADVERSARIAL SPARRING — ROUND HISTORY SUMMARY")
        print("=" * 70)
        header = f"{'Round':<8}{'Precision':<12}{'Recall':<12}{'F1 Score':<12}{'ROC AUC':<10}{'Missed Fraud':<14}"
        print(header)
        print("-" * 70)

        for r in self.round_history:
            line = f"R{r['round']:<7}{r['precision']*100:>6.2f}%     {r['recall']*100:>6.2f}%     {r['f1']*100:>6.2f}%     {r['roc_auc']:>6.4f}    {r['fn']:>5d}"
            print(line)

        print("=" * 70 + "\n")
