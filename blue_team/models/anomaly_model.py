"""
blue_team/models/anomaly_model.py

AI Defense Lab — Anomaly Detection Model (Isolation Forest)
Mastercard Innovation Challenge 2026

Unsupervised anomaly detector designed to catch novel, unseen, or unlabeled attack vectors.
"""

from __future__ import annotations

import pathlib
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """
    IsolationForest-based anomaly detector for detecting zero-day and out-of-distribution payment fraud.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.is_fitted = False

    def fit(self, X: pd.DataFrame) -> AnomalyDetector:
        """Fits Isolation Forest model on input feature matrix X."""
        self.model.fit(X)
        self.is_fitted = True
        return self

    def predict_anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns normalized anomaly score in range [0, 1] where 1 is highly anomalous.
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector has not been fitted yet.")
        # IsolationForest decision_function outputs negative values for anomalies
        raw_scores = self.model.decision_function(X)
        # Normalize to [0, 1] range (invert so higher = more anomalous)
        normalized_scores = 1.0 - ((raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-8))
        return np.clip(normalized_scores, 0.0, 1.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Returns binary predictions: 1 for anomaly, 0 for normal."""
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector has not been fitted yet.")
        raw_preds = self.model.predict(X)  # -1 for anomaly, 1 for normal
        return (raw_preds == -1).astype(int)
