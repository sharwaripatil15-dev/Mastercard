"""
blue_team/models/tabular_model.py

AI Defense Lab — Tabular Fraud Classifier (XGBoost)
Mastercard Innovation Challenge 2026

Supervised tabular model for single-transaction and velocity-based fraud detection.
Handles severe class imbalance dynamically using scale_pos_weight.
"""

from __future__ import annotations

import pathlib
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import xgboost as xgb


class TabularFraudClassifier:
    """
    XGBoost-based Tabular Fraud Classifier designed for imbalanced transaction datasets.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        scale_pos_weight: Optional[float] = None,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state

        self.model: Optional[xgb.XGBClassifier] = None
        self.feature_names: List[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> TabularFraudClassifier:
        """Trains the XGBoost classifier on input features X and binary target y."""
        self.feature_names = list(X.columns)

        # Compute class imbalance weight if not explicitly provided
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()

        if self.scale_pos_weight is None:
            computed_weight = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0
        else:
            computed_weight = self.scale_pos_weight

        print(f"Training XGBoost Fraud Classifier (Class Ratio Neg:Pos = {n_neg}:{n_pos}, scale_pos_weight={computed_weight:.2f})...")

        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=computed_weight,
            random_state=self.random_state,
            eval_metric="logloss",
        )

        self.model.fit(X, y)
        print("Training complete.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns predicted probabilities of fraud (label=1)."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")
        # Ensure feature alignment
        X_aligned = X[self.feature_names] if self.feature_names else X
        return self.model.predict_proba(X_aligned)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predicts binary fraud class based on decision threshold."""
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def get_feature_importances(self) -> pd.Series:
        """Returns feature importances sorted descending."""
        if self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        importances = self.model.feature_importances_
        return pd.Series(importances, index=self.feature_names).sort_values(ascending=False)

    def save_model(self, filepath: Union[str, Path]) -> None:
        """Saves trained model to disk."""
        if self.model is None:
            raise RuntimeError("Cannot save an untrained model.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        print(f"Model saved to {path}")

    def load_model(self, filepath: Union[str, Path]) -> None:
        """Loads trained model from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(path))
        print(f"Model loaded from {path}")
