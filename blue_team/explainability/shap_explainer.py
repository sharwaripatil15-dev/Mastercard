"""
blue_team/explainability/shap_explainer.py

AI Defense Lab — SHAP Explainability Engine
Mastercard Innovation Challenge 2026

Calculates feature-level SHAP (SHapley Additive exPlanations) values for model predictions,
explaining WHY a transaction was flagged as fraud or approved as genuine.
Uses native fast C++ XGBoost Tree SHAP with fallback to SHAP explainer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import shap
import xgboost as xgb


class ShapExplainer:
    """
    SHAP-based explainer for tabular payment fraud classifier.
    """

    def __init__(self, model: Union[xgb.XGBClassifier, Any], feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def explain_instance(self, sample_df: pd.DataFrame, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Explains a single transaction (or DataFrame of samples).
        Returns top contributing features with their SHAP values and human-readable reasons.
        """
        X_sample = sample_df[self.feature_names] if self.feature_names else sample_df

        try:
            booster = self.model.get_booster() if hasattr(self.model, "get_booster") else self.model
            dmat = xgb.DMatrix(X_sample, feature_names=self.feature_names)
            contribs = booster.predict(dmat, pred_contribs=True)
            shap_matrix = contribs[:, :-1]  # drop bias column
        except Exception:
            explainer = shap.Explainer(self.model.predict_proba, X_sample)
            shap_vals = explainer(X_sample)
            shap_matrix = shap_vals.values[:, :, 1] if len(shap_vals.values.shape) == 3 else shap_vals.values

        results = []
        for idx in range(len(X_sample)):
            sample_shap = shap_matrix[idx]
            sample_val = X_sample.iloc[idx].values

            top_indices = np.argsort(np.abs(sample_shap))[::-1][:top_k]

            contributions = []
            for i in top_indices:
                feat = self.feature_names[i]
                val = float(sample_val[i])
                s_val = float(sample_shap[i])

                contributions.append({
                    "feature": feat,
                    "value": val,
                    "shap_value": round(s_val, 4),
                    "impact": "increases_risk" if s_val > 0 else "decreases_risk",
                })

            results.append({
                "transaction_index": idx,
                "top_explanations": contributions,
            })

        return results
