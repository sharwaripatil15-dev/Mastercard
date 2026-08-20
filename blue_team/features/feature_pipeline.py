"""
blue_team/features/feature_pipeline.py

AI Defense Lab — Feature Engineering Pipeline
Mastercard Innovation Challenge 2026

Transforms raw payment transaction records into numerical feature vectors ready
for machine learning models (XGBoost, IsolationForest, Neural Networks).
Calculates velocity, amount relative metrics, device/geography risk signals,
and time-series aggregated features per account.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


class FeaturePipeline:
    """
    Feature engineering pipeline that converts transaction rows into tabular feature matrix X and target y.
    """

    def __init__(self):
        self.feature_names: List[str] = []

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Extracts features and target labels from a transaction dataframe."""
        df_work = df.copy()

        # Ensure timestamp datetime
        if not pd.api.types.is_datetime64_any_dtype(df_work["timestamp"]):
            df_work["timestamp"] = pd.to_datetime(df_work["timestamp"], format="mixed")

        df_work = df_work.sort_values(by="timestamp").reset_index(drop=True)

        features = pd.DataFrame(index=df_work.index)

        # ----------------------------------------------------------------------
        # 1. Base Amount & Device / Geo Features
        # ----------------------------------------------------------------------
        features["amount"] = df_work["amount"].astype(float)
        features["log_amount"] = np.log1p(features["amount"])
        features["is_micro_amount"] = ((features["amount"] >= 1.0) & (features["amount"] <= 2.5)).astype(int)
        features["is_large_amount"] = (features["amount"] >= 500.0).astype(int)

        features["account_age_days"] = df_work["account_age_days"].fillna(0).astype(float)
        features["is_new_device"] = df_work["is_new_device"].astype(int)
        features["ip_mismatch"] = (df_work["ip_country"] != df_work["home_country"]).astype(int)

        dest_age = df_work["destination_account_age_days"].fillna(-1).astype(float)
        features["destination_account_age_days"] = dest_age

        # ----------------------------------------------------------------------
        # 2. Temporal Features
        # ----------------------------------------------------------------------
        features["hour_of_day"] = df_work["timestamp"].dt.hour
        features["day_of_week"] = df_work["timestamp"].dt.dayofweek
        features["is_night"] = ((features["hour_of_day"] < 6) | (features["hour_of_day"] > 22)).astype(int)

        # ----------------------------------------------------------------------
        # 3. Categorical Encodings (One-Hot)
        # ----------------------------------------------------------------------
        channel_dummies = pd.get_dummies(df_work["channel"], prefix="channel", dtype=int)
        category_dummies = pd.get_dummies(df_work["merchant_category"], prefix="cat", dtype=int)
        features = pd.concat([features, channel_dummies, category_dummies], axis=1)

        # ----------------------------------------------------------------------
        # 4. Fast Numpy Searchsorted Rolling Metrics per Account
        # ----------------------------------------------------------------------
        n_samples = len(df_work)
        txn_count_1h = np.zeros(n_samples, dtype=float)
        distinct_merchants_1h = np.zeros(n_samples, dtype=float)
        distinct_channels_1h = np.zeros(n_samples, dtype=float)
        decline_count_1h = np.zeros(n_samples, dtype=float)

        txn_count_24h = np.zeros(n_samples, dtype=float)
        amount_sum_24h = np.zeros(n_samples, dtype=float)
        amount_avg_24h = np.zeros(n_samples, dtype=float)

        auth_status_arr = df_work["auth_status"].values if "auth_status" in df_work.columns else np.full(n_samples, "approved")
        channels_arr = df_work["channel"].values

        for _, group in df_work.groupby("account_id"):
            indices = group.index.values
            timestamps = group["timestamp"].values
            amounts = group["amount"].values.astype(float)
            merchants = group["merchant_id"].values
            channels = channels_arr[indices]
            declined = (auth_status_arr[indices] == "declined")

            n_grp = len(group)
            for i in range(n_grp):
                curr_t = timestamps[i]
                orig_idx = indices[i]

                # 1 hour window
                t_1h_start = curr_t - np.timedelta64(1, 'h')
                start_1h = np.searchsorted(timestamps, t_1h_start, side='left')
                win_1h = slice(start_1h, i + 1)

                txn_count_1h[orig_idx] = i + 1 - start_1h
                distinct_merchants_1h[orig_idx] = len(set(merchants[win_1h]))
                distinct_channels_1h[orig_idx] = len(set(channels[win_1h]))
                decline_count_1h[orig_idx] = np.sum(declined[win_1h])

                # 24 hour window
                t_24h_start = curr_t - np.timedelta64(24, 'h')
                start_24h = np.searchsorted(timestamps, t_24h_start, side='left')
                win_24h_amounts = amounts[start_24h : i + 1]

                txn_count_24h[orig_idx] = len(win_24h_amounts)
                amount_sum_24h[orig_idx] = np.sum(win_24h_amounts)
                amount_avg_24h[orig_idx] = np.mean(win_24h_amounts)

        features["txn_count_1h"] = txn_count_1h
        features["distinct_merchants_1h"] = distinct_merchants_1h
        features["distinct_channels_1h"] = distinct_channels_1h
        features["cross_channel_chain_flag"] = (features["distinct_channels_1h"] >= 2.0).astype(int)
        features["decline_count_1h"] = decline_count_1h
        features["decline_ratio_1h"] = features["decline_count_1h"] / np.maximum(features["txn_count_1h"], 1.0)

        features["txn_count_24h"] = txn_count_24h
        features["amount_sum_24h"] = amount_sum_24h
        features["amount_avg_24h"] = amount_avg_24h
        features["amount_to_avg_24h_ratio"] = features["amount"] / (features["amount_avg_24h"] + 1e-5)

        # ----------------------------------------------------------------------
        # 5. Final Cleanup
        # ----------------------------------------------------------------------
        features = features.fillna(0)
        self.feature_names = list(features.columns)

        y = df_work["label"].astype(int)

        return features, y, self.feature_names


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Helper function to run feature engineering pipeline."""
    pipeline = FeaturePipeline()
    return pipeline.fit_transform(df)


def main():
    parser = argparse.ArgumentParser(description="Run feature engineering pipeline on merged dataset.")
    parser.add_argument("--input", type=str, default="data/merged_dataset.csv", help="Input dataset CSV path")
    parser.add_argument("--output-x", type=str, default="data/features_X.csv", help="Output features X path")
    parser.add_argument("--output-y", type=str, default="data/labels_y.csv", help="Output labels y path")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found at {input_path}")

    print(f"Reading dataset from {input_path}...")
    df = pd.read_csv(input_path)

    pipeline = FeaturePipeline()
    X, y, feature_names = pipeline.fit_transform(df)

    print(f"\nExtracted {len(feature_names)} features for {len(X)} rows.")
    print("Top features engineered:", feature_names[:10])

    Path(args.output_x).parent.mkdir(parents=True, exist_ok=True)
    X.to_csv(args.output_x, index=False)
    y.to_csv(args.output_y, index=False)
    print(f"Saved features -> {args.output_x} and labels -> {args.output_y}")


if __name__ == "__main__":
    main()
