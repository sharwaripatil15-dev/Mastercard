"""
data/merge_datasets.py

AI Defense Lab — Dataset Merging Utility
Mastercard Innovation Challenge 2026

Combines baseline genuine transactions with synthetic attack datasets from
`data/synthetic_attacks/` into a single labeled dataset ready for model training.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd

DEFAULT_BASELINE_PATH = Path(__file__).resolve().parent / "baseline_genuine" / "baseline_transactions.csv"
DEFAULT_ATTACKS_DIR = Path(__file__).resolve().parent / "synthetic_attacks"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "merged_dataset.csv"

# Baseline schema columns (guaranteed to be present in baseline)
BASE_COLUMNS = [
    "transaction_id", "customer_id", "account_id", "customer_name", "customer_email",
    "account_age_days", "timestamp", "amount", "currency", "merchant_category",
    "merchant_id", "channel", "device_id", "ip_country", "home_country",
    "is_new_device", "destination_account_age_days", "label", "attack_type"
]


def merge_datasets(
    baseline_path: Path = DEFAULT_BASELINE_PATH,
    attacks_dir: Path = DEFAULT_ATTACKS_DIR,
    output_path: Path = DEFAULT_OUTPUT_PATH,
    specific_attack_files: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Merges baseline genuine dataset with synthetic attack datasets.
    Handles extra attack-specific columns cleanly.
    """
    if not baseline_path.exists():
        raise FileNotFoundError(f"Baseline file not found at {baseline_path}")

    print(f"Loading baseline dataset from {baseline_path}...")
    baseline_df = pd.read_csv(baseline_path)
    
    # Ensure baseline has label=0 and attack_type=None
    if "label" not in baseline_df.columns:
        baseline_df["label"] = 0
    if "attack_type" not in baseline_df.columns:
        baseline_df["attack_type"] = None

    dfs_to_merge = [baseline_df]

    # Find attack CSVs
    if attacks_dir.exists():
        if specific_attack_files:
            attack_files = [attacks_dir / f for f in specific_attack_files if (attacks_dir / f).exists()]
        else:
            attack_files = list(attacks_dir.glob("*.csv"))

        for attack_file in attack_files:
            print(f"Loading synthetic attack dataset from {attack_file}...")
            attack_df = pd.read_csv(attack_file)
            dfs_to_merge.append(attack_df)
    else:
        print(f"Warning: Attacks directory {attacks_dir} does not exist. Only baseline data will be merged.")

    # Combine dataframes
    merged_df = pd.concat(dfs_to_merge, ignore_index=True)

    # Fill defaults for baseline rows missing attack-specific fields
    if "auth_status" in merged_df.columns:
        merged_df["auth_status"] = merged_df["auth_status"].fillna("approved")
    else:
        merged_df["auth_status"] = "approved"

    if "is_micro_auth" in merged_df.columns:
        merged_df["is_micro_auth"] = merged_df["is_micro_auth"].fillna(False).astype(bool)
    else:
        merged_df["is_micro_auth"] = False

    if "distinct_merchants_touched" in merged_df.columns:
        merged_df["distinct_merchants_touched"] = merged_df["distinct_merchants_touched"].fillna(1).astype(int)
    else:
        merged_df["distinct_merchants_touched"] = 1

    if "decline_rate" in merged_df.columns:
        merged_df["decline_rate"] = merged_df["decline_rate"].fillna(0.0).astype(float)
    else:
        merged_df["decline_rate"] = 0.0

    # Sort chronologically
    merged_df["timestamp"] = pd.to_datetime(merged_df["timestamp"], format="mixed")
    merged_df = merged_df.sort_values(by="timestamp").reset_index(drop=True)
    merged_df["timestamp"] = merged_df["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Save to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_path, index=False)

    total_count = len(merged_df)
    genuine_count = (merged_df["label"] == 0).sum()
    fraud_count = (merged_df["label"] == 1).sum()
    fraud_pct = (fraud_count / total_count * 100) if total_count > 0 else 0.0

    print("\n--- Merged Dataset Summary ---")
    print(f"Total Transactions: {total_count:,}")
    print(f"Genuine (Label 0):  {genuine_count:,} ({100 - fraud_pct:.2f}%)")
    print(f"Fraud   (Label 1):  {fraud_count:,} ({fraud_pct:.2f}%)")
    print("\nFraud Breakdown by Attack Type:")
    attack_counts = merged_df[merged_df["label"] == 1]["attack_type"].value_counts()
    for attack_type, count in attack_counts.items():
        print(f"  - {attack_type}: {count:,}")
    print(f"Saved merged dataset -> {output_path}\n")

    return merged_df


def main():
    parser = argparse.ArgumentParser(description="Merge baseline and synthetic attack datasets.")
    parser.add_argument("--baseline", type=str, default=str(DEFAULT_BASELINE_PATH), help="Path to baseline transactions CSV")
    parser.add_argument("--attacks-dir", type=str, default=str(DEFAULT_ATTACKS_DIR), help="Directory containing attack CSVs")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH), help="Output path for merged CSV")

    args = parser.parse_args()

    merge_datasets(
        baseline_path=Path(args.baseline),
        attacks_dir=Path(args.attacks_dir),
        output_path=Path(args.output),
    )


if __name__ == "__main__":
    main()
