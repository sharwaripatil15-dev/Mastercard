"""
red_team/generators/structuring_generator.py

AI Defense Lab — Low-and-Slow Structuring Generator
Mastercard Innovation Challenge 2026

Simulates "Low-and-Slow Structuring" attack (ID: low_and_slow_structuring) based on
the spec in `red_team/taxonomy/attack_taxonomy.yaml`.

Attack pattern:
- A large transfer is split into N smaller transactions over multiple days to stay
  just below single-transaction fraud monitoring thresholds ($150 - $450 each).
- Often uses the same destination account or channel (e.g. app transfer / wire).
"""

from __future__ import annotations

import argparse
import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "taxonomy" / "attack_taxonomy.yaml"
BASELINE_CSV_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "baseline_genuine" / "baseline_transactions.csv"


def load_baseline_profiles(baseline_path: Path = BASELINE_CSV_PATH) -> List[Dict[str, Any]]:
    if not baseline_path.exists():
        return []
    df = pd.read_csv(baseline_path)
    profiles = df.groupby("customer_id").first().reset_index()
    return profiles.to_dict(orient="records")


class StructuringGenerator:
    """
    Generates synthetic structuring attack transactions spanning multiple days.
    """

    def __init__(
        self,
        num_series: int = 30,
        num_transactions: Optional[int] = None,
        per_txn_amount: Optional[float] = None,
        span_days: Optional[int] = None,
        baseline_path: Path = BASELINE_CSV_PATH,
        seed: int = 42,
    ):
        self.num_series = num_series
        self.num_transactions = num_transactions
        self.per_txn_amount = per_txn_amount
        self.span_days = span_days
        self.rng = random.Random(seed)
        np.random.seed(seed)
        self.profiles = load_baseline_profiles(baseline_path)

    def generate(self) -> pd.DataFrame:
        rows = []
        start_base = datetime.fromisoformat("2026-06-01")

        for _ in range(self.num_series):
            if self.profiles:
                profile = self.rng.choice(self.profiles)
                customer_id = profile["customer_id"]
                account_id = profile["account_id"]
                name = profile["customer_name"]
                email = profile["customer_email"]
                home_country = profile["home_country"]
                account_age_days = profile["account_age_days"]
                device_id = profile["device_id"]
            else:
                customer_id = "cust_struct_" + uuid.uuid4().hex[:8]
                account_id = "acc_" + uuid.uuid4().hex[:10]
                name = "Karan Patel"
                email = "karan.patel@gmail.com"
                home_country = "IN"
                account_age_days = 400
                device_id = "dev_struct_" + uuid.uuid4().hex[:8]

            n_txns = self.num_transactions or self.rng.randint(4, 9)
            span = self.span_days or self.rng.randint(3, 10)
            base_amount = self.per_txn_amount or float(np.random.uniform(180.0, 420.0))

            start_day_offset = self.rng.randint(0, 65)
            series_start = start_base + timedelta(days=start_day_offset, hours=self.rng.randint(8, 20))

            dest_account_age = self.rng.randint(1, 15)  # typically new recipient account

            for i in range(n_txns):
                day_offset = (i / max(n_txns - 1, 1)) * span
                txn_time = series_start + timedelta(days=day_offset, minutes=self.rng.randint(-30, 30))

                amount = round(base_amount + float(np.random.uniform(-15.0, 15.0)), 2)

                rows.append({
                    "transaction_id": "txn_struct_" + uuid.uuid4().hex[:12],
                    "customer_id": customer_id,
                    "account_id": account_id,
                    "customer_name": name,
                    "customer_email": email,
                    "account_age_days": account_age_days,
                    "timestamp": txn_time.isoformat(),
                    "amount": amount,
                    "currency": "INR" if home_country == "IN" else "USD",
                    "merchant_category": "ecommerce_general",
                    "merchant_id": "merch_struct_" + uuid.uuid4().hex[:6],
                    "channel": "app_transfer",
                    "device_id": device_id,
                    "ip_country": home_country,
                    "home_country": home_country,
                    "is_new_device": False,
                    "destination_account_age_days": dest_account_age,
                    "auth_status": "approved",
                    "label": 1,
                    "attack_type": "low_and_slow_structuring",
                    "distinct_merchants_touched": 1,
                    "decline_rate": 0.0,
                    "is_micro_auth": False,
                })

        df = pd.DataFrame(rows)
        return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic low-and-slow structuring transactions.")
    parser.add_argument("--num-series", type=int, default=30, help="Number of structuring series to generate")
    parser.add_argument("--output", type=str, default="data/synthetic_attacks/structuring_attacks.csv", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator = StructuringGenerator(num_series=args.num_series, seed=args.seed)
    df = generator.generate()
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} structuring attack transactions across {args.num_series} series -> {output_path}")


if __name__ == "__main__":
    main()
