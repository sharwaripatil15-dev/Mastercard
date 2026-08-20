"""
red_team/generators/card_testing_generator.py

AI Defense Lab — Card Testing Bots Attack Generator
Mastercard Innovation Challenge 2026

Simulates "Card Testing Bots" attack (ID: card_testing_bots) based on the spec
in `red_team/taxonomy/attack_taxonomy.yaml`.

Realistic overlapping simulation:
- Attacks target existing baseline customer accounts (compromised cards).
- Device usage mixes home device (60%) and new device (40%).
- Decline rates per burst vary from 25% to 65% to mimic realistic retry noise.
- Amounts span $0.80 - $4.50 with timing jitter.
"""

from __future__ import annotations

import argparse
import random
import string
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "taxonomy" / "attack_taxonomy.yaml"
BASELINE_CSV_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "baseline_genuine" / "baseline_transactions.csv"

MERCHANT_CATEGORIES = ["ecommerce_general", "digital_goods", "subscriptions", "apparel", "entertainment"]


def load_baseline_profiles(baseline_path: Path = BASELINE_CSV_PATH) -> List[Dict[str, Any]]:
    """Loads distinct customer profiles from baseline transactions."""
    if not baseline_path.exists():
        return []

    df = pd.read_csv(baseline_path)
    profiles = df.groupby("customer_id").first().reset_index()
    return profiles.to_dict(orient="records")


class CardTestingGenerator:
    """
    Generates realistic synthetic card testing bot attack transactions that overlap
    with baseline customer behavior to avoid trivial synthetic indicators.
    """

    def __init__(
        self,
        num_bursts: int = 50,
        baseline_path: Path = BASELINE_CSV_PATH,
        distinct_merchants_touched: Optional[int] = None,
        window_minutes: Optional[int] = None,
        decline_rate: Optional[float] = None,
        seed: int = 42,
    ):
        self.num_bursts = num_bursts
        self.baseline_path = baseline_path
        self.distinct_merchants_touched = distinct_merchants_touched
        self.window_minutes = window_minutes
        self.decline_rate = decline_rate
        self.rng = random.Random(seed)
        np.random.seed(seed)
        self.profiles = load_baseline_profiles(baseline_path)

    def generate(self) -> pd.DataFrame:
        rows = []

        for _ in range(self.num_bursts):
            # Select an existing customer profile if available, else synthetic
            if self.profiles:
                profile = self.rng.choice(self.profiles)
                customer_id = profile["customer_id"]
                account_id = profile["account_id"]
                name = profile["customer_name"]
                email = profile["customer_email"]
                home_country = profile["home_country"]
                account_age_days = profile["account_age_days"]
                home_device_id = profile["device_id"]
            else:
                customer_id = "cust_victim_" + uuid.uuid4().hex[:8]
                account_id = "acc_" + uuid.uuid4().hex[:10]
                name = "Aarav Sharma"
                email = "aarav.sharma@gmail.com"
                home_country = "IN"
                account_age_days = 365
                home_device_id = "dev_home_" + uuid.uuid4().hex[:8]

            # 60% chance home device, 40% new device
            if self.rng.random() < 0.60:
                bot_device_id = home_device_id
                bot_ip_country = home_country
                is_new_device = False
            else:
                bot_device_id = "dev_bot_" + "".join(self.rng.choices(string.ascii_lowercase + string.digits, k=8))
                bot_ip_country = self.rng.choice(["IN", "US", "GB", "AE", "SG"])
                is_new_device = True

            # Start timestamp during active period
            start_date = datetime.fromisoformat("2026-06-01")
            day_offset = self.rng.randint(0, 75)
            burst_start = start_date + timedelta(
                days=day_offset,
                hours=self.rng.randint(0, 23),
                minutes=self.rng.randint(0, 59),
            )

            # Burst parameters with realistic noise
            n_txns = self.distinct_merchants_touched if self.distinct_merchants_touched is not None else self.rng.randint(3, 10)
            burst_decline_rate = self.decline_rate if self.decline_rate is not None else float(np.random.uniform(0.25, 0.65))

            merchants = ["merch_test_" + uuid.uuid4().hex[:6] for _ in range(n_txns)]

            current_time = burst_start
            for i in range(n_txns):
                jitter_sec = self.rng.randint(5, 60)
                current_time = current_time + timedelta(seconds=jitter_sec)

                amount = round(self.rng.uniform(0.80, 4.50), 2)
                is_declined = self.rng.random() < burst_decline_rate

                rows.append({
                    "transaction_id": "txn_ct_" + uuid.uuid4().hex[:12],
                    "customer_id": customer_id,
                    "account_id": account_id,
                    "customer_name": name,
                    "customer_email": email,
                    "account_age_days": account_age_days,
                    "timestamp": current_time.isoformat(),
                    "amount": amount,
                    "currency": "INR" if home_country == "IN" else "USD",
                    "merchant_category": self.rng.choice(MERCHANT_CATEGORIES),
                    "merchant_id": merchants[i],
                    "channel": "card_not_present",
                    "device_id": bot_device_id,
                    "ip_country": bot_ip_country,
                    "home_country": home_country,
                    "is_new_device": is_new_device,
                    "destination_account_age_days": None,
                    "auth_status": "declined" if is_declined else "approved",
                    "label": 1,
                    "attack_type": "card_testing_bots",
                    "distinct_merchants_touched": n_txns,
                    "decline_rate": round(burst_decline_rate, 2),
                    "is_micro_auth": True,
                })

            # 70% chance of follow-up large transaction
            if self.rng.random() < 0.70:
                followup_time = current_time + timedelta(minutes=self.rng.randint(5, 45))
                large_amount = round(float(np.random.uniform(350.0, 1800.0)), 2)

                rows.append({
                    "transaction_id": "txn_ct_bustout_" + uuid.uuid4().hex[:10],
                    "customer_id": customer_id,
                    "account_id": account_id,
                    "customer_name": name,
                    "customer_email": email,
                    "account_age_days": account_age_days,
                    "timestamp": followup_time.isoformat(),
                    "amount": large_amount,
                    "currency": "INR" if home_country == "IN" else "USD",
                    "merchant_category": "electronics",
                    "merchant_id": "merch_big_" + uuid.uuid4().hex[:6],
                    "channel": "card_not_present",
                    "device_id": bot_device_id,
                    "ip_country": bot_ip_country,
                    "home_country": home_country,
                    "is_new_device": is_new_device,
                    "destination_account_age_days": None,
                    "auth_status": "approved",
                    "label": 1,
                    "attack_type": "card_testing_bots",
                    "distinct_merchants_touched": n_txns,
                    "decline_rate": round(burst_decline_rate, 2),
                    "is_micro_auth": False,
                })

        df = pd.DataFrame(rows)
        return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic card testing bot attack transactions.")
    parser.add_argument("--num-bursts", type=int, default=50, help="Number of attack bursts")
    parser.add_argument("--output", type=str, default="data/synthetic_attacks/card_testing_attacks.csv", help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator = CardTestingGenerator(
        num_bursts=args.num_bursts,
        seed=args.seed,
    )

    df = generator.generate()
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} realistic card testing attack transactions across {args.num_bursts} bursts -> {output_path}")


if __name__ == "__main__":
    main()
