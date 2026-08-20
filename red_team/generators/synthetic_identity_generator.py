"""
red_team/generators/synthetic_identity_generator.py

AI Defense Lab — Synthetic Identity Fraud Generator
Mastercard Innovation Challenge 2026

Simulates "Synthetic Identity Fraud" attack (ID: synthetic_identity_fraud) based on
the spec in `red_team/taxonomy/attack_taxonomy.yaml`.

Attack pattern:
- AI-fabricated synthetic identity opens a new account.
- Low-risk warming transactions ($10 - $40) for an aging period (e.g. 15-45 days) to build trust.
- Followed by a massive "bust-out" transaction spike ($1,500 - $5,000) on high-value categories (electronics, travel).
"""

from __future__ import annotations

import argparse
import random
import string
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class SyntheticIdentityGenerator:
    """
    Generates synthetic identity fraud account lifecycles (warming + bust-out).
    """

    def __init__(
        self,
        num_accounts: int = 25,
        aging_period_days: Optional[int] = None,
        bust_out_multiplier: Optional[float] = None,
        start_date: str = "2026-06-01",
        seed: int = 42,
    ):
        self.num_accounts = num_accounts
        self.aging_period_days = aging_period_days
        self.bust_out_multiplier = bust_out_multiplier
        self.start_date = datetime.fromisoformat(start_date)
        self.rng = random.Random(seed)
        np.random.seed(seed)

    def generate(self) -> pd.DataFrame:
        rows = []

        for _ in range(self.num_accounts):
            customer_id = "cust_synth_" + uuid.uuid4().hex[:8]
            account_id = "acc_synth_" + uuid.uuid4().hex[:10]
            name = f"SynthUser {uuid.uuid4().hex[:4].upper()}"
            email = f"synth.{uuid.uuid4().hex[:6]}@mail.com"
            home_country = self.rng.choice(["IN", "US", "GB", "AE"])
            device_id = "dev_synth_" + "".join(self.rng.choices(string.ascii_lowercase + string.digits, k=8))

            aging_days = self.aging_period_days or self.rng.randint(15, 45)
            account_age_at_start = self.rng.randint(1, 10)

            acct_start_time = self.start_date + timedelta(days=self.rng.randint(0, 30))

            # 1. Warming phase: low-value transactions
            n_warming_txns = self.rng.randint(5, 12)
            warming_amounts = []
            for j in range(n_warming_txns):
                day_offset = (j / max(n_warming_txns - 1, 1)) * aging_days
                txn_time = acct_start_time + timedelta(days=day_offset, hours=self.rng.randint(9, 21))
                amount = round(float(np.random.uniform(12.0, 45.0)), 2)
                warming_amounts.append(amount)

                # Warming phase rows logged as genuine or low-risk part of lifecycle
                rows.append({
                    "transaction_id": "txn_synth_warm_" + uuid.uuid4().hex[:10],
                    "customer_id": customer_id,
                    "account_id": account_id,
                    "customer_name": name,
                    "customer_email": email,
                    "account_age_days": account_age_at_start + int(day_offset),
                    "timestamp": txn_time.isoformat(),
                    "amount": amount,
                    "currency": "INR" if home_country == "IN" else "USD",
                    "merchant_category": self.rng.choice(["grocery", "fuel", "utilities"]),
                    "merchant_id": "merch_warm_" + uuid.uuid4().hex[:6],
                    "channel": "card_not_present",
                    "device_id": device_id,
                    "ip_country": home_country,
                    "home_country": home_country,
                    "is_new_device": False,
                    "destination_account_age_days": None,
                    "auth_status": "approved",
                    "label": 0,  # warming txns look legitimate
                    "attack_type": None,
                    "distinct_merchants_touched": 1,
                    "decline_rate": 0.0,
                    "is_micro_auth": False,
                })

            # 2. Bust-out phase: high-value fraud spike
            avg_warm = np.mean(warming_amounts)
            multiplier = self.bust_out_multiplier or float(np.random.uniform(25.0, 60.0))
            bust_out_amount = round(float(avg_warm * multiplier), 2)

            bust_out_time = acct_start_time + timedelta(days=aging_days + self.rng.randint(1, 3))

            rows.append({
                "transaction_id": "txn_synth_bustout_" + uuid.uuid4().hex[:10],
                "customer_id": customer_id,
                "account_id": account_id,
                "customer_name": name,
                "customer_email": email,
                "account_age_days": account_age_at_start + aging_days + 1,
                "timestamp": bust_out_time.isoformat(),
                "amount": bust_out_amount,
                "currency": "INR" if home_country == "IN" else "USD",
                "merchant_category": self.rng.choice(["electronics", "travel"]),
                "merchant_id": "merch_bust_" + uuid.uuid4().hex[:6],
                "channel": "card_not_present",
                "device_id": device_id,
                "ip_country": self.rng.choice(["IN", "US", "GB", "AE"]),
                "home_country": home_country,
                "is_new_device": True,
                "destination_account_age_days": None,
                "auth_status": "approved",
                "label": 1,  # bust-out is fraud
                "attack_type": "synthetic_identity_fraud",
                "distinct_merchants_touched": 1,
                "decline_rate": 0.0,
                "is_micro_auth": False,
            })

        df = pd.DataFrame(rows)
        return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic identity fraud lifecycles.")
    parser.add_argument("--num-accounts", type=int, default=25, help="Number of synthetic accounts")
    parser.add_argument("--output", type=str, default="data/synthetic_attacks/synthetic_identity_attacks.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator = SyntheticIdentityGenerator(num_accounts=args.num_accounts, seed=args.seed)
    df = generator.generate()
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} synthetic identity transactions across {args.num_accounts} accounts -> {output_path}")


if __name__ == "__main__":
    main()
