"""
red_team/generators/phishing_generator.py

AI Defense Lab — Phishing & Cross-Channel Attack Generator
Mastercard Innovation Challenge 2026

Simulates two cross-channel phishing attack vectors from `attack_taxonomy.yaml`:
1. `hyper_personalized_phishing` — click-to-cashout sequence:
   Phishing link click -> App login (new device) -> Password reset -> High-value transfer.
2. `phishing_ato_wire_chain` — 3-channel chain:
   Email link click -> App login (new device) -> Call-center-authorized wire transfer.

Includes cross-channel sequence metadata (`chain_id`, `event_sequence_position`, `time_since_previous_event_sec`).
"""

from __future__ import annotations

import argparse
import random
import string
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

BASELINE_CSV_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "baseline_genuine" / "baseline_transactions.csv"


def load_baseline_profiles(baseline_path: Path = BASELINE_CSV_PATH) -> List[Dict[str, Any]]:
    if not baseline_path.exists():
        return []
    df = pd.read_csv(baseline_path)
    profiles = df.groupby("customer_id").first().reset_index()
    return profiles.to_dict(orient="records")


class PhishingGenerator:
    """
    Generates multi-stage cross-channel phishing attack sequences.
    """

    def __init__(
        self,
        num_chains: int = 25,
        attack_type: str = "all",  # 'hyper_personalized_phishing', 'phishing_ato_wire_chain', or 'all'
        baseline_path: Path = BASELINE_CSV_PATH,
        seed: int = 42,
    ):
        self.num_chains = num_chains
        self.attack_type = attack_type
        self.rng = random.Random(seed)
        np.random.seed(seed)
        self.profiles = load_baseline_profiles(baseline_path)

    def generate(self) -> pd.DataFrame:
        rows = []
        start_date = datetime.fromisoformat("2026-06-01")

        for _ in range(self.num_chains):
            if self.profiles:
                profile = self.rng.choice(self.profiles)
                customer_id = profile["customer_id"]
                account_id = profile["account_id"]
                name = profile["customer_name"]
                email = profile["customer_email"]
                home_country = profile["home_country"]
                account_age_days = profile["account_age_days"]
            else:
                customer_id = "cust_phish_" + uuid.uuid4().hex[:8]
                account_id = "acc_" + uuid.uuid4().hex[:10]
                name = "Ishaan Patel"
                email = "ishaan.patel@gmail.com"
                home_country = "IN"
                account_age_days = 450

            chain_id = "chain_phish_" + uuid.uuid4().hex[:8]
            bot_device_id = "dev_phish_" + "".join(self.rng.choices(string.ascii_lowercase + string.digits, k=8))
            bot_ip_country = self.rng.choice(["IN", "US", "GB", "AE"])

            chain_start = start_date + timedelta(
                days=self.rng.randint(0, 70),
                hours=self.rng.randint(8, 22),
                minutes=self.rng.randint(0, 59),
            )

            # Pick sub-attack variant
            variant = self.attack_type
            if variant == "all":
                variant = self.rng.choice(["hyper_personalized_phishing", "phishing_ato_wire_chain"])

            if variant == "hyper_personalized_phishing":
                # Sequence: Email Click (1) -> New Device Login (2) -> Password Reset (3) -> High Value Cashout (4)
                t1 = chain_start
                t2 = t1 + timedelta(seconds=self.rng.randint(30, 180))
                t3 = t2 + timedelta(seconds=self.rng.randint(60, 300))
                t4 = t3 + timedelta(seconds=self.rng.randint(120, 600))

                cashout_amount = round(float(np.random.uniform(850.0, 3500.0)), 2)

                # Final cashout transaction record
                rows.append({
                    "transaction_id": "txn_phish_cashout_" + uuid.uuid4().hex[:10],
                    "customer_id": customer_id,
                    "account_id": account_id,
                    "customer_name": name,
                    "customer_email": email,
                    "account_age_days": account_age_days,
                    "timestamp": t4.isoformat(),
                    "amount": cashout_amount,
                    "currency": "INR" if home_country == "IN" else "USD",
                    "merchant_category": "electronics",
                    "merchant_id": "merch_phish_" + uuid.uuid4().hex[:6],
                    "channel": "app_transfer",
                    "device_id": bot_device_id,
                    "ip_country": bot_ip_country,
                    "home_country": home_country,
                    "is_new_device": True,
                    "destination_account_age_days": self.rng.randint(1, 10),
                    "auth_status": "approved",
                    "label": 1,
                    "attack_type": "hyper_personalized_phishing",
                    "chain_id": chain_id,
                    "event_sequence_position": 4,
                    "time_since_previous_event_sec": int((t4 - t3).total_seconds()),
                    "distinct_merchants_touched": 1,
                    "decline_rate": 0.0,
                    "is_micro_auth": False,
                })

            else:  # phishing_ato_wire_chain
                # Sequence: Email Click (1) -> App Login (2) -> Call Center Wire Transfer (3)
                t1 = chain_start
                t2 = t1 + timedelta(seconds=self.rng.randint(60, 300))
                t3 = t2 + timedelta(seconds=self.rng.randint(300, 1200))

                wire_amount = round(float(np.random.uniform(1500.0, 5000.0)), 2)

                rows.append({
                    "transaction_id": "txn_phish_wire_" + uuid.uuid4().hex[:10],
                    "customer_id": customer_id,
                    "account_id": account_id,
                    "customer_name": name,
                    "customer_email": email,
                    "account_age_days": account_age_days,
                    "timestamp": t3.isoformat(),
                    "amount": wire_amount,
                    "currency": "INR" if home_country == "IN" else "USD",
                    "merchant_category": "travel",
                    "merchant_id": "merch_wire_" + uuid.uuid4().hex[:6],
                    "channel": "wire",
                    "device_id": bot_device_id,
                    "ip_country": bot_ip_country,
                    "home_country": home_country,
                    "is_new_device": True,
                    "destination_account_age_days": self.rng.randint(1, 5),
                    "auth_status": "approved",
                    "label": 1,
                    "attack_type": "phishing_ato_wire_chain",
                    "chain_id": chain_id,
                    "event_sequence_position": 3,
                    "time_since_previous_event_sec": int((t3 - t2).total_seconds()),
                    "distinct_merchants_touched": 1,
                    "decline_rate": 0.0,
                    "is_micro_auth": False,
                })

        df = pd.DataFrame(rows)
        return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic phishing & cross-channel attack chains.")
    parser.add_argument("--num-chains", type=int, default=25, help="Number of phishing chains")
    parser.add_argument("--output", type=str, default="data/synthetic_attacks/phishing_attacks.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator = PhishingGenerator(num_chains=args.num_chains, seed=args.seed)
    df = generator.generate()
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} phishing & cross-channel attack transactions -> {output_path}")


if __name__ == "__main__":
    main()
