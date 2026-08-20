"""
red_team/generators/model_evasion_generator.py

AI Defense Lab — Model Evasion Attack Generator
Mastercard Innovation Challenge 2026

Simulates "Model Evasion" attack (ID: model_evasion) based on the spec in
`red_team/taxonomy/attack_taxonomy.yaml`.

Attack pattern:
- The attacker probes the classifier and perturbs transaction features
  (amounts, intervals, device indicators) to sit right below detection thresholds.
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


class ModelEvasionGenerator:
    """
    Generates synthetic adversarial model evasion transactions designed to probe
    and stay under Blue Team decision boundaries.
    """

    def __init__(
        self,
        num_attacks: int = 40,
        perturbation_magnitude: float = 0.10,
        baseline_path: Path = BASELINE_CSV_PATH,
        seed: int = 42,
    ):
        self.num_attacks = num_attacks
        self.perturbation_magnitude = perturbation_magnitude
        self.rng = random.Random(seed)
        np.random.seed(seed)
        self.profiles = load_baseline_profiles(baseline_path)

    def generate(self) -> pd.DataFrame:
        rows = []
        start_date = datetime.fromisoformat("2026-06-01")

        for _ in range(self.num_attacks):
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
                customer_id = "cust_evasion_" + uuid.uuid4().hex[:8]
                account_id = "acc_" + uuid.uuid4().hex[:10]
                name = "Pooja Sharma"
                email = "pooja.sharma@gmail.com"
                home_country = "IN"
                account_age_days = 500
                device_id = "dev_evasion_" + uuid.uuid4().hex[:8]

            # Perturbed transaction sitting right around normal spend thresholds
            txn_time = start_date + timedelta(
                days=self.rng.randint(0, 75),
                hours=self.rng.randint(9, 21),
                minutes=self.rng.randint(0, 59),
            )

            # Perturbed amount slightly below review cutoff (e.g. $95 - $145)
            base_amount = 120.0
            amount = round(base_amount * (1.0 - (self.perturbation_magnitude * float(np.random.uniform(0.1, 1.0)))), 2)

            rows.append({
                "transaction_id": "txn_evasion_" + uuid.uuid4().hex[:12],
                "customer_id": customer_id,
                "account_id": account_id,
                "customer_name": name,
                "customer_email": email,
                "account_age_days": account_age_days,
                "timestamp": txn_time.isoformat(),
                "amount": amount,
                "currency": "INR" if home_country == "IN" else "USD",
                "merchant_category": "ecommerce_general",
                "merchant_id": "merch_evasion_" + uuid.uuid4().hex[:6],
                "channel": "card_not_present",
                "device_id": device_id,  # uses recognized home device
                "ip_country": home_country,
                "home_country": home_country,
                "is_new_device": False,
                "destination_account_age_days": None,
                "auth_status": "approved",
                "label": 1,  # fraud intent, but adversarial evasion
                "attack_type": "model_evasion",
                "distinct_merchants_touched": 1,
                "decline_rate": 0.0,
                "is_micro_auth": False,
            })

        df = pd.DataFrame(rows)
        return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic model evasion transactions.")
    parser.add_argument("--num-attacks", type=int, default=40, help="Number of evasion transactions")
    parser.add_argument("--output", type=str, default="data/synthetic_attacks/model_evasion_attacks.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generator = ModelEvasionGenerator(num_attacks=args.num_attacks, seed=args.seed)
    df = generator.generate()
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} model evasion attack transactions -> {output_path}")


if __name__ == "__main__":
    main()
