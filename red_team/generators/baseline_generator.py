"""
red_team/generators/baseline_generator.py

AI Defense Lab — Baseline Genuine Transaction Generator
Mastercard Innovation Challenge 2026

Generates realistic "normal" (non-fraud) payment transactions. This is the
foundation dataset every attack batch gets merged into — a fraud detector
is useless without a realistic population of genuine behavior to contrast
against.

Design principles:
- Each simulated customer has a stable "home profile": a usual device, a
  usual country/IP, a preferred spending window, and a personal spending
  scale. Their transactions vary around that profile, not randomly from
  scratch every time — this is what makes the baseline realistic instead
  of uniform noise.
- Amounts follow a log-normal distribution (realistic for retail spend:
  many small purchases, a long tail of larger ones).
- A small fraction of genuine activity is intentionally "borderline"
  (new device while traveling, a late-night purchase, etc.) so the
  defender model has to learn real decision boundaries instead of a
  trivial one.

No external services or paid APIs are required — uses only the Python
standard library plus numpy/pandas, so it runs identically for every
team member without extra setup.
"""

from __future__ import annotations

import argparse
import random
import string
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Reference data (kept small and explicit — easy for teammates to extend)
# --------------------------------------------------------------------------

MERCHANT_CATEGORIES = [
    ("grocery", 0.22),
    ("food_delivery", 0.14),
    ("fuel", 0.10),
    ("ecommerce_general", 0.16),
    ("utilities", 0.08),
    ("entertainment", 0.09),
    ("travel", 0.05),
    ("electronics", 0.06),
    ("healthcare", 0.05),
    ("apparel", 0.05),
]

CHANNELS = [
    ("card_present", 0.35),
    ("card_not_present", 0.40),
    ("app_transfer", 0.20),
    ("wire", 0.05),
]

COUNTRIES = ["IN", "US", "GB", "AE", "SG", "DE"]
COUNTRY_WEIGHTS = [0.70, 0.08, 0.06, 0.08, 0.05, 0.03]

FIRST_NAMES = ["Aarav", "Vivaan", "Aditya", "Ishaan", "Kabir", "Diya", "Ananya",
               "Myra", "Saanvi", "Priya", "Rohan", "Neha", "Karan", "Simran",
               "Arjun", "Meera", "Rahul", "Pooja", "Sana", "Dev"]
LAST_NAMES = ["Sharma", "Verma", "Patel", "Gupta", "Iyer", "Khan", "Reddy",
              "Nair", "Singh", "Das", "Mehta", "Rao", "Kapoor", "Joshi"]


def _weighted_choice(rng: random.Random, options_with_weights):
    options, weights = zip(*options_with_weights)
    return rng.choices(options, weights=weights, k=1)[0]


def _random_name(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def _random_email(name: str, rng: random.Random) -> str:
    handle = name.lower().replace(" ", ".") + str(rng.randint(1, 999))
    domain = rng.choice(["gmail.com", "yahoo.com", "outlook.com", "proton.me"])
    return f"{handle}@{domain}"


def _random_device_id(rng: random.Random) -> str:
    return "dev_" + "".join(rng.choices(string.ascii_lowercase + string.digits, k=10))


# --------------------------------------------------------------------------
# Customer profile — stable attributes that make transactions look coherent
# --------------------------------------------------------------------------

class CustomerProfile:
    """A simulated customer's stable 'home' behavior baseline."""

    def __init__(self, customer_id: str, rng: random.Random):
        self.customer_id = customer_id
        self.account_id = "acc_" + uuid.uuid4().hex[:12]
        self.name = _random_name(rng)
        self.email = _random_email(self.name, rng)
        self.home_country = _weighted_choice(rng, list(zip(COUNTRIES, COUNTRY_WEIGHTS)))
        self.home_device_id = _random_device_id(rng)
        # log-normal spend scale: most customers modest, some spend heavily
        self.spend_scale = float(np.random.lognormal(mean=3.2, sigma=0.6))
        # preferred active hours (most people transact 8am-10pm-ish, shifted per person)
        self.preferred_hour_center = rng.randint(9, 20)
        self.preferred_hour_spread = rng.randint(3, 6)
        self.account_age_days = rng.randint(30, 2500)


# --------------------------------------------------------------------------
# Core generator
# --------------------------------------------------------------------------

class BaselineGenerator:
    """
    Generates a population of genuine transactions across many simulated
    customers over a given date range.
    """

    def __init__(self, num_customers: int = 500, txns_per_customer_avg: int = 25,
                 start_date: str = "2026-06-01", end_date: str = "2026-08-19",
                 seed: int = 42):
        self.num_customers = num_customers
        self.txns_per_customer_avg = txns_per_customer_avg
        self.start_date = datetime.fromisoformat(start_date)
        self.end_date = datetime.fromisoformat(end_date)
        self.rng = random.Random(seed)
        np.random.seed(seed)

    def _random_timestamp_for(self, profile: CustomerProfile) -> datetime:
        """Pick a timestamp biased toward the customer's usual active hours."""
        span_days = (self.end_date - self.start_date).days
        day_offset = self.rng.randint(0, max(span_days, 1))
        base_day = self.start_date + timedelta(days=day_offset)

        hour = int(np.clip(
            np.random.normal(profile.preferred_hour_center, profile.preferred_hour_spread),
            0, 23
        ))
        minute = self.rng.randint(0, 59)
        second = self.rng.randint(0, 59)
        return base_day.replace(hour=hour, minute=minute, second=second)

    def _random_amount_for(self, profile: CustomerProfile, category: str) -> float:
        category_multiplier = {
            "grocery": 0.6, "food_delivery": 0.35, "fuel": 0.7,
            "ecommerce_general": 1.0, "utilities": 0.9, "entertainment": 0.5,
            "travel": 3.5, "electronics": 4.0, "healthcare": 1.5, "apparel": 0.8,
        }.get(category, 1.0)
        base = np.random.lognormal(mean=np.log(profile.spend_scale * category_multiplier + 1), sigma=0.5)
        return round(float(max(base, 1.0)), 2)

    def _device_and_ip_for(self, profile: CustomerProfile) -> tuple[str, str, bool]:
        """95% of the time: usual device/country. 5%: new device (e.g. travel) —
        genuine but borderline, so the defender learns this isn't automatically fraud."""
        if self.rng.random() < 0.95:
            return profile.home_device_id, profile.home_country, False
        else:
            new_device = _random_device_id(self.rng)
            new_country = self.rng.choice(COUNTRIES)
            return new_device, new_country, True

    def generate(self) -> pd.DataFrame:
        rows = []
        for i in range(self.num_customers):
            customer_id = "cust_" + uuid.uuid4().hex[:10]
            profile = CustomerProfile(customer_id, self.rng)

            n_txns = max(1, int(np.random.poisson(self.txns_per_customer_avg)))
            for _ in range(n_txns):
                category = _weighted_choice(self.rng, MERCHANT_CATEGORIES)
                channel = _weighted_choice(self.rng, CHANNELS)
                timestamp = self._random_timestamp_for(profile)
                amount = self._random_amount_for(profile, category)
                device_id, ip_country, is_new_device = self._device_and_ip_for(profile)

                rows.append({
                    "transaction_id": "txn_" + uuid.uuid4().hex[:14],
                    "customer_id": profile.customer_id,
                    "account_id": profile.account_id,
                    "customer_name": profile.name,
                    "customer_email": profile.email,
                    "account_age_days": profile.account_age_days,
                    "timestamp": timestamp.isoformat(),
                    "amount": amount,
                    "currency": "INR" if profile.home_country == "IN" else "USD",
                    "merchant_category": category,
                    "merchant_id": "merch_" + uuid.uuid4().hex[:8],
                    "channel": channel,
                    "device_id": device_id,
                    "ip_country": ip_country,
                    "home_country": profile.home_country,
                    "is_new_device": is_new_device,
                    "destination_account_age_days": None,   # not applicable for genuine spend
                    "label": 0,                              # 0 = genuine, 1 = fraud (set by attack generators)
                    "attack_type": None,
                })

        df = pd.DataFrame(rows)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df


# --------------------------------------------------------------------------
# CLI entry point
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate baseline genuine transactions.")
    parser.add_argument("--num_customers", type=int, default=500)
    parser.add_argument("--txns_per_customer_avg", type=int, default=25)
    parser.add_argument("--start_date", type=str, default="2026-06-01")
    parser.add_argument("--end_date", type=str, default="2026-08-19")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out", type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "baseline_genuine" / "baseline_transactions.csv")
    )
    args = parser.parse_args()

    generator = BaselineGenerator(
        num_customers=args.num_customers,
        txns_per_customer_avg=args.txns_per_customer_avg,
        start_date=args.start_date,
        end_date=args.end_date,
        seed=args.seed,
    )
    df = generator.generate()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} genuine transactions across {args.num_customers} customers.")
    print(f"Saved to: {out_path}")
    print("\nSample rows:")
    print(df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()