"""
red_team/mutation/mutation_engine.py

AI Defense Lab — Adversarial Mutation Engine
Mastercard Innovation Challenge 2026

Ingests missed fraud transactions (False Negatives) from the Blue Team defender,
reads `mutation_knobs` from `attack_taxonomy.yaml`, and produces harder attack variants
for subsequent rounds.

Logs explicit mutation parameters per round so judges can trace attacker escalation.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml

TAXONOMY_PATH = Path(__file__).resolve().parent.parent / "taxonomy" / "attack_taxonomy.yaml"


def load_taxonomy_knobs(taxonomy_path: Path = TAXONOMY_PATH) -> Dict[str, List[str]]:
    if not taxonomy_path.exists():
        return {}
    with open(taxonomy_path, "r", encoding="utf-8") as f:
        taxonomy = yaml.safe_load(f)

    knobs_map = {}
    for attack in taxonomy.get("attacks", []):
        knobs_map[attack["id"]] = attack.get("mutation_knobs", [])
    return knobs_map


class MutationEngine:
    """
    Adversarial Mutation Engine that evolves missed attack parameters round-over-round.
    """

    def __init__(self, seed: int = 42, taxonomy_path: Path = TAXONOMY_PATH):
        self.rng = random.Random(seed)
        np.random.seed(seed)
        self.knobs_map = load_taxonomy_knobs(taxonomy_path)

    def mutate_missed_attacks(
        self, missed_df: pd.DataFrame, round_num: int = 2
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Takes missed attack rows (False Negatives) and produces a mutated dataset
        crafted to be harder to detect in the next round.
        Returns (mutated_df, mutation_summary).
        """
        if missed_df.empty:
            return pd.DataFrame(), {"status": "no_missed_attacks"}

        mutated_rows = []
        mutation_details = []

        # Intensity increases per round
        intensity_mult = 1.0 + (0.15 * (round_num - 2))

        for _, row in missed_df.iterrows():
            attack_type = row.get("attack_type", "card_testing_bots")
            knobs = self.knobs_map.get(attack_type, [])

            mutated_row = row.to_dict()
            mutated_row["transaction_id"] = f"txn_mut_r{round_num}_" + str(row.get("transaction_id", ""))[-10:]

            applied_knobs = []

            if attack_type == "card_testing_bots":
                # Knob 1: Lower decline rate closer to benign retry rate
                if "decline_rate" in mutated_row and pd.notna(mutated_row["decline_rate"]):
                    old_rate = float(mutated_row["decline_rate"])
                    new_rate = max(0.12, old_rate - (0.20 * intensity_mult))
                    mutated_row["decline_rate"] = round(new_rate, 2)
                    applied_knobs.append(f"decline_rate: {old_rate:.2f}->{new_rate:.2f}")

                # Knob 2: Use recognized home device 75% of time to defeat is_new_device
                mutated_row["is_new_device"] = False if self.rng.random() < 0.75 else True
                applied_knobs.append("is_new_device->False (recognized device mimicry)")

                # Knob 3: Nudge amount to normal spend range ($5 - $18)
                old_amt = float(mutated_row["amount"])
                new_amt = round(float(np.random.uniform(5.00, 18.00)), 2)
                mutated_row["amount"] = new_amt
                applied_knobs.append(f"amount: ${old_amt}->${new_amt}")

            elif attack_type == "low_and_slow_structuring":
                # Knob 1: Lower per-transaction amount by 20%
                old_amt = float(mutated_row["amount"])
                new_amt = round(old_amt * (0.80 / intensity_mult), 2)
                mutated_row["amount"] = new_amt
                applied_knobs.append(f"per_txn_amount: ${old_amt}->${new_amt}")

                # Knob 2: Extend account age and recipient age
                mutated_row["destination_account_age_days"] = self.rng.randint(60, 240)
                applied_knobs.append("destination_account_age->60-240d (mule aging)")

            elif attack_type == "synthetic_identity_fraud":
                # Knob 1: Extend aging period
                old_age = int(mutated_row["account_age_days"])
                new_age = old_age + int(25 * intensity_mult)
                mutated_row["account_age_days"] = new_age
                applied_knobs.append(f"aging_period: {old_age}d->{new_age}d")

                # Knob 2: Lower bust-out amount spike closer to upper genuine range
                old_amt = float(mutated_row["amount"])
                new_amt = round(old_amt * 0.65, 2)
                mutated_row["amount"] = new_amt
                applied_knobs.append(f"bust_out_amount: ${old_amt}->${new_amt}")

            elif attack_type == "model_evasion":
                # Perturb amount 10% lower to sit right below decision boundary
                old_amt = float(mutated_row["amount"])
                new_amt = round(old_amt * (0.90 / intensity_mult), 2)
                mutated_row["amount"] = new_amt
                mutated_row["is_new_device"] = False
                applied_knobs.append(f"perturbation_magnitude: amount ${old_amt}->${new_amt}")

            else:
                old_amt = float(mutated_row["amount"])
                new_amt = round(old_amt * 0.85, 2)
                mutated_row["amount"] = new_amt
                applied_knobs.append(f"generic_amount_nudge: ${old_amt}->${new_amt}")

            mutated_rows.append(mutated_row)
            mutation_details.append({
                "attack_type": attack_type,
                "applied_knobs": applied_knobs,
            })

        mutated_df = pd.DataFrame(mutated_rows)
        summary = {
            "round": round_num,
            "mutated_count": len(mutated_df),
            "intensity_multiplier": round(intensity_mult, 2),
            "sample_mutations": mutation_details[:3],
        }

        print(f"\n[MutationEngine] Round {round_num} Escalation (Intensity {intensity_mult:.2f}x):")
        print(f"  Mutated {len(mutated_df)} missed attacks using taxonomy knobs:")
        for det in mutation_details[:3]:
            print(f"   - [{det['attack_type']}]: {', '.join(det['applied_knobs'])}")

        return mutated_df, summary
