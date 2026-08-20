"""
loop/orchestrator.py

AI Defense Lab — Closed-Loop Adversarial Orchestrator
Mastercard Innovation Challenge 2026

Coordinates multi-round adversarial sparring between the Red Team (Attackers)
and Blue Team (Defenders):
- Round 1: Baseline defender trained on basic fraud patterns -> Evaluated on initial attack batch.
- Round 2: Red Team MutationEngine ingests Round 1 missed fraud and mutates attack knobs -> Evaluated on mutated Round 2 test set.
- Round 3: Red Team further escalates with higher mutation intensity -> Evaluated on mutated Round 3 test set.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from blue_team.features.feature_pipeline import FeaturePipeline
from blue_team.models.anomaly_model import AnomalyDetector
from blue_team.models.tabular_model import TabularFraudClassifier
from data.merge_datasets import merge_datasets
from loop.metrics import MetricsEngine
from red_team.generators.card_testing_generator import CardTestingGenerator
from red_team.generators.model_evasion_generator import ModelEvasionGenerator
from red_team.generators.structuring_generator import StructuringGenerator
from red_team.generators.synthetic_identity_generator import SyntheticIdentityGenerator
from red_team.mutation.mutation_engine import MutationEngine

from red_team.generators.phishing_generator import PhishingGenerator

REPO_ROOT = Path(__file__).resolve().parent.parent


class AdversarialOrchestrator:
    """
    Master controller executing multi-round closed-loop adversarial sparring.
    """

    def __init__(self, num_rounds: int = 3, threshold: float = 0.5, seed: int = 42):
        self.num_rounds = num_rounds
        self.threshold = threshold
        self.seed = seed

        self.mutation_engine = MutationEngine(seed=seed)
        self.metrics_engine = MetricsEngine()
        self.feature_pipeline = FeaturePipeline()

    def generate_initial_attacks(self) -> List[pd.DataFrame]:
        """Generates initial attack datasets across 5 diverse categories."""
        print("[Red Team] Generating initial attack datasets across 5 attack categories...")

        ct_df = CardTestingGenerator(num_bursts=45, seed=self.seed).generate()
        struct_df = StructuringGenerator(num_series=30, seed=self.seed).generate()
        synth_df = SyntheticIdentityGenerator(num_accounts=25, seed=self.seed).generate()
        evasion_df = ModelEvasionGenerator(num_attacks=35, seed=self.seed).generate()
        phish_df = PhishingGenerator(num_chains=25, seed=self.seed).generate()

        attacks_dir = REPO_ROOT / "data" / "synthetic_attacks"
        attacks_dir.mkdir(parents=True, exist_ok=True)

        ct_df.to_csv(attacks_dir / "card_testing_attacks.csv", index=False)
        struct_df.to_csv(attacks_dir / "structuring_attacks.csv", index=False)
        synth_df.to_csv(attacks_dir / "synthetic_identity_attacks.csv", index=False)
        evasion_df.to_csv(attacks_dir / "model_evasion_attacks.csv", index=False)
        phish_df.to_csv(attacks_dir / "phishing_attacks.csv", index=False)

        return [ct_df, struct_df, synth_df, evasion_df, phish_df]

    def run_loop(self) -> List[Dict[str, Any]]:
        """Executes N rounds of adversarial sparring with dynamic per-round attack mutation & retraining."""
        print("=" * 75)
        print("   AI DEFENSE LAB — ADVERSARIAL CLOSED-LOOP ORCHESTRATOR")
        print("=" * 75)

        self.generate_initial_attacks()

        baseline_path = REPO_ROOT / "data" / "baseline_genuine" / "baseline_transactions.csv"
        attacks_dir = REPO_ROOT / "data" / "synthetic_attacks"
        merged_output_path = REPO_ROOT / "data" / "merged_dataset.csv"

        # Baseline genuine + initial attacks merged
        initial_merged = merge_datasets(
            baseline_path=baseline_path,
            attacks_dir=attacks_dir,
            output_path=merged_output_path,
        )

        current_round_df = initial_merged.copy()
        training_history_dfs = []

        for r in range(1, self.num_rounds + 1):
            print(f"\n" + "-" * 75)
            print(f"  EXECUTING ADVERSARIAL SPARRING ROUND {r}/{self.num_rounds}")
            print("-" * 75)

            # Feature Engineering & GroupSplit for current round dataset
            X, y, feature_names = self.feature_pipeline.fit_transform(current_round_df)
            attack_types_series = current_round_df["attack_type"].fillna("genuine")
            customer_ids = current_round_df["customer_id"].values

            gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=self.seed + r)
            train_idx, test_idx = next(gss.split(X, y, groups=customer_ids))

            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            atk_test = attack_types_series.iloc[test_idx]

            # Construct Training Data
            if r == 1:
                # Round 1: Legacy model trained on baseline genuine + standard card testing only
                r1_train = current_round_df.iloc[train_idx]
                r1_mask = (r1_train["attack_type"].isin(["card_testing_bots", np.nan, None])) | (r1_train["label"] == 0)
                train_data_df = r1_train[r1_mask]
                X_train, y_train, _ = self.feature_pipeline.fit_transform(train_data_df)
                print(f"[Blue Team R1] Training initial defender on baseline patterns ({len(X_train)} samples)...")
            else:
                # Round 2+: Retrains on cumulative training pool + previous round mutated attacks
                cumulative_train_df = pd.concat(training_history_dfs, ignore_index=True)
                X_train, y_train, _ = self.feature_pipeline.fit_transform(cumulative_train_df)
                print(f"[Blue Team R{r}] Retraining defender on expanded feedback dataset ({len(X_train)} samples)...")

            # Store current round train partition into history
            training_history_dfs.append(current_round_df.iloc[train_idx])

            # Train Classifier & Anomaly Detector
            classifier = TabularFraudClassifier(random_state=self.seed + r)
            classifier.fit(X_train, y_train)

            anomaly_detector = AnomalyDetector(random_state=self.seed + r)
            anomaly_detector.fit(X_train[y_train == 0])

            # Predict & Score on current round test set
            y_prob_super = classifier.predict_proba(X_test)
            y_prob_anom = anomaly_detector.predict_anomaly_score(X_test)

            y_prob_fused = 0.80 * y_prob_super + 0.20 * y_prob_anom
            y_pred = (y_prob_fused >= self.threshold).astype(int)

            # Evaluate Round Metrics
            round_data = self.metrics_engine.evaluate_round(
                round_num=r,
                y_true=y_test.values,
                y_pred=y_pred,
                y_prob=y_prob_fused,
                attack_types=atk_test,
                threshold=self.threshold,
            )

            print(f"\n--- ROUND {r} PERFORMANCE BENCHMARK ---")
            print(f"Recall (Detection Rate): {round_data['recall']*100:.2f}%  |  Precision: {round_data['precision']*100:.2f}%  |  F1 Score: {round_data['f1']*100:.2f}%")
            print(f"Caught Fraud: {round_data['tp']}  |  Missed Fraud (FN): {round_data['fn']}  |  False Alarms (FP): {round_data['fp']}")

            print("\nPer-Attack Category Detection Rates:")
            for atype, stats in round_data["per_attack_metrics"].items():
                print(f"  - {atype:28s}: {stats['detection_rate']*100:6.2f}% ({stats['caught']}/{stats['total']} caught)")

            # Extract Missed Fraud & Mutate for Next Round
            if r < self.num_rounds:
                missed_mask = (y_test.values == 1) & (y_pred == 0)
                missed_indices = X_test.index[missed_mask]

                if len(missed_indices) > 0:
                    missed_rows = current_round_df.iloc[missed_indices].copy()
                    print(f"\n[Feedback Router] Identified {len(missed_rows)} missed fraud attempts in Round {r}. Routing to Red Team...")

                    mutated_df, mut_summary = self.mutation_engine.mutate_missed_attacks(missed_rows, round_num=r + 1)
                    if not mutated_df.empty:
                        # Next round evaluates on dataset containing mutated attacks
                        current_round_df = pd.concat([initial_merged, mutated_df], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

        self.metrics_engine.print_history_summary()
        return self.metrics_engine.round_history


def main():
    parser = argparse.ArgumentParser(description="Run multi-round closed-loop adversarial sparring.")
    parser.add_argument("--rounds", type=int, default=3, help="Number of adversarial rounds to run")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    orchestrator = AdversarialOrchestrator(num_rounds=args.rounds, threshold=args.threshold, seed=args.seed)
    orchestrator.run_loop()


if __name__ == "__main__":
    main()
