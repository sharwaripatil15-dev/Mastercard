"""
run_pipeline.py

AI Defense Lab — End-to-End Proof of Concept Pipeline
Mastercard Innovation Challenge 2026

Executes the full vertical slice (Phase 1) end-to-end:
1. Generate Card Testing attack transactions (`red_team/generators/card_testing_generator.py`)
2. Merge baseline genuine data + attack dataset (`data/merge_datasets.py`)
3. Engineer features (`blue_team/features/feature_pipeline.py`)
4. Train XGBoost Fraud Classifier (`blue_team/models/tabular_model.py`)
5. Evaluate model performance & print metrics (`blue_team/models/evaluate.py`)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from blue_team.features.feature_pipeline import FeaturePipeline
from blue_team.models.evaluate import evaluate_predictions, print_evaluation_report
from blue_team.models.tabular_model import TabularFraudClassifier
from data.merge_datasets import merge_datasets
from red_team.generators.card_testing_generator import CardTestingGenerator

REPO_ROOT = Path(__file__).resolve().parent


def run_pipeline(
    num_bursts: int = 50,
    merchants_per_burst: int = 10,
    window_minutes: int = 10,
    decline_rate: float = 0.75,
    threshold: float = 0.5,
    seed: int = 42,
) -> None:
    """Executes full end-to-end vertical slice."""
    print("=" * 60)
    print("  AI DEFENSE LAB — PHASE 1 VERTICAL SLICE PIPELINE")
    print("=" * 60)

    # Step 1: Generate Card Testing Attack Data
    attack_output_path = REPO_ROOT / "data" / "synthetic_attacks" / "card_testing_attacks.csv"
    print("\n[Step 1/5] Generating synthetic Card Testing Bot attack dataset...")
    generator = CardTestingGenerator(
        num_bursts=num_bursts,
        distinct_merchants_touched=merchants_per_burst,
        window_minutes=window_minutes,
        decline_rate=decline_rate,
        seed=seed,
    )
    attack_df = generator.generate()
    attack_output_path.parent.mkdir(parents=True, exist_ok=True)
    attack_df.to_csv(attack_output_path, index=False)
    print(f"Generated {len(attack_df)} attack transactions across {num_bursts} bursts -> {attack_output_path}")

    # Step 2: Merge Baseline + Attacks
    baseline_path = REPO_ROOT / "data" / "baseline_genuine" / "baseline_transactions.csv"
    merged_output_path = REPO_ROOT / "data" / "merged_dataset.csv"
    print("\n[Step 2/5] Merging baseline dataset with synthetic attacks...")
    merged_df = merge_datasets(
        baseline_path=baseline_path,
        attacks_dir=attack_output_path.parent,
        output_path=merged_output_path,
    )

    # Step 3: Feature Engineering
    print("\n[Step 3/5] Extracting tabular features...")
    pipeline = FeaturePipeline()
    X, y, feature_names = pipeline.fit_transform(merged_df)
    customer_ids = merged_df["customer_id"].values
    print(f"Extracted {X.shape[1]} features for {X.shape[0]} transactions across {len(set(customer_ids))} distinct customers.")

    # Train/Test Split (Customer-Level GroupShuffleSplit — 0% customer overlap)
    print("\nSplitting dataset into train (80%) and test (20%) sets by customer_id (GroupShuffleSplit)...")
    from sklearn.model_selection import GroupShuffleSplit
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=customer_ids))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f"Train set: {X_train.shape[0]} samples (Fraud: {(y_train == 1).sum()}, Customers: {len(set(customer_ids[train_idx]))})")
    print(f"Test set:  {X_test.shape[0]} samples (Fraud: {(y_test == 1).sum()}, Customers: {len(set(customer_ids[test_idx]))})")

    # Step 4: Model Training
    print("\n[Step 4/5] Training XGBoost Tabular Fraud Classifier...")
    classifier = TabularFraudClassifier(random_state=seed)
    classifier.fit(X_train, y_train)

    # Step 5: Model Evaluation
    print("\n[Step 5/5] Evaluating classifier on held-out test set...")
    y_prob = classifier.predict_proba(X_test)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = evaluate_predictions(y_test.values, y_pred, y_prob, threshold=threshold)
    print_evaluation_report(metrics, title="Phase 1 Baseline Classifier Performance")

    # Top Feature Importances
    importances = classifier.get_feature_importances()
    print("Top 10 Feature Importances:")
    for feat, imp in importances.head(10).items():
        print(f"  - {feat:30s}: {imp:.4f}")

    print("\nPipeline execution completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="Run full end-to-end Phase 1 pipeline.")
    parser.add_argument("--num-bursts", type=int, default=50, help="Number of attack bursts to generate")
    parser.add_argument("--merchants-per-burst", type=int, default=10, help="Distinct merchants per burst")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for fraud detection")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    run_pipeline(
        num_bursts=args.num_bursts,
        merchants_per_burst=args.merchants_per_burst,
        threshold=args.threshold,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
