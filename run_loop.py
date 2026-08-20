"""
run_loop.py

AI Defense Lab — Multi-Round Adversarial Sparring Loop Runner
Mastercard Innovation Challenge 2026

Executes the closed-loop adversarial feedback engine:
Attacker generates fraud -> Defender scores -> Missed attacks mutate -> Retrain -> Repeat.
"""

from __future__ import annotations

import argparse
from loop.orchestrator import AdversarialOrchestrator


def main():
    parser = argparse.ArgumentParser(description="Run closed-loop adversarial sparring across N rounds.")
    parser.add_argument("--rounds", type=int, default=3, help="Number of sparring rounds (default: 3)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision cutoff threshold (default: 0.5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    orchestrator = AdversarialOrchestrator(
        num_rounds=args.rounds,
        threshold=args.threshold,
        seed=args.seed,
    )
    orchestrator.run_loop()


if __name__ == "__main__":
    main()
