"""Evaluate fairness on the reproducible real freMTPL2 hold-out partition."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.baselines.evaluate_holdout import evaluate_holdout
from experiments.fairness.fairness_metrics import evaluate_fairness


def evaluate_b2_fairness():
    """Return approval parity and subgroup metrics for the real hold-out partition."""
    holdout = evaluate_holdout()
    results = evaluate_fairness(holdout['records'], holdout['decisions'])

    print(f"Test partition: {holdout['test_size']} records")
    print("Positive outcome: approval (decision other than Reject)")
    print(
        "Driver age and bonus-malus are analytical proxies, "
        "not protected-class conclusions."
    )

    for metric_name, metric_value in results.items():
        print(f"{metric_name}:")
        if isinstance(metric_value, dict) and 'ratio' in metric_value:
            print(f"  Ratio: {metric_value['ratio']:.4f}")
            print(f"  Rates: {metric_value['rates']}")
        elif isinstance(metric_value, dict):
            for k, v in metric_value.items():
                print(f"  {k}: {v:.4f}")
        else:
            print(f"  {metric_value}")

    return results


if __name__ == "__main__":
    evaluate_b2_fairness()
