"""Evaluate uncertainty on the real freMTPL2 held-out partition."""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.baselines.evaluate_holdout import evaluate_holdout
from experiments.uncertainty.uncertainty_metrics import (
    UncertaintyMetrics,
    evaluate_uncertainty,
)


def evaluate_b2_uncertainty():
    """Return calibration and selective-classification metrics."""
    holdout = evaluate_holdout()
    probabilities = np.asarray(holdout['probabilities'], dtype=float)
    # Binary model: the top-label confidence is how sure the model is of its own
    # label (the claim-indicator class it selected).
    confidences = np.maximum(probabilities, 1.0 - probabilities)
    results = evaluate_uncertainty(
        holdout['y_pred'],
        holdout['y_true'],
        confidences,
    )
    results['probability_brier_score'] = UncertaintyMetrics().brier_score(
        holdout['y_true'], probabilities
    )

    print(f"Test partition: {holdout['test_size']} records")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Top-label ECE: {results['top_label_ece']:.4f}")
    print(f"Top-label Brier score: {results['top_label_brier_score']:.4f}")
    print(f"Probability Brier score (binary): {results['probability_brier_score']:.4f}")
    print(f"\nConfidence Statistics:")
    for k, v in results['confidence_statistics'].items():
        print(f"  {k}: {v:.4f}")

    print(f"\nEscalation Rates (confidence below threshold):")
    for k, v in results['escalation_rates'].items():
        print(f"  {k}: {v:.4f} ({v*100:.2f}%)")

    print(f"\nSelective Accuracy (accuracy on high-confidence cases):")
    for k, v in results['selective_accuracy'].items():
        print(f"  {k}: accuracy={v['accuracy']:.4f}, coverage={v['coverage']:.4f}")

    print(f"\nRisk-Coverage Curve (first 5 points):")
    for i, (cov, acc) in enumerate(results['risk_coverage_curve'][:5]):
        print(f"  Point {i}: coverage={cov:.4f}, accuracy={acc:.4f}")

    return results


if __name__ == "__main__":
    evaluate_b2_uncertainty()
