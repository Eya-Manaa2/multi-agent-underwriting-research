"""B2 - machine-learning model only (the shipped XGBoost claim model).

The reference baseline: the real freMTPL2 model is used alone, with no rule engine and
no additional agent. Its predicted claim probability is mapped onto the pilot
recommendation vocabulary through the documented business thresholds. Metrics are
computed on the same reproducible hold-out partition as every other baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.baselines.evaluate_holdout import evaluate_holdout  # noqa: E402
from experiments.baselines.evaluation import (  # noqa: E402
    decision_summary,
    save_result,
    score_metrics,
)


def run() -> dict:
    holdout = evaluate_holdout()
    payload = {
        'baseline': 'B2_ml_only',
        'description': 'shipped XGBoost claim model alone (no rules, no agents)',
        'dataset_size': holdout['dataset_size'],
        'test_size': holdout['test_size'],
        'records_skipped': holdout['records_skipped'],
        **score_metrics(holdout['y_true'], holdout['probabilities']),
        **decision_summary(holdout['y_true'], holdout['decisions']),
    }
    return payload


if __name__ == '__main__':
    result = run()
    print(f"Dataset: {result['dataset_size']:,} real policies")
    print(f"Hold-out partition: {result['test_size']:,} policies")
    print(f"Base claim rate: {result['base_rate']:.4f}")
    print(f"ROC-AUC: {result['roc_auc']:.4f}")
    print(f"PR-AUC : {result['pr_auc']:.4f}")
    print(f"Accuracy @0.5: {result['accuracy_at_0_5']:.4f}")
    print(f"Decision counts: {result['decision_counts']}")
    print(
        f"Adverse flag: rate {result['flagged_rate']:.4f} | "
        f"precision {result['flag_precision']:.4f} | recall {result['flag_recall']:.4f}"
    )
    print(f"Saved: {save_result('b2_ml_only', result)}")
