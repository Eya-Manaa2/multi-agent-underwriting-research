"""Reproduce the real freMTPL2 hold-out evaluation of the shipped model.

The former module scored a 3-class model against a small synthetic dataset. It now
evaluates the model actually shipped in ``models/`` on the held-out partition of the
real public freMTPL2 dataset (CC0), reproducing the exact partition used at training
time (same size, stratification and ``random_state``). The target is the real,
observed outcome: did the policy file at least one claim?
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.risk_agent import RiskAgent  # noqa: E402
from experiments.real_data.train_fremtpl2_model import (  # noqa: E402
    RANDOM_STATE,
    TEST_SIZE,
    build_dataset,
)
from utils.actuarial_features import (  # noqa: E402
    CLASS_LABELS,
    FEATURE_NAMES,
    POSITIVE_CLASS,
    decision_from_probability,
)

RAW_FILE = PROJECT_ROOT / 'data' / 'raw' / 'freMTPL2freq.arff'


def evaluate_holdout(raw_path=RAW_FILE):
    """Evaluate the shipped model on the reproducible real hold-out partition."""
    raw_path = Path(raw_path)
    if not raw_path.is_file():
        raise FileNotFoundError(
            f'{raw_path} is missing. Run experiments/data/fetch_fremtpl2.py first.'
        )

    features, targets, context, skipped = build_dataset(raw_path)
    if len(targets) == 0:
        raise ValueError(f'No usable record was found in {raw_path}.')

    # Identical partition to training (same size, stratify and random_state), so the
    # reported metrics are a genuine hold-out and not re-trained on the test rows.
    _, test_indices = train_test_split(
        np.arange(len(targets)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=targets,
    )

    model = RiskAgent().model
    x_test = features[test_indices]
    y_true = targets[test_indices]
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    decisions = [decision_from_probability(float(p)) for p in probabilities]

    return {
        'dataset_size': int(len(targets)),
        'records_skipped': int(skipped),
        'test_size': int(len(test_indices)),
        'y_true': y_true,
        'y_pred': predictions,
        'probabilities': probabilities,
        'decisions': decisions,
        'records': [context[index] for index in test_indices],
        'accuracy': float(accuracy_score(y_true, predictions)),
        'roc_auc': float(roc_auc_score(y_true, probabilities)),
        'average_precision': float(average_precision_score(y_true, probabilities)),
        'f1_claim': float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        'confusion_matrix': confusion_matrix(y_true, predictions),
        'base_rate_test': float(np.mean(y_true)),
        'mapped_decisions': {
            label: int(decisions.count(label)) for label in CLASS_LABELS
        },
        'feature_names': list(FEATURE_NAMES),
        'positive_class': POSITIVE_CLASS,
    }


if __name__ == '__main__':
    result = evaluate_holdout()
    print(
        f"Dataset: {result['dataset_size']:,} real policies "
        f"({result['records_skipped']:,} skipped)"
    )
    print(f"Hold-out partition: {result['test_size']:,} policies")
    print(f"Base claim rate (test): {result['base_rate_test']:.4f}")
    print(f"ROC-AUC: {result['roc_auc']:.4f}")
    print(f"PR-AUC : {result['average_precision']:.4f}")
    print(f"Accuracy @0.5: {result['accuracy']:.4f}")
    print(f"Mapped decisions: {result['mapped_decisions']}")
