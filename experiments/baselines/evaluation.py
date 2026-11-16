"""Shared evaluation harness for the real freMTPL2 underwriting baselines.

Every baseline (B1 rules, B2 ML, B3 single LLM agent, B4 multi-agent pipeline) is
scored on the *same* reproducible hold-out partition of the real public freMTPL2
dataset (CC0), so the comparison is apples-to-apples.

freMTPL2 carries no accept/reject ground truth: the observed target is the real fact
"did the policy file at least one claim?", and the pilot recommendation vocabulary is
mapped onto it through the documented business thresholds, exactly as at training
time. The reported "adverse-flag" metrics are therefore proxy metrics against the
observed claim outcome, not a measure of an insurer acceptance policy.
"""

from __future__ import annotations

import json
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

from experiments.real_data.train_fremtpl2_model import (  # noqa: E402
    RANDOM_STATE,
    TEST_SIZE,
    build_dataset,
)

RAW_FILE = PROJECT_ROOT / 'data' / 'raw' / 'freMTPL2freq.arff'
RESULTS_DIR = PROJECT_ROOT / 'experiments' / 'baselines' / 'results'

# Full pilot decision vocabulary (see products/auto/rules.py).
PILOT_DECISIONS = (
    'Accept_Standard',
    'Accept_with_Surcharge',
    'Accept_with_Exclusion',
    'Reject',
    'Escalate_to_Human',
    'Request_More_Info',
)
ACCEPT_STANDARD = 'Accept_Standard'


def holdout_partition(raw_path=RAW_FILE) -> dict:
    """Return the exact hold-out partition used at training time.

    The split (size, stratification and ``random_state``) is reproduced from
    ``experiments.real_data.train_fremtpl2_model`` so the metrics are a genuine
    hold-out and never re-trained on the test rows.
    """
    raw_path = Path(raw_path)
    if not raw_path.is_file():
        raise FileNotFoundError(
            f'{raw_path} is missing. Run experiments/data/fetch_fremtpl2.py first.'
        )

    features, targets, records, skipped = build_dataset(raw_path)
    if len(targets) == 0:
        raise ValueError(f'No usable record was found in {raw_path}.')

    train_indices, test_indices = train_test_split(
        np.arange(len(targets)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=targets,
    )
    return {
        'features': features,
        'targets': targets,
        'records': records,
        'train_indices': train_indices,
        'test_indices': test_indices,
        'skipped': int(skipped),
        'dataset_size': int(len(targets)),
        'test_size': int(len(test_indices)),
    }


def stratified_sample(test_indices, targets, size, seed=RANDOM_STATE):
    """Return a bounded stratified sample of the given indices (same claim rate)."""
    indices = np.asarray(test_indices)
    if size is None or size >= len(indices):
        return indices
    if size <= 0:
        raise ValueError('sample size must be a positive integer')

    rng = np.random.RandomState(seed)
    labels = np.asarray(targets)[indices]
    positives = indices[labels == 1]
    negatives = indices[labels == 0]
    share = float(np.mean(labels)) if len(indices) else 0.0

    n_positive = min(len(positives), max(1, int(round(size * share))))
    n_negative = min(len(negatives), max(1, size - n_positive))
    chosen = np.concatenate(
        [
            rng.choice(positives, n_positive, replace=False),
            rng.choice(negatives, n_negative, replace=False),
        ]
    )
    chosen.sort()
    return chosen


def score_metrics(y_true, scores, threshold=0.5) -> dict:
    """Discrimination metrics for a continuous risk score against the real target."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores, dtype=float)
    predictions = (scores >= threshold).astype(int)
    return {
        'roc_auc': float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else None,
        'pr_auc': float(average_precision_score(y_true, scores)),
        'accuracy_at_0_5': float(accuracy_score(y_true, predictions)),
        'f1_claim': float(f1_score(y_true, predictions, pos_label=1, zero_division=0)),
        'base_rate': float(np.mean(y_true)),
        'confusion_matrix_at_0_5': confusion_matrix(y_true, predictions).tolist(),
    }


def decision_summary(y_true, decisions) -> dict:
    """Recommendation distribution and adverse-flag proxy metrics.

    A recommendation is treated as an "adverse flag" when it is anything other than
    ``Accept_Standard`` (surcharge, exclusion, reject, escalation). Against the real
    claim outcome this yields a proxy precision/recall, not an acceptance policy.
    """
    y_true = np.asarray(y_true)
    decisions = list(decisions)
    counts = {label: int(decisions.count(label)) for label in PILOT_DECISIONS}
    unknown = [d for d in decisions if d not in PILOT_DECISIONS]
    if unknown:
        raise ValueError(f'unknown decision label(s): {sorted(set(unknown))}')

    flagged = np.asarray([d != ACCEPT_STANDARD for d in decisions])
    claims = y_true == 1
    return {
        'decision_counts': counts,
        'flagged_rate': float(np.mean(flagged)) if len(flagged) else None,
        'flag_precision': (
            float(np.mean(claims[flagged])) if flagged.any() else None
        ),
        'flag_recall': (
            float(np.mean(flagged[claims])) if claims.any() else None
        ),
    }


def agreement_rate(decisions_a, decisions_b) -> float:
    """Fraction of cases where two decision sequences agree (identical order)."""
    decisions_a = list(decisions_a)
    decisions_b = list(decisions_b)
    if len(decisions_a) != len(decisions_b):
        raise ValueError('decision sequences must have equal lengths')
    if not decisions_a:
        return 1.0
    return float(np.mean([a == b for a, b in zip(decisions_a, decisions_b)]))


def save_result(name: str, payload: dict, directory=None) -> Path:
    """Persist a result under ``experiments/baselines/results`` (or ``directory``)."""
    target_dir = Path(directory) if directory is not None else RESULTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f'{name}.json'
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )
    return path
