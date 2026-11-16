"""Decision quality on the real Prudential underwriting-decision benchmark.

This is the Levier-B counterpart of ``experiments/pricing/pure_premium``. freMTPL2
has no underwriting decision to learn: the observed target is a claim outcome, so
accuracy, recall and "equal opportunity" could only be measured against a proxy.
The Prudential dataset does carry the real decision -- the ordinal risk class
``Response`` (1-8) assigned by the insurer's human underwriters -- which makes a
genuine decision-quality evaluation possible.

Task and metric
---------------
The task is ordinal: predict the underwriter risk class of each real application.
The official competition metric is the Quadratic Weighted Kappa (QWK), which
penalises large class disagreements quadratically and is the appropriate measure
for an ordered decision. The model is an XGBoost regressor producing a continuous
score, mapped back to the 1-8 classes with thresholds tuned on a validation split
(the standard approach for this benchmark). A majority-class baseline is reported
alongside, so the lift of the model is explicit.

Everything is real: no synthetic rows, no pseudo-labels. Data is fetched by
``experiments/data/fetch_prudential.py`` (real Kaggle competition download).

Outputs:
    models/prudential_decision_model.json     trained ordinal scorer
    models/prudential_decision_metadata.json  metrics, provenance, mappings
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DATA_FILE = PROJECT_ROOT / 'data' / 'raw' / 'prudential' / 'train.csv'
MANIFEST_FILE = PROJECT_ROOT / 'data' / 'raw' / 'prudential' / 'manifest.json'
MODEL_FILE = PROJECT_ROOT / 'models' / 'prudential_decision_model.json'
METADATA_FILE = PROJECT_ROOT / 'models' / 'prudential_decision_metadata.json'

MIN_CLASS = 1
MAX_CLASS = 8
RANDOM_STATE = 42
VALID_SIZE = 0.2
TEST_SIZE = 0.2


def quadratic_weighted_kappa(
    y_true, y_pred, min_rating: int = MIN_CLASS, max_rating: int = MAX_CLASS
) -> float:
    """Quadratic Weighted Kappa between two ordinal 1-8 rating vectors.

    Self-contained so the official metric is verifiable in isolation (the test
    suite checks it against scikit-learn's implementation).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError('y_true and y_pred must have the same shape')
    if y_true.size == 0:
        raise ValueError('cannot compute QWK on an empty vector')

    ratings = max_rating - min_rating + 1
    observed = np.zeros((ratings, ratings), dtype=float)
    np.add.at(observed, (y_true - min_rating, y_pred - min_rating), 1.0)

    total = observed.sum()
    history_true = observed.sum(axis=1, keepdims=True)
    history_pred = observed.sum(axis=0, keepdims=True)
    expected = history_true @ history_pred / total

    index = np.arange(ratings)
    weights = (index[:, None] - index[None, :]) ** 2 / (ratings - 1) ** 2
    denominator = float((weights * expected).sum())
    if denominator == 0:
        return 1.0
    return float(1.0 - float((weights * observed).sum()) / denominator)


def apply_thresholds(scores, thresholds) -> np.ndarray:
    """Map continuous scores to 1-8 classes using ascending class thresholds."""
    scores = np.asarray(scores, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)
    if thresholds.shape != (MAX_CLASS - MIN_CLASS,):
        raise ValueError(f'expected {MAX_CLASS - MIN_CLASS} thresholds')
    return np.digitize(scores, thresholds) + MIN_CLASS


def optimize_thresholds(
    y_true, scores, initial=None, rounds: int = 4, grid: int = 40
) -> tuple[np.ndarray, float]:
    """Coordinate ascent on the 7 class thresholds to maximise QWK on a split."""
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if initial is None:
        initial = np.asarray([2.0, 2.5, 3.6, 4.6, 5.6, 6.6, 7.6])
    thresholds = np.array(initial, dtype=float)

    best = quadratic_weighted_kappa(y_true, apply_thresholds(scores, thresholds))
    for _ in range(rounds):
        for position in range(len(thresholds)):
            low = thresholds[position - 1] if position > 0 else float(scores.min()) - 0.5
            high = thresholds[position + 1] if position < len(thresholds) - 1 else float(
                scores.max()
            ) + 0.5
            if high - low <= 1e-3:
                continue
            current = thresholds[position]
            for candidate in np.linspace(low + 1e-3, high - 1e-3, grid):
                thresholds[position] = candidate
                score = quadratic_weighted_kappa(
                    y_true, apply_thresholds(scores, thresholds)
                )
                if score > best:
                    best = score
                    current = candidate
            thresholds[position] = current
    return thresholds, best


def load_prudential(path: Path = DATA_FILE) -> dict:
    """Load the real applications, their decisions, and the column encodings."""
    if not path.is_file():
        raise FileNotFoundError(
            f'{path} is missing. Run experiments/data/fetch_prudential.py first.'
        )

    with path.open(encoding='utf-8', newline='') as source:
        reader = csv.reader(source)
        header = next(reader)
        feature_names = header[1:-1]

        # Pass 1: decide which feature columns are categorical (not numeric).
        non_numeric = set()
        for row in reader:
            for position, name in enumerate(feature_names, start=1):
                value = row[position].strip()
                if not value:
                    continue
                try:
                    float(value)
                except ValueError:
                    non_numeric.add(name)

        categorical_mappings = {}

    with path.open(encoding='utf-8', newline='') as source:
        reader = csv.reader(source)
        next(reader)

        # Pass 2: collect the categorical vocabularies, then build the matrix.
        raw_values = {}
        rows_cache = []
        for row in reader:
            rows_cache.append(row)
            for position, name in enumerate(feature_names, start=1):
                if name in non_numeric:
                    value = row[position].strip()
                    if value:
                        raw_values.setdefault(name, set()).add(value)

        categorical_mappings = {
            name: {value: index for index, value in enumerate(sorted(values))}
            for name, values in raw_values.items()
        }

        features = np.full((len(rows_cache), len(feature_names)), np.nan, dtype=np.float32)
        targets = np.zeros(len(rows_cache), dtype=int)
        for row_index, row in enumerate(rows_cache):
            for position, name in enumerate(feature_names, start=1):
                value = row[position].strip()
                if not value:
                    continue
                if name in categorical_mappings:
                    features[row_index, position - 1] = categorical_mappings[name][value]
                else:
                    features[row_index, position - 1] = float(value)
            targets[row_index] = int(float(row[-1]))

    if features.shape[0] == 0:
        raise ValueError(f'No usable record was found in {path}.')
    return {
        'features': features,
        'targets': targets,
        'feature_names': feature_names,
        'categorical_mappings': categorical_mappings,
    }


def _provenance() -> dict:
    manifest = {}
    if MANIFEST_FILE.is_file():
        manifest = json.loads(MANIFEST_FILE.read_text(encoding='utf-8'))
    return {
        'dataset': 'Prudential Life Insurance Assessment (real underwriting decisions)',
        'competition_url': manifest.get('competition_url'),
        'rules_url': manifest.get('rules_url'),
        'license': manifest.get('license'),
        'sha256': manifest.get('sha256'),
        'records_in_file': manifest.get('records'),
        'citation': manifest.get('citation'),
    }


def train() -> dict:
    """Train the real ordinal decision model and write every artefact."""
    print(f'Reading real applications from {DATA_FILE} ...')
    data = load_prudential()
    features = data['features']
    targets = data['targets']
    print(
        f"{features.shape[0]:,} real applications · {features.shape[1]} features · "
        f"classes {sorted(set(targets.tolist()))}"
    )

    index = np.arange(len(targets))
    train_idx, test_idx = train_test_split(
        index, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=targets
    )
    train_idx, valid_idx = train_test_split(
        train_idx,
        test_size=VALID_SIZE,
        random_state=RANDOM_STATE,
        stratify=targets[train_idx],
    )

    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(features[train_idx], targets[train_idx].astype(float))

    valid_scores = model.predict(features[valid_idx])
    thresholds, valid_qwk = optimize_thresholds(targets[valid_idx], valid_scores)

    test_scores = model.predict(features[test_idx])
    test_pred = apply_thresholds(test_scores, thresholds)
    y_test = targets[test_idx]

    # Majority-class baseline: the most frequent real decision, predicted always.
    classes, counts = np.unique(targets[train_idx], return_counts=True)
    majority = int(classes[int(np.argmax(counts))])
    majority_pred = np.full_like(y_test, majority)

    # Nearest-integer rounding is the naive mapping, shown for contrast.
    rounded_pred = np.clip(np.round(test_scores), MIN_CLASS, MAX_CLASS).astype(int)

    metrics = {
        'qwk_model': quadratic_weighted_kappa(y_test, test_pred),
        'qwk_majority_class': quadratic_weighted_kappa(y_test, majority_pred),
        'qwk_nearest_rounding': quadratic_weighted_kappa(y_test, rounded_pred),
        'accuracy_model': float(np.mean(test_pred == y_test)),
        'accuracy_majority_class': float(np.mean(majority_pred == y_test)),
        'mean_absolute_error': float(np.mean(np.abs(test_pred - y_test))),
        'macro_f1': float(f1_score(y_test, test_pred, average='macro', zero_division=0)),
        'majority_class': majority,
        'valid_qwk_at_thresholds': valid_qwk,
        'confusion_matrix_rows_true_cols_pred': np.histogram2d(
            y_test,
            test_pred,
            bins=[np.arange(MIN_CLASS, MAX_CLASS + 2), np.arange(MIN_CLASS, MAX_CLASS + 2)],
        )[0].astype(int).tolist(),
        'per_class_support': {
            int(value): int(count)
            for value, count in zip(*np.unique(y_test, return_counts=True))
        },
    }

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_FILE))

    metadata = {
        'task': 'real Prudential underwriting decision (ordinal risk class 1-8)',
        'feature_names': data['feature_names'],
        'categorical_mappings': data['categorical_mappings'],
        'class_thresholds': thresholds.tolist(),
        'trained_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'split': {
            'train': int(len(train_idx)),
            'valid': int(len(valid_idx)),
            'test': int(len(test_idx)),
            'random_state': RANDOM_STATE,
        },
        'metrics': metrics,
        'source': _provenance(),
    }
    METADATA_FILE.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )

    print(f'Model   : {MODEL_FILE}')
    print(f'Metadata: {METADATA_FILE}')
    print(
        f"QWK model {metrics['qwk_model']:.4f} | majority {metrics['qwk_majority_class']:.4f} "
        f"| nearest-rounding {metrics['qwk_nearest_rounding']:.4f}"
    )
    print(f"Accuracy {metrics['accuracy_model']:.4f} | MAE {metrics['mean_absolute_error']:.3f}")
    return metadata


if __name__ == '__main__':
    train()
