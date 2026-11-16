"""Train the underwriting risk model on the real public freMTPL2 dataset (CC0).

Reads the verified raw ARFF downloaded by ``experiments/data/fetch_fremtpl2.py``,
turns each policy into the real feature vector defined in ``utils.actuarial_features``
and trains an XGBoost classifier on a real, observed outcome.

Target (real, from the dataset): did the policy file at least one claim?
The predicted claim probability is then mapped onto the pilot recommendation
vocabulary through the documented business thresholds in ``utils.actuarial_features``.

Outputs:
    models/fremtpl2_underwriting_model.json     trained model
    models/fremtpl2_underwriting_metadata.json  contract, encoders, metrics, provenance

This is genuine supervised learning on real data: no synthetic rows and no
heuristic pseudo-labels are involved.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.actuarial_features import (  # noqa: E402
    AREA_CODES,
    CLASS_LABELS,
    FEATURE_NAMES,
    REVIEW_THRESHOLD,
    SURCHARGE_THRESHOLD,
    VEHICLE_GAS_CODES,
    decision_from_probability,
    feature_vector,
    target_claim_occurred,
)

RAW_FILE = PROJECT_ROOT / 'data' / 'raw' / 'freMTPL2freq.arff'
MANIFEST_FILE = PROJECT_ROOT / 'data' / 'raw' / 'fremtpl2_manifest.json'
MODEL_FILE = PROJECT_ROOT / 'models' / 'fremtpl2_underwriting_model.json'
METADATA_FILE = PROJECT_ROOT / 'models' / 'fremtpl2_underwriting_metadata.json'

# Column order of freMTPL2freq (see the @attribute header of the ARFF file).
COL_CLAIM_NB = 1
COL_EXPOSURE = 2
COL_AREA = 3
COL_VEH_POWER = 4
COL_VEH_AGE = 5
COL_DRIV_AGE = 6
COL_BONUS_MALUS = 7
COL_VEH_GAS = 9
COL_DENSITY = 10

RANDOM_STATE = 42
TEST_SIZE = 0.2


def _iter_rows(path: Path):
    """Yield data rows of an ARFF file without loading it into memory."""
    in_data = False
    with path.open(encoding='utf-8-sig', newline='') as source:
        for line in source:
            stripped = line.strip()
            if not stripped or stripped.startswith('%'):
                continue
            if not in_data:
                if stripped.lower() == '@data':
                    in_data = True
                continue
            yield next(csv.reader([line], skipinitialspace=True, quotechar="'"))


def build_dataset(path: Path):
    """Return (features, targets, context, skipped) built from real records.

    ``context`` retains the full real application record of every policy kept, so
    downstream evaluations (hold-out metrics, fairness, baselines B1-B4) can work on
    the real, interpretable variables instead of the encoded feature vector.
    """
    features, targets, context = [], [], []
    skipped = 0
    for row in _iter_rows(path):
        if len(row) != 12:
            skipped += 1
            continue
        try:
            exposure = float(row[COL_EXPOSURE])
        except ValueError:
            skipped += 1
            continue
        if exposure <= 0:
            skipped += 1
            continue
        try:
            vector = feature_vector(
                {
                    'driver_age': int(float(row[COL_DRIV_AGE])),
                    'bonus_malus': float(row[COL_BONUS_MALUS]),
                    'vehicle_age': int(float(row[COL_VEH_AGE])),
                    'vehicle_power': int(float(row[COL_VEH_POWER])),
                    'vehicle_gas': row[COL_VEH_GAS],
                    'density': float(row[COL_DENSITY]),
                    'area': row[COL_AREA],
                }
            )
        except (ValueError, KeyError):
            skipped += 1
            continue
        features.append(vector)
        targets.append(target_claim_occurred(int(float(row[COL_CLAIM_NB]))))
        context.append(
            {
                'driver_age': int(float(row[COL_DRIV_AGE])),
                'bonus_malus': float(row[COL_BONUS_MALUS]),
                'vehicle_age': int(float(row[COL_VEH_AGE])),
                'vehicle_power': int(float(row[COL_VEH_POWER])),
                'vehicle_gas': row[COL_VEH_GAS].strip(),
                'density': float(row[COL_DENSITY]),
                'area': row[COL_AREA].strip().upper(),
            }
        )
    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(targets, dtype=np.int64),
        context,
        skipped,
    )


def _provenance() -> dict:
    manifest = {}
    if MANIFEST_FILE.is_file():
        manifest = json.loads(MANIFEST_FILE.read_text(encoding='utf-8'))
    dataset = (manifest.get('datasets') or {}).get('freMTPL2freq', {})
    return {
        'dataset': 'freMTPL2freq',
        'catalogue_description': dataset.get('catalogue_description'),
        'url': dataset.get('url'),
        'md5': dataset.get('md5'),
        'records_in_file': dataset.get('records'),
        'license': manifest.get('license', 'CC0 (as stated in OpenML dataset metadata)'),
        'citation': manifest.get(
            'citation',
            'Christophe Dutang and Arthur Charpentier (2018), CASdatasets: Insurance datasets.',
        ),
    }


def train() -> dict:
    if not RAW_FILE.is_file():
        raise FileNotFoundError(
            f'{RAW_FILE} is missing. Run experiments/data/fetch_fremtpl2.py first.'
        )

    print(f'Reading real data from {RAW_FILE.name} ...')
    features, targets, _, skipped = build_dataset(RAW_FILE)
    if len(targets) == 0:
        raise ValueError('No usable record was found in the raw dataset.')

    base_rate = float(np.mean(targets))
    print(
        f'Built {len(targets):,} real policies ({skipped:,} skipped) · '
        f'claim base rate {base_rate:.4f}'
    )

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=targets,
    )

    # No class re-weighting: predictions are kept on the real claim base rate so the
    # documented thresholds stay meaningful. Discrimination is reported with ROC-AUC,
    # which is rank-based and therefore insensitive to this choice.
    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)

    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    mapped = [decision_from_probability(float(p)) for p in probabilities]
    mapped_counts = Counter(mapped)

    metrics = {
        'roc_auc': float(roc_auc_score(y_test, probabilities)),
        'average_precision': float(average_precision_score(y_test, probabilities)),
        'accuracy_at_0_5': float(accuracy_score(y_test, predictions)),
        'base_rate_test': float(np.mean(y_test)),
        'confusion_matrix_at_0_5': confusion_matrix(y_test, predictions).tolist(),
        'test_mapped_decisions': {
            label: int(mapped_counts.get(label, 0)) for label in CLASS_LABELS
        },
        'thresholds': {
            'surcharge': SURCHARGE_THRESHOLD,
            'review': REVIEW_THRESHOLD,
        },
    }

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_FILE))

    metadata = {
        'model_file': MODEL_FILE.name,
        'feature_names': list(FEATURE_NAMES),
        'class_labels': list(CLASS_LABELS),
        'target_definition': 'real freMTPL2 outcome: ClaimNb >= 1 (claim occurred)',
        'positive_class': 'claim_occurred',
        'decision_thresholds': {
            'surcharge': SURCHARGE_THRESHOLD,
            'review': REVIEW_THRESHOLD,
        },
        'encoders': {
            'area': {code: index for index, code in enumerate(AREA_CODES)},
            'vehicle_gas': {
                code: index for index, code in enumerate(VEHICLE_GAS_CODES)
            },
        },
        'trained_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'training': {
            'real_records_used': int(len(targets)),
            'records_skipped': int(skipped),
            'train_size': int(len(y_train)),
            'test_size': int(len(y_test)),
            'random_state': RANDOM_STATE,
            'base_rate': base_rate,
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
        f"ROC-AUC: {metrics['roc_auc']:.4f} | PR-AUC: {metrics['average_precision']:.4f} "
        f"| accuracy@0.5: {metrics['accuracy_at_0_5']:.4f}"
    )
    print(f"Test-mapped decisions: {metrics['test_mapped_decisions']}")
    return metadata


if __name__ == '__main__':
    train()
