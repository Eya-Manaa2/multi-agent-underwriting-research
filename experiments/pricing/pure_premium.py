"""Pricing-grade underwriting on the real public freMTPL2 dataset (CC0).

Motivation
----------
The pilot originally mapped the predicted claim probability onto the
recommendation vocabulary through fixed thresholds (``0.05`` surcharge, ``0.15``
review, see ``utils/actuarial_features``). Those values were a *research
convention*: nothing tied them to the economics of the risk.

This module replaces that convention by the standard actuarial decomposition of
the pure premium (expected cost of a policy over its exposure period):

    pure_premium = E[claim count | policy] x E[claim amount | claim]

Both terms are trained on the **real** freMTPL2 data, frequency from
``freMTPL2freq`` and severity from ``freMTPL2sev`` (Christophe Dutang and Arthur
Charpentier, 2018, CASdatasets; OpenML, CC0):

* frequency  -> XGBoost ``count:poisson`` regressor with the policy ``Exposure``
  used as a log-link offset (claims per unit of exposure);
* severity   -> XGBoost ``reg:gamma`` regressor on the real claim amounts, using
  the policy covariates of each claim (joined by ``IDpol``).

The pure premium is money (EUR). The recommendation stops depending on a magic
number: the insurer derives a **technical premium** ``pure_premium /
target_loss_ratio`` and rejects a policy only when that premium leaves its
**risk appetite**, expressed as a multiple of the portfolio-average pure
premium. The published thresholds are therefore expressed in EUR and are
calibrated to a target loss ratio, not invented.

Because ``pure_premium ~= claim_probability x mean_severity``, the resulting
decision can also be re-expressed as an *equivalent claim probability*: that is
the honest, economic reading of the former ``0.05`` / ``0.15`` convention (see
``equivalent_claim_probability`` in the metadata).

Outputs
-------
    models/fremtpl2_frequency_model.json       trained frequency model
    models/fremtpl2_severity_model.json        trained severity model
    models/fremtpl2_pricing_metadata.json      metrics, provenance, calibration
    data/underwriting/pricing_policy_v1.json   versioned economic policy

This is genuine supervised learning on real data: no synthetic rows, no
heuristic pseudo-labels.
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.actuarial_features import (  # noqa: E402
    CLASS_LABELS,
    FEATURE_NAMES,
    feature_vector,
    target_claim_occurred,
)

RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
FREQ_FILE = RAW_DIR / 'freMTPL2freq.arff'
SEV_FILE = RAW_DIR / 'freMTPL2sev.arff'
MANIFEST_FILE = RAW_DIR / 'fremtpl2_manifest.json'

FREQUENCY_MODEL_FILE = PROJECT_ROOT / 'models' / 'fremtpl2_frequency_model.json'
SEVERITY_MODEL_FILE = PROJECT_ROOT / 'models' / 'fremtpl2_severity_model.json'
PRICING_METADATA_FILE = PROJECT_ROOT / 'models' / 'fremtpl2_pricing_metadata.json'
POLICY_FILE = PROJECT_ROOT / 'data' / 'underwriting' / 'pricing_policy_v1.json'

# Column order of the ARFF headers (see @attribute lines).
FREQ_IDPOL = 0
FREQ_CLAIM_NB = 1
FREQ_EXPOSURE = 2
FREQ_AREA = 3
FREQ_VEH_POWER = 4
FREQ_VEH_AGE = 5
FREQ_DRIV_AGE = 6
FREQ_BONUS_MALUS = 7
FREQ_VEH_GAS = 9
FREQ_DENSITY = 10
SEV_IDPOL = 0
SEV_CLAIM_AMOUNT = 1

# Same reproducible partition as the classifier and the baselines (80/20,
# stratified, seed 42) so every component is measured on the *same* hold-out.
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Economic policy parameters (business inputs, not results) ----------------
# Target loss ratio used to turn a pure premium into a technical premium.
TARGET_LOSS_RATIO = 0.70
# Risk appetite: reject above this multiple of the portfolio-average pure premium.
REJECT_MULTIPLE = 3.0
# Above the portfolio average the risk is surcharged, below it is accepted as is.
SURCHARGE_MULTIPLE = 1.0


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


def load_frequency(path: Path = FREQ_FILE) -> dict:
    """Return the real per-policy records needed for frequency modelling.

    Keys: ``policy_ids`` (IDpol), ``features`` (encoded 7-vector), ``exposure``,
    ``claim_count`` and ``claim_occurred`` (binary target used only to reproduce
    the project-wide stratified split).
    """
    if not path.is_file():
        raise FileNotFoundError(
            f'{path} is missing. Run experiments/data/fetch_fremtpl2.py first.'
        )

    policy_ids, features, exposure, claim_count, occurred = [], [], [], [], []
    for row in _iter_rows(path):
        if len(row) != 12:
            continue
        try:
            exposure_value = float(row[FREQ_EXPOSURE])
        except ValueError:
            continue
        if exposure_value <= 0:
            continue
        try:
            vector = feature_vector(
                {
                    'driver_age': int(float(row[FREQ_DRIV_AGE])),
                    'bonus_malus': float(row[FREQ_BONUS_MALUS]),
                    'vehicle_age': int(float(row[FREQ_VEH_AGE])),
                    'vehicle_power': int(float(row[FREQ_VEH_POWER])),
                    'vehicle_gas': row[FREQ_VEH_GAS],
                    'density': float(row[FREQ_DENSITY]),
                    'area': row[FREQ_AREA],
                }
            )
            count = int(float(row[FREQ_CLAIM_NB]))
        except (ValueError, KeyError):
            continue
        policy_ids.append(row[FREQ_IDPOL].strip())
        features.append(vector)
        exposure.append(exposure_value)
        claim_count.append(count)
        occurred.append(target_claim_occurred(count))

    if not policy_ids:
        raise ValueError(f'No usable record was found in {path}.')

    return {
        'policy_ids': policy_ids,
        'features': np.asarray(features, dtype=np.float32),
        'exposure': np.asarray(exposure, dtype=np.float64),
        'claim_count': np.asarray(claim_count, dtype=np.float64),
        'claim_occurred': np.asarray(occurred, dtype=np.int64),
    }


def load_severity(feature_by_id: dict, path: Path = SEV_FILE) -> dict:
    """Return the real claim-level severities joined to their policy covariates.

    Keys: ``claim_features`` (encoded covariates of each claiming policy),
    ``claim_amounts`` (EUR), ``claim_policy_ids`` (IDpol aligned with the claims)
    and ``cost_by_policy`` (total real cost per IDpol).
    """
    if not path.is_file():
        raise FileNotFoundError(
            f'{path} is missing. Run experiments/data/fetch_fremtpl2.py first.'
        )

    claim_features, claim_amounts, claim_policy_ids = [], [], []
    cost_by_policy: dict[str, float] = {}
    for row in _iter_rows(path):
        if len(row) != 2:
            continue
        policy_id = row[SEV_IDPOL].strip()
        vector = feature_by_id.get(policy_id)
        if vector is None:
            continue
        try:
            amount = float(row[SEV_CLAIM_AMOUNT])
        except ValueError:
            continue
        if amount <= 0:
            continue
        claim_features.append(vector)
        claim_amounts.append(amount)
        claim_policy_ids.append(policy_id)
        cost_by_policy[policy_id] = cost_by_policy.get(policy_id, 0.0) + amount

    if not claim_amounts:
        raise ValueError(f'No usable claim amount was found in {path}.')

    return {
        'claim_features': np.asarray(claim_features, dtype=np.float32),
        'claim_amounts': np.asarray(claim_amounts, dtype=np.float64),
        'claim_policy_ids': claim_policy_ids,
        'cost_by_policy': cost_by_policy,
    }


def split_indices(claim_occurred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the project-wide 80/20 stratified split on the real target."""
    train_idx, test_idx = train_test_split(
        np.arange(len(claim_occurred)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=claim_occurred,
    )
    return train_idx, test_idx


def _poisson_deviance(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Poisson deviance (count data); lower is better."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 1e-9)
    with np.errstate(divide='ignore', invalid='ignore'):
        term = np.where(observed > 0, observed * np.log(observed / predicted), 0.0)
    return float(np.mean(2.0 * (term - (observed - predicted))))


def _gamma_deviance(observed: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Gamma deviance (positive skew amounts); lower is better."""
    observed = np.asarray(observed, dtype=float)
    predicted = np.maximum(np.asarray(predicted, dtype=float), 1e-9)
    return float(np.mean(2.0 * ((observed - predicted) / predicted - np.log(observed / predicted))))


def technical_premium(pure_premium: float, target_loss_ratio: float = TARGET_LOSS_RATIO) -> float:
    """Convert an expected loss into the premium charged at a target loss ratio."""
    if target_loss_ratio <= 0 or target_loss_ratio > 1:
        raise ValueError('target_loss_ratio must be within (0, 1]')
    return pure_premium / target_loss_ratio


def recommend_from_pure_premium(
    pure_premium: float,
    surcharge_threshold: float,
    reject_threshold: float,
) -> str:
    """Economic recommendation: reject outside appetite, surcharge above average.

    Both thresholds are expected costs in EUR. They must satisfy
    ``0 <= surcharge_threshold <= reject_threshold``.
    """
    if surcharge_threshold < 0 or reject_threshold < surcharge_threshold:
        raise ValueError('thresholds must satisfy 0 <= surcharge <= reject')
    if pure_premium >= reject_threshold:
        return CLASS_LABELS[2]
    if pure_premium >= surcharge_threshold:
        return CLASS_LABELS[1]
    return CLASS_LABELS[0]


def equivalent_claim_probability(pure_premium_threshold: float, mean_severity: float) -> float:
    """Re-express a EUR threshold as an equivalent claim probability.

    Uses ``pure_premium ~= claim_probability x mean_severity`` so the economic
    thresholds can be compared with the former probability convention.
    """
    if mean_severity <= 0:
        raise ValueError('mean_severity must be positive')
    return float(pure_premium_threshold / mean_severity)


def _provenance() -> dict:
    manifest = {}
    if MANIFEST_FILE.is_file():
        manifest = json.loads(MANIFEST_FILE.read_text(encoding='utf-8'))
    datasets = manifest.get('datasets') or {}
    return {
        'frequency_dataset': 'freMTPL2freq',
        'severity_dataset': 'freMTPL2sev',
        'urls': [datasets.get('freMTPL2freq', {}).get('url'),
                 datasets.get('freMTPL2sev', {}).get('url')],
        'md5': {
            'freMTPL2freq': datasets.get('freMTPL2freq', {}).get('md5'),
            'freMTPL2sev': datasets.get('freMTPL2sev', {}).get('md5'),
        },
        'license': manifest.get('license', 'CC0 (as stated in OpenML dataset metadata)'),
        'citation': manifest.get(
            'citation',
            'Christophe Dutang and Arthur Charpentier (2018), CASdatasets: Insurance datasets.',
        ),
    }


def train() -> dict:
    """Train the real frequency and severity models and write every artifact."""
    print(f'Reading real data from {FREQ_FILE.name} and {SEV_FILE.name} ...')
    frequency = load_frequency()
    feature_by_id = {
        policy_id: vector
        for policy_id, vector in zip(frequency['policy_ids'], frequency['features'])
    }
    severity = load_severity(feature_by_id)

    train_idx, test_idx = split_indices(frequency['claim_occurred'])
    print(
        f"{len(frequency['policy_ids']):,} real policies "
        f"({len(frequency['claim_count'])} counts) · "
        f"{len(severity['claim_amounts']):,} real claim amounts"
    )

    # --- Frequency model: Poisson count with the exposure as log-link offset ----
    exposure_train = frequency['exposure'][train_idx]
    frequency_model = XGBRegressor(
        objective='count:poisson',
        n_estimators=400,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    frequency_model.fit(
        frequency['features'][train_idx],
        frequency['claim_count'][train_idx],
        base_margin=np.log(exposure_train),
    )

    exposure_test = frequency['exposure'][test_idx]
    predicted_count = frequency_model.predict(
        frequency['features'][test_idx], base_margin=np.log(exposure_test)
    )
    predicted_count = np.maximum(predicted_count, 0.0)

    # --- Severity model: Gamma regression on real claim amounts ----------------
    # A claim belongs to the train partition when its policy does, so severity is
    # never fitted on hold-out claims.
    policy_position = {pid: i for i, pid in enumerate(frequency['policy_ids'])}
    claim_policy_index = np.asarray(
        [policy_position[pid] for pid in severity['claim_policy_ids']], dtype=np.int64
    )
    severity_is_train = np.isin(claim_policy_index, train_idx)

    severity_model = XGBRegressor(
        objective='reg:gamma',
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    severity_model.fit(
        severity['claim_features'][severity_is_train],
        severity['claim_amounts'][severity_is_train],
    )

    predicted_severity = np.maximum(
        severity_model.predict(severity['claim_features']), 1e-6
    )

    # --- Pure premium on the shared hold-out ----------------------------------
    mean_severity = float(np.mean(severity['claim_amounts'][severity_is_train]))
    severity_per_policy = np.maximum(severity_model.predict(frequency['features'][test_idx]), 1e-6)
    pure_premium = predicted_count * severity_per_policy

    # Portfolio baseline is the *train* average, i.e. the neutral technical level.
    train_mean_severity = mean_severity
    train_predicted_count = np.maximum(
        frequency_model.predict(
            frequency['features'][train_idx], base_margin=np.log(exposure_train)
        ),
        0.0,
    )
    baseline_pure_premium = float(np.mean(train_predicted_count * train_mean_severity))

    surcharge_threshold = SURCHARGE_MULTIPLE * baseline_pure_premium
    reject_threshold = REJECT_MULTIPLE * baseline_pure_premium

    decisions = [
        recommend_from_pure_premium(float(value), surcharge_threshold, reject_threshold)
        for value in pure_premium
    ]
    decision_counts = {label: int(decisions.count(label)) for label in CLASS_LABELS}

    # --- Real-cost calibration: predicted vs observed loss on the hold-out ----
    test_ids = [frequency['policy_ids'][i] for i in test_idx]
    observed_cost = np.asarray(
        [severity['cost_by_policy'].get(pid, 0.0) for pid in test_ids], dtype=float
    )
    observed_count = frequency['claim_count'][test_idx]
    matched = np.asarray([pid in severity['cost_by_policy'] for pid in test_ids])

    metrics = {
        'frequency': {
            'poisson_deviance': _poisson_deviance(observed_count, predicted_count),
            'mean_predicted_count': float(np.mean(predicted_count)),
            'mean_observed_count': float(np.mean(observed_count)),
        },
        'severity': {
            'gamma_deviance': _gamma_deviance(
                severity['claim_amounts'][~severity_is_train],
                predicted_severity[~severity_is_train],
            ),
            'mean_claim_amount': mean_severity,
        },
        'pure_premium': {
            'mean_predicted_eur': float(np.mean(pure_premium)),
            'baseline_train_eur': baseline_pure_premium,
            'observed_total_cost_eur': float(np.sum(observed_cost)),
            'predicted_total_cost_eur': float(np.sum(pure_premium)),
            'loss_ratio_predicted_vs_observed': (
                float(np.sum(pure_premium) / np.sum(observed_cost))
                if observed_cost.sum() > 0
                else None
            ),
            'severity_coverage': float(np.mean(matched)),
        },
        'decision_counts_on_holdout': decision_counts,
    }

    # --- Artifacts -------------------------------------------------------------
    FREQUENCY_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    frequency_model.save_model(str(FREQUENCY_MODEL_FILE))
    severity_model.save_model(str(SEVERITY_MODEL_FILE))

    metadata = {
        'target_definition': 'real freMTPL2 pure premium = frequency x severity',
        'feature_names': list(FEATURE_NAMES),
        'class_labels': list(CLASS_LABELS),
        'trained_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'split': {
            'test_size': TEST_SIZE,
            'random_state': RANDOM_STATE,
            'train_policies': int(len(train_idx)),
            'test_policies': int(len(test_idx)),
        },
        'economic_policy': {
            'target_loss_ratio': TARGET_LOSS_RATIO,
            'surcharge_multiple': SURCHARGE_MULTIPLE,
            'reject_multiple': REJECT_MULTIPLE,
            'baseline_pure_premium_eur': baseline_pure_premium,
            'surcharge_threshold_eur': surcharge_threshold,
            'reject_threshold_eur': reject_threshold,
            'equivalent_surcharge_claim_probability': equivalent_claim_probability(
                surcharge_threshold, train_mean_severity
            ),
            'equivalent_reject_claim_probability': equivalent_claim_probability(
                reject_threshold, train_mean_severity
            ),
        },
        'metrics': metrics,
        'source': _provenance(),
    }
    PRICING_METADATA_FILE.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )

    policy = {
        'policy_id': 'pricing_policy_v1',
        'version': '1.0.0',
        'product_line': 'auto',
        'currency': 'EUR',
        'basis': 'real freMTPL2 frequency x severity pure premium',
        'parameters': {
            'target_loss_ratio': TARGET_LOSS_RATIO,
            'surcharge_multiple_of_baseline': SURCHARGE_MULTIPLE,
            'reject_multiple_of_baseline': REJECT_MULTIPLE,
        },
        'calibrated_thresholds': {
            'baseline_pure_premium_eur': baseline_pure_premium,
            'surcharge_threshold_eur': surcharge_threshold,
            'reject_threshold_eur': reject_threshold,
            'equivalent_surcharge_claim_probability': metadata['economic_policy'][
                'equivalent_surcharge_claim_probability'
            ],
            'equivalent_reject_claim_probability': metadata['economic_policy'][
                'equivalent_reject_claim_probability'
            ],
        },
        'decision_rule': [
            'reject if pure_premium >= reject_threshold_eur',
            'accept_with_surcharge if pure_premium >= surcharge_threshold_eur',
            'accept_standard otherwise',
        ],
        'provenance': _provenance(),
    }
    POLICY_FILE.parent.mkdir(parents=True, exist_ok=True)
    POLICY_FILE.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2) + '\n', encoding='utf-8'
    )

    print(f'Frequency model : {FREQUENCY_MODEL_FILE}')
    print(f'Severity model  : {SEVERITY_MODEL_FILE}')
    print(f'Pricing metadata: {PRICING_METADATA_FILE}')
    print(f'Pricing policy  : {POLICY_FILE}')
    print(
        f"Pure premium (train baseline): {baseline_pure_premium:.2f} EUR | "
        f"surcharge >= {surcharge_threshold:.2f} EUR | reject >= {reject_threshold:.2f} EUR"
    )
    print(
        f"Equivalent claim probabilities: surcharge "
        f"{metadata['economic_policy']['equivalent_surcharge_claim_probability']:.4f} | "
        f"reject {metadata['economic_policy']['equivalent_reject_claim_probability']:.4f}"
    )
    print(f'Test decision counts: {decision_counts}')
    return metadata


if __name__ == '__main__':
    train()
