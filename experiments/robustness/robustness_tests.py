"""Robustness of the shipped freMTPL2 claim model under degraded inputs.

Research axis "robustesse" (report/cadrage.md section 7.2): how much does the real
claim model degrade when the incoming application data is noisy or incomplete?

Two realistic degradations are applied to a bounded stratified sample of the real
hold-out partition, and the discrimination metrics are compared with the clean
baseline:

* **noise**  : additive Gaussian noise on the numeric fields, scaled by the field's
  own standard deviation (1 %, 5 %, 10 %, 20 %), values re-clipped to the observed
  ranges;
* **missing**: parts of the numeric fields become unavailable — either a random
  fraction of cells (10 %, 30 %) or one whole field at a time, set to NaN. XGBoost
  handles missing values natively, so this exercises the real inference path rather
  than an imputation the model never saw.

No value is invented: the clean metrics are the reference and every degraded run is
compared against them on the same cases.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.risk_agent import RiskAgent  # noqa: E402
from experiments.baselines.evaluation import (  # noqa: E402
    RANDOM_STATE,
    holdout_partition,
    save_result,
    score_metrics,
    stratified_sample,
)
from utils.actuarial_features import FEATURE_NAMES  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / 'experiments' / 'robustness' / 'results'

DEFAULT_SAMPLE_SIZE = 10000
NOISE_LEVELS = (0.01, 0.05, 0.10, 0.20)
MISSING_CELL_LEVELS = (0.10, 0.30)
# Columns of the encoded feature vector that carry continuous magnitudes. The two
# remaining columns (vehicle_gas, area) are category codes, not magnitudes.
NUMERIC_COLUMNS = (0, 1, 2, 3, 5)  # driver_age, bonus_malus, vehicle_age, vehicle_power, density
# Among them, the genuinely integer fields: noise must keep them integral.
INTEGER_COLUMNS = (0, 2, 3)  # driver_age, vehicle_age, vehicle_power


def _probabilities(model, features) -> np.ndarray:
    return model.predict_proba(features)[:, 1]


def _noisy(features, level, rng) -> np.ndarray:
    """Additive Gaussian noise on the numeric fields, scaled by column std."""
    degraded = features.astype(np.float64, copy=True)
    std = features[:, list(NUMERIC_COLUMNS)].astype(np.float64).std(axis=0)
    for offset, column in enumerate(NUMERIC_COLUMNS):
        degraded[:, column] += rng.normal(0.0, level * std[offset], size=len(features))
    # Keep the perturbed values inside the real observed ranges, and restore the
    # integrality of the count/age fields.
    for column in NUMERIC_COLUMNS:
        low = float(features[:, column].min())
        high = float(features[:, column].max())
        degraded[:, column] = np.clip(degraded[:, column], low, high)
    for column in INTEGER_COLUMNS:
        degraded[:, column] = np.round(degraded[:, column])
    return degraded.astype(np.float32)


def _masked_cells(features, level, rng) -> np.ndarray:
    """Set a random fraction of the numeric cells to NaN (unavailable field)."""
    degraded = features.astype(np.float32, copy=True)
    for column in NUMERIC_COLUMNS:
        mask = rng.random(size=len(degraded)) < level
        degraded[mask, column] = np.nan
    return degraded


def _dropped_field(features, column) -> np.ndarray:
    """Mark one whole numeric field as unavailable."""
    degraded = features.astype(np.float32, copy=True)
    degraded[:, column] = np.nan
    return degraded


def run(sample_size=DEFAULT_SAMPLE_SIZE, seed=RANDOM_STATE) -> dict:
    """Measure model degradation under noise and missing fields on the hold-out."""
    holdout = holdout_partition()
    index = stratified_sample(
        holdout['test_indices'], holdout['targets'], sample_size, seed
    )
    features = holdout['features'][index]
    y_true = np.asarray(holdout['targets'][index])
    model = RiskAgent().model

    baseline_scores = _probabilities(model, features)
    baseline = score_metrics(y_true, baseline_scores)

    rng = np.random.RandomState(seed)

    noise = {}
    for level in NOISE_LEVELS:
        # A fresh generator per level keeps the comparison stable and reproducible.
        scores = _probabilities(model, _noisy(features, level, np.random.RandomState(seed)))
        metrics = score_metrics(y_true, scores)
        metrics['roc_auc_delta'] = metrics['roc_auc'] - baseline['roc_auc']
        noise[f'{level:.2f}'] = metrics

    missing_cells = {}
    for level in MISSING_CELL_LEVELS:
        scores = _probabilities(
            model, _masked_cells(features, level, np.random.RandomState(seed))
        )
        metrics = score_metrics(y_true, scores)
        metrics['roc_auc_delta'] = metrics['roc_auc'] - baseline['roc_auc']
        missing_cells[f'{level:.2f}'] = metrics

    dropped_fields = {}
    for column in NUMERIC_COLUMNS:
        scores = _probabilities(model, _dropped_field(features, column))
        metrics = score_metrics(y_true, scores)
        metrics['roc_auc_delta'] = metrics['roc_auc'] - baseline['roc_auc']
        dropped_fields[FEATURE_NAMES[column]] = metrics

    return {
        'experiment': 'robustness',
        'status': 'ok',
        'description': 'shipped freMTPL2 claim model under noisy / incomplete inputs',
        'sample_size': int(len(index)),
        'sample_seed': seed,
        'baseline': baseline,
        'noise': noise,
        'missing_cells': missing_cells,
        'dropped_fields': dropped_fields,
    }


if __name__ == '__main__':
    result = run()
    base = result['baseline']
    print(f"Sample: {result['sample_size']} policies (seed {result['sample_seed']})")
    print(f"Clean ROC-AUC: {base['roc_auc']:.4f} | PR-AUC: {base['pr_auc']:.4f}")
    print('Noise (relative to field std):')
    for level, metrics in result['noise'].items():
        print(f"  {level}: ROC-AUC {metrics['roc_auc']:.4f} (Δ {metrics['roc_auc_delta']:+.4f})")
    print('Missing cells:')
    for level, metrics in result['missing_cells'].items():
        print(f"  {level}: ROC-AUC {metrics['roc_auc']:.4f} (Δ {metrics['roc_auc_delta']:+.4f})")
    print('Dropped fields:')
    for name, metrics in result['dropped_fields'].items():
        print(f"  {name}: ROC-AUC {metrics['roc_auc']:.4f} (Δ {metrics['roc_auc_delta']:+.4f})")
    print(f"Saved: {save_result('robustness', result, directory=RESULTS_DIR)}")
