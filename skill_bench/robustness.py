"""Robustness degradations applied to the real application records.

Two realistic degradations, both defined on the *application* (not on the encoded
feature vector) so the whole pipeline sees them:

* **noise** -- additive Gaussian noise on the numeric fields, scaled by the field's
  own standard deviation over the sample. Values are re-clipped to the declared
  bounds and integer fields are rounded, so the application stays admissible and the
  pipeline still produces a recommendation (the decision can then be compared with
  the clean run);
* **missing field** -- one numeric field is dropped. The intake agent is fail-closed,
  so a dropped required field makes the pipeline refuse to emit a recommendation.
  This is measured as a *safety* rate (share of refusals), not as quality.

The noise degradation therefore measures decision/quality stability, while the
missing-field degradation measures the fail-closed guarantee. Neither invents a
value: a missing field is never imputed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.actuarial_features import (  # noqa: E402
    FIELD_BOUNDS,
    INTEGER_FIELDS,
)

# Continuous magnitudes carried by the real freMTPL2 application. The categorical
# fields (``vehicle_gas``, ``area``) are never perturbed.
NUMERIC_FIELDS: tuple[str, ...] = (
    'driver_age',
    'bonus_malus',
    'vehicle_age',
    'vehicle_power',
    'density',
)


def reference_std(records: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Per-field standard deviation over the sample, used to scale the noise."""
    std: dict[str, float] = {}
    for field in NUMERIC_FIELDS:
        values = np.asarray([float(record[field]) for record in records], dtype=float)
        std[field] = float(values.std(ddof=0))
    return std


def noisy_application(
    record: Mapping[str, Any],
    level: float,
    std: Mapping[str, float],
    rng: np.random.RandomState,
) -> dict[str, Any]:
    """Return a copy of ``record`` with bounded Gaussian noise on the numeric fields."""
    perturbed = dict(record)
    for field in NUMERIC_FIELDS:
        low, high = FIELD_BOUNDS[field]
        value = float(record[field]) + rng.normal(0.0, level * float(std[field]))
        value = min(max(value, float(low)), float(high))
        if field in INTEGER_FIELDS:
            value = float(round(value))
            value = min(max(value, float(low)), float(high))
        perturbed[field] = value
    return perturbed


def drop_field(record: Mapping[str, Any], field: str) -> dict[str, Any]:
    """Return a copy of ``record`` with ``field`` removed (never imputed)."""
    degraded = dict(record)
    degraded.pop(field, None)
    return degraded


def decision_stability(
    clean_decisions: Sequence[str], degraded_decisions: Sequence[str]
) -> float:
    """Share of cases whose decision is unchanged under degradation."""
    if len(clean_decisions) != len(degraded_decisions):
        raise ValueError('decision sequences must have equal lengths')
    if not clean_decisions:
        return 1.0
    unchanged = sum(
        1 for clean, degraded in zip(clean_decisions, degraded_decisions) if clean == degraded
    )
    return float(unchanged / len(clean_decisions))
