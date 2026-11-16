"""Statistics for the benchmark: bootstrap confidence intervals and Pareto fronts.

Both are computed from the measured per-case records only. Nothing is hand-typed:
the confidence intervals resample the observed cases, and the Pareto front is derived
from the aggregated metrics of every evaluated combination.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

DEFAULT_N_BOOT = 2000
DEFAULT_ALPHA = 0.05


def _safe_roc_auc(y_true, scores) -> float | None:
    """ROC-AUC, or ``None`` when a single class makes it undefined.

    Computed as the Mann-Whitney U statistic with average ranks, which is exactly
    the trapezoidal ROC-AUC scikit-learn returns (ties are credited as half a
    pair). The formula avoids the per-resample validation overhead of
    ``sklearn.metrics.roc_auc_score`` inside the bootstrap loop;
    ``tests/test_skill_bench.py`` pins the equivalence against scikit-learn.
    """
    labels = np.asarray(y_true)
    values = np.asarray(scores, dtype=float)
    positive = labels == 1
    n_positive = int(np.count_nonzero(positive))
    n_negative = int(labels.size - n_positive)
    if n_positive == 0 or n_negative == 0:
        return None
    ranks = rankdata(values, method='average')
    rank_sum = float(np.sum(ranks[positive]))
    return float(
        (rank_sum - n_positive * (n_positive + 1) / 2.0) / (n_positive * n_negative)
    )


def bootstrap_ci(
    values: Sequence[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 42,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Percentile bootstrap confidence interval of ``statistic`` over ``values``."""
    array = np.asarray(list(values), dtype=float)
    array = array[~np.isnan(array)]
    n = int(array.size)
    interval = {
        'point': None,
        'low': None,
        'high': None,
        'n': n,
        'n_boot': int(n_boot),
        'alpha': float(alpha),
    }
    if n == 0:
        return interval
    point = float(statistic(array))
    interval['point'] = point
    if n == 1:
        interval['low'] = point
        interval['high'] = point
        return interval

    rng = np.random.RandomState(seed)
    draws = np.empty(int(n_boot), dtype=float)
    for index in range(int(n_boot)):
        draws[index] = statistic(array[rng.randint(0, n, n)])
    interval['low'] = float(np.percentile(draws, 100.0 * alpha / 2.0))
    interval['high'] = float(np.percentile(draws, 100.0 * (1.0 - alpha / 2.0)))
    return interval


def bootstrap_roc_auc_ci(
    y_true: Sequence[int],
    scores: Sequence[float],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 42,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Percentile bootstrap CI of ROC-AUC over paired ``(label, score)`` cases."""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    n = int(len(y))
    interval = {
        'point': _safe_roc_auc(y, s),
        'low': None,
        'high': None,
        'n': n,
        'n_boot': int(n_boot),
        'alpha': float(alpha),
    }
    if n == 0:
        return interval

    rng = np.random.RandomState(seed)
    draws: list[float] = []
    for _ in range(int(n_boot)):
        index = rng.randint(0, n, n)
        value = _safe_roc_auc(y[index], s[index])
        if value is not None:
            draws.append(value)
    if draws:
        interval['low'] = float(np.percentile(draws, 100.0 * alpha / 2.0))
        interval['high'] = float(np.percentile(draws, 100.0 * (1.0 - alpha / 2.0)))
    return interval


def bootstrap_paired_ci(
    y_true: Sequence[int],
    scores: Sequence[float],
    statistic: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = 42,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Bootstrap CI of a paired score statistic (e.g. F1 at a fixed threshold)."""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    n = int(len(y))
    interval = {
        'point': float(statistic(y, s)) if n else None,
        'low': None,
        'high': None,
        'n': n,
        'n_boot': int(n_boot),
        'alpha': float(alpha),
    }
    if n == 0:
        return interval

    rng = np.random.RandomState(seed)
    draws: list[float] = []
    for _ in range(int(n_boot)):
        index = rng.randint(0, n, n)
        draws.append(float(statistic(y[index], s[index])))
    if draws:
        interval['low'] = float(np.percentile(draws, 100.0 * alpha / 2.0))
        interval['high'] = float(np.percentile(draws, 100.0 * (1.0 - alpha / 2.0)))
    return interval


# -- Pareto front ------------------------------------------------------------
def _get_path(record: Mapping[str, Any], path: str) -> Any:
    """Read a dotted ``'a.b.c'`` path from a nested mapping (``None`` if absent)."""
    current: Any = record
    for part in path.split('.'):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _strictly_better(left: float, right: float, direction: str) -> bool:
    return left > right if direction == 'max' else left < right


def dominates(
    values_a: Sequence[float], values_b: Sequence[float], directions: Sequence[str]
) -> bool:
    """True when ``a`` dominates ``b`` (no worse everywhere, strictly better once)."""
    strictly = False
    for a_value, b_value, direction in zip(values_a, values_b, directions):
        if _strictly_better(b_value, a_value, direction):
            return False
        if _strictly_better(a_value, b_value, direction):
            strictly = True
    return strictly


def pareto_front(
    records: Sequence[Mapping[str, Any]],
    objectives: Sequence[tuple[str, str]],
    id_key: str = 'key',
) -> dict[str, Any]:
    """Split ``records`` into the non-dominated front and the dominated rest.

    ``objectives`` is a list of ``(dotted_path, 'max'|'min')``. A record with any
    missing objective (e.g. an unpriceable cost) is excluded from the front rather
    than being silently treated as zero.
    """
    directions = [direction for _, direction in objectives]
    paths = [path for path, _ in objectives]

    parsed: list[tuple[Mapping[str, Any], list[float]]] = []
    excluded: list[Any] = []
    for record in records:
        values: list[float] = []
        missing = False
        for path in paths:
            value = _get_path(record, path)
            if value is None:
                missing = True
                break
            values.append(float(value))
        if missing:
            excluded.append(record.get(id_key))
        else:
            parsed.append((record, values))

    front: list[Mapping[str, Any]] = []
    dominated: list[Any] = []
    for index, (record, values) in enumerate(parsed):
        is_dominated = False
        for other_index, (_, other_values) in enumerate(parsed):
            if index == other_index:
                continue
            if dominates(other_values, values, directions):
                is_dominated = True
                break
        if is_dominated:
            dominated.append(record.get(id_key))
        else:
            front.append(record)

    def _row(record: Mapping[str, Any]) -> dict[str, Any]:
        row: dict[str, Any] = {id_key: record.get(id_key)}
        for path in paths:
            row[path] = _get_path(record, path)
        return row

    return {
        'front': [_row(record) for record in front],
        'dominated': dominated,
        'excluded': excluded,
        'front_size': len(front),
        'dominated_size': len(dominated),
        'excluded_size': len(excluded),
    }
