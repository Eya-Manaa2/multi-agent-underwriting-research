"""Configuration loading for the benchmark (YAML + built-in defaults)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG: dict[str, Any] = {
    'case': 'underwriting',
    'seeds': [1, 2, 3, 4, 5],
    'sample_size': 60,
    'collaboration_mode': 'sequential',
    'decision_mode': 'human_in_the_loop',
    'robustness': {
        'enabled': True,
        'sample_size': 20,
        'noise_level': 0.10,
    },
    'bootstrap': {
        'n_boot': 2000,
        'alpha': 0.05,
        'seed': 42,
    },
    # USD per one million tokens, per model. Empty by default: without LiteLLM and
    # without an entry here, a token-consuming model is reported as unpriceable.
    'pricing': {},
    'pareto': {
        # Objective paths must match the structure produced by
        # ``evaluate.evaluate_combination``: quality, latency and robustness are
        # bootstrap intervals (hence the trailing ``.point``), while the token and
        # cost axes are scalars.
        'objectives': [
            ['metrics.quality.roc_auc.point', 'max'],
            ['metrics.latency.per_case_ms.point', 'min'],
            ['metrics.cost.usd_per_case', 'min'],
            ['metrics.tokens.per_case', 'min'],
            ['metrics.robustness.noise_stability.point', 'max'],
        ]
    },
    'output_dir': None,
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Return the default configuration, overridden by an optional YAML file."""
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f'configuration file not found: {config_path}')
    loaded = yaml.safe_load(config_path.read_text(encoding='utf-8')) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f'{config_path} must contain a YAML mapping')
    return _deep_merge(DEFAULT_CONFIG, loaded)


def default_config_path() -> Path:
    return PROJECT_ROOT / 'skill_bench' / 'configs' / 'underwriting.yaml'
