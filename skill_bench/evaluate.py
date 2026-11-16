"""The exhaustive campaign: score every valid combination over K seeds and save JSON.

For each valid combination (the mandatory core plus any subset of the eight optional
agents) and each seed, the pipeline runs on a bounded stratified sample of the real
freMTPL2 hold-out partition. The benchmark records, per combination:

* **quality**    -- ROC-AUC and F1 of the final risk score against the observed claim
                    outcome (a real, independent target), with 95% bootstrap CIs;
* **latency**    -- wall-clock milliseconds per case;
* **tokens**     -- input/output tokens per case (0 for the deterministic agents);
* **cost**       -- estimated USD per case (0 without a provider call);
* **robustness** -- decision stability under bounded Gaussian noise, and the change
                    in ROC-AUC under the same noise, with bootstrap CIs.

Everything is written to ``skill_bench/results/``; the Pareto front is written to
``skill_bench/results/pareto.json``. No figure in this project is typed by hand.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skill_bench.config import default_config_path, load_config  # noqa: E402
from skill_bench.counter import CostCounter, LITELLM_AVAILABLE  # noqa: E402
from skill_bench.metrics import (  # noqa: E402
    bootstrap_ci,
    bootstrap_paired_ci,
    bootstrap_roc_auc_ci,
    pareto_front,
)
from skill_bench.robustness import (  # noqa: E402
    NUMERIC_FIELDS,
    decision_stability,
    drop_field,
    noisy_application,
    reference_std,
)
from skill_bench.underwriting import (  # noqa: E402
    UNDERWRITING_CORE,
    SkillPipeline,
    valid_underwriting_combinations,
)

RESULTS_DIR = PROJECT_ROOT / 'skill_bench' / 'results'
EVALUATIONS_FILE = 'underwriting_evaluations.json'
PARETO_FILE = 'pareto.json'

#: Structural limits that hold whatever the measurements turn out to be. The
#: data-dependent limits are derived from the results by ``_derive_limitations``.
BASE_LIMITATIONS = (
    'freMTPL2 carries no insurer accept/reject ground truth: the target is the real '
    'observed fact "the policy filed at least one claim". Quality is therefore a '
    'discrimination proxy of the final risk score against an observed outcome, not a '
    'measure of an acceptance policy.',
    'The eleven agents are deterministic by default (no provider call): token and '
    'dollar totals are legitimately zero unless an LLM skill is enabled with a '
    'provider key.',
    'Latency is measured after a warm-up pass (see ``_warm_up``) so the first '
    'combination does not pay the one-off model and import cost; the values are '
    'wall-clock milliseconds on a shared machine and carry the usual jitter.',
    'Missing-field robustness is a fail-closed property of the mandatory intake '
    'agent: it is identical across combinations and is therefore reported once, not '
    'per combination.',
)


def _application(record: Mapping[str, Any], position: int) -> dict[str, Any]:
    application = {'application_id': f'SB-{int(position)}', 'product_line': 'auto'}
    application.update(record)
    return application


def _f1_at_05(y_true: np.ndarray, scores: np.ndarray) -> float:
    """F1 of the claim class at a fixed 0.5 threshold.

    Written out arithmetically rather than through ``sklearn.metrics.f1_score`` so
    the bootstrap loop does not pay scikit-learn's validation cost per resample.
    ``tests/test_skill_bench.py`` pins the equivalence against scikit-learn.
    """
    labels = np.asarray(y_true)
    predicted = np.asarray(scores, dtype=float) >= 0.5
    positive = labels == 1
    true_positive = int(np.count_nonzero(predicted & positive))
    false_positive = int(np.count_nonzero(predicted & ~positive))
    false_negative = int(np.count_nonzero(~predicted & positive))
    denominator = 2 * true_positive + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return float(2 * true_positive / denominator)


def _run_sample(
    pipeline: SkillPipeline,
    applications: Sequence[Mapping[str, Any]],
    included: Sequence[str],
    decision_mode: str,
    mode: str,
    price_table: Mapping[str, Any] | None = None,
) -> tuple[list[str], list[float], list[float], CostCounter]:
    """Run the pipeline on every application; return decisions, scores, latencies."""
    decisions: list[str] = []
    scores: list[float] = []
    latencies: list[float] = []
    counter = CostCounter(price_table=price_table)
    with contextlib.redirect_stdout(io.StringIO()):
        for application in applications:
            started = time.perf_counter()
            state, case_counter = pipeline.run(
                application,
                included=included,
                decision_mode=decision_mode,
                mode=mode,
            )
            latencies.append((time.perf_counter() - started) * 1000.0)
            counter.extend(case_counter)
            decision_result = state['decision_result']
            decisions.append(decision_result['final_decision'])
            scores.append(float(decision_result['final_risk_score']) / 100.0)
    return decisions, scores, latencies, counter


def _warm_up(
    pipeline: SkillPipeline,
    holdout: Mapping[str, Any],
    included: Sequence[str],
    config: Mapping[str, Any],
) -> int:
    """Prime the process before any timed run; return the number of warm-up cases.

    The very first pipeline invocations in a fresh process are measurably slower
    than the steady state (model load, imports, CPU ramp). Running them here, once,
    keeps the per-combination latency comparable instead of charging the whole
    cold-start to whichever combination happens to be evaluated first.
    """
    size = min(int(config['sample_size']), 12)
    seed = int(config['seeds'][0])
    indices = _stratified_indices(holdout, size, seed)
    applications = [
        _application(holdout['records'][int(position)], int(position))
        for position in indices
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        for application in applications:
            pipeline.run(
                application,
                included=included,
                decision_mode=config['decision_mode'],
                mode=config['collaboration_mode'],
            )
    return len(applications)


def evaluate_combination(
    pipeline: SkillPipeline,
    holdout: Mapping[str, Any],
    included: Sequence[str],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate one valid combination over the configured seeds."""
    seeds = list(config['seeds'])
    sample_size = int(config['sample_size'])
    decision_mode = config['decision_mode']
    mode = config['collaboration_mode']
    robustness_cfg = config['robustness']
    robustness_size = min(int(robustness_cfg.get('sample_size', sample_size)), sample_size)
    noise_level = float(robustness_cfg.get('noise_level', 0.10))
    bootstrap_cfg = config['bootstrap']
    n_boot = int(bootstrap_cfg['n_boot'])
    alpha = float(bootstrap_cfg['alpha'])
    boot_seed = int(bootstrap_cfg['seed'])

    disabled = [name for name in pipeline.order if name not in set(included)]

    pooled_y: list[int] = []
    pooled_scores: list[float] = []
    pooled_latency: list[float] = []
    pooled_y_noisy: list[int] = []
    pooled_scores_noisy: list[float] = []
    stability_bits: list[float] = []
    per_seed: list[dict[str, Any]] = []

    total_tokens_in = 0
    total_tokens_out = 0
    total_cost = 0.0
    any_cost_unknown = False
    case_count = 0

    for seed in seeds:
        indices = np.asarray(
            _stratified_indices(holdout, sample_size, seed), dtype=int
        )
        records = [holdout['records'][int(position)] for position in indices]
        y_true = np.asarray(holdout['targets'][indices], dtype=int)
        applications = [
            _application(record, int(position))
            for record, position in zip(records, indices)
        ]

        decisions, scores, latencies, counter = _run_sample(
            pipeline, applications, included, decision_mode, mode,
            price_table=config.get('pricing'),
        )
        scores_array = np.asarray(scores, dtype=float)
        latency_array = np.asarray(latencies, dtype=float)

        pooled_y.extend(int(value) for value in y_true)
        pooled_scores.extend(float(value) for value in scores_array)
        pooled_latency.extend(float(value) for value in latency_array)

        total_tokens_in += counter.total_input_tokens
        total_tokens_out += counter.total_output_tokens
        if counter.total_cost_usd is None:
            any_cost_unknown = True
        else:
            total_cost += counter.total_cost_usd
        case_count += len(applications)

        seed_entry: dict[str, Any] = {
            'seed': int(seed),
            'sample_size': int(len(applications)),
            'base_rate': float(np.mean(y_true)),
            'roc_auc': _safe_mean_roc_auc(y_true, scores_array),
            'f1_claim': _f1_at_05(y_true, scores_array),
            'decision_counts': _decision_counts(decisions),
            'latency_per_case_ms': float(np.mean(latency_array)),
            'tokens_in_per_case': float(counter.total_input_tokens / len(applications)),
            'tokens_out_per_case': float(counter.total_output_tokens / len(applications)),
            'cost_usd_per_case': (
                None
                if counter.total_cost_usd is None
                else float(counter.total_cost_usd / len(applications))
            ),
            'llm_calls': int(counter.call_count),
        }

        if robustness_cfg.get('enabled', True) and robustness_size > 0:
            subset_records = records[:robustness_size]
            subset_y = y_true[:robustness_size]
            clean_subset_decisions = decisions[:robustness_size]
            std = reference_std(subset_records)
            rng = np.random.RandomState(int(seed))
            noisy_records = [
                noisy_application(record, noise_level, std, rng)
                for record in subset_records
            ]
            noisy_apps = [
                _application(record, int(position))
                for record, position in zip(noisy_records, indices[:robustness_size])
            ]
            noisy_decisions, noisy_scores, _, _ = _run_sample(
                pipeline, noisy_apps, included, decision_mode, mode,
                price_table=config.get('pricing'),
            )
            stability = decision_stability(clean_subset_decisions, noisy_decisions)
            stability_bits.extend(
                1.0 if clean == degraded else 0.0
                for clean, degraded in zip(clean_subset_decisions, noisy_decisions)
            )
            pooled_y_noisy.extend(int(value) for value in subset_y)
            pooled_scores_noisy.extend(float(value) for value in noisy_scores)
            seed_entry['noise_stability'] = stability
            seed_entry['noisy_roc_auc'] = _safe_mean_roc_auc(subset_y, noisy_scores)
            clean_subset_scores = scores_array[:robustness_size]
            seed_entry['roc_auc_delta_noise'] = _delta_roc_auc(
                subset_y, clean_subset_scores, noisy_scores
            )

        per_seed.append(seed_entry)

    metrics: dict[str, Any] = {
        'quality': {
            'roc_auc': bootstrap_roc_auc_ci(
                pooled_y, pooled_scores, n_boot=n_boot, seed=boot_seed, alpha=alpha
            ),
            'f1_claim': bootstrap_paired_ci(
                pooled_y,
                pooled_scores,
                _f1_at_05,
                n_boot=n_boot,
                seed=boot_seed,
                alpha=alpha,
            ),
        },
        'latency': {
            'per_case_ms': bootstrap_ci(
                pooled_latency, np.mean, n_boot=n_boot, seed=boot_seed, alpha=alpha
            )
        },
        'tokens': {
            'per_case': float((total_tokens_in + total_tokens_out) / max(case_count, 1)),
            'input_per_case': float(total_tokens_in / max(case_count, 1)),
            'output_per_case': float(total_tokens_out / max(case_count, 1)),
            'total_input': int(total_tokens_in),
            'total_output': int(total_tokens_out),
            'cases': int(case_count),
        },
        'cost': {
            'usd_per_case': (
                None if any_cost_unknown else float(total_cost / max(case_count, 1))
            ),
            'total_usd': None if any_cost_unknown else float(total_cost),
            'unknown_calls': bool(any_cost_unknown),
        },
    }

    metrics['robustness'] = {
        'noise_level': noise_level,
        'noise_sample_size': robustness_size,
        'noise_stability': bootstrap_ci(
            stability_bits, np.mean, n_boot=n_boot, seed=boot_seed, alpha=alpha
        ),
        'noisy_roc_auc': bootstrap_roc_auc_ci(
            pooled_y_noisy, pooled_scores_noisy, n_boot=n_boot, seed=boot_seed, alpha=alpha
        ),
        'roc_auc_delta_mean': (
            float(
                np.mean(
                    [
                        entry['roc_auc_delta_noise']
                        for entry in per_seed
                        if entry.get('roc_auc_delta_noise') is not None
                    ]
                )
            )
            if any(entry.get('roc_auc_delta_noise') is not None for entry in per_seed)
            else None
        ),
    }

    return {
        'key': '+'.join(included),
        'skills': list(included),
        'disabled': disabled,
        'size': len(included),
        'seed_count': len(seeds),
        'sample_size': sample_size,
        'cases_evaluated': case_count,
        'metrics': metrics,
        'per_seed': per_seed,
    }


# -- helpers -----------------------------------------------------------------
def _stratified_indices(holdout: Mapping[str, Any], size: int, seed: int) -> list[int]:
    from experiments.baselines.evaluation import stratified_sample

    return [int(value) for value in stratified_sample(
        holdout['test_indices'], holdout['targets'], size, seed=seed
    )]


def _safe_mean_roc_auc(y_true, scores) -> float | None:
    from skill_bench.metrics import _safe_roc_auc

    return _safe_roc_auc(y_true, scores)


def _delta_roc_auc(y_true, clean_scores, noisy_scores) -> float | None:
    clean = _safe_mean_roc_auc(y_true, clean_scores)
    noisy = _safe_mean_roc_auc(y_true, noisy_scores)
    if clean is None or noisy is None:
        return None
    return float(noisy - clean)


def _decision_counts(decisions: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision] = counts.get(decision, 0) + 1
    return dict(sorted(counts.items()))


def run_missing_field_check(
    pipeline: SkillPipeline, holdout: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Fail-closed check: a dropped required field must block the recommendation.

    Intake is part of the mandatory core, so this property is identical for every
    valid combination and is measured once.
    """
    sample_size = min(int(config['sample_size']), 10)
    seed = int(config['seeds'][0])
    indices = _stratified_indices(holdout, sample_size, seed)
    records = [holdout['records'][int(position)] for position in indices]

    per_field: dict[str, Any] = {}
    for field in NUMERIC_FIELDS:
        refused = 0
        for record, position in zip(records, indices):
            application = _application(drop_field(record, field), int(position))
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    pipeline.run(
                        application,
                        included=list(UNDERWRITING_CORE),
                        decision_mode=config['decision_mode'],
                        mode=config['collaboration_mode'],
                    )
            except (ValueError, RuntimeError, KeyError):
                refused += 1
        per_field[field] = {
            'cases': len(records),
            'refused': refused,
            'fail_closed_rate': float(refused / len(records)) if records else None,
        }
    rates = [entry['fail_closed_rate'] for entry in per_field.values()
             if entry['fail_closed_rate'] is not None]
    return {
        'description': 'dropping a required numeric field must block the recommendation',
        'sample_size': len(records),
        'per_field': per_field,
        'min_fail_closed_rate': float(min(rates)) if rates else None,
    }


def _effective_objectives(config: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The Pareto objectives that can actually be measured with this configuration.

    When the robustness degradations are disabled the robustness axis has no value,
    so it is dropped from the front instead of excluding every combination. The list
    actually used is written to ``meta.objectives``, so the artefact stays auditable.
    """
    objectives = [
        (str(path), str(direction))
        for path, direction in config['pareto']['objectives']
    ]
    if not config.get('robustness', {}).get('enabled', True):
        objectives = [
            item for item in objectives if not item[0].startswith('metrics.robustness')
        ]
    return objectives


def _metric_value(entry: Mapping[str, Any], path: str) -> Any:
    from skill_bench.metrics import _get_path

    return _get_path(entry, path)


def _derive_limitations(
    evaluated: Sequence[Mapping[str, Any]],
    objectives: Sequence[tuple[str, str]],
) -> list[str]:
    """The structural limits plus the ones the measurements actually revealed.

    The data-dependent sentences are built from the evaluated combinations, so they
    cannot drift away from the JSON they describe: if a future run makes an axis
    vary, the corresponding limitation disappears by itself.
    """
    limitations = list(BASE_LIMITATIONS)
    if not evaluated:
        return limitations

    constant_axes = [
        path
        for path, _ in objectives
        if len({_metric_value(entry, path) for entry in evaluated} - {None}) <= 1
    ]
    if constant_axes:
        limitations.append(
            f'Across the {len(evaluated)} evaluated combinations these Pareto axes '
            'take a single value and therefore cannot discriminate between them: '
            + ', '.join(constant_axes) + '.'
        )

    quality_values = {
        _metric_value(entry, 'metrics.quality.roc_auc.point') for entry in evaluated
    } - {None}
    if len(quality_values) <= 1:
        limitations.append(
            'The quality axis is identical for every combination: the final risk '
            'score IS the mandatory risk agent score, and the optional agents change '
            'the decision label without changing the score ranking. Quality separates '
            'nothing in this case -- that is a property of the pipeline, not of the '
            'observed target.'
        )

    stability_ns = {
        _metric_value(entry, 'metrics.robustness.noise_stability.n')
        for entry in evaluated
    } - {None}
    stability_values = {
        _metric_value(entry, 'metrics.robustness.noise_stability.point')
        for entry in evaluated
    } - {None}
    if len(stability_values) > 1 and len(stability_ns) == 1:
        bits = int(next(iter(stability_ns)))
        spread = round((max(stability_values) - min(stability_values)) * bits)
        limitations.append(
            f'Decision stability is quantised over {bits} measured noisy decisions, '
            f'so the whole spread between the best and worst combination is {spread} '
            'decision(s) out of '
            f'{bits}; a difference that small sits inside the published bootstrap '
            'interval and is not by itself a reliable ranking signal.'
        )
    return limitations


def evaluate(config: Mapping[str, Any]) -> dict[str, Any]:
    """Run the exhaustive campaign and return the full result payload."""
    from experiments.baselines.evaluation import holdout_partition
    from orchestrator.langgraph_orchestrator import LangGraphOrchestrator

    holdout = holdout_partition()
    orchestrator = LangGraphOrchestrator()
    pipeline = SkillPipeline(orchestrator, build_report=False)

    combinations = valid_underwriting_combinations()
    total = len(combinations)
    print(
        f'[skill-bench] {total} valid combinations x {len(config["seeds"])} seeds x '
        f'{config["sample_size"]} cases',
        file=sys.stderr,
    )

    warm_up_cases = _warm_up(pipeline, holdout, combinations[0], config)
    print(f'[skill-bench] warm-up: {warm_up_cases} cases discarded', file=sys.stderr)

    started = time.perf_counter()
    evaluated: list[dict[str, Any]] = []
    for position, combination in enumerate(combinations, start=1):
        entry = evaluate_combination(pipeline, holdout, combination, config)
        entry['status'] = 'ok'
        evaluated.append(entry)
        if position % 16 == 0 or position == total:
            elapsed = time.perf_counter() - started
            print(
                f'[skill-bench] {position}/{total} combinations '
                f'({elapsed:.0f}s elapsed)',
                file=sys.stderr,
            )

    missing_field = run_missing_field_check(pipeline, holdout, config)

    objectives = _effective_objectives(config)
    front = pareto_front(evaluated, objectives, id_key='key')

    base_rate = float(np.mean(holdout['targets'][holdout['test_indices']]))
    return {
        'meta': {
            'framework': 'agent-skill-bench',
            'version': '0.1.0',
            'case': config['case'],
            'generated_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'python': platform.python_version(),
            'litellm_available': LITELLM_AVAILABLE,
            'config': _plain(config),
            'objectives': [list(item) for item in objectives],
        },
        'dataset': {
            'name': 'freMTPL2freq',
            'source': 'OpenML 41214 (CC0), CASdatasets (Dutang & Charpentier)',
            'dataset_size': holdout['dataset_size'],
            'holdout_size': holdout['test_size'],
            'holdout_base_rate': base_rate,
            'target': 'observed: the policy filed at least one claim',
        },
        'graph': pipeline.graph.describe(),
        'core': list(UNDERWRITING_CORE),
        'combination_count': len(evaluated),
        'combinations': evaluated,
        'missing_field_check': missing_field,
        'pareto': front,
        'limitations': _derive_limitations(evaluated, objectives),
    }


def _plain(value: Any) -> Any:
    """Recursively convert numpy scalars to plain Python values for JSON."""
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def save_evaluations(evaluations: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / EVALUATIONS_FILE
    path.write_text(
        json.dumps(_plain(evaluations), ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return path


def save_pareto(evaluations: Mapping[str, Any], directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / PARETO_FILE
    payload = {
        'meta': evaluations['meta'],
        'objectives': evaluations['meta']['objectives'],
        'source_file': EVALUATIONS_FILE,
        'combination_count': evaluations['combination_count'],
        'pareto': evaluations['pareto'],
        'limitations': evaluations['limitations'],
    }
    path.write_text(
        json.dumps(_plain(payload), ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Run the agent-skill-bench exhaustive underwriting campaign.'
    )
    parser.add_argument('--config', default=None, help='YAML configuration (optional)')
    parser.add_argument('--sample-size', type=int, default=None)
    parser.add_argument('--seeds', type=int, default=None, help='number of seeds')
    parser.add_argument(
        '--no-robustness',
        action='store_true',
        help='skip the noise degradations (the robustness Pareto objective is dropped)',
    )
    parser.add_argument('--output-dir', default=None)
    parser.add_argument(
        '--fast',
        action='store_true',
        help='quick smoke profile: 8 cases, 2 seeds, small robustness sample',
    )
    args = parser.parse_args(argv)

    config = load_config(args.config or default_config_path())
    if args.fast:
        config['sample_size'] = 8
        config['seeds'] = [1, 2]
        config['bootstrap']['n_boot'] = 200
        config['robustness']['sample_size'] = 4
    if args.sample_size is not None:
        config['sample_size'] = args.sample_size
    if args.seeds is not None:
        config['seeds'] = list(range(1, args.seeds + 1))
    if args.no_robustness:
        config['robustness']['enabled'] = False

    directory = (
        Path(args.output_dir) if args.output_dir is not None
        else (Path(config['output_dir']) if config.get('output_dir') else RESULTS_DIR)
    )

    evaluations = evaluate(config)
    evaluations_path = save_evaluations(evaluations, directory)
    pareto_path = save_pareto(evaluations, directory)

    print(f'[skill-bench] evaluations: {evaluations_path}', file=sys.stderr)
    print(
        f'[skill-bench] pareto front: {evaluations["pareto"]["front_size"]} '
        f'non-dominated combinations -> {pareto_path}',
        file=sys.stderr,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
