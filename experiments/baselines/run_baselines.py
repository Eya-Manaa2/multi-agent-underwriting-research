"""Reproducible runner for the freMTPL2 underwriting baselines B1-B4.

Runs every baseline on the real public freMTPL2 benchmark and prints a consolidated
comparison, then writes the full report to ``experiments/baselines/results``.

    python -m experiments.baselines.run_baselines

B1 (rules) and B2 (ML) are scored on the whole hold-out partition. B3 (single LLM
agent) and B4 (multi-agent pipeline) run on a bounded stratified sample because they
are far heavier per case; B3 is skipped with an explicit ``unavailable`` status when
no LLM provider is configured (fail-closed, no fabricated result).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.baselines import (  # noqa: E402
    baseline_b1_rules,
    baseline_b2_ml,
    baseline_b3_llm_agent,
    baseline_b4_pipeline,
)
from experiments.baselines.evaluation import RANDOM_STATE, save_result  # noqa: E402

DEFAULT_SAMPLE_SIZE = 200


def _print_baseline(name: str, result: dict) -> None:
    print(f"\n=== {name} ===")
    if result.get('status') not in (None, 'ok'):
        print(f"  not run ({result.get('status')}): {result.get('reason')}")
        return
    if 'roc_auc' in result and result['roc_auc'] is not None:
        print(f"  ROC-AUC: {result['roc_auc']:.4f} | PR-AUC: {result['pr_auc']:.4f}")
        print(f"  Accuracy @0.5: {result['accuracy_at_0_5']:.4f}")
    print(f"  Decisions: {result['decision_counts']}")
    if result.get('flagged_rate') is not None:
        print(
            f"  Adverse flag rate: {result['flagged_rate']:.4f} "
            f"(precision {result['flag_precision']:.4f}, "
            f"recall {result['flag_recall']:.4f})"
        )


def main(argv=None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sample-size', type=int, default=DEFAULT_SAMPLE_SIZE,
                        help='sample size for the heavy B4 pipeline baseline')
    parser.add_argument('--seed', type=int, default=RANDOM_STATE)
    parser.add_argument('--b3-sample', type=int,
                        default=baseline_b3_llm_agent.DEFAULT_SAMPLE_SIZE,
                        help='sample size for the B3 LLM baseline')
    parser.add_argument('--b3-runs', type=int, default=baseline_b3_llm_agent.DEFAULT_RUNS)
    parser.add_argument('--b3-rpm', type=int,
                        default=baseline_b3_llm_agent.DEFAULT_CALLS_PER_MINUTE,
                        help='max B3 LLM calls per minute (provider rate limit)')
    parser.add_argument('--skip-b3', action='store_true')
    parser.add_argument('--yes-b3', action='store_true',
                        help='confirm a B3 run above the call guard-rail')
    parser.add_argument('--skip-b4', action='store_true')
    args = parser.parse_args(argv)

    report = {
        'generated_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'benchmark': {
            'dataset': 'freMTPL2freq (OpenML 41214, CC0)',
            'target': 'real observed outcome: policy filed at least one claim',
            'partition': 'stratified 80/20 hold-out, random_state=42',
            'note': (
                'freMTPL2 carries no accept/reject ground truth: decision metrics are '
                'proxy metrics against the observed claim outcome.'
            ),
        },
        'sample_size': args.sample_size,
        'seed': args.seed,
        'baselines': {},
    }

    print('Running B1 (rules only) on the full hold-out ...')
    report['baselines']['B1'] = baseline_b1_rules.run()
    _print_baseline('B1 - rules only', report['baselines']['B1'])

    print('Running B2 (ML only) on the full hold-out ...')
    report['baselines']['B2'] = baseline_b2_ml.run()
    _print_baseline('B2 - ML only', report['baselines']['B2'])

    if args.skip_b3:
        report['baselines']['B3'] = {'baseline': 'B3_single_llm_agent',
                                     'status': 'skipped'}
    else:
        planned_calls = args.b3_sample * args.b3_runs + 1
        print(
            f"Running B3 (single LLM agent): {args.b3_sample} policies x "
            f"{args.b3_runs} runs = {planned_calls} calls at {args.b3_rpm}/min ..."
        )
        report['baselines']['B3'] = baseline_b3_llm_agent.run(
            sample_size=args.b3_sample,
            runs=args.b3_runs,
            seed=args.seed,
            calls_per_minute=args.b3_rpm,
            allow_large=args.yes_b3,
        )
    _print_baseline('B3 - single LLM agent', report['baselines']['B3'])

    if args.skip_b4:
        report['baselines']['B4'] = {'baseline': 'B4_multi_agent_pipeline',
                                     'status': 'skipped'}
    else:
        print('Running B4 (multi-agent pipeline) ...')
        report['baselines']['B4'] = baseline_b4_pipeline.run_all_modes(
            sample_size=args.sample_size, seed=args.seed
        )
        for mode, result in report['baselines']['B4']['modes'].items():
            _print_baseline(f'B4 - pipeline ({mode})', result)
        print(f"  Mode agreement: {report['baselines']['B4']['mode_agreement']:.4f}")

    print(f"\nSaved: {save_result('baselines_report', report)}")
    return report


if __name__ == '__main__':
    main()
