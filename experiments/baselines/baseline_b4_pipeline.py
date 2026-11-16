"""B4 - multi-agent underwriting pipeline (LangGraph, 11 agents).

Runs the real orchestrator (intake, client, risque, incertitude, documents, externe,
sinistres, décision, équité, conformité, explication) on a bounded stratified sample
of the hold-out, in both ``sequential`` and ``parallel`` collaboration modes, so the
collaboration axis of the research question can be compared on identical cases.

The full pipeline is heavier than a single model and re-runs eleven agents per case,
so the sample is bounded by design; the runner also reports the measured latency.
"""

from __future__ import annotations

import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.baselines.evaluation import (  # noqa: E402
    RANDOM_STATE,
    agreement_rate,
    decision_summary,
    holdout_partition,
    save_result,
    score_metrics,
    stratified_sample,
)
from orchestrator.langgraph_orchestrator import LangGraphOrchestrator  # noqa: E402

DEFAULT_SAMPLE_SIZE = 200
MODES = ('sequential', 'parallel')


def _application(record, position) -> dict:
    """Build a valid application from a real freMTPL2 record."""
    application = {'application_id': f'B4-{int(position)}', 'product_line': 'auto'}
    application.update(record)
    return application


def run(mode='sequential', sample_size=DEFAULT_SAMPLE_SIZE, seed=RANDOM_STATE) -> dict:
    """Run the real 11-agent pipeline on a bounded stratified hold-out sample."""
    if mode not in MODES:
        raise ValueError(f'Unknown collaboration mode: {mode}')

    holdout = holdout_partition()
    index = stratified_sample(
        holdout['test_indices'], holdout['targets'], sample_size, seed
    )
    y_true = np.asarray(holdout['targets'][index])

    orchestrator = LangGraphOrchestrator()
    decisions, scores = [], []
    started = time.perf_counter()
    # The agents are chatty on stdout; keep the research output readable.
    with contextlib.redirect_stdout(io.StringIO()):
        for position in index:
            application = _application(holdout['records'][position], position)
            state = orchestrator.run_workflow(
                application, mode=mode, decision_mode='automated'
            )
            decision_result = state['decision_result']
            decisions.append(decision_result['final_decision'])
            scores.append(float(decision_result['final_risk_score']) / 100.0)
    elapsed = time.perf_counter() - started

    return {
        'baseline': 'B4_multi_agent_pipeline',
        'status': 'ok',
        'description': 'full eleven-agent LangGraph pipeline',
        'collaboration_mode': mode,
        'sample_size': int(len(index)),
        'sample_seed': seed,
        'latency_seconds_total': elapsed,
        'latency_seconds_per_case': elapsed / len(index),
        'decisions': decisions,
        **score_metrics(y_true, scores),
        **decision_summary(y_true, decisions),
    }


def run_all_modes(sample_size=DEFAULT_SAMPLE_SIZE, seed=RANDOM_STATE) -> dict:
    """Run both collaboration modes and report their decision agreement."""
    results = {mode: run(mode, sample_size, seed) for mode in MODES}
    combined = {
        'baseline': 'B4_multi_agent_pipeline',
        'status': 'ok',
        'sample_size': results['sequential']['sample_size'],
        'sample_seed': seed,
        'modes': results,
        'mode_agreement': agreement_rate(
            results['sequential']['decisions'], results['parallel']['decisions']
        ),
    }
    return combined


if __name__ == '__main__':
    combined = run_all_modes()
    for mode in MODES:
        result = combined['modes'][mode]
        print(f"[{mode}] {result['sample_size']} cases · "
              f"{result['latency_seconds_per_case'] * 1000:.1f} ms/case")
        print(f"  Base claim rate: {result['base_rate']:.4f}")
        print(f"  ROC-AUC (risk score): {result['roc_auc']:.4f}")
        print(f"  Decision counts: {result['decision_counts']}")
        print(
            f"  Adverse flag: rate {result['flagged_rate']:.4f} | "
            f"precision {result['flag_precision']:.4f} | "
            f"recall {result['flag_recall']:.4f}"
        )
    print(f"Mode agreement (sequential vs parallel): {combined['mode_agreement']:.4f}")
    print(f"Saved: {save_result('b4_multi_agent_pipeline', combined)}")
