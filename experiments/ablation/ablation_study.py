"""Agent ablation study on the real freMTPL2 underwriting pipeline.

Research axis "ablation" (report/cadrage.md section 6.2 / 7.3): what does each agent
actually contribute to the recommendation produced by the eleven-agent LangGraph
pipeline? Every ablatable agent is removed in turn (leave-one-out) on a bounded
stratified sample of the real hold-out partition, and the result is compared with the
full pipeline on the *same* cases:

* **decision change rate** — share of cases whose final recommendation changes when
  the agent is removed (0 means the agent never altered the recommendation);
* **discrimination** — ROC-AUC / PR-AUC of the final risk score;
* **report completeness** — number of agents still contributing to the consolidated
  report, which exposes the agents that only feed governance rather than the decision.

The core chain (intake, risk, decision) is structural and cannot be removed: without a
risk estimate or a decision there is no recommendation to measure.
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
from orchestrator.langgraph_orchestrator import (  # noqa: E402
    AGENT_NODES,
    LangGraphOrchestrator,
)

RESULTS_DIR = PROJECT_ROOT / 'experiments' / 'ablation' / 'results'
DEFAULT_SAMPLE_SIZE = 100
# The pilot's frozen decision mode is human-in-the-loop; ``automated`` is kept as the
# reference configuration used by the B4 baseline. Collaboration mode is fixed to
# ``sequential`` so the ablation isolates the *decision* configuration.
COLLABORATION_MODE = 'sequential'
MODES = ('automated', 'human_in_the_loop')
# Agents that can be removed without breaking the decision chain. The three others
# (intake, risk, decision) are structural.
STRUCTURAL_AGENTS = ('intake', 'risk', 'decision')
ABLATABLE_AGENTS = tuple(name for name in AGENT_NODES if name not in STRUCTURAL_AGENTS)


def _application(record, position) -> dict:
    """Build a valid application from a real freMTPL2 record."""
    application = {'application_id': f'ABL-{int(position)}', 'product_line': 'auto'}
    application.update(record)
    return application


def _run_configuration(orchestrator, records, positions, decision_mode, disabled) -> dict:
    """Run the pipeline once with ``disabled`` agents ablated; collect the outcomes."""
    decisions, scores, report_agents = [], [], []
    started = time.perf_counter()
    # The agents are chatty on stdout; keep the research output readable.
    with contextlib.redirect_stdout(io.StringIO()):
        for record, position in zip(records, positions):
            state = orchestrator.run_workflow(
                _application(record, position),
                mode=COLLABORATION_MODE,
                decision_mode=decision_mode,
                disabled_agents=disabled,
            )
            decision_result = state['decision_result']
            decisions.append(decision_result['final_decision'])
            scores.append(float(decision_result['final_risk_score']) / 100.0)
            report_agents.append(len(state['report_result']['provenance']['agents']))
    elapsed = time.perf_counter() - started
    return {
        'disabled_agents': list(disabled),
        'decisions': decisions,
        'agents_in_report': float(np.mean(report_agents)),
        'latency_seconds_per_case': elapsed / len(records) if records else 0.0,
        'scores': scores,
    }


def _ablate_mode(orchestrator, records, index, y_true, mode) -> dict:
    """Run the full pipeline and every leave-one-out configuration in one mode."""
    baseline_raw = _run_configuration(orchestrator, records, index, mode, ())
    baseline = {k: v for k, v in baseline_raw.items() if k != 'scores'}
    baseline.update(decision_summary(y_true, baseline_raw['decisions']))
    baseline.update(score_metrics(y_true, baseline_raw['scores']))

    ablations = {}
    for agent in ABLATABLE_AGENTS:
        raw = _run_configuration(orchestrator, records, index, mode, (agent,))
        entry = {k: v for k, v in raw.items() if k != 'scores'}
        entry['decision_change_rate'] = 1.0 - agreement_rate(
            baseline_raw['decisions'], raw['decisions']
        )
        entry.update(decision_summary(y_true, raw['decisions']))
        entry.update(score_metrics(y_true, raw['scores']))
        ablations[agent] = entry
    return {'mode': mode, 'baseline': baseline, 'ablations': ablations}


def run(sample_size=DEFAULT_SAMPLE_SIZE, seed=RANDOM_STATE) -> dict:
    """Leave-one-out ablation of the pipeline agents on the real hold-out."""
    holdout = holdout_partition()
    index = stratified_sample(
        holdout['test_indices'], holdout['targets'], sample_size, seed
    )
    y_true = np.asarray(holdout['targets'][index])
    records = [holdout['records'][position] for position in index]

    orchestrator = LangGraphOrchestrator()
    return {
        'experiment': 'ablation',
        'status': 'ok',
        'description': 'leave-one-out ablation of the eleven-agent pipeline',
        'structural_agents': list(STRUCTURAL_AGENTS),
        'sample_size': int(len(index)),
        'sample_seed': seed,
        'modes': {
            mode: _ablate_mode(orchestrator, records, index, y_true, mode)
            for mode in MODES
        },
    }


if __name__ == '__main__':
    result = run()
    print(f"Sample: {result['sample_size']} policies (seed {result['sample_seed']})")
    for mode, block in result['modes'].items():
        base = block['baseline']
        print(f"\n=== mode {mode} ===")
        print(
            f"Full pipeline: ROC-AUC {base['roc_auc']:.4f} | "
            f"decisions {base['decision_counts']} | "
            f"agents in report {base['agents_in_report']:.1f}"
        )
        for agent, entry in block['ablations'].items():
            print(
                f"  -{agent:12s} decision change {entry['decision_change_rate']:.4f} | "
                f"ROC-AUC {entry['roc_auc']:.4f} | agents {entry['agents_in_report']:.1f}"
            )
    print(f"\nSaved: {save_result('ablation', result, directory=RESULTS_DIR)}")
