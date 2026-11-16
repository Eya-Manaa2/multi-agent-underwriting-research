"""Collaboration schemes for the multi-agent underwriting pilot.

Research axis "collaboration" (report/cadrage.md section 4.1 / 6.2, hypothesis H1):
does a richer collaboration pattern (debate, weighted vote) decide better on ambiguous
cases than a plain sequential pipeline or a single agent?

Five configurations are compared on the *same* bounded stratified sample of the real
freMTPL2 hold-out:

* ``single_agent``  - one LLM agent (reference for H1, same infra as baseline B3);
* ``pipeline``      - deterministic sequential eleven-agent LangGraph pipeline;
* ``orchestrator``  - deterministic orchestrator fan-out (parallel graph): the
                      orchestrator dispatches to the specialist agents and aggregates;
* ``debate``        - several LLM agents with distinct roles debate for a few rounds,
                      then the final round is aggregated by majority vote;
* ``weighted_vote`` - several LLM agents vote independently, each ballot weighted by
                      the voter's own confidence.

Only ``single_agent``, ``debate`` and ``weighted_vote`` call a provider; they inherit
the B3 guard-rails (preflight, call cap, pacing) and stay fail-closed (no fabricated
result without a key). The two deterministic schemes reuse the shipped orchestrator and
need no LLM.

freMTPL2 carries no accept/reject ground truth: the reported decision metrics are proxy
metrics against the real observed claim outcome ("did the policy file a claim?").
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

from experiments.baselines.baseline_b3_llm_agent import (  # noqa: E402
    DEFAULT_CALLS_PER_MINUTE,
    MAX_CALLS_WITHOUT_CONFIRMATION,
    GeoRiskTool,
    SingleLLMAgentBaseline,
)
from experiments.baselines.evaluation import (  # noqa: E402
    RANDOM_STATE,
    decision_summary,
    holdout_partition,
    save_result,
    score_metrics,
    stratified_sample,
)
from orchestrator.langgraph_orchestrator import LangGraphOrchestrator  # noqa: E402
from utils.actuarial_features import decision_from_probability  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / 'experiments' / 'collaboration' / 'results'
DEFAULT_SAMPLE_SIZE = 15
DEFAULT_VOTERS = 3
DEFAULT_DEBATE_ROUNDS = 3

# The four documented collaboration schemes (report/cadrage.md section 6.2), plus the
# single-agent reference used to test H1.
STRUCTURED_SCHEMES = {'pipeline': 'sequential', 'orchestrator': 'parallel'}
LLM_REFERENCE = 'single_agent'
LLM_SCHEMES = ('debate', 'weighted_vote')

_JSON_CONTRACT = (
    'Réponds uniquement en JSON valide de la forme '
    '{"decision":"Accept_Standard|Accept_with_Surcharge|Reject",'
    '"claim_risk":0.0,"rationale":"..."}. '
    '"decision" doit appartenir exactement à cette liste et "claim_risk" est la '
    'probabilité de sinistre estimée, un nombre entre 0 et 1.'
)

# Distinct debating/voting roles. Each role shares the same JSON contract so every
# ballot can be validated by the B3 infrastructure.
VOTER_ROLES = (
    {
        'name': 'actuaire',
        'system': (
            'Tu es un actuaire automobile prudent. Tu privilégies la protection de '
            'l’assureur : en cas de doute sur la sinistralité, tu recommandes la '
            'surprime ou le refus. ' + _JSON_CONTRACT
        ),
    },
    {
        'name': 'souscripteur',
        'system': (
            'Tu es un souscripteur orienté relation client. Tu cherches l’équilibre '
            'entre le risque mesuré et l’acceptation commerciale du dossier. '
            + _JSON_CONTRACT
        ),
    },
    {
        'name': 'auditeur_risque',
        'system': (
            'Tu es un auditeur indépendant du risque. Tu es sceptique, tu exiges des '
            'preuves et tu recommandes le refus lorsque le risque te paraît sous-'
            'estimé. ' + _JSON_CONTRACT
        ),
    },
)


class _Pacer:
    """Global request pacer: keeps LLM calls within the provider requests-per-minute."""

    def __init__(self, calls_per_minute):
        self.min_interval = 60.0 / calls_per_minute if calls_per_minute else 0.0
        self.calls = 0
        self._last = 0.0

    def wait(self):
        if self.calls and self.min_interval:
            elapsed = time.perf_counter() - self._last
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
        self.calls += 1
        self._last = time.perf_counter()


def _application(record, position) -> dict:
    """Build a valid application from a real freMTPL2 record."""
    application = {'application_id': f'COL-{int(position)}', 'product_line': 'auto'}
    application.update(record)
    return application


def _score_block(y_true, decisions, scores) -> dict:
    """Common metric block for any scheme (discrimination + decision proxy)."""
    return {
        **score_metrics(y_true, scores),
        **decision_summary(y_true, decisions),
    }


# --------------------------------------------------------------------------- #
# Deterministic schemes (pipeline / orchestrator)
# --------------------------------------------------------------------------- #
def _run_structured(orchestrator, records, positions, graph_mode):
    """Run the shipped LangGraph pipeline; return decisions, scores and latency."""
    decisions, scores = [], []
    started = time.perf_counter()
    # The agents are chatty on stdout; keep the research output readable.
    with contextlib.redirect_stdout(io.StringIO()):
        for record, position in zip(records, positions):
            state = orchestrator.run_workflow(
                _application(record, position),
                mode=graph_mode,
                decision_mode='automated',
            )
            decision_result = state['decision_result']
            decisions.append(decision_result['final_decision'])
            scores.append(float(decision_result['final_risk_score']) / 100.0)
    elapsed = time.perf_counter() - started
    return decisions, scores, elapsed


# --------------------------------------------------------------------------- #
# LLM schemes (single agent reference / debate / weighted vote)
# --------------------------------------------------------------------------- #
def _ballot(agent, outcome, weight) -> dict:
    return {
        'agent': agent,
        'decision': outcome['decision'],
        'claim_risk': float(outcome['claim_risk']),
        'weight': float(weight),
    }


def _aggregate(ballots) -> str:
    """Weighted majority vote; ties fall back to the mean claim-risk mapping."""
    totals: dict[str, float] = {}
    for ballot in ballots:
        totals[ballot['decision']] = totals.get(ballot['decision'], 0.0) + ballot['weight']
    best = max(totals.values())
    winners = sorted(label for label, total in totals.items() if total == best)
    if len(winners) == 1:
        return winners[0]
    return decision_from_probability(
        float(np.mean([ballot['claim_risk'] for ballot in ballots]))
    )


def _debate_prompt(base_prompt, round_no, previous) -> str:
    if round_no == 1:
        tail = 'Tour 1 du débat : formule ta proposition initiale.'
    else:
        lines = '\n'.join(
            f"- {p['agent']} : {p['decision']} (probabilité {p['claim_risk']:.2f}) "
            f"— {p['rationale']}"
            for p in previous
        )
        tail = (
            f"Tour {round_no} du débat. Propositions des agents au tour précédent :\n"
            f"{lines}\n"
            'Confronte ces avis au tien, puis rends ta recommandation finale '
            '(tu peux la réviser ou la maintenir).'
        )
    return f'{base_prompt}\n{tail}\n{_JSON_CONTRACT}'


def _run_single_agent(agent, records, geo_tool, pacer):
    """One LLM agent decides every case (the H1 reference)."""
    decisions, risks = [], []
    for record in records:
        pacer.wait()
        geo = geo_tool.risk_for(record['area'])
        outcome = agent.recommend(record, geo)
        decisions.append(outcome['decision'])
        risks.append(outcome['claim_risk'])
    return decisions, np.asarray(risks, dtype=float)


def _run_weighted_vote(panel, records, geo_tool, pacer):
    """Each role votes independently; ballots are weighted by voter confidence."""
    decisions, mean_risks, disagreements = [], [], 0
    for record in records:
        geo = geo_tool.risk_for(record['area'])
        ballots = []
        for name, member in panel:
            pacer.wait()
            outcome = member.recommend(record, geo)
            # Confidence: distance from the maximally uncertain estimate (p = 0.5).
            weight = 2.0 * abs(outcome['claim_risk'] - 0.5)
            ballots.append(_ballot(name, outcome, weight))
        decisions.append(_aggregate(ballots))
        mean_risks.append(float(np.mean([b['claim_risk'] for b in ballots])))
        if len({b['decision'] for b in ballots}) > 1:
            disagreements += 1
    return decisions, np.asarray(mean_risks, dtype=float), disagreements


def _run_debate(panel, records, geo_tool, rounds, pacer):
    """Roles debate for ``rounds`` turns; the final round is a majority vote."""
    decisions, mean_risks = [], []
    initial_disagreements, final_disagreements, resolved = 0, 0, 0
    for record in records:
        geo = geo_tool.risk_for(record['area'])
        history = []
        first_round = None
        for round_no in range(1, rounds + 1):
            proposals = []
            for name, member in panel:
                pacer.wait()
                prompt = _debate_prompt(
                    member._build_prompt(record, geo), round_no, history
                )
                outcome = member.complete(prompt)
                proposals.append(_ballot(name, outcome, 1.0))
            if round_no == 1:
                first_round = proposals
            history = proposals
        # Final aggregation: unweighted majority over the last round.
        decisions.append(_aggregate(history))
        mean_risks.append(float(np.mean([b['claim_risk'] for b in history])))
        initial_split = len({b['decision'] for b in first_round}) > 1
        final_split = len({b['decision'] for b in history}) > 1
        initial_disagreements += int(initial_split)
        final_disagreements += int(final_split)
        resolved += int(initial_split and not final_split)
    total = len(records) or 1
    return decisions, np.asarray(mean_risks, dtype=float), {
        'initial_disagreement_rate': initial_disagreements / total,
        'final_disagreement_rate': final_disagreements / total,
        'resolved_disagreement_rate': (
            resolved / initial_disagreements if initial_disagreements else 1.0
        ),
    }


def _panel(provider, model, voters):
    roles = VOTER_ROLES[:max(1, min(voters, len(VOTER_ROLES)))]
    return [
        (
            role['name'],
            SingleLLMAgentBaseline(
                provider=provider, model=model, system_prompt=role['system']
            ),
        )
        for role in roles
    ]


def _llm_unavailable(mode, status, reason) -> dict:
    return {'kind': 'llm', 'status': status, 'reason': reason}


def run(
    sample_size=DEFAULT_SAMPLE_SIZE,
    voters=DEFAULT_VOTERS,
    rounds=DEFAULT_DEBATE_ROUNDS,
    seed=RANDOM_STATE,
    calls_per_minute=DEFAULT_CALLS_PER_MINUTE,
    allow_large=False,
) -> dict:
    """Compare the collaboration schemes on the real freMTPL2 hold-out."""
    holdout = holdout_partition()
    index = stratified_sample(
        holdout['test_indices'], holdout['targets'], sample_size, seed
    )
    y_true = np.asarray(holdout['targets'][index])
    records = [holdout['records'][position] for position in index]

    result = {
        'experiment': 'collaboration',
        'status': 'ok',
        'description': 'collaboration schemes compared on the real freMTPL2 hold-out',
        'hypothesis': (
            'H1: debate / weighted vote reduce errors on ambiguous cases versus a '
            'single agent'
        ),
        'sample_size': int(len(index)),
        'sample_seed': seed,
        'voters': int(voters),
        'debate_rounds': int(rounds),
        'schemes': {},
    }

    # --- Deterministic schemes (no LLM) -----------------------------------
    orchestrator = LangGraphOrchestrator()
    for name, graph_mode in STRUCTURED_SCHEMES.items():
        decisions, scores, elapsed = _run_structured(
            orchestrator, records, index, graph_mode
        )
        result['schemes'][name] = {
            'kind': 'deterministic',
            'collaboration_mode': graph_mode,
            'latency_seconds_per_case': elapsed / len(records) if records else 0.0,
            **_score_block(y_true, decisions, scores),
        }

    # --- LLM schemes -------------------------------------------------------
    reference = SingleLLMAgentBaseline()
    available, reason = reference.available()
    if not available:
        for name in (LLM_REFERENCE, *LLM_SCHEMES):
            result['schemes'][name] = _llm_unavailable(name, 'unavailable', reason)
        result['status'] = 'partial'
        return result

    # single agent + weighted vote + debate (``rounds`` turns) + preflight probe.
    planned_calls = (
        sample_size
        + sample_size * voters
        + sample_size * voters * rounds
        + 1
    )
    if planned_calls > MAX_CALLS_WITHOUT_CONFIRMATION and not allow_large:
        reason = (
            f'{planned_calls} appels LLM planifiés dépassent le garde-fou de '
            f'{MAX_CALLS_WITHOUT_CONFIRMATION}. Vérifie le quota du fournisseur puis '
            'relance avec allow_large=True.'
        )
        for name in (LLM_REFERENCE, *LLM_SCHEMES):
            result['schemes'][name] = _llm_unavailable(name, 'refused', reason)
        result['status'] = 'partial'
        return result

    ready, detail = reference.probe()
    if not ready:
        for name in (LLM_REFERENCE, *LLM_SCHEMES):
            result['schemes'][name] = _llm_unavailable(name, 'unavailable', detail)
        result['status'] = 'partial'
        return result

    provider = reference.provider or reference._detect_provider()
    model = reference.model or reference._default_model(provider)
    result['provider'] = provider
    result['model'] = model
    pacer = _Pacer(calls_per_minute)
    pacer.calls = 1  # count the preflight probe
    pacer._last = time.perf_counter()

    with contextlib.redirect_stdout(io.StringIO()):
        try:
            # Single-agent reference.
            started = time.perf_counter()
            decisions, risks = _run_single_agent(reference, records, GeoRiskTool(), pacer)
            elapsed = time.perf_counter() - started
            result['schemes'][LLM_REFERENCE] = {
                'kind': 'llm',
                'collaboration_mode': 'single',
                'latency_seconds_per_case': elapsed / len(records) if records else 0.0,
                **_score_block(y_true, decisions, risks),
            }

            panel = _panel(provider, model, voters)

            # Weighted vote.
            started = time.perf_counter()
            decisions, risks, disagreements = _run_weighted_vote(
                panel, records, GeoRiskTool(), pacer
            )
            elapsed = time.perf_counter() - started
            result['schemes']['weighted_vote'] = {
                'kind': 'llm',
                'collaboration_mode': 'weighted_vote',
                'voter_roles': [name for name, _ in panel],
                'initial_disagreement_rate': disagreements / len(records)
                if records
                else 0.0,
                'latency_seconds_per_case': elapsed / len(records) if records else 0.0,
                **_score_block(y_true, decisions, risks),
            }

            # Debate.
            started = time.perf_counter()
            decisions, risks, debate_stats = _run_debate(
                panel, records, GeoRiskTool(), rounds, pacer
            )
            elapsed = time.perf_counter() - started
            result['schemes']['debate'] = {
                'kind': 'llm',
                'collaboration_mode': 'debate',
                'voter_roles': [name for name, _ in panel],
                'debate_rounds': int(rounds),
                **debate_stats,
                'latency_seconds_per_case': elapsed / len(records) if records else 0.0,
                **_score_block(y_true, decisions, risks),
            }
        except (ValueError, RuntimeError) as error:
            result['status'] = 'error'
            result['reason'] = (
                f'{error} ({pacer.calls} appel(s) LLM réussi(s) avant l’échec)'
            )
            return result

    result['llm_calls'] = pacer.calls
    result['calls_per_minute'] = calls_per_minute
    # Decision agreement between the sequential pipeline and the orchestrator fan-out.
    return result


if __name__ == '__main__':
    outcome = run()
    print(f"Sample: {outcome['sample_size']} policies (seed {outcome['sample_seed']})")
    for name, scheme in outcome['schemes'].items():
        if scheme.get('status'):
            print(f"  {name:14s} not run ({scheme['status']}): {scheme.get('reason')}")
            continue
        print(
            f"  {name:14s} ROC-AUC {scheme['roc_auc']:.4f} | "
            f"decisions {scheme['decision_counts']} | "
            f"{scheme['latency_seconds_per_case'] * 1000:.1f} ms/case"
        )
        if 'initial_disagreement_rate' in scheme:
            print(
                f"     disagreement: initial {scheme['initial_disagreement_rate']:.2f}"
            )
        if 'resolved_disagreement_rate' in scheme:
            print(
                f"     disagreement: initial {scheme['initial_disagreement_rate']:.2f} "
                f"→ final {scheme['final_disagreement_rate']:.2f} "
                f"(resolved {scheme['resolved_disagreement_rate']:.2f})"
            )
    if outcome.get('llm_calls'):
        print(f"LLM calls: {outcome['llm_calls']} (capped at {outcome['calls_per_minute']}/min)")
    print(f"Saved: {save_result('collaboration', outcome, directory=RESULTS_DIR)}")
