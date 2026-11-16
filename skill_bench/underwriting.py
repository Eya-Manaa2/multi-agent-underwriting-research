"""Wrap the eleven existing underwriting agents as skills, and run a combination.

This module is the bridge between the benchmark framework and the real repository:
it does **not** reimplement any agent. Each skill is a thin wrapper around the very
orchestrator node methods used by the LangGraph workflow, so a skill call executes the
same agent code, in the same order, with the same shared state.

Hard dependencies (grounded in the code):

* ``risk`` requires ``intake``  -- the risk model validates via ``IntakeAgent``;
* ``decision`` requires ``risk`` -- ``EnhancedDecisionAgent`` raises without a valid
  risk recommendation;
* ``customer`` / ``document`` / ``external`` / ``claims`` require ``intake``;
* ``uncertainty`` requires ``risk``;
* ``fairness`` / ``compliance`` / ``explanation`` require ``decision``.

The three skills the pipeline cannot do without -- ``intake``, ``risk``, ``decision``
-- form the mandatory core, exactly as in ``experiments/ablation/ablation_study.py``.
The eight remaining agents are optional; every subset of them is a valid
combination (their hard dependencies are always inside the core).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skill_bench.counter import CostCounter  # noqa: E402
from skill_bench.graph import SkillGraph  # noqa: E402
from skill_bench.skill import FunctionSkill, Skill, SkillMetadata  # noqa: E402

# (skill name, hard dependencies, wrapped class, description)
UNDERWRITING_SKILL_SPECS: tuple[tuple[str, tuple[str, ...], str, str], ...] = (
    (
        'intake',
        (),
        'IntakeAgent',
        'Validate and prepare the real freMTPL2 application (fail-closed).',
    ),
    (
        'customer',
        ('intake',),
        'CustomerAgent',
        'Driver profile bands and risk factors.',
    ),
    (
        'risk',
        ('intake',),
        'RiskAgent',
        'Real freMTPL2 claim-occurrence model (XGBoost).',
    ),
    (
        'uncertainty',
        ('risk',),
        'UncertaintyAgent',
        'Model predictive uncertainty and escalation recommendation.',
    ),
    (
        'document',
        ('intake',),
        'DocumentAgent',
        'Local document checks; optional grounded LLM review (off by default).',
    ),
    (
        'external',
        ('intake',),
        'ExternalAgent',
        'Zone risk and AGIRA insurance-status external factors.',
    ),
    (
        'claims',
        ('intake',),
        'ClaimsAgent',
        'Claims history from the versioned reference dataset.',
    ),
    (
        'decision',
        ('risk',),
        'EnhancedDecisionAgent',
        'Aggregate the agent outputs into one recommendation.',
    ),
    (
        'fairness',
        ('decision',),
        'FairnessAgent',
        'Fairness review signals.',
    ),
    (
        'compliance',
        ('decision',),
        'ComplianceAgent',
        'Compliance status (explicitly uncertified).',
    ),
    (
        'explanation',
        ('decision',),
        'ExplanationAgent',
        'Explanations for customer, regulator and auditor.',
    ),
)

UNDERWRITING_CORE: tuple[str, ...] = ('intake', 'risk', 'decision')

_DOCUMENT_COST_NOTE = (
    'deterministic local checks by default; the optional grounded LLM review is only '
    'called when explicitly enabled and a provider key is configured'
)


def _metadata_for(name: str, agent_class: str, description: str) -> SkillMetadata:
    return SkillMetadata(
        description=description,
        agent_class=agent_class,
        deterministic=True,
        uses_llm=False,
        model=None,
        requires_api_key=None,
        est_input_tokens=0,
        est_output_tokens=0,
        est_cost_per_call_usd=0.0,
        cost_notes=_DOCUMENT_COST_NOTE if name == 'document' else 'no provider call',
    )


def spec_skills() -> list[Skill]:
    """Build the skill objects from the specs alone (no agent import required).

    Useful for graph-level tests and for describing the search space without loading
    the XGBoost model.
    """

    def _refuse(state):  # pragma: no cover - never called in graph-only usage
        raise RuntimeError('spec skill has no handler bound')

    return [
        FunctionSkill(
            name,
            dependencies,
            _refuse,
            _metadata_for(name, agent_class, description),
        )
        for name, dependencies, agent_class, description in UNDERWRITING_SKILL_SPECS
    ]


def build_underwriting_skills(orchestrator) -> list[Skill]:
    """Bind each spec to the matching node method of a real orchestrator."""
    skills: list[Skill] = []
    for name, dependencies, agent_class, description in UNDERWRITING_SKILL_SPECS:
        handler = getattr(orchestrator, f'{name}_node', None)
        if handler is None:
            raise AttributeError(
                f'the orchestrator exposes no {name!r} node to wrap'
            )
        skills.append(
            FunctionSkill(
                name,
                dependencies,
                handler,
                _metadata_for(name, agent_class, description),
            )
        )
    return skills


def underwriting_graph() -> SkillGraph:
    """The dependency graph of the underwriting skills (spec-level)."""
    return SkillGraph(spec_skills())


def valid_underwriting_combinations() -> list[tuple[str, ...]]:
    """Every valid combination: the core plus any subset of the eight optional skills."""
    return underwriting_graph().valid_combinations(required=UNDERWRITING_CORE)


class SkillPipeline:
    """Run a chosen combination of skills over the shared state.

    The execution order is the graph's topological order, i.e. exactly the order the
    orchestrator's sequential workflow uses. A skill that is not part of the
    combination is simply skipped -- the same effect as the orchestrator's
    ``disabled_agents`` ablation, whose guarded nodes return an empty update.
    """

    def __init__(
        self,
        orchestrator=None,
        skills: Sequence[Skill] | None = None,
        graph: SkillGraph | None = None,
        build_report: bool = False,
    ):
        if skills is None:
            if orchestrator is None:
                raise ValueError('provide either an orchestrator or an explicit skill list')
            skills = build_underwriting_skills(orchestrator)
        self.orchestrator = orchestrator
        self.skills: dict[str, Skill] = {skill.name: skill for skill in skills}
        self.graph = graph or SkillGraph(list(skills))
        self.order: list[str] = self.graph.topological_order
        self.build_report = build_report

    # -- shared state -------------------------------------------------------
    @staticmethod
    def initial_state(
        application: Mapping[str, Any],
        decision_mode: str = 'human_in_the_loop',
        mode: str = 'sequential',
        **extra,
    ) -> dict[str, Any]:
        """The same initial state the LangGraph workflow builds."""
        state: dict[str, Any] = {
            'application': dict(application),
            'intake_result': {},
            'customer_result': {},
            'risk_result': {},
            'uncertainty_result': {},
            'document_result': {},
            'llm_document_result': {},
            'external_result': {},
            'claims_result': {},
            'decision_result': {},
            'fairness_result': {},
            'compliance_result': {},
            'explanation_result': {},
            'report_result': {},
            'decision_mode': decision_mode,
            'document_text': None,
            'document_name': None,
            'document_provided': None,
            'document_size_bytes': None,
            'document_status': None,
            'document_bundle': None,
            'enable_llm_document_analysis': False,
            'messages': [],
            'collaboration_mode': mode,
            'disabled_agents': (),
        }
        state.update(extra)
        return state

    # -- execution ----------------------------------------------------------
    def run(
        self,
        application: Mapping[str, Any],
        included: Sequence[str] | None = None,
        decision_mode: str = 'human_in_the_loop',
        mode: str = 'sequential',
        build_report: bool | None = None,
        counter: CostCounter | None = None,
        **extra,
    ) -> tuple[dict[str, Any], CostCounter]:
        """Run the combination and return ``(final_state, counter)``.

        A skill that raises propagates the exception after its failing call has been
        recorded, so the caller can distinguish a fail-closed refusal (invalid
        application) from a genuine error.
        """
        build_report = self.build_report if build_report is None else build_report
        counter = counter if counter is not None else CostCounter()
        state = self.initial_state(
            application, decision_mode=decision_mode, mode=mode, **extra
        )

        selected = (
            list(self.order)
            if included is None
            else [name for name in self.order if name in set(included)]
        )
        state['disabled_agents'] = tuple(
            name for name in self.order if name not in set(selected)
        )

        for name in selected:
            skill = self.skills[name]
            started = time.perf_counter()
            try:
                updates = skill.run(state)
            except Exception as error:
                counter.record(
                    name,
                    (time.perf_counter() - started) * 1000.0,
                    model=skill.metadata.model,
                    ok=False,
                    error=f'{type(error).__name__}: {error}',
                )
                raise
            latency_ms = (time.perf_counter() - started) * 1000.0
            messages = None
            if isinstance(updates, dict):
                updates = dict(updates)
                messages = updates.pop('messages', None)
            state.update(updates or {})
            if messages:
                state['messages'] = list(state.get('messages') or []) + list(messages)
            counter.record(
                name,
                latency_ms,
                input_tokens=0,
                output_tokens=0,
                model=skill.metadata.model,
            )

        # Mirror the orchestrator's post-processing: extracted text is not kept.
        state.pop('document_text', None)
        state.pop('document_bundle', None)
        if build_report:
            from orchestrator.underwriting_report import build_underwriting_report

            state['report_result'] = build_underwriting_report(state)
        return state, counter

    # -- description --------------------------------------------------------
    def describe(self) -> dict[str, Any]:
        return {
            'skills': [self.skills[name].describe() for name in self.order],
            'execution_order': list(self.order),
            'graph': self.graph.describe(),
            'core': list(UNDERWRITING_CORE),
        }
