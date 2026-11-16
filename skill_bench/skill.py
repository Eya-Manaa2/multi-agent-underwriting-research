"""The Skill interface: name, dependencies, call, and cost metadata.

A ``Skill`` is the benchmark's uniform unit of work. Each existing agent is wrapped
in exactly one skill, which lets the framework reason about *which agents can be
combined* (through the declared dependencies) and *what each call costs* (through the
declared cost metadata and the shared counter).

The interface is deliberately small:

* ``name``          -- unique identifier of the skill (agent name);
* ``dependencies``  -- names of the skills this skill hard-requires to produce a
                       usable output;
* ``run(state)``    -- the call: read the shared state, return the state updates
                       (the same protocol the LangGraph orchestrator nodes use);
* ``metadata``      -- an immutable :class:`SkillMetadata` describing the wrapped
                       class and its cost model.

No cost figure is guessed: a skill that performs no provider call declares
``uses_llm=False`` and contributes zero tokens and zero dollars, which the shared
counter verifies at run time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class SkillMetadata:
    """Static, declarative description of a skill, including its cost model.

    ``est_*`` fields are *declarations* used only for planning (they let the
    framework list the expected cost of a combination before running it). The
    benchmark never reports a declared figure as a measured one: the measured
    latency / tokens / cost always come from the shared counter at run time.
    """

    description: str = ''
    agent_class: str = ''
    deterministic: bool = True
    uses_llm: bool = False
    model: str | None = None
    requires_api_key: str | None = None
    est_input_tokens: int = 0
    est_output_tokens: int = 0
    est_cost_per_call_usd: float = 0.0
    cost_notes: str = ''


@dataclass(frozen=True)
class SkillCall:
    """One measured invocation of a skill, as recorded by the shared counter."""

    skill: str
    latency_ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = 0.0
    ok: bool = True
    error: str | None = None


class Skill:
    """Base class: a named unit of work with declared dependencies."""

    name: str = ''
    dependencies: tuple[str, ...] = ()
    metadata: SkillMetadata = SkillMetadata()

    def run(self, state: Mapping[str, Any]) -> dict[str, Any]:
        """Execute the skill against the shared state and return its updates."""
        raise NotImplementedError

    def __call__(self, state: Mapping[str, Any]) -> dict[str, Any]:
        return self.run(state)

    def cost_estimate_usd(self) -> float:
        """Declared per-call cost estimate (planning only, never reported)."""
        return self.metadata.est_cost_per_call_usd

    def describe(self) -> dict[str, Any]:
        """A JSON-serialisable description of the skill."""
        return {
            'name': self.name,
            'dependencies': list(self.dependencies),
            'metadata': asdict(self.metadata),
        }


class FunctionSkill(Skill):
    """Wrap a plain callable (typically an orchestrator node) as a Skill.

    The handler keeps the orchestrator-node protocol: it receives the shared state
    and returns the dictionary of state updates. This is what makes the benchmark
    reuse the *real* agents instead of a reimplementation.
    """

    def __init__(
        self,
        name: str,
        dependencies: tuple[str, ...],
        handler: Callable[[Mapping[str, Any]], Mapping[str, Any] | None],
        metadata: SkillMetadata | None = None,
    ):
        self.name = name
        self.dependencies = tuple(dependencies)
        self.metadata = metadata or SkillMetadata()
        self._handler = handler

    def run(self, state: Mapping[str, Any]) -> dict[str, Any]:
        updates = self._handler(state)
        return dict(updates) if updates else {}
