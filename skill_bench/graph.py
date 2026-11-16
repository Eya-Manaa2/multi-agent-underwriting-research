"""NetworkX dependency graph of the skills and enumeration of valid combinations.

Each skill declares the skills it hard-requires (``dependencies``). The graph is a
directed acyclic graph whose edge ``dependency -> skill`` means "the dependency must
be present for the skill to produce a usable output". A *combination* is valid when

1. it contains the mandatory core skills (the pipeline cannot produce a
   recommendation without them), and
2. it is closed under the declared dependencies: if a skill is included, every skill
   it hard-requires is included too.

Enumerating the closed subsets of the optional skills gives the search space the
benchmark explores. Any combination that is not a closed superset of the core is
rejected here, before any evaluation runs.
"""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence

import networkx as nx

from skill_bench.skill import Skill


class SkillGraph:
    """A directed dependency graph over a fixed set of skills."""

    def __init__(self, skills: Sequence[Skill]):
        if not skills:
            raise ValueError('a skill graph needs at least one skill')
        self.skills: dict[str, Skill] = {}
        for skill in skills:
            if skill.name in self.skills:
                raise ValueError(f'duplicate skill name: {skill.name}')
            self.skills[skill.name] = skill

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.skills)
        for skill in skills:
            for dependency in skill.dependencies:
                if dependency not in self.skills:
                    raise ValueError(
                        f'skill {skill.name!r} depends on unknown skill '
                        f'{dependency!r}'
                    )
                # Edge points from the dependency to the dependent skill.
                self.graph.add_edge(dependency, skill.name)

        if not nx.is_directed_acyclic_graph(self.graph):
            cycle = nx.find_cycle(self.graph)
            raise ValueError(f'the dependency graph contains a cycle: {cycle}')

    # -- topology -----------------------------------------------------------
    @property
    def topological_order(self) -> list[str]:
        """Nodes in dependency order (dependencies first)."""
        return list(nx.topological_sort(self.graph))

    @property
    def edges(self) -> list[tuple[str, str]]:
        return list(self.graph.edges)

    def dependencies_of(self, name: str) -> tuple[str, ...]:
        return tuple(self.skills[name].dependencies)

    # -- validity -----------------------------------------------------------
    def validate(self, combination: Iterable[str]) -> tuple[bool, str | None]:
        """Return ``(True, None)`` for a valid combination, else ``(False, reason)``."""
        names = set(combination)
        unknown = names - set(self.skills)
        if unknown:
            return False, f'unknown skill(s): {sorted(unknown)}'

        missing: dict[str, list[str]] = {}
        for name in names:
            absent = set(self.dependencies_of(name)) - names
            if absent:
                missing[name] = sorted(absent)
        if missing:
            return False, f'dependency closure violated: {missing}'
        return True, None

    def validate_with_core(
        self, combination: Iterable[str], required: Iterable[str]
    ) -> tuple[bool, str | None]:
        """Validate a combination that must also contain the mandatory core."""
        names = set(combination)
        absent_core = set(required) - names
        if absent_core:
            return False, f'missing core skill(s): {sorted(absent_core)}'
        return self.validate(names)

    def valid_combinations(self, required: Iterable[str] = ()) -> list[tuple[str, ...]]:
        """Enumerate every valid combination as a sorted tuple of skill names.

        The returned list is deterministic: combinations are ordered by size then
        alphabetically, so two runs of the benchmark compare the same rows.
        """
        required = tuple(required)
        unknown_required = set(required) - set(self.skills)
        if unknown_required:
            raise ValueError(f'unknown core skill(s): {sorted(unknown_required)}')

        optional = [name for name in self.skills if name not in set(required)]
        combinations: list[tuple[str, ...]] = []
        for size in range(len(optional) + 1):
            for subset in itertools.combinations(sorted(optional), size):
                candidate = tuple(sorted(set(required) | set(subset)))
                ok, _ = self.validate(candidate)
                if ok:
                    combinations.append(candidate)
        return combinations

    # -- description --------------------------------------------------------
    def describe(self) -> dict:
        """A JSON-serialisable description of the graph."""
        return {
            'nodes': [
                {
                    'name': name,
                    'dependencies': list(self.dependencies_of(name)),
                }
                for name in self.topological_order
            ],
            'edges': self.edges,
            'skill_count': len(self.skills),
        }
