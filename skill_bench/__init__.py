"""agent-skill-bench: a reproducible benchmark framework for agent skills.

The framework wraps the existing underwriting agents as *skills*, declares their
dependencies as a NetworkX graph, enumerates the valid skill combinations, measures
latency / tokens / estimated cost with a shared counter, and scores every valid
combination over K seeds with bootstrap confidence intervals and a Pareto front.

Nothing in this package estimates a number by hand: every reported figure is read
back from the JSON files under ``skill_bench/results/``.

Sub-modules:
    skill        Skill interface (name, dependencies, call, cost metadata)
    counter      shared latency / token / cost counter
    graph        NetworkX dependency graph and valid-combination enumeration
    underwriting wrappers for the eleven underwriting agents
    metrics      bootstrap confidence intervals and the Pareto front
    robustness   noise / missing-field degradations
    evaluate     the exhaustive campaign and the saved artefacts
"""

__all__ = [
    'skill',
    'counter',
    'graph',
    'underwriting',
    'metrics',
    'robustness',
    'evaluate',
]

__version__ = '0.1.0'
