"""Shared counter: latency, input/output tokens and estimated cost per call.

Every skill call is recorded through one common counter so the three resource axes
of the benchmark (latency, tokens, cost) are measured consistently across all
combinations.

Cost estimation is layered and honest:

1. **LiteLLM** (preferred) -- when the optional ``litellm`` package is importable,
   ``litellm.cost_per_token`` prices the model's usage. This is the requested path.
2. **Provider usage + configured price table** -- when LiteLLM is unavailable, the
   input/output token counts are read directly from the raw provider reply
   (``usage_from_groq`` / ``usage_from_gemini``) and priced against an explicit
   per-million-token table supplied in the YAML configuration.
3. **Unknown** -- when neither is available for a model that *did* consume tokens,
   the cost is reported as ``None`` (unknown), never as ``0``. A skill that performs
   no provider call (``uses_llm=False``) legitimately costs ``0``.

No price is ever hard-coded: the optional table lives in the configuration file and
is quoted in the saved results so the reader can audit it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

try:  # LiteLLM is optional; its absence must never break the benchmark.
    import litellm as _litellm
except Exception:  # pragma: no cover - exercised only when litellm is missing
    _litellm = None

LITELLM_AVAILABLE = _litellm is not None

# USD per one million tokens, per model. Keys are exact model identifiers as used
# in the configuration. Values are {"input": <usd>, "output": <usd>}.
PriceTable = Mapping[str, Mapping[str, float]]


def estimate_cost_usd(
    model: str | None,
    input_tokens: int,
    output_tokens: int,
    price_table: PriceTable | None = None,
) -> float | None:
    """Estimate the USD cost of one call, or ``None`` when it cannot be priced.

    LiteLLM is tried first because it carries an up-to-date provider price map.
    When it is missing (or does not know the model) the explicit price table is
    used; without an entry the cost stays ``None`` rather than being invented.
    """
    input_tokens = int(input_tokens or 0)
    output_tokens = int(output_tokens or 0)
    if not model:
        # No model means no provider call: the call is genuinely free.
        return 0.0 if input_tokens == 0 and output_tokens == 0 else None

    if LITELLM_AVAILABLE:  # pragma: no cover - depends on the optional package
        try:
            prompt_cost, completion_cost = _litellm.cost_per_token(
                model=model,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )
            return float(prompt_cost) + float(completion_cost)
        except Exception:
            pass

    entry = (price_table or {}).get(model)
    if isinstance(entry, Mapping) and {'input', 'output'} <= set(entry):
        return (
            input_tokens / 1_000_000.0 * float(entry['input'])
            + output_tokens / 1_000_000.0 * float(entry['output'])
        )
    return None


def usage_from_groq(response: Any) -> tuple[int, int]:
    """Read ``(prompt_tokens, completion_tokens)`` from a raw Groq reply.

    Groq (OpenAI-compatible) exposes ``response.usage.prompt_tokens`` and
    ``response.usage.completion_tokens``. Missing fields read as ``0`` so a reply
    without usage metadata never fabricates a token count.
    """
    usage = getattr(response, 'usage', None)
    if usage is None:
        return 0, 0
    prompt = getattr(usage, 'prompt_tokens', None)
    completion = getattr(usage, 'completion_tokens', None)
    return int(prompt or 0), int(completion or 0)


def usage_from_gemini(response: Any) -> tuple[int, int]:
    """Read ``(prompt_token_count, candidates_token_count)`` from a Gemini reply.

    The Google GenAI SDK exposes ``response.usage_metadata`` with
    ``prompt_token_count`` and (optionally) ``candidates_token_count``.
    """
    usage = getattr(response, 'usage_metadata', None)
    if usage is None:
        return 0, 0
    prompt = getattr(usage, 'prompt_token_count', None)
    completion = getattr(usage, 'candidates_token_count', None)
    return int(prompt or 0), int(completion or 0)


def usage_from_response(provider: str, response: Any) -> tuple[int, int]:
    """Dispatch to the provider-specific usage reader (fail-soft to ``(0, 0)``)."""
    provider = (provider or '').strip().lower()
    if provider == 'groq':
        return usage_from_groq(response)
    if provider == 'gemini':
        return usage_from_gemini(response)
    return 0, 0


@dataclass
class CostCounter:
    """Accumulate per-call latency, tokens and estimated cost for one run."""

    price_table: PriceTable | None = None
    calls: list[Any] = field(default_factory=list)  # list[SkillCall]

    def record(
        self,
        skill: str,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str | None = None,
        cost_usd: float | None = None,
        ok: bool = True,
        error: str | None = None,
    ):
        """Record one call; price it when the caller did not already do so."""
        if cost_usd is None:
            cost_usd = estimate_cost_usd(
                model, input_tokens, output_tokens, self.price_table
            )
        from skill_bench.skill import SkillCall  # local import avoids a cycle

        call = SkillCall(
            skill=skill,
            latency_ms=float(latency_ms),
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            cost_usd=cost_usd,
            ok=bool(ok),
            error=error,
        )
        self.calls.append(call)
        return call

    def extend(self, other: 'CostCounter'):
        """Merge another counter's calls into this one."""
        self.calls.extend(other.calls)
        return self

    # -- aggregates ---------------------------------------------------------
    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def total_latency_ms(self) -> float:
        return float(sum(call.latency_ms for call in self.calls))

    @property
    def total_input_tokens(self) -> int:
        return int(sum(call.input_tokens for call in self.calls))

    @property
    def total_output_tokens(self) -> int:
        return int(sum(call.output_tokens for call in self.calls))

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def unknown_cost_calls(self) -> int:
        """Calls whose cost could not be priced (``cost_usd`` is ``None``)."""
        return sum(1 for call in self.calls if call.cost_usd is None)

    @property
    def total_cost_usd(self) -> float | None:
        """Total cost, or ``None`` when at least one call could not be priced."""
        if self.unknown_cost_calls:
            return None
        return float(sum(call.cost_usd or 0.0 for call in self.calls))

    def summary(self) -> dict[str, Any]:
        """A JSON-serialisable summary of this counter."""
        return {
            'calls': self.call_count,
            'latency_ms_total': self.total_latency_ms,
            'input_tokens_total': self.total_input_tokens,
            'output_tokens_total': self.total_output_tokens,
            'tokens_total': self.total_tokens,
            'cost_usd_total': self.total_cost_usd,
            'unknown_cost_calls': self.unknown_cost_calls,
        }

    def per_call(self) -> Iterable[dict[str, Any]]:
        """Yield each recorded call as a plain dictionary."""
        for call in self.calls:
            yield {
                'skill': call.skill,
                'latency_ms': call.latency_ms,
                'input_tokens': call.input_tokens,
                'output_tokens': call.output_tokens,
                'cost_usd': call.cost_usd,
                'ok': call.ok,
                'error': call.error,
            }
