"""Strict priority tiers for outbound LLM provider requests.

Kept separate from `llm.py` (which owns `ProviderGate` itself) so that
`reasoners/*` can set the active tier at each reasoner boundary without
creating a circular import with `llm.py`, which already imports reasoner
prompts.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

# Lower numbers run first; there is no aging or anti-starvation, by design
# (see llm.ProviderGate).
TIER_FOREGROUND = 1  # synthesize: the only synchronous, HTTP-response-blocking call.
TIER_FRESHNESS = 2  # extraction, continuity.
TIER_BACKGROUND = 3  # person, relationship, user_model, coverage, learning_goals.

# Set at each reasoner boundary via `llm_call_tier`; internal call sites
# (~15 of them across owner-reasoning, batching, and chunking) inherit the
# active tier through the contextvar instead of threading a parameter.
_call_tier: ContextVar[int] = ContextVar("gossipmemo_llm_call_tier", default=TIER_BACKGROUND)

# Carried alongside the tier, set at the same reasoner boundary, so
# `ProviderGate.current_label` can report a human-readable job name
# ("reason-person", "extract", ...) instead of a tier-derived placeholder.
_call_label: ContextVar[str | None] = ContextVar("gossipmemo_llm_call_label", default=None)


def current_call_tier() -> int:
    return _call_tier.get()


def current_call_label() -> str | None:
    return _call_label.get()


@contextmanager
def llm_call_tier(tier: int, label: str | None = None) -> Iterator[None]:
    """Mark every provider request issued in this scope with `tier`/`label`.

    Unset scopes (fakes, tests, call sites that never opt in) default to
    ``TIER_BACKGROUND`` and a `None` label, matching the spec's "safe
    default" requirement.
    """

    tier_token = _call_tier.set(tier)
    label_token = _call_label.set(label)
    try:
        yield
    finally:
        _call_tier.reset(tier_token)
        _call_label.reset(label_token)


__all__ = [
    "TIER_BACKGROUND",
    "TIER_FOREGROUND",
    "TIER_FRESHNESS",
    "current_call_label",
    "current_call_tier",
    "llm_call_tier",
]
