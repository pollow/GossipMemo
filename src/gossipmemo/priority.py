"""Strict priority tiers for outbound LLM provider requests.

Kept separate from `llm.py` (which owns `ProviderGate` itself) so that
`reasoners/*` can set the active tier at each reasoner boundary without
creating a circular import with `llm.py`, which already imports reasoner
prompts.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator

# Lower numbers run first; there is no aging or anti-starvation, by design
# (see llm.ProviderGate).
TIER_FOREGROUND = 1  # synthesize: the only synchronous, HTTP-response-blocking call.
TIER_FRESHNESS = 2  # extraction, continuity.
TIER_BACKGROUND = 3  # person, relationship, user_model, coverage, learning_goals.

# Set at each reasoner boundary via `llm_call_tier`; internal call sites
# (~15 of them across owner-reasoning, chunking, and digesting) inherit the
# active tier through the contextvar instead of threading a parameter.
_call_tier: ContextVar[int] = ContextVar("gossipmemo_llm_call_tier", default=TIER_BACKGROUND)


def current_call_tier() -> int:
    return _call_tier.get()


@contextmanager
def llm_call_tier(tier: int) -> Iterator[None]:
    """Mark every provider request issued in this scope with `tier`.

    Unset scopes (fakes, tests, call sites that never opt in) default to
    ``TIER_BACKGROUND``, matching the spec's "safe default" requirement.
    """

    token = _call_tier.set(tier)
    try:
        yield
    finally:
        _call_tier.reset(token)


__all__ = [
    "TIER_BACKGROUND",
    "TIER_FOREGROUND",
    "TIER_FRESHNESS",
    "current_call_tier",
    "llm_call_tier",
]
