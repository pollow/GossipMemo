"""Reasoner-agnostic budget-driven reduction, shared across reasoners.

Store-side backlog paging is a separate concern (`WorldStore.*_context`
callers choose how much to read); this module is the adapter-side half:
once a reasoner has decided what it wants to send, `reduce_until_fits`
gives it one shape for shrinking that request until it fits the transport's
`context_budget`, bounded and only while shrinking is provably making
progress.

`reasoners/owner.py` is the first user: an oversized owner-reasoning
prompt can only be shrunk lossily (owner cards are full snapshots, not
paginated evidence), so it digests memories round by round. A later
reasoner (goal planning's three-round candidate reduction) can reuse the
same round-loop shape without adopting owner-specific types -- the round
body is entirely the caller's `reduce_round` callback.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import TypeVar

T = TypeVar("T")


async def reduce_until_fits(
    reduce_round: Callable[[Sequence[T]], Awaitable[list[T]]],
    target_fits: Callable[[list[T]], bool],
    progress_size: Callable[[list[T]], int],
    source: Sequence[T],
    *,
    max_rounds: int = 3,
    no_progress_message: str = "reduction made no progress",
) -> list[T]:
    """Shrink `source` in rounds until `target_fits` accepts it.

    Each round calls `reduce_round(source)` to get a smaller candidate,
    checks whether the caller's real target request now fits, and
    otherwise measures `progress_size` to confirm the round actually
    shrank things before feeding the result back in as the next round's
    source. A round that doesn't shrink -- by returning something at least
    as large as before, or by exhausting `max_rounds` -- raises rather
    than looping forever or silently returning an oversized result.

    Callers are expected to have already confirmed `source` itself does
    not fit; this always performs at least one round.
    """

    previous_size: int | None = None
    current: Sequence[T] = source
    for _ in range(max_rounds):
        reduced = await reduce_round(current)
        if target_fits(reduced):
            return reduced
        size = progress_size(reduced)
        if previous_size is not None and size >= previous_size:
            raise ValueError(no_progress_message)
        previous_size, current = size, reduced
    raise ValueError(f"{no_progress_message} within {max_rounds} rounds")


__all__ = ["reduce_until_fits"]
