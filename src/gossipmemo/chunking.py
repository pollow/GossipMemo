"""Reasoner-agnostic budget-driven reduction, shared across reasoners.

Store-side backlog paging is a separate concern (`WorldStore.*_context`
callers choose how much to read); this module is the adapter-side half:
once a reasoner has decided what it wants to send, `reduce_until_fits`
gives it one shape for shrinking that request until it fits the transport's
`context_budget`, bounded and only while shrinking is provably making
progress. `greedy_chunks` is the paginating sibling: evidence that CAN be
split across several requests (coverage and goal planning are
accumulators, unlike owner's snapshot cards) is packed request-by-request
until it no longer fits, splitting a single oversized item by bisecting
its `.content` when even one item alone would not fit.

Neither helper knows about `ContextBudget` or `ChatCompletionRequest`:
callers close over their own transport and pass plain predicates
(`fits`, `check`), which keeps this module a leaf usable from any
reasoner.

`reasoners/owner.py` is the first user of `reduce_until_fits`: an oversized
owner-reasoning prompt can only be shrunk lossily (owner cards are full
snapshots, not paginated evidence), so it digests memories round by round.
`reasoners/learning_goals.py` reuses the same round-loop shape for its
candidate-reconciliation reduction without adopting owner-specific types --
the round body is entirely the caller's `reduce_round` callback.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

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


def greedy_chunks(
    items: Sequence[T],
    fits: Callable[[Sequence[T]], bool],
    check: Callable[[Sequence[T]], None],
) -> list[list[T]]:
    """Pack `items` into as few `fits`-sized chunks as possible, in order.

    A single item that does not fit on its own is split by bisecting its
    `.content` string attribute (present on `MemoryView`/`HypothesisView`
    and copied via `.model_copy`) into the largest pieces that do. `check`
    is called on every newly-grown `current` chunk so a caller's own
    (typically more informative) budget-exceeded error is what surfaces,
    not a generic one from this module.
    """

    chunks: list[list[T]] = []
    current: list[T] = []
    for item in items:
        for piece in _split_oversized(item, fits):
            candidate = [*current, piece]
            if current and not fits(candidate):
                chunks.append(current)
                current = []
            current.append(piece)
            check(current)
    if current:
        chunks.append(current)
    return chunks


def _split_oversized(item: T, fits: Callable[[Sequence[T]], bool]) -> list[T]:
    if fits([item]):
        return [item]
    content: Any = getattr(item, "content", None)
    if not isinstance(content, str) or not content:
        raise ValueError("single context item exceeds input budget")
    lo, hi = 1, len(content)
    while lo < hi:
        middle = (lo + hi + 1) // 2
        candidate = item.model_copy(update={"content": content[:middle]})
        if fits([candidate]):
            lo = middle
        else:
            hi = middle - 1
    if lo < 1:
        raise ValueError("single context item exceeds input budget")
    return [
        item.model_copy(update={"content": content[index:index + lo]})
        for index in range(0, len(content), lo)
    ]


__all__ = ["greedy_chunks", "reduce_until_fits"]
