"""Small orchestration seam for ordered reasoning catch-up."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .reasoners import Reasoner


DEFAULT_REASONING_PIPELINE = (
    "person",
    "relationship",
    "user_model",
    "coverage",
    "learning_goals",
)


async def catch_up(
    reasoner: Reasoner, space_id: str, should_continue: Callable[[], bool]
) -> None:
    """Drive one reasoner until it reports no work left, or we are stopping.

    The loop lives here rather than in the reasoner: a reasoner owns one
    bounded attempt (load, call, commit, "call me again?"), and the caller
    owns whether to keep going at all.
    """
    while should_continue() and await reasoner.attempt(space_id):
        pass


class ReasoningPipeline:
    """Run reasoners in order, each to catch-up, in a single space."""

    def __init__(
        self,
        reasoners: Sequence[Reasoner],
        should_continue: Callable[[], bool] = lambda: True,
    ) -> None:
        self._reasoners = tuple(reasoners)
        self._should_continue = should_continue

    async def run_until_caught_up(self, space_id: str) -> None:
        for reasoner in self._reasoners:
            await catch_up(reasoner, space_id, self._should_continue)


__all__ = ["DEFAULT_REASONING_PIPELINE", "ReasoningPipeline", "catch_up"]
