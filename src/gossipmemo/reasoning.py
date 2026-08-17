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
            await reasoner.run_until_caught_up(space_id, self._should_continue)


__all__ = ["DEFAULT_REASONING_PIPELINE", "ReasoningPipeline"]
