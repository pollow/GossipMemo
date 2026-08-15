"""Small orchestration seam for ordered reasoning catch-up."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol


class ReasoningStage(Protocol):
    name: str

    async def run_until_caught_up(self, space_id: str) -> None: ...


DEFAULT_REASONING_PIPELINE = (
    "person",
    "relationship",
    "user_model",
    "coverage",
    "learning_goals",
)
class ReasoningPipeline:
    """Run opaque stage adapters in order; stage-specific knowledge stays outside."""

    def __init__(self, stages: Sequence[ReasoningStage]) -> None:
        self._stages = tuple(stages)

    async def run_until_caught_up(self, space_id: str) -> None:
        for stage in self._stages:
            await stage.run_until_caught_up(space_id)


class FunctionStage:
    def __init__(self, name: str, runner: Callable[[str], Awaitable[None]]) -> None:
        self.name = name
        self._runner = runner

    async def run_until_caught_up(self, space_id: str) -> None:
        await self._runner(space_id)


__all__ = ["DEFAULT_REASONING_PIPELINE", "FunctionStage", "ReasoningPipeline", "ReasoningStage"]
