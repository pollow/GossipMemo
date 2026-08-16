"""Learning-goal planning reasoner."""

from __future__ import annotations

from ..llm import LlmModel
from ..queue import ReasonerCallQueue
from ..store import WorldStore
from .base import DescriptorReasoner


def build_learning_goals_reasoner(store: WorldStore, model: LlmModel, queue: ReasonerCallQueue) -> DescriptorReasoner:
    """Single pass, no retry loop."""

    def load_context(space_id: str):
        return store.learning_goal_context(space_id)

    def call(space_id: str, context):
        coverage, hypotheses, open_goals, closed_goals = context
        return (
            "plan-learning-goals",
            model.plan_learning_goals,
            (coverage, hypotheses, open_goals, closed_goals),
        )

    def apply(space_id: str, context, result) -> bool:
        coverage, _, open_goals, closed_goals = context
        store.apply_goal_planning(
            space_id, coverage.revision, result,
            {goal.id for goal in open_goals} | {goal.id for goal in closed_goals},
        )
        return True

    def continue_when(context, result, applied: bool) -> bool:
        return False

    return DescriptorReasoner("learning_goals", queue, load_context, call, apply, continue_when)


__all__ = ["build_learning_goals_reasoner"]
