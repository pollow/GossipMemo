"""Learning-goal planning reasoner and its prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..models import CoverageMapView, HypothesisView, LearningGoalCandidate, LearningGoalView
from ..prompts import COVERAGE_METHOD, COVERAGE_RUBRIC, _json
from ..store import WorldStore
from .base import DescriptorReasoner

if TYPE_CHECKING:
    from ..llm import LlmModel

GOAL_PLANNING_SYSTEM_PROMPT = """Plan a very small number of optional, user-owned
learning invitations from an updated coverage map. Return only the supplied JSON
schema. Every new or changed goal must cite supplied criterion and boundary IDs.
Unknown, intimate, traumatic, stigmatized, or otherwise private areas are never
automatic targets: respect explicit readiness, consent, defer, and do-not-pursue
signals. Prefer gentle, specific questions with an easy opt-out; never pressure,
diagnose, or imply that disclosure is owed. Omission is no-op: only explicitly
transition an existing supplied goal when its lifecycle changes."""


def goal_planning_prompt(
    coverage: CoverageMapView, hypotheses: list[HypothesisView],
    open_goals: list[LearningGoalView], recent_closed_goals: list[LearningGoalView],
) -> str:
    return (
        "<coverage-rubric>\n" + COVERAGE_RUBRIC + "\n" + COVERAGE_METHOD + "\n</coverage-rubric>\n"
        "<updated-coverage-map>\n" + _json(coverage) + "\n</updated-coverage-map>\n"
        "<open-hypotheses>\n"
        + _json([item.model_dump(mode="json") for item in hypotheses])
        + "\n</open-hypotheses>\n<open-goals>\n"
        + _json([item.model_dump(mode="json") for item in open_goals])
        + "\n</open-goals>\n<recent-closed-goals>\n"
        + _json([item.model_dump(mode="json") for item in recent_closed_goals])
        + "\n</recent-closed-goals>\n"
        "Plan only after the map is caught up. Use trauma-informed partnership, choice, and an easy decline; "
        "do not equate a blind spot with a question to ask now. Goals can focus a person or relationship but remain user-owned."
    )


def goal_candidate_prompt(
    coverage: CoverageMapView, hypotheses: list[HypothesisView],
    open_goals: list[LearningGoalView], recent_closed_goals: list[LearningGoalView],
) -> str:
    return (
        "<coverage-rubric>\n" + COVERAGE_RUBRIC + "\n" + COVERAGE_METHOD + "\n</coverage-rubric>\n"
        "<updated-coverage-map>\n" + _json(coverage) + "\n</updated-coverage-map>\n"
        "<open-hypotheses>\n" + _json(hypotheses) + "\n</open-hypotheses>\n"
        "<open-goals>\n" + _json(open_goals) + "\n</open-goals>\n"
        "<recent-closed-goals>\n" + _json(recent_closed_goals) + "\n</recent-closed-goals>\n"
        "Propose optional candidate invitations only. Candidates are non-mutating: do not "
        "transition, retire, defer, update, or otherwise change any goal lifecycle. Respect "
        "consent, privacy, and easy decline; cite only supplied criterion and boundary IDs."
    )


def goal_reconciliation_prompt(
    coverage: CoverageMapView, hypotheses: list[HypothesisView],
    open_goals: list[LearningGoalView], recent_closed_goals: list[LearningGoalView],
    candidates: list[LearningGoalCandidate],
) -> str:
    return (
        goal_planning_prompt(coverage, hypotheses, open_goals, recent_closed_goals)
        + "\n<candidates>\n"
        + _json([item.model_dump(mode="json") for item in candidates])
        + "\n</candidates>\n"
        "Select or reconcile these candidates into the final result. This is the only lifecycle-mutating planning pass."
    )


def goal_candidate_reduction_prompt(candidates: list[LearningGoalCandidate]) -> str:
    return (
        "Deduplicate and compress these non-mutating learning-goal candidates. "
        "Return candidates only; never transition any lifecycle.\n<candidates>\n"
        + _json([item.model_dump(mode="json") for item in candidates])
        + "\n</candidates>"
    )


def build_learning_goals_reasoner(store: WorldStore, model: LlmModel) -> DescriptorReasoner:
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

    return DescriptorReasoner("learning_goals", load_context, call, apply, continue_when)


__all__ = [
    "GOAL_PLANNING_SYSTEM_PROMPT",
    "build_learning_goals_reasoner",
    "goal_candidate_prompt",
    "goal_candidate_reduction_prompt",
    "goal_planning_prompt",
    "goal_reconciliation_prompt",
]
