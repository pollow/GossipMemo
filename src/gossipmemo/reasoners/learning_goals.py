"""Learning-goal planning reasoner and its prompts.

Like coverage, learning-goal planning is an accumulator over hypotheses and
existing goals, so candidate generation paginates through
`chunking.greedy_chunks` the same way `reasoners.coverage` does. The final
reconciliation step is different: it must mutate goal lifecycles from a
*single* candidate list, so an oversized reconciliation request can only be
shrunk by compressing candidates round by round -- the same shape as the
owner family's evidence digest, expressed through the same
`chunking.reduce_until_fits` helper rather than a second hand-rolled
round loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

from ..chunking import greedy_chunks, reduce_until_fits
from ..models import (
    CoverageMapView,
    GoalPlanningCandidates,
    GoalPlanningResult,
    HypothesisView,
    LearningGoalCandidate,
    LearningGoalView,
)
from ..priority import current_call_label, current_call_tier
from ..prompts import COVERAGE_METHOD, COVERAGE_RUBRIC, _json
from ..store import WorldStore
from ..transport import ChatCompletionRequest, LlmTransport, structured
from .base import DescriptorReasoner
from .coverage import _coverage_skeleton, _structured_request

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


def _bounded_goal_context(
    transport: LlmTransport, coverage: CoverageMapView, hypotheses: Sequence[HypothesisView],
    open_goals: Sequence[LearningGoalView], closed_goals: Sequence[LearningGoalView],
) -> tuple[CoverageMapView, list[HypothesisView], list[LearningGoalView], list[LearningGoalView]]:
    # Goal planning has different schemas and prompt scaffolding than a
    # coverage audit, so its minimum-fit proof must use its own requests.
    context_budget = transport.context_budget

    def fits(
        candidate_coverage: CoverageMapView, candidate_hypotheses: Sequence[HypothesisView],
        candidate_open: Sequence[LearningGoalView], candidate_closed: Sequence[LearningGoalView],
    ) -> bool:
        candidate = _structured_request(
            transport, GOAL_PLANNING_SYSTEM_PROMPT,
            goal_candidate_prompt(
                candidate_coverage, list(candidate_hypotheses), list(
                    candidate_open), list(candidate_closed),
            ),
            GoalPlanningCandidates,
        )
        final = _structured_request(
            transport, GOAL_PLANNING_SYSTEM_PROMPT,
            goal_reconciliation_prompt(
                candidate_coverage, list(candidate_hypotheses), list(
                    candidate_open), list(candidate_closed), [],
            ),
            GoalPlanningResult,
        )
        return all(
            context_budget.report(context_budget.estimate_request(request)).fits
            for request in (candidate, final)
        )

    skeleton_hypotheses = [item.model_copy(
        update={"content": "", "evidence": []}) for item in hypotheses]
    skeleton_open = [goal.model_copy(update={"prompt": "", "rationale": ""}) for goal in open_goals]
    skeleton_closed = [goal.model_copy(
        update={"prompt": "", "rationale": ""}) for goal in closed_goals]
    if fits(coverage, skeleton_hypotheses, skeleton_open, skeleton_closed):
        return coverage, skeleton_hypotheses, skeleton_open, skeleton_closed

    bounded_coverage = _coverage_skeleton(coverage)
    if not fits(bounded_coverage, [], [], []):
        raise ValueError("learning-goal minimum skeleton exceeds context budget")

    bounded_hypotheses: list[HypothesisView] = []
    bounded_open: list[LearningGoalView] = []
    bounded_closed: list[LearningGoalView] = []
    # Lifecycle IDs have priority; omitted comparison state is always no-op.
    for item in skeleton_open:
        if fits(bounded_coverage, bounded_hypotheses, [*bounded_open, item], bounded_closed):
            bounded_open.append(item)
    for item in skeleton_hypotheses:
        if fits(bounded_coverage, [*bounded_hypotheses, item], bounded_open, bounded_closed):
            bounded_hypotheses.append(item)
    for item in skeleton_closed:
        if fits(bounded_coverage, bounded_hypotheses, bounded_open, [*bounded_closed, item]):
            bounded_closed.append(item)
    return bounded_coverage, bounded_hypotheses, bounded_open, bounded_closed


def _filter_goal_candidates(
    candidates: Sequence[LearningGoalCandidate], coverage: CoverageMapView, hypotheses: Sequence[HypothesisView],
) -> list[LearningGoalCandidate]:
    criteria = set(coverage.criteria)
    boundaries = {item.id for item in coverage.boundaries if item.status == "open"}
    # Persistence remains the authority for focus Person/Relationship IDs.
    accepted: list[LearningGoalCandidate] = []
    for item in candidates:
        if not set(item.criteria_refs).issubset(criteria) or not set(item.boundary_ids).issubset(boundaries):
            continue
        accepted.append(item)
    return accepted


async def _plan_learning_goals(
    transport: LlmTransport, coverage: CoverageMapView, hypotheses: Sequence[HypothesisView],
    open_goals: Sequence[LearningGoalView], recent_closed_goals: Sequence[LearningGoalView],
) -> GoalPlanningResult:
    context_budget = transport.context_budget
    tier, label = current_call_tier(), current_call_label()
    normal = _structured_request(
        transport, GOAL_PLANNING_SYSTEM_PROMPT,
        goal_planning_prompt(coverage, list(hypotheses), list(
            open_goals), list(recent_closed_goals)),
        GoalPlanningResult,
    )
    if context_budget.report(context_budget.estimate_request(normal)).fits:
        _, result = await structured(transport, normal.messages, GoalPlanningResult, tier=tier, label=label)
        return result

    bounded_coverage, bounded_hypotheses, bounded_open, bounded_closed = _bounded_goal_context(
        transport, coverage, hypotheses, open_goals, recent_closed_goals,
    )

    def candidate_request(chunk: Sequence[HypothesisView]) -> ChatCompletionRequest:
        return _structured_request(
            transport, GOAL_PLANNING_SYSTEM_PROMPT,
            goal_candidate_prompt(bounded_coverage, list(chunk), bounded_open, bounded_closed),
            GoalPlanningCandidates,
        )

    def candidate_fits(items: Sequence[HypothesisView | None]) -> bool:
        chunk = [item for item in items if item is not None]
        return context_budget.report(context_budget.estimate_request(candidate_request(chunk))).fits

    def candidate_check(items: Sequence[HypothesisView | None]) -> None:
        chunk = [item for item in items if item is not None]
        context_budget.check(context_budget.estimate_request(candidate_request(chunk)))

    candidates: list[LearningGoalCandidate] = []
    # Candidate passes cover every original hypothesis; only the final
    # reconciliation uses the complete ID-preserving bounded map. A
    # hypothesis-free planning still runs one candidate pass, driven by the
    # sentinel `[None]` source below.
    for chunk in greedy_chunks(list(hypotheses) or [None], candidate_fits, candidate_check):
        request = candidate_request([item for item in chunk if item is not None])
        _, result = await structured(transport, request.messages, GoalPlanningCandidates, tier=tier, label=label)
        candidates.extend(_filter_goal_candidates(result.candidates, coverage, hypotheses))

    def reconciliation_request(items: Sequence[LearningGoalCandidate]) -> ChatCompletionRequest:
        return _structured_request(
            transport, GOAL_PLANNING_SYSTEM_PROMPT,
            goal_reconciliation_prompt(
                bounded_coverage, bounded_hypotheses, bounded_open, bounded_closed, list(items),
            ),
            GoalPlanningResult,
        )

    final_request = reconciliation_request(candidates)
    if not context_budget.report(context_budget.estimate_request(final_request)).fits:

        def reduction_request(items: Sequence[LearningGoalCandidate]) -> ChatCompletionRequest:
            return _structured_request(
                transport, GOAL_PLANNING_SYSTEM_PROMPT,
                goal_candidate_reduction_prompt(list(items)), GoalPlanningCandidates,
            )

        def reduction_fits(items: Sequence[LearningGoalCandidate]) -> bool:
            return context_budget.report(context_budget.estimate_request(reduction_request(items))).fits

        def reduction_check(items: Sequence[LearningGoalCandidate]) -> None:
            context_budget.check(context_budget.estimate_request(reduction_request(items)))

        async def reduce_round(
            source: Sequence[LearningGoalCandidate],
        ) -> list[LearningGoalCandidate]:
            reduced: list[LearningGoalCandidate] = []
            for chunk in greedy_chunks(list(source), reduction_fits, reduction_check):
                request = reduction_request(chunk)
                _, result = await structured(
                    transport, request.messages, GoalPlanningCandidates, tier=tier, label=label,
                )
                reduced.extend(_filter_goal_candidates(result.candidates, coverage, hypotheses))
            if not reduced:
                raise ValueError("learning-goal candidate reduction made no progress")
            return reduced

        def target_fits(candidates: list[LearningGoalCandidate]) -> bool:
            request = reconciliation_request(candidates)
            return context_budget.report(context_budget.estimate_request(request)).fits

        def progress_size(candidates: list[LearningGoalCandidate]) -> int:
            return context_budget.estimate_text(str([item.model_dump() for item in candidates]))

        candidates = await reduce_until_fits(
            reduce_round, target_fits, progress_size, candidates,
            max_rounds=3, no_progress_message="learning-goal candidate reduction made no progress",
        )
        final_request = reconciliation_request(candidates)

    context_budget.check(context_budget.estimate_request(final_request))
    _, result = await structured(transport, final_request.messages, GoalPlanningResult, tier=tier, label=label)
    return result


def build_learning_goals_reasoner(store: WorldStore, model: LlmTransport) -> DescriptorReasoner:
    """Single pass, no retry loop."""

    plan_learning_goals = partial(_plan_learning_goals, model)

    def load_context(space_id: str):
        return store.learning_goal_context(space_id)

    def call(space_id: str, context):
        coverage, hypotheses, open_goals, closed_goals = context
        return (
            "plan-learning-goals",
            plan_learning_goals,
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
