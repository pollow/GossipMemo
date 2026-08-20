"""Learning-goal planning reasoner and its prompts.

Planning reads coverage entries and nothing else: Memories are the auditor's
input, and by the time planning runs, everything the auditor understood is
already in the entries. That boundary is what makes fan-out affordable --
one request per root, carrying that root's overview entry plus its children,
is a page of prose rather than hundreds of Memories.

Fanning out is the normal path, not an overflow path. A single global pass
saw every root at once and answered with two or three directions for the
whole life; a request that can only see one root answers for that root, and
breadth comes from there being twenty of them. The passes are therefore
always two-stage: per-root candidates, then one reconciliation that is the
only lifecycle-mutating request. Each per-root pass also sees that root's
open goals and can cast a non-mutating closure recommendation against them
-- a vote grounded in the entries it actually read, not the weaker proxy of
"a candidate happened to overlap this goal". Reconciliation reads
candidates, goals, and those recommendations, so when even that does not
fit, candidates are compressed round by round through
`chunking.reduce_until_fits` -- the same shape as the owner family's
evidence digest. Recommendations are left out of that reduction and carried
through whole: they are short, and compressing votes would blur exactly the
per-root grounding that makes them worth having.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import partial

from ..chunking import greedy_chunks, reduce_until_fits
from ..models import (
    COVERAGE_CRITERIA,
    COVERAGE_ROOTS,
    CoverageEntryView,
    GoalClosureRecommendation,
    GoalPlanningCandidates,
    GoalPlanningResult,
    LearningGoalCandidate,
    LearningGoalView,
)
from ..priority import current_call_label, current_call_tier
from ..prompts import PromptLibrary
from ..prompts.render import _json
from ..store import WorldStore
from ..transport import ChatCompletionRequest, LlmTransport, structured
from .base import DescriptorReasoner
from .coverage import _structured_request
from .settings import ReasoningSettings


def _entry_lines(entries: Sequence[CoverageEntryView]) -> str:
    return "\n".join(
        f"- id={item.id!r} path={item.path!r} content={item.content!r}" for item in entries
    ) or "- (none)"


def _goal_lines(goals: Sequence[LearningGoalView]) -> str:
    return "\n".join(
        f"- id={item.id!r} status={item.status!r} prompt={item.prompt!r} "
        f"rationale={item.rationale!r}" for item in goals
    ) or "- (none)"


def goal_candidate_prompt(
    root: str, entries: Sequence[CoverageEntryView], open_goals: Sequence[LearningGoalView],
    *, prompts: PromptLibrary,
) -> str:
    """One root's entries and the four directions a candidate may expand in."""
    return (
        f"<coverage-root id={root!r} facet={COVERAGE_CRITERIA.get(root, '')!r}>\n"
        + prompts.coverage_root_viewpoints.get(root, "") + "\nAreas that stay unsaid under "
        "this root unless something invites them: "
        + prompts.coverage_root_blind_spots.get(root, "")
        + "\n</coverage-root>\n<method>\n" + prompts.coverage_method + "\n</method>\n"
        "<entries>\n" + _entry_lines(entries) + "\n</entries>\n"
        "<open-goals comparison-only=\"true\">\n" + _goal_lines(open_goals) + "\n</open-goals>\n"
        + prompts.goal_candidate_expansion_rule + " " + prompts.goal_candidate_closure_rule
    )


def _recommendation_lines(recommendations: Sequence[GoalClosureRecommendation]) -> str:
    return "\n".join(
        f"- goal_id={item.goal_id!r} reason={item.reason!r}" for item in recommendations
    ) or "- (none)"


def goal_reconciliation_prompt(
    candidates: Sequence[LearningGoalCandidate], open_goals: Sequence[LearningGoalView],
    recent_closed_goals: Sequence[LearningGoalView],
    closure_recommendations: Sequence[GoalClosureRecommendation] = (),
    *, prompts: PromptLibrary,
) -> str:
    """The one mutating pass: candidates from every root, plus goal lifecycles."""
    return (
        "<candidates>\n" + _json([item.model_dump(mode="json") for item in candidates])
        + "\n</candidates>\n<open-goals>\n" + _goal_lines(open_goals)
        + "\n</open-goals>\n<recent-closed-goals>\n" + _goal_lines(recent_closed_goals)
        + "\n</recent-closed-goals>\n<closure-recommendations>\n"
        + _recommendation_lines(closure_recommendations) + "\n</closure-recommendations>\n"
        + prompts.goal_reconciliation_merge_rule + " "
        + prompts.goal_reconciliation_lifecycle_rule
    )


def goal_candidate_reduction_prompt(
    candidates: Sequence[LearningGoalCandidate], *, prompts: PromptLibrary,
) -> str:
    return (
        prompts.goal_candidate_reduction_rule
        + "\n<candidates>\n"
        + _json([item.model_dump(mode="json") for item in candidates])
        + "\n</candidates>"
    )


async def _root_candidates(
    transport: LlmTransport, settings: ReasoningSettings, root: str,
    entries: Sequence[CoverageEntryView], open_goals: Sequence[LearningGoalView],
) -> tuple[list[LearningGoalCandidate], list[GoalClosureRecommendation]]:
    """Plan one root, splitting its child entries when they outgrow a request.

    The overview entry rides in every chunk: it is what tells the model which
    areas exist under this root, which is exactly what a sideways expansion
    needs. Closure recommendations ride alongside the candidates: this pass
    is the only one that sees this root's entries, so it is the only place
    that can judge an open goal against what is actually now understood.
    """
    context_budget = transport.context_budget
    overview = [item for item in entries if not item.path]
    children = [item for item in entries if item.path]

    def request_for(chunk: Sequence[CoverageEntryView]) -> ChatCompletionRequest:
        return _structured_request(
            transport, settings.prompts.goal_planning_system,
            goal_candidate_prompt(
                root, [*overview, *chunk], open_goals, prompts=settings.prompts),
            GoalPlanningCandidates,
        )

    def fits(chunk: Sequence[CoverageEntryView]) -> bool:
        return context_budget.report(context_budget.estimate_request(request_for(chunk))).fits

    def check(chunk: Sequence[CoverageEntryView]) -> None:
        context_budget.check(context_budget.estimate_request(request_for(chunk)))

    candidates: list[LearningGoalCandidate] = []
    recommendations: list[GoalClosureRecommendation] = []
    # A root whose entries are only the overview still gets its one request.
    for chunk in greedy_chunks(children, fits, check) or [[]]:
        request = request_for(chunk)
        _, result = await structured(
            transport, request.messages, GoalPlanningCandidates,
            tier=current_call_tier(), label=current_call_label(),
        )
        candidates.extend(result.candidates)
        recommendations.extend(result.closure_recommendations)
    return candidates, recommendations


async def _plan_learning_goals(
    transport: LlmTransport, settings: ReasoningSettings, entries: Sequence[CoverageEntryView],
    open_goals: Sequence[LearningGoalView], recent_closed_goals: Sequence[LearningGoalView],
) -> GoalPlanningResult:
    context_budget = transport.context_budget
    tier, label = current_call_tier(), current_call_label()
    candidates: list[LearningGoalCandidate] = []
    recommendations: list[GoalClosureRecommendation] = []
    for root in COVERAGE_ROOTS:
        # A root with no entries is skipped rather than planned from its
        # rubric line alone: with nothing understood there yet, anything
        # proposed would be invented from the facet name. The audit reads
        # the same evidence for every root, so such a root fills in on its
        # own as soon as there is evidence to summarize.
        root_entries = [item for item in entries if item.root == root]
        if root_entries:
            root_candidates, root_recommendations = await _root_candidates(
                transport, settings, root, root_entries, open_goals)
            candidates.extend(root_candidates)
            recommendations.extend(root_recommendations)
    if not candidates and not open_goals:
        return GoalPlanningResult()

    def reconciliation_request(items: Sequence[LearningGoalCandidate]) -> ChatCompletionRequest:
        return _structured_request(
            transport, settings.prompts.goal_planning_system,
            goal_reconciliation_prompt(
                items, open_goals, recent_closed_goals, recommendations,
                prompts=settings.prompts),
            GoalPlanningResult,
        )

    final_request = reconciliation_request(candidates)
    if not context_budget.report(context_budget.estimate_request(final_request)).fits:

        def reduction_request(items: Sequence[LearningGoalCandidate]) -> ChatCompletionRequest:
            return _structured_request(
                transport, settings.prompts.goal_planning_system,
                goal_candidate_reduction_prompt(items, prompts=settings.prompts),
                GoalPlanningCandidates,
            )

        def reduction_fits(items: Sequence[LearningGoalCandidate]) -> bool:
            return context_budget.report(
                context_budget.estimate_request(reduction_request(items))
            ).fits

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
                reduced.extend(result.candidates)
            if not reduced:
                raise ValueError("learning-goal candidate reduction made no progress")
            return reduced

        def target_fits(items: list[LearningGoalCandidate]) -> bool:
            return context_budget.report(
                context_budget.estimate_request(reconciliation_request(items))).fits

        def progress_size(items: list[LearningGoalCandidate]) -> int:
            return context_budget.estimate_text(str([item.model_dump() for item in items]))

        candidates = await reduce_until_fits(
            reduce_round, target_fits, progress_size, candidates,
            max_rounds=3, no_progress_message="learning-goal candidate reduction made no progress",
        )
        final_request = reconciliation_request(candidates)

    context_budget.check(context_budget.estimate_request(final_request))
    _, result = await structured(
        transport, final_request.messages, GoalPlanningResult, tier=tier, label=label)
    return result


def build_learning_goals_reasoner(
    store: WorldStore, model: LlmTransport, settings: ReasoningSettings
) -> DescriptorReasoner:
    """Single pass, no retry loop."""

    plan_learning_goals = partial(_plan_learning_goals, model, settings)

    def load_context(space_id: str):
        return store.learning_goal_context(space_id)

    def call(space_id: str, context):
        _, entries, open_goals, closed_goals = context
        return "plan-learning-goals", plan_learning_goals, (entries, open_goals, closed_goals)

    def apply(space_id: str, context, result) -> bool:
        revision, _, open_goals, closed_goals = context
        store.apply_goal_planning(
            space_id, revision, result,
            {goal.id for goal in open_goals} | {goal.id for goal in closed_goals},
        )
        return True

    def continue_when(context, result, applied: bool) -> bool:
        return False

    return DescriptorReasoner("learning_goals", load_context, call, apply, continue_when)


__all__ = [
    "build_learning_goals_reasoner",
    "goal_candidate_prompt",
    "goal_candidate_reduction_prompt",
    "goal_reconciliation_prompt",
]
