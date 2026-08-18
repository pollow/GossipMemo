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
from ..prompts import (
    COVERAGE_METHOD,
    COVERAGE_ROOT_BLIND_SPOTS,
    COVERAGE_ROOT_VIEWPOINTS,
    _json,
)
from ..store import WorldStore
from ..transport import ChatCompletionRequest, LlmTransport, structured
from .base import DescriptorReasoner
from .coverage import _structured_request

GOAL_PLANNING_SYSTEM_PROMPT = """Plan optional directions in which this user's memoir
and persona could be understood better, reading summaries of what is already
understood. Return only the supplied JSON schema. A direction is natural language: what
it is, why it is worth understanding, and one suggested wording. Private, intimate,
painful, and stigmatized areas are not off limits -- an unlit part of a life is still
part of it, and recording a direction is not asking about it. Whether to raise one now,
how to word it, and how to keep that exchange safe is the consuming agent's decision in
the moment, not this planner's. A direction may be about a friend, but it belongs to the
user's own life, relationships, or memoir: never a standalone information-gathering task
about a third party, and never a suggestion that the user test, probe, or secretly
verify anyone. Do not diagnose, and do not assume that reconciliation, disclosure, or
repair is the right ending of any thread. Omission is no-op: only explicitly transition
an existing supplied goal when its lifecycle changes."""


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
) -> str:
    """One root's entries and the four directions a candidate may expand in."""
    return (
        f"<coverage-root id={root!r} facet={COVERAGE_CRITERIA.get(root, '')!r}>\n"
        + COVERAGE_ROOT_VIEWPOINTS.get(root, "") + "\nAreas that stay unsaid under this "
        "root unless something invites them: " + COVERAGE_ROOT_BLIND_SPOTS.get(root, "")
        + "\n</coverage-root>\n<method>\n" + COVERAGE_METHOD + "\n</method>\n"
        "<entries>\n" + _entry_lines(entries) + "\n</entries>\n"
        "<open-goals comparison-only=\"true\">\n" + _goal_lines(open_goals) + "\n</open-goals>\n"
        "These entries are everything that is understood about this root; the entry with "
        "an empty path is its overview. Propose optional candidate directions only, at "
        "most three, and none at all when this root has nothing worth opening. Expand in "
        "four ways: deeper into one entry, at something it states but never unfolds; "
        "sideways to a neighbouring or missing sibling, a stage or place or period the "
        "entries step over; forward in time along a thread the entries already contain, "
        "at what became of it -- a concrete hook like that usually yields the best "
        "direction; and along a person an entry names, where the direction is that "
        "person's part in the user's own life. Write each direction in the "
        "language of the entries. Cite in `entry_ids` the entries a direction grew out of when you "
        "can, leave it empty rather than guessing, and never withhold a direction for "
        "having nothing to cite. Do not repeat a direction an open goal already covers. "
        "Candidates are non-mutating: do not transition, retire, defer, update, or "
        "otherwise change any goal lifecycle. Separately, look over the open goals above "
        "against what these entries now show: when one now reads as answered, overtaken, "
        "or no longer worth holding open, add a closure recommendation citing its "
        "`goal_id` and a short reason. This is a vote for a later pass to weigh, not a "
        "transition -- do not remove or alter the goal here, and skip any goal this "
        "root's entries say nothing new about."
    )


def _recommendation_lines(recommendations: Sequence[GoalClosureRecommendation]) -> str:
    return "\n".join(
        f"- goal_id={item.goal_id!r} reason={item.reason!r}" for item in recommendations
    ) or "- (none)"


def goal_reconciliation_prompt(
    candidates: Sequence[LearningGoalCandidate], open_goals: Sequence[LearningGoalView],
    recent_closed_goals: Sequence[LearningGoalView],
    closure_recommendations: Sequence[GoalClosureRecommendation] = (),
) -> str:
    """The one mutating pass: candidates from every root, plus goal lifecycles."""
    return (
        "<candidates>\n" + _json([item.model_dump(mode="json") for item in candidates])
        + "\n</candidates>\n<open-goals>\n" + _goal_lines(open_goals)
        + "\n</open-goals>\n<recent-closed-goals>\n" + _goal_lines(recent_closed_goals)
        + "\n</recent-closed-goals>\n<closure-recommendations>\n"
        + _recommendation_lines(closure_recommendations) + "\n</closure-recommendations>\n"
        "These candidates come from separate per-root passes, so near-duplicates across "
        "roots are expected: merge them, and keep the ones that read as an invitation "
        "into this user's own life rather than a survey question. Keep breadth -- several "
        "directions on one subject are worth less than the same number spread across "
        "different parts of the life. Reuse an existing `goal_id` to rewrite that goal, "
        "and omit it to create a new one. This is the only pass that may transition a "
        "goal's lifecycle: transition one when the evidence shows it is answered, "
        "overtaken, or no longer worth holding open. The closure recommendations are each "
        "one root's vote grounded in what its entries actually show, not an instruction: "
        "weigh a recommendation as evidence, and a goal recommended closed by one root can "
        "still be worth holding open if the rest of its scope is unanswered."
    )


def goal_candidate_reduction_prompt(candidates: Sequence[LearningGoalCandidate]) -> str:
    return (
        "Deduplicate and compress these non-mutating learning-goal candidates, keeping "
        "their breadth across different parts of the life. Return candidates only; never "
        "transition any lifecycle.\n<candidates>\n"
        + _json([item.model_dump(mode="json") for item in candidates])
        + "\n</candidates>"
    )


async def _root_candidates(
    transport: LlmTransport, root: str, entries: Sequence[CoverageEntryView],
    open_goals: Sequence[LearningGoalView],
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
            transport, GOAL_PLANNING_SYSTEM_PROMPT,
            goal_candidate_prompt(root, [*overview, *chunk], open_goals),
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
    transport: LlmTransport, entries: Sequence[CoverageEntryView],
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
                transport, root, root_entries, open_goals)
            candidates.extend(root_candidates)
            recommendations.extend(root_recommendations)
    if not candidates and not open_goals:
        return GoalPlanningResult()

    def reconciliation_request(items: Sequence[LearningGoalCandidate]) -> ChatCompletionRequest:
        return _structured_request(
            transport, GOAL_PLANNING_SYSTEM_PROMPT,
            goal_reconciliation_prompt(items, open_goals, recent_closed_goals, recommendations),
            GoalPlanningResult,
        )

    final_request = reconciliation_request(candidates)
    if not context_budget.report(context_budget.estimate_request(final_request)).fits:

        def reduction_request(items: Sequence[LearningGoalCandidate]) -> ChatCompletionRequest:
            return _structured_request(
                transport, GOAL_PLANNING_SYSTEM_PROMPT,
                goal_candidate_reduction_prompt(items), GoalPlanningCandidates,
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
    _, result = await structured(transport, final_request.messages, GoalPlanningResult, tier=tier, label=label)
    return result


def build_learning_goals_reasoner(store: WorldStore, model: LlmTransport) -> DescriptorReasoner:
    """Single pass, no retry loop."""

    plan_learning_goals = partial(_plan_learning_goals, model)

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
    "GOAL_PLANNING_SYSTEM_PROMPT",
    "build_learning_goals_reasoner",
    "goal_candidate_prompt",
    "goal_candidate_reduction_prompt",
    "goal_reconciliation_prompt",
]
