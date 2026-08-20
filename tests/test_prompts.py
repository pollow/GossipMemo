from gossipmemo.models import GoalClosureRecommendation, MemoryView, ModelMessage
from gossipmemo.query import QUERY_SYNTHESIS_SYSTEM_PROMPT
from gossipmemo.reasoners import (
    CONTINUITY_SYSTEM_PROMPT,
    COVERAGE_AUDIT_SYSTEM_PROMPT,
    EXTRACTION_SYSTEM_PROMPT,
    GOAL_PLANNING_SYSTEM_PROMPT,
    PERSON_REASONING_SYSTEM_PROMPT,
    RELATIONSHIP_REASONING_SYSTEM_PROMPT,
    USER_MODEL_REASONING_SYSTEM_PROMPT,
    coverage_audit_prompt,
    extraction_prompt,
    goal_candidate_prompt,
    goal_reconciliation_prompt,
)


def message(
    message_id: str, *, author: str = "user", content: str = "I had coffee today."
) -> ModelMessage:
    return ModelMessage(
        id=message_id,
        space_id="space",
        author=author,
        content=content,
        occurred_at="2026-08-14T12:00:00+00:00",
        source_provider="test",
    )


def test_extraction_prompt_applies_batch_policy_and_dates():
    prompt = extraction_prompt(
        [
            message("m-conservative"),
            message("m-comprehensive"),
        ], "comprehensive"
    )

    assert "server's comprehensive extraction policy for the whole batch" in prompt
    assert "occurred_at" in EXTRACTION_SYSTEM_PROMPT
    assert "never use an inferred basis for extraction" in " ".join(
        EXTRACTION_SYSTEM_PROMPT.split()
    )
    assert "dominant language of current user evidence" in EXTRACTION_SYSTEM_PROMPT
    assert "including new display names" in EXTRACTION_SYSTEM_PROMPT


def test_extraction_prompt_separates_recent_context_from_new_evidence():
    comparison = MemoryView(
        id="memory-1",
        content="Deus likes tea.",
        kind="fact",
        basis="stated",
        status="active",
        about_user=True,
        created_at="2026-08-13T12:00:00+00:00",
    )
    prompt = extraction_prompt(
        [message("new")],
        context=[message("old")],
        known_people=[
            {
                "id": "person_1",
                "display_name": "Alex Wang",
                "aliases": ["Alex Wang"],
            }
        ],
        comparison_memories=[comparison],
    )
    assert "Recent context (context only)" in prompt
    assert "Alex Wang" in prompt
    assert "canonical display_name" in prompt
    assert "Omit a known person unless" in prompt
    assert "Do not echo the known-people list" in prompt
    assert "Current batch evidence (user-authored; the only messages allowed" in prompt
    assert "Comparison memories (deduplication/update reference only" in prompt
    assert "supersedes_memory_id" in prompt
    assert "never new evidence" in prompt


def test_extraction_prompt_routes_assistant_context_and_canonical_user_name():
    prompt = extraction_prompt(
        [
            message("hypothesis", author="assistant", content="You avoid conflict."),
            message("confirmation", content="Yes, especially at work."),
            message("advice", author="assistant", content="Try a direct conversation."),
        ],
        user_name="Deus",
    )

    assert "The fixed current user is named 'Deus'" in prompt
    assert "Assistant content may supply a proposition only when" in prompt
    assert "Current batch evidence (user-authored" in prompt
    assert "confirmation" in prompt
    assert "Current batch context (assistant-authored; context only)" in prompt
    assert "hypothesis" in prompt and "advice" in prompt
    assert "never save their restatements, summaries, analyses, or advice" in prompt


def test_extraction_prompt_requires_stable_specific_person_identity():
    prompt = extraction_prompt([message("identity")], user_name="Deus")

    assert "one concrete human whose identity is distinguishable" in prompt
    assert "evidence-supported durable identity anchor" in prompt
    assert "uniquely and temporally determined" in prompt
    assert "grammatical anaphor" in prompt
    assert "unbounded group or category" in prompt
    assert "most stable, specific, neutral canonical label" in prompt
    assert "Do not create separate identities" in prompt
    assert "synonymous wording" in prompt
    assert "leave its `people` refs empty" in prompt
    assert "identity uncertainty alone is not a reason to discard" in prompt


def test_reasoning_prompts_allow_useful_social_inference():
    assert "patterns" in PERSON_REASONING_SYSTEM_PROMPT
    assert "social inferences" in PERSON_REASONING_SYSTEM_PROMPT
    assert "friction" in RELATIONSHIP_REASONING_SYSTEM_PROMPT
    assert "Generalize recurring patterns" in USER_MODEL_REASONING_SYSTEM_PROMPT


def test_owner_prompts_keep_relevance_apart_from_subject_and_relationship():
    """Linked-but-not-about is the failure these two prompts exist to prevent.

    These assertions used to sit on `person_reasoning_prompt` and
    `relationship_reasoning_prompt`, user-prompt builders that went dead
    when owner reasoning moved to `reasoners/owner.py` and were deleted
    with them. The instructions themselves still ship, in the system
    prompts, so they are checked there now rather than dropped.
    """

    assert "Linked memories indicate" in PERSON_REASONING_SYSTEM_PROMPT
    assert "semantic subject" in PERSON_REASONING_SYSTEM_PROMPT
    assert "Do not transfer" in PERSON_REASONING_SYSTEM_PROMPT
    assert "endpoint People" in RELATIONSHIP_REASONING_SYSTEM_PROMPT
    assert (
        "Mere co-occurrence is not relationship evidence"
        in RELATIONSHIP_REASONING_SYSTEM_PROMPT
    )


def test_owner_system_policies_do_not_prescribe_a_stage_output_shape():
    for prompt in (
        PERSON_REASONING_SYSTEM_PROMPT,
        RELATIONSHIP_REASONING_SYSTEM_PROMPT,
        USER_MODEL_REASONING_SYSTEM_PROMPT,
    ):
        assert "Return only" not in prompt
        assert "JSON schema" not in prompt
        assert "optional inferred memory" not in prompt


def test_all_prompt_contracts_keep_i18n_rule_compact():
    for prompt in (
        EXTRACTION_SYSTEM_PROMPT,
        PERSON_REASONING_SYSTEM_PROMPT,
        RELATIONSHIP_REASONING_SYSTEM_PROMPT,
        USER_MODEL_REASONING_SYSTEM_PROMPT,
        CONTINUITY_SYSTEM_PROMPT,
    ):
        assert "language" in prompt and "best matches" in prompt
    assert "language of the question" in QUERY_SYNTHESIS_SYSTEM_PROMPT


def test_coverage_audit_prompt_summarizes_one_root_without_hunting_gaps():
    prompt = coverage_audit_prompt("M4", [], [])
    # The audited root arrives through the call structure, with its own
    # viewpoint line; naming what is missing belongs to goal planning, so the
    # rubric's blind-spot cues stay out of this prompt.
    assert "M4" in prompt and "people_and_relationship_arcs" in prompt
    assert "Which attachments, ruptures" in prompt
    assert "blind spot" not in prompt
    assert "empty path" in prompt and "superseded" in prompt
    assert "only one memory supports" in prompt
    assert "never write what is\nmissing" in COVERAGE_AUDIT_SYSTEM_PROMPT


def test_goal_candidate_prompt_carries_one_root_and_four_expansions():
    prompt = goal_candidate_prompt("M4", [], [])
    assert "M4" in prompt and "people_and_relationship_arcs" in prompt
    # The rubric's blind-spot cues moved here from the audit: they are what
    # a sideways expansion needs, and the auditor no longer hunts gaps.
    assert "estrangement, reconciliation" in prompt
    assert "deeper into one entry" in prompt and "sideways to a neighbouring" in prompt
    assert "forward in time along a thread" in prompt and "a person an entry names" in prompt
    assert "never withhold a direction for having nothing to cite" in " ".join(prompt.split())


def test_goal_candidate_prompt_votes_closure_without_mutating():
    """Candidates see this root's entries, so they can judge staleness --

    but the prompt must still say a recommendation is a vote, not a
    transition, since reconciliation remains the only mutating pass.
    """
    prompt = " ".join(goal_candidate_prompt("M4", [], []).split())
    assert "closure recommendation" in prompt
    assert "vote for a later pass to weigh, not a transition" in prompt
    assert "do not remove or alter the goal here" in prompt
    assert "Candidates are non-mutating: do not transition, retire, defer, update" in prompt


def test_goal_planning_leaves_asking_now_to_the_consuming_agent():
    """The planner records directions; the agent decides what to raise.

    The old prompt called private areas "never automatic targets" and told
    the planner not to equate a blind spot with a question to ask now --
    that is the consuming agent's judgement, and taking it here is what
    censored the plan down to nothing. What stays is the standing limit:
    no diagnosis, no assumed happy ending, no third-party dossier.
    """
    text = " ".join(GOAL_PLANNING_SYSTEM_PROMPT.split())
    assert "never automatic targets" not in text
    assert "question to ask now" not in text
    assert "are not off limits" in text
    assert "the consuming agent's decision in the moment, not this planner's" in text
    assert "never a standalone information-gathering task about a third party" in text
    assert "test, probe, or secretly verify anyone" in text
    assert "Do not diagnose" in text and "reconciliation, disclosure, or repair" in text


def test_goal_reconciliation_is_the_only_lifecycle_pass_and_keeps_breadth():
    candidates = goal_candidate_prompt("M1", [], [])
    final = goal_reconciliation_prompt([], [], [])
    assert "do not transition, retire, defer, update" in candidates
    assert "only pass that may transition a goal's lifecycle" in final
    assert "Keep breadth" in final


def test_goal_reconciliation_weighs_closure_recommendations_as_evidence():
    """Entry-grounded votes from the per-root pass are evidence, not orders --

    reconciliation still decides, and a goal one root recommends closing can
    stay open if the rest of its scope is unanswered.
    """
    recommendation = GoalClosureRecommendation(goal_id="goal-1", reason="fully covered now")
    final = " ".join(goal_reconciliation_prompt([], [], [], [recommendation]).split())
    assert "goal_id='goal-1'" in final and "fully covered now" in final
    assert "<closure-recommendations>" in final
    assert "not an instruction" in final
    assert "can still be worth holding open" in final

    empty = " ".join(goal_reconciliation_prompt([], [], []).split())
    assert "(none)" in empty and "<closure-recommendations>" in empty


def test_query_synthesis_does_not_treat_unmerged_people_as_distinct_evidence():
    assert "Separate Person records are not evidence" in QUERY_SYNTHESIS_SYSTEM_PROMPT
    assert "identities may overlap" in QUERY_SYNTHESIS_SYSTEM_PROMPT
