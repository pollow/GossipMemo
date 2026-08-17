from gossipmemo.models import CoverageMapView, MemoryView, ModelMessage
from gossipmemo.query import QUERY_SYNTHESIS_SYSTEM_PROMPT
from gossipmemo.reasoners import (
    EXTRACTION_SYSTEM_PROMPT,
    PERSON_REASONING_SYSTEM_PROMPT,
    RELATIONSHIP_REASONING_SYSTEM_PROMPT,
    USER_MODEL_REASONING_SYSTEM_PROMPT,
    CONTINUITY_SYSTEM_PROMPT,
    extraction_prompt,
    coverage_audit_prompt,
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
            message("hypothesis", author="assistant",
                    content="You avoid conflict."),
            message("confirmation", content="Yes, especially at work."),
            message("advice", author="assistant",
                    content="Try a direct conversation."),
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


def test_coverage_prompt_keeps_private_facets_and_choice_boundaries():
    prompt = coverage_audit_prompt(CoverageMapView(space_id="space"), [], [])
    assert "sexuality" in prompt
    assert "shame" in prompt
    assert "mortality" in prompt
    assert "never raises a coverage level" in prompt


def test_query_synthesis_does_not_treat_unmerged_people_as_distinct_evidence():
    assert "Separate Person records are not evidence" in QUERY_SYNTHESIS_SYSTEM_PROMPT
    assert "identities may overlap" in QUERY_SYNTHESIS_SYSTEM_PROMPT
