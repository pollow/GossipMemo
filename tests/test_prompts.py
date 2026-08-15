from gossipmemo.models import ModelMessage
from gossipmemo.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    PERSON_REASONING_SYSTEM_PROMPT,
    QUERY_SYNTHESIS_SYSTEM_PROMPT,
    RELATIONSHIP_REASONING_SYSTEM_PROMPT,
    USER_MODEL_REASONING_SYSTEM_PROMPT,
    CONTINUITY_SYSTEM_PROMPT,
    extraction_prompt,
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


def test_extraction_prompt_separates_recent_context_from_new_evidence():
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
    )
    assert "Recent context (context only)" in prompt
    assert "Alex Wang" in prompt
    assert "canonical display_name" in prompt
    assert "Omit a known person unless" in prompt
    assert "Do not echo the known-people list" in prompt
    assert "Current batch evidence (user-authored; the only messages allowed" in prompt


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


def test_reasoning_prompts_allow_useful_social_inference():
    assert "patterns" in PERSON_REASONING_SYSTEM_PROMPT
    assert "social inferences" in PERSON_REASONING_SYSTEM_PROMPT
    assert "friction" in RELATIONSHIP_REASONING_SYSTEM_PROMPT
    assert "Generalize recurring patterns" in USER_MODEL_REASONING_SYSTEM_PROMPT


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
