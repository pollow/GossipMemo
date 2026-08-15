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


def message(message_id: str, policy: str) -> ModelMessage:
    return ModelMessage(
        id=message_id,
        space_id="space",
        author="user",
        content="I had coffee today.",
        occurred_at="2026-08-14T12:00:00+00:00",
        source_provider="test",
        extraction_policy=policy,
    )


def test_extraction_prompt_applies_policy_per_message_and_dates():
    prompt = extraction_prompt(
        [
            message("m-conservative", "conservative"),
            message("m-comprehensive", "comprehensive"),
        ]
    )

    assert "each message's extraction policy independently" in prompt
    assert "Message m-conservative (conservative)" in prompt
    assert "Message m-comprehensive (comprehensive)" in prompt
    assert "occurred_at" in EXTRACTION_SYSTEM_PROMPT
    assert "never use an inferred basis for extraction" in " ".join(
        EXTRACTION_SYSTEM_PROMPT.split()
    )


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
