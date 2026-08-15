from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest

from gossipmemo.imports import load_chat_messages
from gossipmemo.models import (
    CoverageAuditPatch,
    CoverageCriterionPatch,
    ContinuityReasoningResult,
    ExtractionResult,
    ExtractedMemory,
    GoalPlanningResult,
    ManualMemoryRequest,
    MessageInput,
    SourceRef,
    UserModelReasoningResult,
)
from gossipmemo.store import SqliteWorldStore
from gossipmemo.world import SocialMemoryWorld


def test_load_chat_messages_requires_sender_timestamp_and_stable_identity(tmp_path):
    path = tmp_path / "history.json"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Earlier",
                        "occurred_at": "2026-08-01T09:00:00-07:00",
                    },
                    {
                        "author": "assistant",
                        "content": "Later",
                        "occurred_at": "2026-08-01T16:01:00Z",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    first = load_chat_messages(path)
    second = load_chat_messages(path)

    assert [message.author for message in first] == ["user", "assistant"]
    assert first[0].occurred_at == datetime(
        2026, 8, 1, 16, 0, tzinfo=timezone.utc
    )
    assert [message.idempotency_key for message in first] == [
        message.idempotency_key for message in second
    ]
    assert first[0].source.provider == "import"
    assert first[0].source.item_id == "0"

    path.write_text(
        json.dumps([{"role": "user", "content": "No timestamp"}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="record 1 occurred_at is required"):
        load_chat_messages(path)


def test_load_chat_messages_reports_jsonl_line_and_rejects_naive_time(tmp_path):
    path = tmp_path / "history.jsonl"
    path.write_text('{"role":"user"}\nnot-json\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 2 is not valid JSON"):
        load_chat_messages(path)

    path.write_text(
        json.dumps(
            {
                "role": "user",
                "content": "Missing timezone",
                "occurred_at": "2026-08-01T09:00:00",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="occurred_at must include a timezone"):
        load_chat_messages(path)


def test_user_md_overwrite_is_fresh_at_current_memory_watermark(tmp_path):
    store = SqliteWorldStore(tmp_path / "world.db")
    store.initialize()
    store.add_manual_memory(
        "personal",
        ManualMemoryRequest(content="The user likes tea.", about_user=True),
    )

    store.overwrite_user_model("personal", {"summary": "# Imported profile"})

    user_model, memories, watermark = store.user_model_context("personal")
    assert user_model.profile_card == {"summary": "# Imported profile"}
    assert user_model.profile_source_updated_at == watermark
    assert user_model.stale is False
    assert len(memories) == 1


def test_import_drains_partial_batch_refreshes_projections_and_is_idempotent(
    tmp_path,
):
    class FakeModel:
        configured = True

        def __init__(self):
            self.extractions = 0
            self.coverage_audits = 0
            self.goal_plans = 0

        async def extract(self, messages, context=(), known_people=(), comparison_memories=()):
            del context, known_people, comparison_memories
            self.extractions += 1
            return ExtractionResult(
                memories=[
                    ExtractedMemory(
                        content="The user likes tea.",
                        basis="stated",
                        about_user=True,
                    )
                ]
            )

        async def reason_user_model(self, memories):
            assert memories[0].content == "The user likes tea."
            return UserModelReasoningResult(
                profile_card={"summary": "Likes tea"}
            )

        async def reason_continuity(self, continuity, messages):
            assert continuity is not None
            return ContinuityReasoningResult(
                text="Imported conversation",
                through_message_id=messages[-1].id,
            )

        async def audit_coverage(self, coverage, memories, hypotheses=()):
            del coverage, hypotheses
            # Make it observable that import waits for induction spawned by
            # extraction instead of returning as soon as messages complete.
            await asyncio.sleep(0.01)
            self.coverage_audits += 1
            return CoverageAuditPatch(
                criteria=[
                    CoverageCriterionPatch(
                        criterion_id="P8",
                        level="fragmentary",
                        known_state="A preference is known.",
                        evidence_memory_ids=[memories[0].id],
                    )
                ]
            )

        async def plan_learning_goals(
            self, coverage, hypotheses, open_goals, recent_closed_goals
        ):
            del coverage, hypotheses, open_goals, recent_closed_goals
            self.goal_plans += 1
            return GoalPlanningResult()

        async def aclose(self):
            return None

    async def scenario():
        store = SqliteWorldStore(tmp_path / "world.db")
        model = FakeModel()
        world = SocialMemoryWorld(
            store,
            model,
            extraction_batch_size=6,
            continuity_threshold=2,
        )
        messages = [
            MessageInput(
                author="user",
                content="I like tea.",
                occurred_at=datetime(2026, 8, 1, 16, index, tzinfo=timezone.utc),
                source=SourceRef(
                    provider="import",
                    conversation_key="history",
                    item_id=str(index),
                ),
            )
            for index in range(2)
        ]

        await world.start()
        try:
            store.overwrite_user_model("personal", {"summary": "USER.md"})
            first = await world.import_messages("personal", messages)
            second = await world.import_messages("personal", messages)
        finally:
            await world.stop()

        assert first == {"messages": 2, "extracted": 2}
        assert second == first
        assert model.extractions == 1
        assert model.coverage_audits == 1
        assert model.goal_plans == 1
        assert store.user_model_context("personal")[0].profile_card == {
            "summary": "Likes tea"
        }
        assert store.context_bundle("personal").continuity.text == (
            "Imported conversation"
        )
        with store._connect() as connection:
            assert connection.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 2
            states = connection.execute(
                "SELECT DISTINCT extraction_state FROM messages"
            ).fetchall()
        assert [row[0] for row in states] == ["completed"]

    asyncio.run(scenario())
