from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone

import pytest

from gossipmemo.context_budget import ContextBudget
from gossipmemo.imports import load_chat_messages
from gossipmemo.models import (
    COVERAGE_ROOTS,
    ManualMemoryRequest,
    MessageInput,
    SourceRef,
)
from gossipmemo.store import SqliteWorldStore
from gossipmemo.transport import ChatCompletionRequest, ProviderGate, RetryPolicy
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
        gate = ProviderGate()
        context_budget = ContextBudget()
        retry_policy = RetryPolicy(attempts=1, base_seconds=0.001, max_seconds=0.001)

        def __init__(self):
            self.extractions = 0
            self.coverage_audits = 0
            self.goal_plans = 0

        def prepare(self, messages, *, structured: bool) -> ChatCompletionRequest:
            return ChatCompletionRequest(
                model="fake",
                messages=list(messages),
                response_format={"type": "json_object"} if structured else None,
            )

        async def complete(self, request: ChatCompletionRequest) -> str:
            # Extraction, continuity, person/relationship/user_model/coverage/
            # learning_goals reasoning all drive `prepare` and `complete`
            # directly now (see reasoners/extraction.py, reasoners/continuity.py,
            # reasoners/owner.py, reasoners/coverage.py,
            # reasoners/learning_goals.py); each stage is distinguished by a
            # prompt marker rather than a typed method.
            combined = " ".join(str(message.content) for message in request.messages)
            if "Extract useful, provenance-aware memories" in combined:
                self.extractions += 1
                return json.dumps({
                    "memories": [
                        {"content": "The user likes tea.", "basis": "stated", "about_user": True}
                    ]
                })
            if "Rebuild compact cross-session continuity." in combined:
                ids = re.findall(r'"id": "([^"]*)"', combined)
                return json.dumps({
                    "text": "Imported conversation",
                    "through_message_id": ids[-1] if ids else "",
                })
            if "Review the projection above" in combined:
                return json.dumps({})
            if "Summarize what is known about one area of a person's" in combined:
                # Make it observable that import waits for induction spawned
                # by extraction instead of returning as soon as messages
                # complete.
                await asyncio.sleep(0.01)
                self.coverage_audits += 1
                return json.dumps({
                    "additions": [{"path": "", "content": "A preference is known."}]
                })
            if "Propose optional candidate directions only" in combined:
                return json.dumps({"candidates": [
                    {"prompt": "Would you like to say more about that?",
                     "rationale": "One optional direction"}
                ]})
            if "<candidates>" in combined:
                self.goal_plans += 1
                return json.dumps({})
            assert "The user likes tea." in combined
            return json.dumps({"profile_card": {"summary": "Likes tea"}})

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
        # Coverage fans out over roots: one request per root per attempt.
        assert model.coverage_audits == len(COVERAGE_ROOTS)
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


def test_import_reports_background_reasoning_failure(tmp_path):
    class FailingModel:
        configured = True
        gate = ProviderGate()
        context_budget = ContextBudget()
        retry_policy = RetryPolicy(attempts=1, base_seconds=0.001, max_seconds=0.001)

        def prepare(self, messages, *, structured: bool) -> ChatCompletionRequest:
            return ChatCompletionRequest(
                model="fake",
                messages=list(messages),
                response_format={"type": "json_object"} if structured else None,
            )

        async def complete(self, request: ChatCompletionRequest) -> str:
            raise RuntimeError("reasoning failed")

        async def aclose(self):
            return None

    async def scenario() -> None:
        store = SqliteWorldStore(tmp_path / "failed-reasoning.db")
        store.initialize()
        store.add_manual_memory(
            "personal",
            ManualMemoryRequest(content="Bob keeps notes.", people=["Bob"]),
        )
        world = SocialMemoryWorld(store, FailingModel())
        await world.start()
        try:
            with pytest.raises(
                RuntimeError,
                match="background import operation reasoning-pipeline failed",
            ):
                await world.import_messages("personal", [])
        finally:
            await world.stop()

    asyncio.run(scenario())
