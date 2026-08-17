from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import httpx

from gossipmemo.app import create_app
from gossipmemo.config import Settings
from gossipmemo.context_budget import ContextBudget
from gossipmemo.models import (
    ContinuityReasoningResult,
    IngestRequest,
    ManualMemoryRequest,
    MessageInput,
    QueryRequest,
)
from gossipmemo.llm import ChatCompletionRequest, ProviderGate, RetryPolicy
from gossipmemo.store import SqliteWorldStore
from gossipmemo.world import SocialMemoryWorld


def test_continuity_uses_rowid_watermark_and_filters_person_refs(tmp_path: Path):
    store = SqliteWorldStore(tmp_path / "continuity.db")
    store.initialize()
    store.add_manual_memory(
        "space", ManualMemoryRequest(content="Alice is a colleague.", people=["Alice"])
    )
    ids = store.record_messages(
        "space", [MessageInput(author="user", content=f"message {i}") for i in range(3)]
    )
    assert store.pending_continuities(3) == ["space"]
    assert store.apply_continuity_reasoning(
        "space", None,
        ContinuityReasoningResult(
            text="unfinished thread", related_person_ids=["person_missing"],
            through_message_id=ids[1],
        ),
    )
    continuity, newer = store.continuity_context("space")
    assert continuity is not None
    assert continuity.through_message_id == ids[1]
    assert [message.id for message in newer] == [ids[2]]
    assert store.pending_continuities(1) == ["space"]

    alice_id = store.read("space", QueryRequest(question="dossier", people=["Alice"])).people[0].id
    assert store.apply_continuity_reasoning(
        "space", ids[1],
        ContinuityReasoningResult(
            text="updated", related_person_ids=[alice_id], through_message_id=ids[2]
        ),
    )
    bundle = store.context_bundle("space")
    assert bundle.continuity.related_person_ids == [alice_id]
    assert [person.id for person in bundle.people] == [alice_id]
    assert bundle.version == store.context_bundle("space").version


class _ContinuityModel:
    """An `LlmTransport` double; every stage is told apart by prompt marker
    in `complete` (see `tests/test_features.py`'s `FakeModel`).
    """

    configured = True
    gate = ProviderGate()
    context_budget = ContextBudget()
    retry_policy = RetryPolicy(attempts=1, base_seconds=0.001, max_seconds=0.001)
    user_name = "CurrentUser"
    extraction_policy = "balanced"

    def __init__(self):
        self.calls = 0

    async def aclose(self):
        return None

    def prepare(self, messages, *, structured: bool) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model="fake",
            messages=list(messages),
            response_format={"type": "json_object"} if structured else None,
        )

    async def complete(self, request: ChatCompletionRequest) -> str:
        combined = " ".join(str(message.content) for message in request.messages)
        if "Rebuild compact cross-session continuity." in combined:
            self.calls += 1
            ids = re.findall(r'"id": "([^"]*)"', combined)
            return json.dumps({"text": "summary", "through_message_id": ids[-1] if ids else ""})
        if "Review the projection above" in combined:
            return json.dumps({})
        if '"profile_card"' in combined:
            return json.dumps({"profile_card": {}})
        return json.dumps(
            {"facets": [], "closeness": None, "tone": None, "status": "unknown", "summary": ""}
        )


def test_continuity_schedules_asynchronously_at_injected_threshold(tmp_path: Path):
    async def scenario():
        store = SqliteWorldStore(tmp_path / "scheduled.db")
        model = _ContinuityModel()
        world = SocialMemoryWorld(
            store, model, extraction_batch_size=100, continuity_threshold=2
        )
        await world.start()
        try:
            response = await world.ingest(
                "space",
                IngestRequest(
                    messages=[
                        MessageInput(author="user", content="one"),
                        MessageInput(author="user", content="two"),
                    ]
                ),
            )
            assert response.status == "accepted"
            for _ in range(50):
                if model.calls:
                    break
                await asyncio.sleep(0)
            assert model.calls == 1
        finally:
            await world.stop()

    asyncio.run(scenario())


def test_continuity_backfill_is_bounded_and_reaches_last_message(tmp_path: Path):
    # World-level reasoning always requests the full pending delta and lets
    # the reasoner own chunking (see OpenAICompatibleAdapter's context_budget
    # handling). This test exercises the store's bounded-call primitive
    # directly, which a reasoner can still opt into.
    store = SqliteWorldStore(tmp_path / "large.db")
    store.initialize()
    ids = store.record_messages("space", [MessageInput(author="user", content=f"m{i}") for i in range(1500)])
    calls_data: list[list] = []
    expected_through: str | None = None
    while True:
        context = store.continuity_context("space", limit=32)
        assert context is not None
        continuity, messages = context
        if not messages:
            break
        calls_data.append(messages)
        result = ContinuityReasoningResult(text="summary", through_message_id=messages[-1].id)
        assert store.apply_continuity_reasoning("space", expected_through, result)
        expected_through = result.through_message_id
    assert len(calls_data) > 1
    assert all(len(batch) <= 32 for batch in calls_data)
    assert store.continuity_context("space")[0].through_message_id == ids[-1]


def test_continuity_context_can_delegate_all_batching_to_reasoner(tmp_path: Path):
    store = SqliteWorldStore(tmp_path / "reasoner-batching.db")
    store.initialize()
    ids = store.record_messages(
        "space", [MessageInput(author="user", content=f"m{i}") for i in range(40)]
    )
    _, bounded = store.continuity_context("space")
    assert len(bounded) == 32

    _, complete = store.continuity_context("space", limit=None)
    assert [message.id for message in complete] == ids


def test_continuity_truncates_oversized_messages_without_stalling(tmp_path: Path):
    # Exercises the store's max_message_chars/max_total_chars bounded call
    # directly; a reasoner opts into this the same way the removed world-level
    # fallback used to.
    store = SqliteWorldStore(tmp_path / "oversized.db")
    store.initialize()
    ids = store.record_messages("space", [MessageInput(author="user", content="x" * 100_000) for _ in range(40)])
    calls_data: list[list] = []
    expected_through: str | None = None
    while True:
        context = store.continuity_context(
            "space", limit=None, max_message_chars=8000, max_total_chars=48000,
        )
        assert context is not None
        continuity, messages = context
        if not messages:
            break
        calls_data.append(messages)
        result = ContinuityReasoningResult(text="summary", through_message_id=messages[-1].id)
        assert store.apply_continuity_reasoning("space", expected_through, result)
        expected_through = result.through_message_id
    assert calls_data
    assert all(max(len(message.content) for message in batch) <= 8000 for batch in calls_data)
    assert all(sum(len(message.content) for message in batch) <= 48000 for batch in calls_data)
    through = store.continuity_context("space")[0].through_message_id
    assert through in ids
    assert ids.index(through) > 0
    assert len(store.continuity_context("space")[1]) < 20


def test_context_endpoint_is_read_only_and_returns_bundle(tmp_path: Path):
    async def scenario():
        store = SqliteWorldStore(tmp_path / "http.db")
        world = SocialMemoryWorld(store, _ContinuityModel())
        app = create_app(
            Settings(
                database_path=store.path,
                llm_base_url="http://llm.test/v1",
                llm_api_key="key",
                llm_model="model",
            ),
            world,
        )
        async with app.router.lifespan_context(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                response = await client.get("/v1/spaces/space/context")
                assert response.status_code == 200
                assert set(response.json()) == {"version", "user_model", "continuity", "people", "guidance"}
                post = await client.post("/v1/spaces/space/context")
                assert post.status_code == 405

    asyncio.run(scenario())
