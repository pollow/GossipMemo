from __future__ import annotations

import asyncio
from pathlib import Path

import httpx

from gossipmemo.app import create_app
from gossipmemo.config import Settings
from gossipmemo.models import (
    ContinuityReasoningResult,
    ExtractionResult,
    IngestRequest,
    ManualMemoryRequest,
    MessageInput,
    PersonReasoningResult,
    QueryRequest,
    RelationshipReasoningResult,
)
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
    configured = True

    def __init__(self):
        self.calls = 0

    async def extract(self, messages, context=(), known_people=(), comparison_memories=()):
        del context, known_people, comparison_memories
        return ExtractionResult()

    async def reason_person(self, person, memories):
        return PersonReasoningResult()

    async def reason_relationship(self, relationship, memories):
        return RelationshipReasoningResult()

    async def reason_continuity(self, continuity, messages):
        self.calls += 1
        return ContinuityReasoningResult(
            text="summary", through_message_id=messages[-1].id
        )

    async def synthesize(self, question, context):
        return ""


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
