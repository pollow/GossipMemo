from __future__ import annotations

import asyncio
import json
from datetime import datetime

import httpx
import pytest
from pydantic import ValidationError

from gossipmemo.models import (
    ExtractionResult,
    ExtractedPerson,
    ManualMemoryRequest,
    ExtractedMemory,
    MessageInput,
    ModelMessage,
    PersonLink,
    PersonReasoningResult,
    QueryRequest,
    ExtractedRelationship,
    RelationshipReasoningResult,
    SourceRef,
    SupersedeRequest,
)
from gossipmemo.store import SqliteWorldStore
from gossipmemo.llm import OpenAICompatibleAdapter
from gossipmemo.app import create_app
from gossipmemo.config import Settings
from gossipmemo.world import SocialMemoryWorld
from gossipmemo_client import AsyncGossipMemo, GossipMemo


class FakeModel:
    configured = True

    async def extract(self, message):
        del message
        return ExtractionResult()

    async def reason_person(self, person, memories):
        del person, memories
        return PersonReasoningResult()

    async def reason_relationship(self, relationship, memories):
        del relationship, memories
        return RelationshipReasoningResult()

    async def synthesize(self, question, context):
        del question
        return "\n".join(memory.content for memory in context.memories)


def _settings(database_path, *, api_key: str = "") -> Settings:
    return Settings(
        database_path=database_path,
        api_key=api_key,
        llm_base_url="http://llm.test/v1",
        llm_api_key="test-key",
        llm_model="test-model",
    )


def _store(tmp_path) -> SqliteWorldStore:
    store = SqliteWorldStore(tmp_path / "features.db")
    store.initialize()
    return store


def test_query_uses_fts_but_keeps_structural_fallback(tmp_path):
    store = _store(tmp_path)
    store.add_manual_memory(
        "personal",
        ManualMemoryRequest(
            content="Alice plans a distinctive sabbatical in October.",
            people=[PersonLink(ref="Alice", role="subject")],
        ),
    )
    store.add_manual_memory(
        "personal",
        ManualMemoryRequest(
            content="Alice prefers green tea.",
            people=[PersonLink(ref="Alice", role="subject")],
        ),
    )

    relevant = store.read(
        "personal",
        QueryRequest(question="When is Alice's sabbatical?", people=["Alice"]),
    )
    assert [memory.content for memory in relevant.memories] == [
        "Alice plans a distinctive sabbatical in October."
    ]

    fallback = store.read(
        "personal",
        QueryRequest(question="What else should I remember?", people=["Alice"]),
    )
    assert len(fallback.memories) == 2


def test_one_hop_expansion_returns_neighbor_and_relationship_memory(tmp_path):
    store = _store(tmp_path)
    receipt = store.record_messages(
        "personal",
        [
            MessageInput(
                author="user",
                content="Alice and Bob work together.",
                source=SourceRef(provider="test", item_id="relationship-1"),
            )
        ],
    )[0]
    store.apply_extraction(
        "personal",
        receipt,
        ExtractionResult(
            people=[
                ExtractedPerson(ref="alice", display_name="Alice"),
                ExtractedPerson(ref="bob", display_name="Bob"),
            ],
            memories=[
                ExtractedMemory(
                    content="Alice and Bob work together.",
                    basis="stated",
                    relationships=[
                        ExtractedRelationship(
                            person_a_ref="alice",
                            person_b_ref="bob",
                            facets=[{"kind": "coworker"}],
                        )
                    ],
                )
            ],
        ),
    )

    context = store.read(
        "personal",
        QueryRequest(
            question="Who works together?",
            people=["Alice"],
            expand_relationships=1,
        ),
    )
    assert {person.display_name for person in context.people} == {"Alice", "Bob"}
    assert len(context.relationships) == 1
    assert context.memories[0].content == "Alice and Bob work together."


def test_supersede_preserves_history_and_retract_reason(tmp_path):
    store = _store(tmp_path)
    original_id = store.add_manual_memory(
        "personal",
        ManualMemoryRequest(
            content="Bob prefers coffee.",
            people=[PersonLink(ref="Bob", role="subject")],
        ),
    )
    replacement_id = store.supersede_memory(
        "personal",
        original_id,
        SupersedeRequest(
            content="Bob now prefers tea.",
            reason="Bob corrected me.",
        ),
    )
    assert replacement_id

    with store._connect() as connection:
        original = connection.execute(
            "SELECT status, invalidation_reason FROM memories WHERE id = ?",
            (original_id,),
        ).fetchone()
        replacement = connection.execute(
            "SELECT supersedes_memory_id FROM memories WHERE id = ?",
            (replacement_id,),
        ).fetchone()
    assert dict(original) == {
        "status": "superseded",
        "invalidation_reason": "Bob corrected me.",
    }
    assert replacement["supersedes_memory_id"] == original_id
    assert store.retract_memory("personal", replacement_id, "No longer reliable")
    with store._connect() as connection:
        reason = connection.execute(
            "SELECT invalidation_reason FROM memories WHERE id = ?",
            (replacement_id,),
        ).fetchone()[0]
    assert reason == "No longer reliable"


def test_message_time_requires_timezone():
    with pytest.raises(ValidationError, match="timezone"):
        MessageInput(
            author="user",
            content="A message without a timezone.",
            occurred_at=datetime(2026, 8, 9, 12, 0),
        )


def test_sync_and_async_sdk_follow_server_contract():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/ingest"):
            return httpx.Response(
                202,
                json={"status": "accepted", "message_ids": ["message_1"]},
            )
        if request.url.path.endswith("/query"):
            return httpx.Response(
                200,
                json={"answer": "Tea", "people": [], "relationships": [], "memories": []},
            )
        raise AssertionError(request.url.path)

    transport = httpx.MockTransport(handler)
    with GossipMemo(
        "http://memory.test", api_key="secret", transport=transport
    ) as client:
        result = client.ingest(
            content="Bob likes tea.",
            author="user",
        )
        assert result == {"status": "accepted", "message_ids": ["message_1"]}
        assert client.query("What does Bob like?")["answer"] == "Tea"

    async def async_scenario() -> None:
        async with AsyncGossipMemo(
            "http://memory.test", api_key="secret", transport=transport
        ) as client:
            assert (await client.query("What does Bob like?"))["answer"] == "Tea"

    asyncio.run(async_scenario())
    assert requests
    assert all(request.headers["authorization"] == "Bearer secret" for request in requests)


def test_openai_compatible_adapter_validates_structured_output():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        payload = json.loads(request.content)
        assert payload["response_format"] == {"type": "json_object"}
        assert "Extraction policy: conservative" in payload["messages"][1]["content"]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": (
                                '{"people":[{"ref":"bob","display_name":"Bob"}],'
                                '"memories":[{"content":"Bob likes tea.",'
                                '"basis":"stated","people":[],'
                                '"relationships":[]}]}'
                            ),
                        }
                    }
                ]
            },
        )

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter(
                "http://llm.test/v1", "key", "test-model", client=client
            )
            result = await adapter.extract(
                ModelMessage(
                    id="message_1",
                    space_id="personal",
                    author="user",
                    content="Bob likes tea.",
                    occurred_at="2026-08-09T12:00:00+00:00",
                    source_provider="test",
                    extraction_policy="conservative",
                )
            )
            assert result.people[0].display_name == "Bob"
            assert result.memories[0].basis == "stated"

    asyncio.run(scenario())


def test_http_auth_and_correction_endpoints(tmp_path):
    async def scenario() -> None:
        store = _store(tmp_path)
        world = SocialMemoryWorld(store, FakeModel())
        app = create_app(
            settings=_settings(tmp_path / "features.db", api_key="secret"),
            world=world,
        )
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://server.test"
            ) as client:
                denied = await client.post(
                    "/v1/spaces/personal/memories",
                    json={"content": "Bob prefers coffee."},
                )
                assert denied.status_code == 401
                headers = {"Authorization": "Bearer secret"}
                created = await client.post(
                    "/v1/spaces/personal/memories",
                    headers=headers,
                    json={
                        "content": "Bob prefers coffee.",
                        "people": [{"ref": "Bob", "role": "subject"}],
                    },
                )
                memory_id = created.json()["id"]
                corrected = await client.post(
                    f"/v1/spaces/personal/memories/{memory_id}/supersede",
                    headers=headers,
                    json={"content": "Bob now prefers tea.", "reason": "Correction"},
                )
                assert corrected.status_code == 201
                replacement_id = corrected.json()["id"]
                retracted = await client.post(
                    f"/v1/spaces/personal/memories/{replacement_id}/retract",
                    headers=headers,
                    json={"reason": "Uncertain"},
                )
                assert retracted.json()["status"] == "retracted"

    asyncio.run(scenario())
