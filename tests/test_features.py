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
    IngestRequest,
    ManualMemoryRequest,
    ExtractedMemory,
    MessageInput,
    ModelMessage,
    MemoryView,
    PersonReasoningResult,
    PersonView,
    QueryContext,
    QueryRequest,
    ExtractedRelationship,
    RelationshipReasoningResult,
    SourceRef,
    SupersedeRequest,
)
from gossipmemo.store import SqliteWorldStore
from gossipmemo.llm import LLMRequestError, OpenAICompatibleAdapter
from gossipmemo.app import create_app
from gossipmemo.config import Settings
from gossipmemo.world import SocialMemoryWorld
from gossipmemo_client import AsyncGossipMemo, GossipMemo


class FakeModel:
    configured = True

    async def extract(self, message, context=(), known_people=(), comparison_memories=()):
        del message, context, known_people, comparison_memories
        return ExtractionResult()

    async def reason_person(self, person, memories, inferred_memories=(), hypotheses=()):
        del person, memories, inferred_memories, hypotheses
        return PersonReasoningResult()

    async def reason_relationship(self, relationship, memories, inferred_memories=(), hypotheses=()):
        del relationship, memories, inferred_memories, hypotheses
        return RelationshipReasoningResult()

    async def reason_user_model(self, memories, inferred_memories=(), hypotheses=()):
        del memories, inferred_memories, hypotheses
        from gossipmemo.models import UserModelReasoningResult
        return UserModelReasoningResult()

    async def reason_continuity(self, continuity, messages):
        del continuity
        from gossipmemo.models import ContinuityReasoningResult
        return ContinuityReasoningResult(through_message_id=messages[-1].id if messages else "")

    async def audit_coverage(self, coverage, memories, hypotheses=()):
        del coverage, memories, hypotheses
        from gossipmemo.models import CoverageAuditPatch
        return CoverageAuditPatch()

    async def plan_learning_goals(self, coverage, hypotheses, open_goals, recent_closed_goals):
        del coverage, hypotheses, open_goals, recent_closed_goals
        from gossipmemo.models import GoalPlanningResult
        return GoalPlanningResult()

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
            people=["Alice"],
        ),
    )
    store.add_manual_memory(
        "personal",
        ManualMemoryRequest(
            content="Alice prefers green tea.",
            people=["Alice"],
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
    batch_id = store.create_extraction_batch("personal", [receipt])
    assert batch_id is not None
    store.apply_extraction(
        "personal",
        batch_id,
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
            people=["Bob"],
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


def test_supersede_inherits_or_overrides_about_user(tmp_path):
    store = _store(tmp_path)
    original = store.add_manual_memory(
        "personal", ManualMemoryRequest(content="I like tea.", about_user=True)
    )
    inherited = store.supersede_memory(
        "personal", original, SupersedeRequest(content="I like green tea.")
    )
    assert inherited
    overridden = store.supersede_memory(
        "personal", inherited, SupersedeRequest(content="This is about Bob.", about_user=False)
    )
    assert overridden
    with store._connect() as connection:
        rows = connection.execute(
            "SELECT about_user FROM memories WHERE id IN (?, ?) ORDER BY created_at",
            (inherited, overridden),
        ).fetchall()
    assert [row["about_user"] for row in rows] == [1, 0]


def test_message_time_requires_timezone():
    with pytest.raises(ValidationError, match="timezone"):
        MessageInput(
            author="user",
            content="A message without a timezone.",
            occurred_at=datetime(2026, 8, 9, 12, 0),
        )


def test_extraction_batches_default_to_six_messages(tmp_path):
    calls: list[int] = []

    class BatchModel(FakeModel):
        async def extract(self, messages, context=(), known_people=(), comparison_memories=()):
            del context, known_people
            calls.append(len(messages))
            return ExtractionResult()

    async def scenario() -> None:
        store = _store(tmp_path)
        world = SocialMemoryWorld(
            store, BatchModel(), extraction_batch_timeout_seconds=0.001
        )
        await world.start()
        try:
            await world.ingest(
                "personal",
                IngestRequest(
                    messages=[
                        MessageInput(author="user", content=f"message {index}")
                        for index in range(13)
                    ]
                ),
            )
            for _ in range(100):
                if len(calls) == 3:
                    break
                await asyncio.sleep(0.001)
        finally:
            await world.stop()

    asyncio.run(scenario())
    assert calls == [6, 6, 1]


def test_partial_extraction_batch_waits_until_full(tmp_path):
    calls: list[int] = []

    class BatchModel(FakeModel):
        async def extract(self, messages, context=(), known_people=(), comparison_memories=()):
            del context, known_people
            calls.append(len(messages))
            return ExtractionResult()

    async def scenario() -> None:
        store = _store(tmp_path)
        world = SocialMemoryWorld(
            store, BatchModel(), extraction_batch_timeout_seconds=60
        )
        await world.start()
        try:
            await world.ingest(
                "personal",
                IngestRequest(
                    messages=[
                        MessageInput(author="user", content=f"message {index}")
                        for index in range(5)
                    ]
                ),
            )
            await asyncio.sleep(0)
            assert calls == []

            await world.ingest(
                "personal",
                IngestRequest(
                    messages=[MessageInput(author="assistant", content="message 5")]
                ),
            )
            for _ in range(100):
                if calls:
                    break
                await asyncio.sleep(0)
        finally:
            await world.stop()

    asyncio.run(scenario())
    assert calls == [6]


def test_induction_waits_for_daily_scheduler(tmp_path):
    calls: list[str] = []

    class InductionModel(FakeModel):
        async def reason_person(self, person, memories, inferred_memories=(), hypotheses=()):
            del inferred_memories, hypotheses
            calls.append(person.display_name)
            return PersonReasoningResult(profile_card={"memory_count": len(memories)})

    async def scenario() -> None:
        store = _store(tmp_path)
        world = SocialMemoryWorld(
            store,
            InductionModel(),
            induction_interval_seconds=0.01,
        )
        await world.start()
        try:
            world.add_memory(
                "personal",
                ManualMemoryRequest(content="Bob likes tea.", people=["Bob"]),
            )
            await asyncio.sleep(0)
            assert calls == []

            for _ in range(100):
                if calls:
                    break
                await asyncio.sleep(0.001)
        finally:
            await world.stop()

    asyncio.run(scenario())
    assert calls == ["Bob"]


def test_induction_schedules_user_model_only_for_about_user_memory(tmp_path):
    calls: list[int] = []

    class UserModel(FakeModel):
        async def reason_user_model(self, memories, inferred_memories=(), hypotheses=()):
            del inferred_memories, hypotheses
            calls.append(len(memories))
            from gossipmemo.models import UserModelReasoningResult
            return UserModelReasoningResult(profile_card={"count": len(memories)})

    async def scenario() -> None:
        store = _store(tmp_path)
        world = SocialMemoryWorld(store, UserModel(), induction_interval_seconds=0.01)
        await world.start()
        try:
            world.add_memory("personal", ManualMemoryRequest(content="ordinary fact"))
            await asyncio.sleep(0.03)
            assert calls == []
            world.add_memory(
                "personal", ManualMemoryRequest(content="I like tea", about_user=True)
            )
            for _ in range(100):
                if calls:
                    break
                await asyncio.sleep(0.001)
        finally:
            await world.stop()
        assert calls == [1]

    asyncio.run(scenario())


def test_startup_catches_up_stale_profiles(tmp_path):
    calls: list[str] = []

    class InductionModel(FakeModel):
        async def reason_person(self, person, memories, inferred_memories=(), hypotheses=()):
            del inferred_memories, hypotheses
            calls.append(person.display_name)
            return PersonReasoningResult(profile_card={"memory_count": len(memories)})

    async def scenario() -> None:
        store = _store(tmp_path)
        store.add_manual_memory(
            "personal", ManualMemoryRequest(content="Bob likes tea.", people=["Bob"])
        )
        world = SocialMemoryWorld(
            store,
            InductionModel(),
            induction_interval_seconds=60,
        )
        await world.start()
        try:
            for _ in range(100):
                if calls:
                    break
                await asyncio.sleep(0)
        finally:
            await world.stop()

    asyncio.run(scenario())
    assert calls == ["Bob"]


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
        assert "server's balanced extraction policy for the whole batch" in payload["messages"][1]["content"]
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
                [ModelMessage(
                    id="message_1",
                    space_id="personal",
                    author="user",
                    content="Bob likes tea.",
                    occurred_at="2026-08-09T12:00:00+00:00",
                    source_provider="test",
                )]
            )
            assert result.people[0].display_name == "Bob"
            assert result.memories[0].basis == "stated"

    asyncio.run(scenario())


def test_owner_reasoning_uses_an_identical_prefix_for_both_stages():
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        content = '{"profile_card":{"summary":"tea"}}' if len(payloads) == 1 else '{"hypothesis_actions":{"upserts":[],"transitions":[]}}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter("http://llm.test/v1", "key", "test-model", client=client)
            result = await adapter.reason_person(
                PersonView(id="person_1", display_name="Bob"),
                [MemoryView(id="memory_1", content="Bob likes tea.", kind="fact", basis="stated", status="active", created_at="2026-08-01T00:00:00+00:00")],
            )
            assert result.profile_card == {"summary": "tea"}

    asyncio.run(scenario())
    first, second = (payload["messages"] for payload in payloads)
    assert second[:len(first)] == first
    assert second[len(first)]["role"] == "assistant"
    assert "<evidence-memories>" in first[1]["content"]
    assert "comparison-only" in first[1]["content"]
    assert "inferred_memory_actions" in second[-1]["content"]


def test_user_owner_review_schema_excludes_inferred_memory_actions():
    payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payloads.append(json.loads(request.content))
        content = '{"profile_card":{}}' if len(payloads) == 1 else '{"hypothesis_actions":{"upserts":[],"transitions":[]}}'
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            adapter = OpenAICompatibleAdapter("http://llm.test/v1", "key", "test-model", client=client)
            await adapter.reason_user_model([])

    asyncio.run(scenario())
    assert "hypothesis_actions" in payloads[1]["messages"][-1]["content"]
    assert "inferred_memory_actions" not in payloads[1]["messages"][-1]["content"]


def test_openai_compatible_adapter_retries_transient_statuses(monkeypatch):
    statuses = iter([429, 503, 200])
    requests = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        status = next(statuses)
        if status == 200:
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Recovered"}}]},
            )
        return httpx.Response(
            status,
            headers={"Retry-After": "7"} if status == 429 else {},
            json={"error": {"message": "temporarily overloaded"}},
        )

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("gossipmemo.llm.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(
        "gossipmemo.llm.random.uniform", lambda _lower, upper: upper
    )

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            adapter = OpenAICompatibleAdapter(
                "http://llm.test/v1",
                "key",
                "test-model",
                client=client,
                max_retries=2,
                retry_base_seconds=2,
                retry_max_seconds=10,
            )
            assert await adapter.synthesize("Question", QueryContext()) == "Recovered"

    asyncio.run(scenario())
    assert requests == 3
    assert delays == [7, 4]


def test_openai_compatible_adapter_does_not_retry_permanent_status(monkeypatch):
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        return httpx.Response(401, json={"error": {"message": "bad key"}})

    async def unexpected_sleep(delay: float) -> None:
        raise AssertionError(f"unexpected retry delay: {delay}")

    monkeypatch.setattr("gossipmemo.llm.asyncio.sleep", unexpected_sleep)

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            adapter = OpenAICompatibleAdapter(
                "http://llm.test/v1", "key", "test-model", client=client
            )
            with pytest.raises(LLMRequestError, match="HTTP 401: bad key"):
                await adapter.synthesize("Question", QueryContext())

    asyncio.run(scenario())
    assert requests == 1


def test_openai_compatible_adapter_retries_transport_errors(monkeypatch):
    requests = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        if requests == 1:
            raise httpx.ConnectError("model is restarting", request=request)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "Recovered"}}]},
        )

    async def fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr("gossipmemo.llm.asyncio.sleep", fake_sleep)
    monkeypatch.setattr(
        "gossipmemo.llm.random.uniform", lambda _lower, upper: upper
    )

    async def scenario() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ) as client:
            adapter = OpenAICompatibleAdapter(
                "http://llm.test/v1",
                "key",
                "test-model",
                client=client,
                max_retries=1,
                retry_base_seconds=3,
                retry_max_seconds=10,
            )
            assert await adapter.synthesize("Question", QueryContext()) == "Recovered"

    asyncio.run(scenario())
    assert requests == 2
    assert delays == [3]


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
                        "people": ["Bob"],
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
