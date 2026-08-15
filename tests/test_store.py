from __future__ import annotations

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from gossipmemo.models import (
    ExtractionResult,
    ExtractedPerson,
    InferredMemory,
    ManualMemoryRequest,
    ExtractedMemory,
    MessageInput,
    PersonReasoningResult,
    QueryRequest,
    ExtractedRelationship,
    SourceRef,
    UserModelReasoningResult,
)
from gossipmemo.store import AmbiguousPersonError, SqliteWorldStore


@pytest.fixture
def store(tmp_path):
    world = SqliteWorldStore(tmp_path / "world.db")
    world.initialize()
    return world


def _message(
    *,
    content: str = "Alice told me that Bob may leave.",
    idempotency_key: str | None = None,
    source_item_id: str | None = None,
) -> MessageInput:
    return MessageInput(
        idempotency_key=idempotency_key,
        author="user",
        content=content,
        occurred_at=datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc),
        source=SourceRef(
            provider="agent_chat",
            conversation_key="conversation-1",
            item_id=source_item_id,
        ),
    )


def _rows(store: SqliteWorldStore, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    with store._connect() as connection:
        return connection.execute(sql, params).fetchall()


def _batch(store: SqliteWorldStore, *message_ids: str) -> str:
    batch_id = store.create_extraction_batch("personal", list(message_ids))
    assert batch_id is not None
    return batch_id


def test_record_messages_is_idempotent_by_key_and_source_identity(store):
    first = store.record_messages(
        "personal", [_message(idempotency_key="request-1", source_item_id="item-1")]
    )[0]
    duplicate_by_key = store.record_messages(
        "personal", [_message(content="changed", idempotency_key="request-1")]
    )[0]
    duplicate_by_source = store.record_messages(
        "personal", [_message(content="changed again", source_item_id="item-1")]
    )[0]

    assert duplicate_by_key == first
    assert duplicate_by_source == first
    assert len(_rows(store, "SELECT id FROM messages")) == 1
    assert _rows(store, "SELECT content FROM messages WHERE id = ?", (first,))[0][
        "content"
    ] == _message().content


def test_record_messages_idempotency_is_atomic_across_concurrent_callers(store):
    # Space creation is deliberately outside this race; the idempotency claim
    # itself must still be safe when separate request handlers hit SQLite at
    # the same time.
    store.ensure_space("personal")

    def record(worker: int):
        return store.record_messages(
            "personal",
            [_message(content=f"worker {worker}", idempotency_key="same-request")],
        )[0]

    with ThreadPoolExecutor(max_workers=8) as workers:
        receipts = list(workers.map(record, range(8)))

    assert len({receipt for receipt in receipts}) == 1
    assert len(_rows(store, "SELECT id FROM messages")) == 1


def test_source_identity_is_atomic_when_conversation_key_is_null(store):
    store.ensure_space("personal")

    def record(worker: int):
        return store.record_messages(
            "personal",
            [
                MessageInput(
                    author="user",
                    content=f"worker {worker}",
                    source=SourceRef(provider="import", item_id="same-item"),
                )
            ],
        )[0]

    with ThreadPoolExecutor(max_workers=8) as workers:
        receipts = list(workers.map(record, range(8)))

    assert len({receipt for receipt in receipts}) == 1
    assert len(_rows(store, "SELECT id FROM messages")) == 1


def test_first_ingest_initializes_space_atomically_across_callers(store):
    def record(worker: int):
        return store.record_messages(
            "new-space",
            [_message(content=f"worker {worker}", idempotency_key=f"request-{worker}")],
        )[0]

    with ThreadPoolExecutor(max_workers=8) as workers:
        receipts = list(workers.map(record, range(8)))

    assert len(receipts) == 8
    assert len(_rows(store, "SELECT id FROM messages WHERE space_id = 'new-space'")) == 8
    assert _rows(store, "SELECT id FROM people WHERE space_id = 'new-space'") == []


def test_zero_memory_extraction_completes_message_without_creating_memory(store):
    receipt = store.record_messages("personal", [_message()])[0]

    affected_people, affected_relationships = store.apply_extraction(
        "personal", _batch(store, receipt), ExtractionResult()
    )

    assert affected_people == set()
    assert affected_relationships == set()
    row = _rows(
        store,
        "SELECT extraction_state, extraction_attempts FROM messages WHERE id = ?",
        (receipt,),
    )[0]
    assert row["extraction_state"] == "completed"
    assert row["extraction_attempts"] == 0
    assert store.pending_extractions() == []
    assert _rows(store, "SELECT id FROM memories") == []


def test_extraction_persists_about_user_flag(store):
    receipt = store.record_messages("personal", [_message()])[0]
    store.apply_extraction(
        "personal", _batch(store, receipt),
        ExtractionResult(memories=[ExtractedMemory(
            content="I prefer tea.", basis="stated", about_user=True,
        )]),
    )
    row = _rows(store, "SELECT about_user FROM memories")[0]
    assert row["about_user"] == 1


def test_user_model_reads_active_about_user_memories_and_uses_watermark(store):
    about_id = store.add_manual_memory(
        "personal", ManualMemoryRequest(content="I like tea.", about_user=True)
    )
    store.add_manual_memory(
        "personal", ManualMemoryRequest(content="Bob likes coffee.")
    )
    context = store.user_model_context("personal")
    assert context is not None
    view, memories, watermark = context
    assert view.stale is True
    assert [memory.content for memory in memories] == ["I like tea."]
    assert watermark is not None
    assert store.apply_user_model_reasoning(
        "personal", watermark, UserModelReasoningResult(profile_card={"summary": "tea"})
    ) is True
    refreshed = store.user_model_context("personal")
    assert refreshed is not None and refreshed[0].stale is False
    assert store.apply_user_model_reasoning(
        "personal", watermark, UserModelReasoningResult(profile_card={"summary": "old"})
    ) is False
    store.retract_memory("personal", about_id)
    after_retract = store.user_model_context("personal")
    assert after_retract is not None and after_retract[1] == []
    assert after_retract[0].stale is False


def test_memory_fts_triggers_track_insert_update_and_delete(store):
    receipt = store.record_messages("personal", [_message()])[0]
    store.apply_extraction(
        "personal",
        _batch(store, receipt),
        ExtractionResult(
            memories=[
                ExtractedMemory(
                    content="Distinctive sabbatical phrase",
                    basis="observed",
                )
            ]
        ),
    )

    assert _rows(
        store,
        "SELECT rowid FROM memory_fts WHERE memory_fts MATCH ?",
        ("sabbatical",),
    )
    memory_id = _rows(store, "SELECT id FROM memories")[0]["id"]
    with store._connect() as connection:
        connection.execute(
            "UPDATE memories SET content = ? WHERE id = ?",
            ("Replacement phrase", memory_id),
        )
    assert not _rows(
        store,
        "SELECT rowid FROM memory_fts WHERE memory_fts MATCH ?",
        ("sabbatical",),
    )
    assert _rows(
        store,
        "SELECT rowid FROM memory_fts WHERE memory_fts MATCH ?",
        ("replacement",),
    )
    with store._connect() as connection:
        connection.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
    assert not _rows(
        store,
        "SELECT rowid FROM memory_fts WHERE memory_fts MATCH ?",
        ("replacement",),
    )


def test_local_llm_queue_is_fifo_and_continues_after_job_error():
    from gossipmemo.queue import LLMQueue

    async def scenario():
        queue = LLMQueue()
        started: list[int] = []

        async def work(value: int) -> int:
            started.append(value)
            await asyncio.sleep(0)
            return value

        await queue.start()
        futures = [queue.submit(f"work-{value}", work, value) for value in range(4)]
        assert await asyncio.gather(*futures) == [0, 1, 2, 3]
        assert started == [0, 1, 2, 3]

        async def fail() -> None:
            raise ValueError("expected job failure")

        failed = asyncio.create_task(queue.submit("fail", fail))
        next_job = asyncio.create_task(queue.submit("after-failure", work, 4))
        with pytest.raises(ValueError, match="expected job failure"):
            await failed
        assert await next_job == 4
        await queue.stop()

    asyncio.run(scenario())


def test_fastapi_lifespan_ingest_wait_and_query(store):
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")

    from gossipmemo.models import (
        ExtractionResult,
        ExtractedPerson,
        ExtractedMemory,
        PersonReasoningResult,
        RelationshipReasoningResult,
    )
    from gossipmemo.app import create_app
    from gossipmemo.config import Settings
    from gossipmemo.world import SocialMemoryWorld

    class FakeModel:
        configured = True

        async def extract(self, message):
            del message
            return ExtractionResult(
                people=[ExtractedPerson(ref="bob", display_name="Bob")],
                memories=[
                    ExtractedMemory(
                        content="Bob prefers tea.",
                        basis="stated",
                        people=["bob"],
                    )
                ],
            )

        async def reason_person(self, person, memories):
            del person, memories
            return PersonReasoningResult(profile_card={"summary": "likes tea"})

        async def reason_relationship(self, relationship, memories):
            del relationship, memories
            return RelationshipReasoningResult()

        async def synthesize(self, question, context):
            assert question == "What does Bob prefer?"
            assert any(memory.content == "Bob prefers tea." for memory in context.memories)
            return "Bob prefers tea."

    async def scenario():
        world = SocialMemoryWorld(
            store, FakeModel(), extraction_batch_timeout_seconds=0.001
        )
        app = create_app(
            settings=Settings(
                database_path=store.path,
                llm_base_url="http://llm.test/v1",
                llm_api_key="test-key",
                llm_model="test-model",
            ),
            world=world,
        )

        # FastAPI runs synchronous dependencies in AnyIO's worker pool.  The
        # restricted test runner has no usable worker threads, so override the
        # app's private auth closure with an async no-op; this still exercises
        # the real lifespan, routing, validation, queue, and store behavior.
        async def allow_request():
            return None

        authorize = next(
            dependency.call
            for route in app.routes
            if getattr(route, "dependant", None)
            for dependency in route.dependant.dependencies
        )
        app.dependency_overrides[authorize] = allow_request
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                ingest = await client.post(
                    "/v1/spaces/personal/ingest",
                    json={
                        "messages": [
                            {
                                "author": "user",
                                "content": "Bob likes tea.",
                                "source": {
                                    "provider": "test",
                                    "conversation_key": "chat-1",
                                    "item_id": "turn-1",
                                },
                            }
                        ]
                    },
                )
                assert ingest.status_code == 202
                payload = ingest.json()
                assert payload["status"] == "accepted"
                message_id = payload["message_ids"][0]

                for _ in range(100):
                    row = _rows(
                        store,
                        "SELECT extraction_state FROM messages WHERE id = ?",
                        (message_id,),
                    )[0]
                    if row["extraction_state"] == "completed":
                        break
                    await asyncio.sleep(0)
                assert row["extraction_state"] == "completed"

                query = await client.post(
                    "/v1/spaces/personal/query",
                    json={
                        "question": "What does Bob prefer?",
                        "people": ["Bob"],
                        "include_evidence": True,
                    },
                )
                assert query.status_code == 200
                payload = query.json()
                assert payload["answer"] == "Bob prefers tea."
                assert payload["memories"][0]["content"] == "Bob prefers tea."

    asyncio.run(scenario())


def test_extraction_keeps_people_evidence_and_relationship(store):
    receipt = store.record_messages("personal", [_message()])[0]
    result = ExtractionResult(
        people=[
            ExtractedPerson(ref="alice", display_name="Alice"),
            ExtractedPerson(ref="bob", display_name="Bob"),
        ],
        memories=[
            ExtractedMemory(
                content="Bob may leave soon",
                kind="situation",
                basis="reported",
                people=["bob", "alice"],
                relationships=[
                    ExtractedRelationship(
                        person_a_ref="alice",
                        person_b_ref="bob",
                        facets=[{"kind": "coworker"}],
                    )
                ],
            )
        ],
    )

    affected_people, affected_relationships = store.apply_extraction(
        "personal", _batch(store, receipt), result
    )
    context = store.read(
        "personal",
        QueryRequest(
            question="what do we know?",
            people=["Alice", "Bob"],
            include_relationships=True,
            include_evidence=True,
        ),
    )

    assert len(affected_people) == 2  # Message authors are not graph people.
    assert len(affected_relationships) == 1
    assert len(context.memories) == 1
    memory = context.memories[0]
    assert memory.basis == "reported"
    assert {person["name"] for person in memory.people} == {"Alice", "Bob"}
    assert memory.evidence == [
        {
            "message_id": receipt,
            "batch_id": memory.evidence[0]["batch_id"],
            "text": _message().content,
            "author": "user",
            "occurred_at": "2026-08-09T12:00:00+00:00",
            "source_provider": "agent_chat",
        }
    ]
    relationship = context.relationships[0]
    assert relationship.facets == [{"kind": "coworker"}]
    assert {relationship.person_a_id, relationship.person_b_id} == {
        next(person.id for person in context.people if person.display_name == "Alice"),
        next(person.id for person in context.people if person.display_name == "Bob"),
    }


def test_memory_people_is_plain_person_association_and_alias_resolves(store):
    receipt = store.record_messages("personal", [_message()])[0]
    store.apply_extraction(
        "personal",
        _batch(store, receipt),
        ExtractionResult(
            people=[ExtractedPerson(ref="alice", display_name="Alice", aliases=["Al"])],
            memories=[
                ExtractedMemory(
                    content="Al is taking Friday off.", basis="stated", people=["alice"]
                )
            ],
        ),
    )

    rows = _rows(
        store,
        "SELECT mp.memory_id, mp.person_id FROM memory_people mp "
        "JOIN person_aliases pa ON pa.person_id = mp.person_id "
        "WHERE pa.normalized_value = 'al'",
    )
    assert len(rows) == 1
    assert set(rows[0].keys()) == {"memory_id", "person_id"}
    assert _rows(store, "PRAGMA table_info(memory_people)")
    assert "role" not in {
        row["name"] for row in _rows(store, "PRAGMA table_info(memory_people)")
    }

    memory = store.add_manual_memory(
        "personal",
        ManualMemoryRequest(content="Al likes coffee.", people=["Al"]),
    )
    linked = _rows(
        store,
        "SELECT person_id FROM memory_people WHERE memory_id = ?",
        (memory,),
    )
    assert [row["person_id"] for row in linked] == [rows[0]["person_id"]]


def test_same_alias_for_two_people_is_ambiguous_not_merged(store):
    receipt = store.record_messages("personal", [_message()])[0]
    store.apply_extraction(
        "personal",
        _batch(store, receipt),
        ExtractionResult(
            people=[
                ExtractedPerson(ref="one", display_name="One", aliases=["Alex"]),
                ExtractedPerson(ref="two", display_name="Two", aliases=["Alex"]),
            ],
            memories=[],
        ),
    )

    assert len(
        _rows(
            store,
            "SELECT person_id FROM person_aliases WHERE normalized_value = 'alex'",
        )
    ) == 2
    with pytest.raises(AmbiguousPersonError):
        store.add_manual_memory(
            "personal", ManualMemoryRequest(content="Alex did it.", people=["Alex"])
        )


def test_manual_memory_retract_updates_memory_people_once(
    store,
):
    memory_id = store.add_manual_memory(
        "personal",
        ManualMemoryRequest(
            content="Bob is taking a sabbatical.",
            kind="situation",
            people=["Bob"],
        ),
    )
    bob = store.read(
        "personal", QueryRequest(question="bob", people=["Bob"])
    ).people[0]
    assert bob.profile_source_updated_at is None
    assert bob.stale is True
    _, _, watermark = store.person_context("personal", bob.id)
    assert watermark is not None
    assert store.apply_person_reasoning(
        "personal",
        bob.id,
        expected_watermark=watermark,
        result=PersonReasoningResult(profile_card={"summary": "on sabbatical"}),
    )

    before_retract = _rows(
        store, "SELECT updated_at FROM memories WHERE id = ?", (memory_id,)
    )[0]["updated_at"]

    assert store.retract_memory("personal", memory_id) is True

    bob_after = store.read(
        "personal", QueryRequest(question="bob", people=["Bob"])
    ).people[0]
    assert bob_after.stale is True
    assert _rows(
        store, "SELECT updated_at FROM memories WHERE id = ?", (memory_id,)
    )[0]["updated_at"] >= before_retract
    assert _rows(
        store, "SELECT status FROM memories WHERE id = ?", (memory_id,)
    )[0]["status"] == "retracted"


def test_person_reasoning_uses_timestamp_as_compare_and_swap(store):
    memory_id = store.add_manual_memory(
        "personal",
        ManualMemoryRequest(
            content="Bob likes tea.",
            people=["Bob"],
        ),
    )
    bob = store.read(
        "personal", QueryRequest(question="bob", people=["Bob"])
    ).people[0]

    assert bob.profile_source_updated_at is None
    _, _, watermark = store.person_context("personal", bob.id)
    assert watermark is not None
    assert (
        store.apply_person_reasoning(
            "personal",
            bob.id,
            expected_watermark="1970-01-01T00:00:00+00:00",
            result=PersonReasoningResult(profile_card={"summary": "stale"}),
        )
        is False
    )
    assert store.apply_person_reasoning(
        "personal",
        bob.id,
        expected_watermark=watermark,
        result=PersonReasoningResult(
            profile_card={"summary": "likes tea"},
            inferred_memories=[
                InferredMemory(
                    content="Bob has a tea preference.",
                    source_memory_ids=[memory_id],
                )
            ],
        ),
    ) is True
    updated_with_inference = store.read(
        "personal", QueryRequest(question="bob", people=["Bob"])
    ).people[0]
    assert updated_with_inference.profile_card == {"summary": "likes tea"}
    assert updated_with_inference.profile_source_updated_at is not None
    assert updated_with_inference.stale is False
    assert _rows(
        store,
        "SELECT source_memory_id FROM memory_derivations WHERE derived_memory_id IN "
        "(SELECT id FROM memories WHERE content = ?)",
        ("Bob has a tea preference.",),
    )[0]["source_memory_id"] == memory_id
    assert (
        store.apply_person_reasoning(
            "personal",
            bob.id,
            expected_watermark=watermark,
            result=PersonReasoningResult(profile_card={"summary": "stale"}),
        )
        is False
    )


def test_same_display_name_references_are_not_automatically_merged(store):
    receipt = store.record_messages("personal", [_message()])[0]
    store.apply_extraction(
        "personal",
        _batch(store, receipt),
        ExtractionResult(
            people=[
                ExtractedPerson(ref="first", display_name="Alex"),
                ExtractedPerson(ref="second", display_name="Alex"),
            ],
            memories=[
                ExtractedMemory(
                    content="Two people named Alex were mentioned.",
                    basis="observed",
                    people=["first", "second"],
                )
            ],
        ),
    )

    alexes = _rows(
        store,
        "SELECT person_id AS id FROM person_aliases "
        "WHERE space_id = ? AND normalized_value = ?",
        ("personal", "alex"),
    )
    assert len(alexes) == 2
    with pytest.raises(AmbiguousPersonError):
        store.add_manual_memory(
            "personal",
            ManualMemoryRequest(
                content="An ambiguous Alex fact.",
                people=["Alex"],
            ),
        )


def test_person_reasoning_timestamp_compare_and_swap_is_atomic(store):
    source_id = store.add_manual_memory(
        "personal",
        ManualMemoryRequest(
            content="Bob has a source fact.",
            people=["Bob"],
        ),
    )
    bob = store.read(
        "personal", QueryRequest(question="bob", people=["Bob"])
    ).people[0]
    _, _, watermark = store.person_context("personal", bob.id)
    assert watermark is not None

    def apply(worker: int) -> bool:
        return store.apply_person_reasoning(
            "personal",
            bob.id,
            expected_watermark=watermark,
            result=PersonReasoningResult(
                profile_card={"worker": worker},
                inferred_memories=[
                    InferredMemory(
                        content=f"Bob inference {worker}",
                        source_memory_ids=[source_id],
                    )
                ],
            ),
        )

    with ThreadPoolExecutor(max_workers=2) as workers:
        outcomes = list(workers.map(apply, (1, 2)))

    assert sorted(outcomes) == [False, True]
    current = store.read(
        "personal", QueryRequest(question="bob", people=["Bob"])
    ).people[0]
    assert current.profile_source_updated_at is not None
    assert current.stale is False
    assert len(
        _rows(
            store,
            "SELECT id FROM memories WHERE basis = 'inferred' AND space_id = ?",
            ("personal",),
        )
    ) == 1
