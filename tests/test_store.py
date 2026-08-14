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
    PersonLink,
    PersonReasoningResult,
    QueryRequest,
    ExtractedRelationship,
    SourceRef,
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
                        people=[PersonLink(ref="bob", role="subject")],
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


def test_extraction_keeps_roles_evidence_and_relationship(store):
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
                people=[
                        PersonLink(ref="bob", role="subject"),
                        PersonLink(ref="alice", role="asserter"),
                ],
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
    assert {person["role"] for person in memory.people} == {
        "subject",
        "asserter",
    }
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


def test_manual_memory_retract_increments_person_revision_once_even_with_multiple_roles(
    store,
):
    memory_id = store.add_manual_memory(
        "personal",
        ManualMemoryRequest(
            content="Bob is taking a sabbatical.",
            kind="situation",
            people=[
                PersonLink(ref="Bob", role="subject"),
                PersonLink(ref="Bob", role="asserter"),
            ],
        ),
    )
    bob = store.read(
        "personal", QueryRequest(question="bob", people=["Bob"])
    ).people[0]
    assert bob.memory_revision == 1
    assert bob.stale is True

    assert store.retract_memory("personal", memory_id) is True

    bob_after = store.read(
        "personal", QueryRequest(question="bob", people=["Bob"])
    ).people[0]
    assert bob_after.memory_revision == 2
    assert _rows(
        store, "SELECT status FROM memories WHERE id = ?", (memory_id,)
    )[0]["status"] == "retracted"


def test_person_reasoning_uses_revision_as_compare_and_swap(store):
    memory_id = store.add_manual_memory(
        "personal",
        ManualMemoryRequest(
            content="Bob likes tea.",
            people=[PersonLink(ref="Bob", role="subject")],
        ),
    )
    bob = store.read(
        "personal", QueryRequest(question="bob", people=["Bob"])
    ).people[0]

    assert bob.memory_revision == 1
    assert bob.profile_memory_revision == 0
    assert (
        store.apply_person_reasoning(
            "personal",
            bob.id,
            expected_revision=0,
            result=PersonReasoningResult(profile_card={"summary": "stale"}),
        )
        is False
    )
    assert store.apply_person_reasoning(
        "personal",
        bob.id,
        expected_revision=1,
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
    assert updated_with_inference.memory_revision == 2
    assert updated_with_inference.profile_memory_revision == 2
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
            expected_revision=1,
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
                    people=[
                        PersonLink(ref="first", role="participant"),
                        PersonLink(ref="second", role="participant"),
                    ],
                )
            ],
        ),
    )

    alexes = _rows(
        store,
        "SELECT id FROM people WHERE space_id = ? AND normalized_name = ?",
        ("personal", "alex"),
    )
    assert len(alexes) == 2
    with pytest.raises(AmbiguousPersonError):
        store.add_manual_memory(
            "personal",
            ManualMemoryRequest(
                content="An ambiguous Alex fact.",
                people=[PersonLink(ref="Alex", role="subject")],
            ),
        )


def test_person_reasoning_revision_compare_and_swap_is_atomic(store):
    source_id = store.add_manual_memory(
        "personal",
        ManualMemoryRequest(
            content="Bob has a source fact.",
            people=[PersonLink(ref="Bob", role="subject")],
        ),
    )
    bob = store.read(
        "personal", QueryRequest(question="bob", people=["Bob"])
    ).people[0]

    def apply(worker: int) -> bool:
        return store.apply_person_reasoning(
            "personal",
            bob.id,
            expected_revision=1,
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
    assert current.memory_revision == 2
    assert len(
        _rows(
            store,
            "SELECT id FROM memories WHERE basis = 'inferred' AND space_id = ?",
            ("personal",),
        )
    ) == 1
