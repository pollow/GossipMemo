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
    InferredMemoryActions,
    InferredMemoryRetraction,
    HypothesisActions,
    HypothesisEvidence,
    HypothesisTransition,
    HypothesisUpsert,
    ManualMemoryRequest,
    ExtractedMemory,
    MessageInput,
    ModelMessage,
    PersonReasoningResult,
    RelationshipReasoningResult,
    QueryRequest,
    ExtractedRelationship,
    SourceRef,
    UserModelReasoningResult,
    CoverageAuditPatch,
    CoverageCriterionPatch,
    CoverageBoundaryUpsert,
    GoalPlanningResult,
    LearningGoalUpsert,
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


def test_known_person_context_supports_later_alias_update(store):
    ids = store.record_messages("personal", [_message(content="Alex Wang")])
    batch_id = _batch(store, ids[0])
    store.apply_extraction(
        "personal",
        batch_id,
        ExtractionResult(
            people=[ExtractedPerson(ref="alex", display_name="Alex Wang")]
        ),
    )
    original_person_id = _rows(store, "SELECT id FROM people")[0]["id"]
    messages = [
        # ModelMessage is deliberately used here to represent an extraction target.
        ModelMessage(
            id="target",
            space_id="personal",
            author="user",
            content="以后管 Alex Wang 叫 AW",
            occurred_at="2026-08-09T12:00:00+00:00",
            source_provider="agent_chat",
        )
    ]
    catalog = store.load_known_people("personal", messages)
    assert catalog == [
        {
            "id": original_person_id,
            "display_name": "Alex Wang",
            "aliases": ["Alex Wang"],
        }
    ]

    alias_message_id = store.record_messages(
        "personal", [_message(content="以后管 Alex Wang 叫 AW")]
    )[0]
    store.apply_extraction(
        "personal",
        _batch(store, alias_message_id),
        ExtractionResult(
            people=[
                ExtractedPerson(
                    ref="alex", display_name="Alex Wang", aliases=["AW"]
                )
            ]
        ),
    )

    assert len(_rows(store, "SELECT id FROM people")) == 1
    assert store.match_people_in_text("personal", "AW")[0].id == original_person_id


def test_load_extraction_context_is_recent_per_conversation(store):
    def msg(content: str, conversation: str | None) -> MessageInput:
        return MessageInput(
            author="user", content=content,
            source=SourceRef(provider="agent_chat", conversation_key=conversation),
        )

    store.record_messages(
        "personal",
        [
            msg("old-1", "chat"),
            msg("old-2", "chat"),
            msg("old-3", "chat"),
            msg("other", "other-chat"),
            msg("no-context", None),
        ],
    )
    batch_ids = store.record_messages(
        "personal", [msg("target-1", "chat"), msg("target-2", "chat")]
    )
    batch_id = _batch(store, *batch_ids)
    context = store.load_extraction_context("personal", batch_id)
    assert [item.content for item in context] == ["old-2", "old-3"]


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


def test_extraction_comparisons_exclude_inferred_and_supersede_only_supplied_memory(store):
    original_receipt = store.record_messages(
        "personal", [_message(content="Alex likes tea.")]
    )[0]
    original_batch = _batch(store, original_receipt)
    store.apply_extraction(
        "personal",
        original_batch,
        ExtractionResult(
            people=[ExtractedPerson(ref="alex", display_name="Alex")],
            memories=[
                ExtractedMemory(
                    content="Alex likes tea.", basis="stated", people=["alex"]
                )
            ],
        ),
    )
    original_id = _rows(
        store, "SELECT id FROM memories WHERE basis = 'stated'"
    )[0]["id"]
    inferred_receipt = store.record_messages(
        "personal", [_message(content="Alex enjoys tea.")]
    )[0]
    inferred_batch = _batch(store, inferred_receipt)
    store.apply_extraction(
        "personal",
        inferred_batch,
        ExtractionResult(
            memories=[
                ExtractedMemory(
                    content="Alex enjoys tea.", basis="inferred", people=["Alex"]
                )
            ]
        ),
    )
    receipt = store.record_messages(
        "personal", [_message(content="Actually Alex prefers coffee now.")]
    )[0]
    batch = _batch(store, receipt)
    comparisons = store.load_extraction_comparisons("personal", batch)
    assert [memory.id for memory in comparisons] == [original_id]
    store.apply_extraction(
        "personal",
        batch,
        ExtractionResult(
            memories=[
                ExtractedMemory(
                    content="Alex prefers coffee now.",
                    basis="stated",
                    people=["Alex"],
                    supersedes_memory_id=original_id,
                )
            ]
        ),
        {original_id},
    )
    rows = _rows(
        store,
        "SELECT id, status, supersedes_memory_id, source_batch_id FROM memories "
        "WHERE basis = 'stated' ORDER BY created_at",
    )
    assert rows[-1]["supersedes_memory_id"] == original_id
    assert rows[-1]["source_batch_id"] == batch
    assert rows[0]["status"] == "superseded"
    assert _rows(
        store, "SELECT person_id FROM memory_people WHERE memory_id = ?", (rows[-1]["id"],)
    )


def test_extraction_ignores_unseen_supersede_id_without_losing_new_memory(store):
    old_id = store.add_manual_memory(
        "personal", ManualMemoryRequest(content="Deus likes tea.", about_user=True)
    )
    receipt = store.record_messages(
        "personal", [_message(content="I prefer coffee now.")]
    )[0]
    batch = _batch(store, receipt)
    store.apply_extraction(
        "personal",
        batch,
        ExtractionResult(
            memories=[
                ExtractedMemory(
                    content="Deus prefers coffee now.",
                    basis="stated",
                    about_user=True,
                    supersedes_memory_id=old_id,
                )
            ]
        ),
        comparison_memory_ids=set(),
    )
    old = _rows(store, "SELECT status FROM memories WHERE id = ?", (old_id,))[0]
    new = _rows(
        store, "SELECT supersedes_memory_id FROM memories WHERE source_batch_id = ?", (batch,)
    )[0]
    assert old["status"] == "active"
    assert new["supersedes_memory_id"] is None


def test_extraction_similar_guard_keeps_polarity_and_drops_in_batch_repeat(store):
    first = store.record_messages("personal", [_message(content="I strongly prefer quiet mornings.")])[0]
    first_batch = _batch(store, first)
    store.apply_extraction(
        "personal",
        first_batch,
        ExtractionResult(
            memories=[
                ExtractedMemory(
                    content="I strongly prefer quiet mornings.",
                    basis="stated",
                    about_user=True,
                ),
                ExtractedMemory(
                    content="I drink 1 cup of coffee.",
                    basis="stated",
                    about_user=True,
                ),
            ]
        ),
    )
    old_ids = {row["id"] for row in _rows(store, "SELECT id FROM memories")}
    second = store.record_messages("personal", [_message(content="I strongly prefer quiet mornings.")])[0]
    second_batch = _batch(store, second)
    store.apply_extraction(
        "personal",
        second_batch,
        ExtractionResult(
            memories=[
                ExtractedMemory(
                    content="I strongly prefer quiet mornings",
                    basis="stated",
                    about_user=True,
                ),
                ExtractedMemory(
                    content="I strongly do not prefer quiet mornings.",
                    basis="stated",
                    about_user=True,
                ),
                ExtractedMemory(
                    content="I strongly do not prefer quiet mornings",
                    basis="stated",
                    about_user=True,
                ),
                ExtractedMemory(
                    content="I drink 2 cups of coffee.",
                    basis="stated",
                    about_user=True,
                ),
            ]
        ),
        old_ids,
    )
    rows = _rows(store, "SELECT content FROM memories WHERE status = 'active'")
    assert {row["content"] for row in rows} == {
        "I strongly prefer quiet mornings.",
        "I strongly do not prefer quiet mornings.",
        "I drink 1 cup of coffee.",
        "I drink 2 cups of coffee.",
    }


def test_memory_view_preserves_temporal_bounds_in_read_and_user_model_context(store):
    receipt = store.record_messages("personal", [_message()])[0]
    store.apply_extraction(
        "personal", _batch(store, receipt),
        ExtractionResult(memories=[ExtractedMemory(
            content="I am traveling this week.", basis="stated", about_user=True,
            valid_from="2026-08-10", valid_to="2026-08-16",
        )]),
    )

    context = store.read("personal", QueryRequest(question="traveling"))
    assert [(memory.valid_from, memory.valid_to) for memory in context.memories] == [
        ("2026-08-10", "2026-08-16")
    ]
    user_context = store.user_model_context("personal")
    assert user_context is not None
    assert [(memory.valid_from, memory.valid_to) for memory in user_context[1]] == [
        ("2026-08-10", "2026-08-16")
    ]


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

        async def extract(self, message, context=(), known_people=(), comparison_memories=()):
            del message, context, known_people, comparison_memories
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


def test_automatic_ambiguous_person_link_is_skipped_but_memory_completes(store):
    store.apply_extraction(
        "personal", _batch(store, store.record_messages("personal", [_message()])[0]),
        ExtractionResult(people=[
            ExtractedPerson(ref="one", display_name="One", aliases=["Alex"]),
            ExtractedPerson(ref="two", display_name="Two", aliases=["Alex"]),
        ]),
    )
    message_id = store.record_messages("personal", [_message(content="Alex called.")])[0]
    batch_id = _batch(store, message_id)
    store.apply_extraction(
        "personal", batch_id,
        ExtractionResult(
            people=[ExtractedPerson(ref="alex", display_name="Alex")],
            memories=[ExtractedMemory(content="Alex called.", basis="reported", people=["Alex"])],
        ),
    )
    assert _rows(store, "SELECT extraction_state FROM messages WHERE id = ?", (message_id,))[0][0] == "completed"
    assert _rows(store, "SELECT COUNT(*) AS n FROM memories")[0]["n"] == 1
    assert _rows(store, "SELECT COUNT(*) AS n FROM memory_people")[0]["n"] == 0
    assert _rows(store, "SELECT COUNT(*) AS n FROM people")[0]["n"] == 2


def test_automatic_known_person_ids_resolve_relationships(store):
    a = store.add_manual_memory("personal", ManualMemoryRequest(content="A exists.", people=["A"]))
    del a
    b = store.add_manual_memory("personal", ManualMemoryRequest(content="B exists.", people=["B"]))
    del b
    people = _rows(store, "SELECT id, display_name FROM people ORDER BY display_name")
    ids = {row["display_name"]: row["id"] for row in people}
    message_id = store.record_messages("personal", [_message(content="A and B.")])[0]
    store.apply_extraction(
        "personal", _batch(store, message_id),
        ExtractionResult(memories=[ExtractedMemory(
            content="A and B.", basis="observed", people=[ids["A"], ids["B"]],
            relationships=[ExtractedRelationship(person_a_ref=ids["A"], person_b_ref=ids["B"])],
        )]),
    )
    relationship = _rows(store, "SELECT * FROM relationships")[0]
    assert {relationship["person_a_id"], relationship["person_b_id"]} == {ids["A"], ids["B"]}
    assert _rows(store, "SELECT COUNT(*) AS n FROM memory_relationships")[0]["n"] == 1


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


def test_inferred_reasoning_is_reconciled_without_recursive_inputs(store):
    source_id = store.add_manual_memory(
        "personal", ManualMemoryRequest(content="Bob repeatedly follows up on plans.", people=["Bob"])
    )
    bob = store.read("personal", QueryRequest(question="bob", people=["Bob"])).people[0]
    _, memories, watermark = store.person_context("personal", bob.id)
    assert all(memory.basis != "inferred" for memory in memories)
    result = PersonReasoningResult(
        profile_card={"summary": "follows up"},
        inferred_memories=[InferredMemory(
            content="Bob reliably follows up on plans.", source_memory_ids=[source_id]
        )],
    )
    assert store.apply_person_reasoning("personal", bob.id, watermark, result)
    first = _rows(store, "SELECT id FROM memories WHERE basis = 'inferred'")[0]["id"]
    _, _, unchanged_watermark = store.person_context("personal", bob.id)
    assert unchanged_watermark == watermark
    assert store.stale_entities()[0] == []
    # A near-identical regenerated result reuses the durable inference and adds
    # its newly supplied derivation instead of creating another Memory.
    supporting_source = store.add_manual_memory(
        "personal", ManualMemoryRequest(content="Bob follows up after meetings.", people=["Bob"])
    )
    _, _, supporting_watermark = store.person_context("personal", bob.id)
    assert store.apply_person_reasoning(
        "personal", bob.id, supporting_watermark,
        PersonReasoningResult(profile_card={}, inferred_memories=[
            InferredMemory(
                content="Bob reliably follows up on plans",
                source_memory_ids=[supporting_source],
            )
        ]),
    )
    assert _rows(
        store,
        "SELECT id FROM memories WHERE basis = 'inferred' AND status = 'active'",
    )[0]["id"] == first
    assert _rows(
        store,
        "SELECT source_memory_id FROM memory_derivations WHERE derived_memory_id = ? "
        "AND source_memory_id = ?",
        (first, supporting_source),
    )
    # An inferred source cannot be reused, but its omission is deliberately a
    # no-op: existing hypotheses remain active until explicitly retracted.
    new_source = store.add_manual_memory(
        "personal", ManualMemoryRequest(content="Bob also follows up on decisions.", people=["Bob"])
    )
    _, _, next_watermark = store.person_context("personal", bob.id)
    assert store.apply_person_reasoning(
        "personal", bob.id, next_watermark,
        PersonReasoningResult(profile_card={}, inferred_memories=[
            InferredMemory(content="Bob reliably follows up on plans.", source_memory_ids=[first])
        ]),
    )
    assert _rows(store, "SELECT status FROM memories WHERE id = ?", (first,))[0]["status"] == "active"
    store.apply_inferred_memory_actions(
        "personal", "person", bob.id, {new_source}, {first},
        InferredMemoryActions(
            retractions=[InferredMemoryRetraction(memory_id=first, reason="support was reconsidered")],
        ),
    )
    assert _rows(store, "SELECT status FROM memories WHERE id = ?", (first,))[0]["status"] == "retracted"
    # A changed output is a new active record with derivations to current source.
    latest_source = store.add_manual_memory(
        "personal", ManualMemoryRequest(content="Bob follows up on outcomes too.", people=["Bob"])
    )
    _, _, latest_watermark = store.person_context("personal", bob.id)
    assert store.apply_person_reasoning(
        "personal", bob.id, latest_watermark,
        PersonReasoningResult(profile_card={}, inferred_memories=[
            InferredMemory(content="Bob reliably follows up on plans and decisions.", source_memory_ids=[latest_source])
        ]),
    )
    active = _rows(store, "SELECT id FROM memories WHERE basis = 'inferred' AND status = 'active'")
    assert len(active) == 1 and active[0]["id"] != first


def test_inferred_memory_is_not_a_hypothesis(store):
    source_id = store.add_manual_memory(
        "personal", ManualMemoryRequest(content="Bob keeps project notes.", people=["Bob"])
    )
    bob = store.read("personal", QueryRequest(question="bob", people=["Bob"])).people[0]
    _, _, watermark = store.person_context("personal", bob.id)
    assert store.apply_person_reasoning(
        "personal", bob.id, watermark,
        PersonReasoningResult(inferred_memories=[InferredMemory(
            content="Bob is organized.", source_memory_ids=[source_id]
        )]),
    )
    inferred_id = _rows(store, "SELECT id FROM memories WHERE basis = 'inferred'")[0]["id"]
    assert _rows(store, "SELECT id FROM hypotheses") == []
    # Not present in the supplied context: retraction is ignored.
    store.apply_inferred_memory_actions(
        "personal", "person", bob.id, {source_id}, set(),
        InferredMemoryActions(retractions=[InferredMemoryRetraction(
            memory_id=inferred_id, reason="not enough evidence"
        )]),
    )
    assert _rows(store, "SELECT status FROM memories WHERE id = ?", (inferred_id,))[0]["status"] == "active"


def test_person_reasoning_applies_hypothesis_actions_in_same_transaction(store):
    source_id = store.add_manual_memory(
        "personal", ManualMemoryRequest(content="Bob keeps project notes.", people=["Bob"])
    )
    bob = store.read(
        "personal", QueryRequest(question="bob", people=["Bob"])
    ).people[0]
    _, _, watermark = store.person_context("personal", bob.id)
    result = PersonReasoningResult(
        profile_card={"summary": "Bob keeps notes."},
        hypothesis_actions=HypothesisActions(upserts=[HypothesisUpsert(
            content="Bob may use notes to prepare for decisions.",
            confidence="low",
            evidence=[HypothesisEvidence(memory_id=source_id)],
        )]),
    )

    assert store.apply_person_reasoning("personal", bob.id, watermark, result)
    hypothesis = _rows(store, "SELECT owner_kind, owner_id FROM hypotheses")[0]
    assert (hypothesis["owner_kind"], hypothesis["owner_id"]) == ("person", bob.id)


def test_hypothesis_actions_require_active_context_evidence_and_scoped_transition(store):
    source_id = store.add_manual_memory(
        "personal", ManualMemoryRequest(content="Bob keeps project notes.", people=["Bob"])
    )
    bob = store.read("personal", QueryRequest(question="bob", people=["Bob"])).people[0]
    store.apply_inferred_memory_actions(
        "personal", "person", bob.id, {source_id}, set(),
        InferredMemoryActions(upserts=[InferredMemory(
            content="Bob is organized.", source_memory_ids=[source_id]
        )]),
    )
    inferred_id = _rows(store, "SELECT id FROM memories WHERE basis = 'inferred'")[0]["id"]
    store.apply_hypothesis_actions(
        "personal", "person", bob.id, {inferred_id}, set(),
        HypothesisActions(upserts=[HypothesisUpsert(
            content="This must not persist.",
            evidence=[HypothesisEvidence(memory_id=inferred_id)],
        )]),
    )
    assert _rows(store, "SELECT id FROM hypotheses") == []
    actions = HypothesisActions(upserts=[HypothesisUpsert(
        content="Bob may be preparing for a role change.",
        confidence="medium",
        evidence=[HypothesisEvidence(memory_id=source_id, role="support")],
    )])
    store.apply_hypothesis_actions("personal", "person", bob.id, {source_id}, set(), actions)
    hypothesis = _rows(store, "SELECT * FROM hypotheses")[0]
    assert hypothesis["status"] == "open"
    assert hypothesis["owner_kind"] == "person" and hypothesis["owner_id"] == bob.id
    assert _rows(
        store, "SELECT memory_id, role FROM hypothesis_evidence WHERE hypothesis_id = ?", (hypothesis["id"],)
    )[0]["role"] == "support"
    # A transition omitted from its supplied hypothesis context is a no-op.
    store.apply_hypothesis_actions(
        "personal", "person", bob.id, {source_id}, set(),
        HypothesisActions(transitions=[HypothesisTransition(
            hypothesis_id=hypothesis["id"], status="rejected", reason="insufficient evidence"
        )]),
    )
    assert _rows(store, "SELECT status FROM hypotheses WHERE id = ?", (hypothesis["id"],))[0]["status"] == "open"
    store.apply_hypothesis_actions(
        "personal", "person", bob.id, {source_id}, {hypothesis["id"]},
        HypothesisActions(
            transitions=[HypothesisTransition(
                hypothesis_id=hypothesis["id"], status="rejected", reason="insufficient evidence"
            )],
        ),
    )
    assert _rows(store, "SELECT status FROM hypotheses WHERE id = ?", (hypothesis["id"],))[0]["status"] == "rejected"
    # Closed hypotheses cannot be upserted or transitioned again, even when
    # the caller supplies their ID in context.
    store.apply_hypothesis_actions(
        "personal", "person", bob.id, {source_id}, {hypothesis["id"]},
        HypothesisActions(upserts=[HypothesisUpsert(
            hypothesis_id=hypothesis["id"], content="Changed claim", confidence="high",
            evidence=[HypothesisEvidence(memory_id=source_id)],
        )]),
    )
    assert _rows(store, "SELECT content FROM hypotheses WHERE id = ?", (hypothesis["id"],))[0]["content"] != "Changed claim"


def test_hypothesis_promotion_requires_active_owned_memory(store):
    source_id = store.add_manual_memory(
        "personal", ManualMemoryRequest(content="Bob keeps project notes.", people=["Bob"])
    )
    bob = store.read("personal", QueryRequest(question="bob", people=["Bob"])).people[0]
    store.apply_hypothesis_actions(
        "personal", "person", bob.id, {source_id}, set(),
        HypothesisActions(upserts=[HypothesisUpsert(
            content="Bob may be changing roles.", evidence=[HypothesisEvidence(memory_id=source_id)]
        )]),
    )
    hypothesis_id = _rows(store, "SELECT id FROM hypotheses")[0]["id"]
    store.apply_hypothesis_actions(
        "personal", "person", bob.id, {source_id}, {hypothesis_id},
        HypothesisActions(transitions=[HypothesisTransition(
            hypothesis_id=hypothesis_id, status="promoted", reason="confirmed"
        )]),
    )
    assert _rows(store, "SELECT status FROM hypotheses WHERE id = ?", (hypothesis_id,))[0]["status"] == "open"
    store.apply_hypothesis_actions(
        "personal", "person", bob.id, {source_id}, {hypothesis_id},
        HypothesisActions(transitions=[HypothesisTransition(
            hypothesis_id=hypothesis_id, status="promoted", reason="confirmed", promoted_memory_id=source_id
        )]),
    )
    promoted = _rows(store, "SELECT status, promoted_memory_id FROM hypotheses WHERE id = ?", (hypothesis_id,))[0]
    assert promoted["status"] == "promoted" and promoted["promoted_memory_id"] == source_id


def test_inferred_outputs_are_scoped_to_each_person_target(store):
    first_source = store.add_manual_memory(
        "personal", ManualMemoryRequest(content="Alex plans carefully.", people=["Alex"])
    )
    second_source = store.add_manual_memory(
        "personal", ManualMemoryRequest(content="Bea plans carefully.", people=["Bea"])
    )
    alex = store.read("personal", QueryRequest(question="alex", people=["Alex"])).people[0]
    bea = store.read("personal", QueryRequest(question="bea", people=["Bea"])).people[0]
    _, _, alex_watermark = store.person_context("personal", alex.id)
    _, _, bea_watermark = store.person_context("personal", bea.id)
    shared = "This person plans carefully."
    assert store.apply_person_reasoning(
        "personal", alex.id, alex_watermark,
        PersonReasoningResult(inferred_memories=[InferredMemory(content=shared, source_memory_ids=[first_source])]),
    )
    assert store.apply_person_reasoning(
        "personal", bea.id, bea_watermark,
        PersonReasoningResult(inferred_memories=[InferredMemory(content=shared, source_memory_ids=[second_source])]),
    )
    assert len(_rows(store, "SELECT id FROM memories WHERE basis = 'inferred' AND status = 'active'")) == 2


def test_relationship_inference_accepts_only_supplied_non_inferred_sources(store):
    receipt = store.record_messages("personal", [_message()])[0]
    affected_relationships: set[str]
    _, affected_relationships = store.apply_extraction(
        "personal",
        _batch(store, receipt),
        ExtractionResult(
            people=[
                ExtractedPerson(ref="alice", display_name="Alice"),
                ExtractedPerson(ref="bob", display_name="Bob"),
            ],
            memories=[
                ExtractedMemory(
                    content="Alice and Bob resolve disagreements directly.",
                    basis="stated",
                    people=["alice", "bob"],
                    relationships=[
                        ExtractedRelationship(
                            person_a_ref="alice", person_b_ref="bob"
                        )
                    ],
                )
            ],
        ),
    )
    relationship_id = next(iter(affected_relationships))
    _, memories, watermark = store.relationship_context("personal", relationship_id)
    source_id = memories[0].id
    assert store.apply_relationship_reasoning(
        "personal",
        relationship_id,
        watermark,
        RelationshipReasoningResult(
            summary="They address friction directly.",
            inferred_memories=[
                InferredMemory(
                    content="Alice and Bob tend to address friction directly.",
                    source_memory_ids=[source_id],
                )
            ],
        ),
    )
    inferred_id = _rows(
        store,
        "SELECT id FROM memories WHERE basis = 'inferred' AND status = 'active'",
    )[0]["id"]
    _, next_memories, next_watermark = store.relationship_context(
        "personal", relationship_id
    )
    assert all(memory.id != inferred_id for memory in next_memories)
    assert next_watermark == watermark


def test_coverage_map_is_initialized_and_goals_require_map_refs(store):
    store.ensure_space("personal")
    coverage, memories, hypotheses, pending = store.coverage_context("personal")
    assert len(coverage.criteria) == 20
    assert {item["level"] for item in coverage.criteria.values()} == {"unknown"}
    assert not memories and not hypotheses and not pending

    memory_id = store.add_manual_memory(
        "personal", ManualMemoryRequest(content="I grew up near the coast.", about_user=True)
    )
    coverage, memories, _, _ = store.coverage_context("personal")
    assert [memory.id for memory in memories] == [memory_id]
    assert store.apply_coverage_audit(
        "personal", coverage.source_watermark,
        coverage.source_cursor_id,
        CoverageAuditPatch(
            criteria=[CoverageCriterionPatch(criterion_id="M1", level="grounded", known_state="Early place is known", evidence_memory_ids=[memory_id])],
            boundary_upserts=[CoverageBoundaryUpsert(kind="blind_spot", summary="Childhood detail remains open", criterion_refs=["M1"])],
        ),
        {memory_id}, set(), set(),
    )
    updated, _, _, _ = store.coverage_context("personal")
    assert updated.criteria["M1"]["level"] == "grounded"
    store.apply_goal_planning(
        "personal",
        updated.revision, GoalPlanningResult(upserts=[LearningGoalUpsert(prompt="Would you like to share a coastal memory, or skip it?", rationale="Optional origin context", criteria_refs=["M1"], boundary_ids=[updated.boundaries[0].id])]),
        set(),
    )
    _, _, goals, _ = store.learning_goal_context("personal")
    assert len(goals) == 1


def test_coverage_cursor_handles_equal_timestamps_and_prunes_retracted_evidence(store):
    ids = [store.add_manual_memory("personal", ManualMemoryRequest(content=f"scene {index}", about_user=True)) for index in range(3)]
    with store._connect() as connection:
        connection.execute("UPDATE memories SET updated_at = ? WHERE space_id = ?", ("2026-01-01T00:00:00+00:00", "personal"))
    seen = []
    for _ in ids:
        coverage, memories, _, _ = store.coverage_context("personal", limit=1)
        seen.extend(memory.id for memory in memories)
        assert store.apply_coverage_audit("personal", coverage.source_watermark, coverage.source_cursor_id, CoverageAuditPatch(criteria=[CoverageCriterionPatch(criterion_id="M6", level="grounded", known_state="scene", evidence_memory_ids=[memories[0].id])]), {memories[0].id}, {boundary.id for boundary in coverage.boundaries}, set())
    assert set(seen) == set(ids)
    assert store.retract_memory("personal", ids[-1])
    coverage, memories, _, _ = store.coverage_context("personal")
    assert ids[-1] in [memory.id for memory in memories]
    assert store.apply_coverage_audit("personal", coverage.source_watermark, coverage.source_cursor_id, CoverageAuditPatch(), {memory.id for memory in memories}, {boundary.id for boundary in coverage.boundaries}, set())
    updated, _, _, _ = store.coverage_context("personal")
    assert ids[-1] not in updated.criteria["M6"]["evidence_memory_ids"]


def test_goal_planning_uses_coverage_revision_cas(store):
    store.ensure_space("personal")
    coverage, _, _, _ = store.coverage_context("personal")
    assert not store.apply_goal_planning("personal", coverage.revision + 1, GoalPlanningResult(), set())


def test_stale_coverage_restart_detects_same_timestamp_after_cursor(store):
    first = store.add_manual_memory("personal", ManualMemoryRequest(content="first", about_user=False))
    second = store.add_manual_memory("personal", ManualMemoryRequest(content="second", about_user=False))
    with store._connect() as connection:
        connection.execute("UPDATE memories SET updated_at = ? WHERE id IN (?, ?)", ("2026-02-01T00:00:00+00:00", first, second))
    coverage, memories, _, _ = store.coverage_context("personal", limit=1)
    assert store.apply_coverage_audit("personal", coverage.source_watermark, coverage.source_cursor_id, CoverageAuditPatch(), {memories[0].id}, set(), set())
    # This models a process restart: stale discovery must notice the second row
    # even though it shares the persisted timestamp.
    assert store.stale_coverage_spaces() == ["personal"]
