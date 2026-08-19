from __future__ import annotations

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from gossipmemo.models import (
    COVERAGE_ROOTS,
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
    ExtractedCoverageAudit,
    ExtractedCoverageEntry,
    ExtractedCoverageEntryEdit,
    GoalPlanningResult,
    LearningGoalTransition,
    LearningGoalUpsert,
)
from gossipmemo.store import AmbiguousPersonError, SqliteWorldStore


@pytest.fixture
def store(tmp_path):
    world = SqliteWorldStore(tmp_path / "world.db")
    world.initialize()
    return world


def test_initialize_seeds_coverage_roots_for_an_existing_space(tmp_path):
    """A space that predates its roots must not be silently unauditable."""
    path = tmp_path / "world.db"
    first = SqliteWorldStore(path)
    first.initialize()
    first.ensure_space("personal")
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM coverage_roots")

    SqliteWorldStore(path).initialize()

    with sqlite3.connect(path) as connection:
        roots = {
            row[0]
            for row in connection.execute(
                "SELECT root FROM coverage_roots WHERE space_id = ?", ("personal",)
            )
        }
    assert roots == set(COVERAGE_ROOTS)


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
    first = store.record_messages("personal", [_message(
        content="I strongly prefer quiet mornings.")])[0]
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
    second = store.record_messages("personal", [_message(
        content="I strongly prefer quiet mornings.")])[0]
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


def test_recall_memories_filters_by_about_user_and_person_and_handles_empty_query(store):
    store.add_manual_memory(
        "personal",
        ManualMemoryRequest(content="Distinctive tea preference",
                            people=["Alice"], about_user=True),
    )
    store.add_manual_memory(
        "personal",
        ManualMemoryRequest(content="Distinctive tea event", people=["Bob"], about_user=False),
    )
    alice_id = store.read(
        "personal", QueryRequest(question="x", people=["Alice"])
    ).people[0].id
    bob_id = store.read(
        "personal", QueryRequest(question="x", people=["Bob"])
    ).people[0].id

    all_matches = store.recall_memories("personal", "distinctive", limit=10)
    assert {memory.content for memory in all_matches} == {
        "Distinctive tea preference",
        "Distinctive tea event",
    }

    about_user_only = store.recall_memories("personal", "distinctive", about_user=True, limit=10)
    assert [memory.content for memory in about_user_only] == ["Distinctive tea preference"]

    not_about_user = store.recall_memories("personal", "distinctive", about_user=False, limit=10)
    assert [memory.content for memory in not_about_user] == ["Distinctive tea event"]

    alice_only = store.recall_memories("personal", "distinctive", person_ids=[alice_id], limit=10)
    assert [memory.content for memory in alice_only] == ["Distinctive tea preference"]

    both = store.recall_memories("personal", "distinctive", person_ids=[alice_id, bob_id], limit=10)
    assert {memory.content for memory in both} == {
        "Distinctive tea preference",
        "Distinctive tea event",
    }

    assert store.recall_memories("personal", "", limit=10) == []
    assert store.recall_memories("personal", "   ", limit=10) == []

    # recall_user_memories keeps its existing hardcoded about_user=1 behavior.
    assert [memory.content for memory in store.recall_user_memories("personal", "distinctive")] == [
        "Distinctive tea preference"
    ]


def test_recall_memories_route_is_protected_llm_free_and_caps_limit(store):
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")

    from gossipmemo.app import create_app
    from gossipmemo.config import Settings
    from gossipmemo.context_budget import ContextBudget
    from gossipmemo.transport import ChatCompletionRequest, ProviderGate, RetryPolicy
    from gossipmemo.world import SocialMemoryWorld

    class ExplodingModel:
        """Any LLM call here proves the route is not LLM-free; fail loudly."""

        configured = True
        gate = ProviderGate()
        context_budget = ContextBudget()
        retry_policy = RetryPolicy(attempts=1, base_seconds=0.001, max_seconds=0.001)
        user_name = "CurrentUser"
        extraction_policy = "balanced"

        async def aclose(self):
            return None

        def prepare(self, messages, *, structured: bool) -> ChatCompletionRequest:
            raise AssertionError("recall route must not call the LLM")

        async def complete(self, request: ChatCompletionRequest) -> str:
            raise AssertionError("recall route must not call the LLM")

    store.add_manual_memory(
        "personal",
        ManualMemoryRequest(content="Distinctive tea preference",
                            people=["Alice"], about_user=True),
    )

    async def scenario():
        world = SocialMemoryWorld(store, ExplodingModel())
        app = create_app(
            settings=Settings(
                database_path=store.path,
                llm_base_url="http://llm.test/v1",
                llm_api_key="test-key",
                llm_model="test-model",
                api_key="secret-token",
            ),
            world=world,
        )
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                unauthorized = await client.get("/v1/spaces/personal/memories?q=distinctive")
                assert unauthorized.status_code == 401

                headers = {"Authorization": "Bearer secret-token"}
                ok = await client.get(
                    "/v1/spaces/personal/memories", params={"q": "distinctive"}, headers=headers
                )
                assert ok.status_code == 200
                body = ok.json()
                assert [memory["content"] for memory in body["memories"]] == [
                    "Distinctive tea preference"
                ]

                about_user_filtered = await client.get(
                    "/v1/spaces/personal/memories",
                    params={"q": "distinctive", "about_user": "false"},
                    headers=headers,
                )
                assert about_user_filtered.json()["memories"] == []

                empty_query = await client.get(
                    "/v1/spaces/personal/memories", params={"q": ""}, headers=headers
                )
                assert empty_query.status_code == 200
                assert empty_query.json()["memories"] == []

                capped = await client.get(
                    "/v1/spaces/personal/memories",
                    params={"q": "distinctive", "limit": "1000"},
                    headers=headers,
                )
                assert capped.status_code == 200
                assert len(capped.json()["memories"]) <= 100

    asyncio.run(scenario())


def test_fastapi_lifespan_ingest_wait_and_query(store):
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")

    import json

    from gossipmemo.app import create_app
    from gossipmemo.config import Settings
    from gossipmemo.context_budget import ContextBudget
    from gossipmemo.transport import ChatCompletionRequest, ProviderGate, RetryPolicy
    from gossipmemo.world import SocialMemoryWorld

    class FakeModel:
        configured = True
        gate = ProviderGate()
        context_budget = ContextBudget()
        retry_policy = RetryPolicy(attempts=1, base_seconds=0.001, max_seconds=0.001)
        user_name = "CurrentUser"
        extraction_policy = "balanced"

        async def aclose(self):
            return None

        def prepare(self, messages, *, structured: bool) -> ChatCompletionRequest:
            return ChatCompletionRequest(
                model="fake",
                messages=list(messages),
                response_format={"type": "json_object"} if structured else None,
            )

        async def complete(self, request: ChatCompletionRequest) -> str:
            # Extraction, query synthesis, and person/relationship reasoning
            # all drive `prepare`/`complete` directly now (see
            # reasoners/extraction.py, query.py, reasoners/owner.py); tell
            # each stage apart by prompt marker.
            combined = " ".join(str(message.content) for message in request.messages)
            if "Extract useful, provenance-aware memories" in combined:
                return json.dumps({
                    "people": [{"ref": "bob", "display_name": "Bob"}],
                    "memories": [
                        {"content": "Bob prefers tea.", "basis": "stated", "people": ["bob"]}
                    ],
                })
            if "Answer the read-only question" in combined:
                assert "What does Bob prefer?" in combined
                assert "Bob prefers tea." in combined
                return "Bob prefers tea."
            if "Review the projection above" in combined:
                return json.dumps({})
            if '"profile_card"' in combined:
                return json.dumps({"profile_card": {"summary": "likes tea"}})
            return json.dumps(
                {"facets": [], "closeness": None, "tone": None, "status": "unknown", "summary": ""}
            )

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


def test_list_people_with_no_query_returns_all_active(store):
    store.add_manual_memory("personal", ManualMemoryRequest(content="x", people=["Alice"]))
    store.add_manual_memory("personal", ManualMemoryRequest(content="y", people=["Bob"]))
    people = store.list_people("personal")
    assert {person.display_name for person in people} == {"Alice", "Bob"}


def test_list_people_query_matches_display_name_case_and_nfkc_insensitively(store):
    store.add_manual_memory(
        "personal", ManualMemoryRequest(content="x", people=["Alice Wang"])
    )
    store.add_manual_memory("personal", ManualMemoryRequest(content="y", people=["Bob"]))
    people = store.list_people("personal", "ALICE")
    assert [person.display_name for person in people] == ["Alice Wang"]


def test_list_people_query_matches_an_alias(store):
    store.apply_extraction(
        "personal", _batch(store, store.record_messages("personal", [_message()])[0]),
        ExtractionResult(
            people=[ExtractedPerson(ref="alice", display_name="Alice Wang", aliases=["AW"])],
            memories=[],
        ),
    )
    people = store.list_people("personal", "aw")
    assert [person.display_name for person in people] == ["Alice Wang"]
    assert "AW" in people[0].aliases


def test_list_people_ambiguous_alias_surfaces_both_people_not_dropped(store):
    store.apply_extraction(
        "personal", _batch(store, store.record_messages("personal", [_message()])[0]),
        ExtractionResult(
            people=[
                ExtractedPerson(ref="one", display_name="One", aliases=["Alex"]),
                ExtractedPerson(ref="two", display_name="Two", aliases=["Alex"]),
            ],
            memories=[],
        ),
    )
    people = store.list_people("personal", "alex")
    assert {person.display_name for person in people} == {"One", "Two"}


def test_list_people_excludes_inactive_or_merged_away_people(store):
    store.add_manual_memory(
        "personal", ManualMemoryRequest(content="x", people=["Alice Wang"])
    )
    store.add_manual_memory("personal", ManualMemoryRequest(content="y", people=["AW"]))
    people_rows = _rows(
        store,
        "SELECT id, display_name FROM people WHERE space_id = 'personal' ORDER BY display_name",
    )
    target, source = people_rows[1]["id"], people_rows[0]["id"]
    store.merge_person("personal", source, target)
    people = store.list_people("personal")
    assert [person.id for person in people] == [target]


def test_list_people_limit_is_honored(store):
    for name in ("Alice", "Bob", "Carol"):
        store.add_manual_memory("personal", ManualMemoryRequest(content=name, people=[name]))
    people = store.list_people("personal", limit=2)
    assert len(people) == 2


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
    assert _rows(store, "SELECT extraction_state FROM messages WHERE id = ?",
                 (message_id,))[0][0] == "completed"
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
        "personal", ManualMemoryRequest(
            content="Bob repeatedly follows up on plans.", people=["Bob"])
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
    assert _rows(store, "SELECT status FROM memories WHERE id = ?",
                 (first,))[0]["status"] == "active"
    store.apply_inferred_memory_actions(
        "personal", "person", bob.id, {new_source}, {first},
        InferredMemoryActions(
            retractions=[InferredMemoryRetraction(
                memory_id=first, reason="support was reconsidered")],
        ),
    )
    assert _rows(store, "SELECT status FROM memories WHERE id = ?",
                 (first,))[0]["status"] == "retracted"
    # A changed output is a new active record with derivations to current source.
    latest_source = store.add_manual_memory(
        "personal", ManualMemoryRequest(content="Bob follows up on outcomes too.", people=["Bob"])
    )
    _, _, latest_watermark = store.person_context("personal", bob.id)
    assert store.apply_person_reasoning(
        "personal", bob.id, latest_watermark,
        PersonReasoningResult(profile_card={}, inferred_memories=[
            InferredMemory(content="Bob reliably follows up on plans and decisions.",
                           source_memory_ids=[latest_source])
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
    assert _rows(store, "SELECT status FROM memories WHERE id = ?",
                 (inferred_id,))[0]["status"] == "active"


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
        store, "SELECT memory_id, role FROM hypothesis_evidence WHERE hypothesis_id = ?", (
            hypothesis["id"],)
    )[0]["role"] == "support"
    # A transition omitted from its supplied hypothesis context is a no-op.
    store.apply_hypothesis_actions(
        "personal", "person", bob.id, {source_id}, set(),
        HypothesisActions(transitions=[HypothesisTransition(
            hypothesis_id=hypothesis["id"], status="rejected", reason="insufficient evidence"
        )]),
    )
    assert _rows(store, "SELECT status FROM hypotheses WHERE id = ?",
                 (hypothesis["id"],))[0]["status"] == "open"
    store.apply_hypothesis_actions(
        "personal", "person", bob.id, {source_id}, {hypothesis["id"]},
        HypothesisActions(
            transitions=[HypothesisTransition(
                hypothesis_id=hypothesis["id"], status="rejected", reason="insufficient evidence"
            )],
        ),
    )
    assert _rows(store, "SELECT status FROM hypotheses WHERE id = ?",
                 (hypothesis["id"],))[0]["status"] == "rejected"
    # Closed hypotheses cannot be upserted or transitioned again, even when
    # the caller supplies their ID in context.
    store.apply_hypothesis_actions(
        "personal", "person", bob.id, {source_id}, {hypothesis["id"]},
        HypothesisActions(upserts=[HypothesisUpsert(
            hypothesis_id=hypothesis["id"], content="Changed claim", confidence="high",
            evidence=[HypothesisEvidence(memory_id=source_id)],
        )]),
    )
    assert _rows(store, "SELECT content FROM hypotheses WHERE id = ?",
                 (hypothesis["id"],))[0]["content"] != "Changed claim"


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
    assert _rows(store, "SELECT status FROM hypotheses WHERE id = ?",
                 (hypothesis_id,))[0]["status"] == "open"
    store.apply_hypothesis_actions(
        "personal", "person", bob.id, {source_id}, {hypothesis_id},
        HypothesisActions(transitions=[HypothesisTransition(
            hypothesis_id=hypothesis_id, status="promoted", reason="confirmed", promoted_memory_id=source_id
        )]),
    )
    promoted = _rows(
        store, "SELECT status, promoted_memory_id FROM hypotheses WHERE id = ?", (hypothesis_id,))[0]
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
        PersonReasoningResult(inferred_memories=[InferredMemory(
            content=shared, source_memory_ids=[first_source])]),
    )
    assert store.apply_person_reasoning(
        "personal", bea.id, bea_watermark,
        PersonReasoningResult(inferred_memories=[InferredMemory(
            content=shared, source_memory_ids=[second_source])]),
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


def _coverage_entries(store, space_id: str = "personal"):
    """Active entries across every root, as goal planning reads them."""
    return store.learning_goal_context(space_id)[1]


def test_coverage_entries_accumulate_per_root_and_feed_goal_planning(store):
    store.ensure_space("personal")
    assert store.coverage_context("personal") is None

    memory_id = store.add_manual_memory(
        "personal", ManualMemoryRequest(content="I grew up near the coast.", about_user=True)
    )
    root, entries, memories = store.coverage_context("personal")
    # Roots are audited in their declared order, one root per read.
    assert (root.root, entries) == ("M1", [])
    assert [memory.id for memory in memories] == [memory_id]
    assert store.apply_coverage_audit(
        "personal", root.root, root.source_watermark, root.source_cursor_id,
        ExtractedCoverageAudit(additions=[ExtractedCoverageEntry(
            content="Early chapters are anchored to a coastal childhood.")]),
        memories, set(),
    )

    # The same evidence is still backlog for every other root.
    next_root, _, next_memories = store.coverage_context("personal")
    assert next_root.root == "M2"
    assert [memory.id for memory in next_memories] == [memory_id]

    revision, planning_entries, _, _ = store.learning_goal_context("personal")
    assert [(item.root, item.path) for item in planning_entries] == [
        ("M1", "")]
    store.apply_goal_planning(
        "personal", revision,
        GoalPlanningResult(upserts=[LearningGoalUpsert(
            prompt="Would you like to share a coastal memory, or skip it?",
            rationale="Optional origin context",
            entry_ids=[planning_entries[0].id, "entry_invented"])]),
        set(),
    )
    _, _, goals, _ = store.learning_goal_context("personal")
    # An unresolvable ref is dropped; the goal it came with is never lost.
    assert [(goal.entry_ids, goal.focus_kind) for goal in goals] == [
        ([planning_entries[0].id], "user")]


def test_coverage_entries_are_modified_and_superseded_by_a_later_audit(store):
    store.add_manual_memory(
        "personal", ManualMemoryRequest(content="I studied in Hangzhou.", about_user=True))
    root, _, memories = store.coverage_context("personal")
    assert store.apply_coverage_audit(
        "personal", root.root, root.source_watermark, root.source_cursor_id,
        ExtractedCoverageAudit(additions=[
            ExtractedCoverageEntry(content="Chapters known: university."),
            ExtractedCoverageEntry(path="university", content="Four years in Hangzhou."),
        ]),
        memories, set(),
    )
    store.add_manual_memory(
        "personal", ManualMemoryRequest(content="Dorm ties loosened later.", about_user=True))
    root, entries, memories = store.coverage_context("personal")
    assert {item.path for item in entries} == {"", "university"}
    overview = next(item for item in entries if item.path == "")
    detail = next(item for item in entries if item.path == "university")
    # Merging is one rewrite plus one supersede; no atomic merge operation.
    assert store.apply_coverage_audit(
        "personal", root.root, root.source_watermark, root.source_cursor_id,
        ExtractedCoverageAudit(modifications=[
            ExtractedCoverageEntryEdit(
                entry_id=detail.id, path="study - Hangzhou",
                content="Four years in Hangzhou, mostly legible through dorm ties that later loosened."),
            ExtractedCoverageEntryEdit(
                entry_id=overview.id, content="absorbed", status="superseded"),
        ]),
        memories, {item.id for item in entries},
    )
    entries = _coverage_entries(store)
    assert [(item.path, item.content) for item in entries] == [
        ("study - Hangzhou",
         "Four years in Hangzhou, mostly legible through dorm ties that later loosened.")]


def test_coverage_audit_ignores_entry_ids_outside_the_audited_root(store):
    store.add_manual_memory("personal", ManualMemoryRequest(content="scene", about_user=True))
    root, _, memories = store.coverage_context("personal")
    evidence = memories
    assert store.apply_coverage_audit(
        "personal", root.root, root.source_watermark, root.source_cursor_id,
        ExtractedCoverageAudit(additions=[ExtractedCoverageEntry(content="M1 overview.")]),
        evidence, set(),
    )
    entries = _coverage_entries(store)
    other, _, _ = store.coverage_context("personal")
    assert other.root == "M2"
    assert store.apply_coverage_audit(
        "personal", other.root, other.source_watermark, other.source_cursor_id,
        ExtractedCoverageAudit(modifications=[ExtractedCoverageEntryEdit(
            entry_id=entries[0].id, content="hijacked", status="superseded")]),
        evidence, {entries[0].id},
    )
    assert [item.content for item in _coverage_entries(store)] == ["M1 overview."]


def test_coverage_cursor_is_per_root_and_survives_equal_timestamps(store):
    ids = [store.add_manual_memory("personal", ManualMemoryRequest(
        content=f"scene {index}", about_user=True)) for index in range(3)]
    with store._connect() as connection:
        connection.execute("UPDATE memories SET updated_at = ? WHERE space_id = ?",
                           ("2026-01-01T00:00:00+00:00", "personal"))
    seen = []
    for _ in ids:
        root, _, memories = store.coverage_context("personal", limit=1)
        assert root.root == "M1"
        seen.extend(memory.id for memory in memories)
        assert store.apply_coverage_audit(
            "personal", root.root, root.source_watermark, root.source_cursor_id,
            ExtractedCoverageAudit(), memories, set(),
        )
    assert set(seen) == set(ids)
    # M1 is caught up; every other root still starts from the beginning.
    root, _, memories = store.coverage_context("personal")
    assert root.root == "M2" and {memory.id for memory in memories} == set(ids)


def test_coverage_entry_can_be_modified_after_a_source_memory_is_retracted(store):
    ids = [store.add_manual_memory("personal", ManualMemoryRequest(
        content=f"scene {index}", about_user=True)) for index in range(2)]
    root, _, memories = store.coverage_context("personal")
    assert store.apply_coverage_audit(
        "personal", root.root, root.source_watermark, root.source_cursor_id,
        ExtractedCoverageAudit(additions=[ExtractedCoverageEntry(content="Two scenes are known.")]),
        memories, set(),
    )
    assert store.retract_memory("personal", ids[-1])
    root, entries, memories = store.coverage_context("personal")
    assert store.apply_coverage_audit(
        "personal", root.root, root.source_watermark, root.source_cursor_id,
        ExtractedCoverageAudit(modifications=[ExtractedCoverageEntryEdit(
            entry_id=entries[0].id, content="One scene is known.")]),
        memories, {entries[0].id},
    )
    assert [item.content for item in _coverage_entries(store)] == ["One scene is known."]


def test_goal_planning_uses_coverage_revision_cas(store):
    store.ensure_space("personal")
    revision, _, _, _ = store.learning_goal_context("personal")
    assert not store.apply_goal_planning(
        "personal", revision + 1, GoalPlanningResult(), set())


def test_goal_focus_is_resolved_from_the_goal_text_by_alias_matching(store):
    """A person-focused goal is anchored by the person its own words name.

    The planner never sees a person ID, so a goal that names exactly one
    known person becomes that person's goal here; naming two keeps it the
    user's, since the direction is then about the user's own life.
    """
    store.add_manual_memory(
        "personal", ManualMemoryRequest(content="Alice and Bob shared a flat with me.",
                                        people=["Alice", "Bob"]))
    alice = next(person.id for person in store.match_people_in_text("personal", "Alice"))
    revision, _, _, _ = store.learning_goal_context("personal")
    assert store.apply_goal_planning(
        "personal", revision,
        GoalPlanningResult(upserts=[
            LearningGoalUpsert(prompt="How did you and Alice first meet?",
                               rationale="Alice recurs but the start of the friendship is blank"),
            LearningGoalUpsert(prompt="What was that shared flat like?",
                               rationale="Alice and Bob both appear, the home itself does not"),
        ]),
        set(),
    )
    _, _, goals, _ = store.learning_goal_context("personal")
    assert {goal.prompt: (goal.focus_kind, goal.focus_id) for goal in goals} == {
        "How did you and Alice first meet?": ("person", alice),
        "What was that shared flat like?": ("user", None),
    }


def test_partial_goals_are_bucketed_with_open_not_recent_closed(store):
    """`partial` is still served to the agent by guidance_bundle's

    `status IN ('open', 'partial')`, so the planner's context must treat it
    as a live goal too -- bucketing it with closed history instead let the
    planner keep proposing directions the agent was still actively pursuing.
    """
    store.ensure_space("personal")
    revision, _, _, _ = store.learning_goal_context("personal")
    store.apply_goal_planning(
        "personal", revision,
        GoalPlanningResult(upserts=[
            LearningGoalUpsert(prompt="Open goal", rationale="stays open"),
            LearningGoalUpsert(prompt="Partial goal", rationale="half answered"),
            LearningGoalUpsert(prompt="Answered goal", rationale="fully answered"),
        ]),
        set(),
    )
    _, _, goals, _ = store.learning_goal_context("personal")
    by_prompt = {goal.prompt: goal.id for goal in goals}
    revision, _, _, _ = store.learning_goal_context("personal")
    store.apply_goal_planning(
        "personal", revision, GoalPlanningResult(transitions=[
            LearningGoalTransition(
                goal_id=by_prompt["Partial goal"], status="partial", reason="half answered"),
            LearningGoalTransition(
                goal_id=by_prompt["Answered goal"], status="answered", reason="fully answered"),
        ]),
        set(by_prompt.values()),
    )
    _, _, open_goals, closed_goals = store.learning_goal_context("personal")
    assert {goal.prompt for goal in open_goals} == {"Open goal", "Partial goal"}
    assert {goal.prompt for goal in closed_goals} == {"Answered goal"}
    assert {goal.status for goal in open_goals} == {"open", "partial"}


def test_stale_coverage_restart_detects_same_timestamp_after_cursor(store):
    first = store.add_manual_memory(
        "personal", ManualMemoryRequest(content="first", about_user=False))
    second = store.add_manual_memory(
        "personal", ManualMemoryRequest(content="second", about_user=False))
    with store._connect() as connection:
        connection.execute("UPDATE memories SET updated_at = ? WHERE id IN (?, ?)",
                           ("2026-02-01T00:00:00+00:00", first, second))
    root, _, memories = store.coverage_context("personal", limit=1)
    assert store.apply_coverage_audit("personal", root.root, root.source_watermark,
                                      root.source_cursor_id, ExtractedCoverageAudit(),
                                      memories, set())
    # This models a process restart: stale discovery must notice the second row
    # even though it shares the persisted timestamp.
    assert store.stale_coverage_spaces() == ["personal"]


def _seeded_store(tmp_path, seed: int, goal_count: int, *, id_prefix: str | None = None) -> SqliteWorldStore:
    world = SqliteWorldStore(tmp_path / f"guidance-{seed}.db")
    world.initialize()
    world.ensure_space("s")
    prefix = id_prefix if id_prefix is not None else "g"
    with world._connect() as connection:
        for index in range(goal_count):
            connection.execute(
                "INSERT INTO learning_goals(id,space_id,prompt,rationale,entry_ids,"
                "created_at,updated_at) VALUES (?,'s',?,'context','[]','1','1')",
                (f"{prefix}{index:02d}", f"goal {index}"),
            )
    return world


def test_guidance_samples_three_to_five_learning_goals(tmp_path):
    """Goals are sampled, not ranked: only the count and the pool are promised.

    The sample is now a deterministic function of the version and the goal
    pool, so varying the pool's id set (rather than an injected RNG seed) is
    what drives different sample sizes across iterations here.
    """
    sizes = set()
    for seed in range(20):
        world = _seeded_store(tmp_path, seed, goal_count=30, id_prefix=f"g{seed}-")
        ids = [item.id for item in world.guidance_bundle("s").items]
        assert all(item.startswith("g") for item in ids)
        assert len(set(ids)) == len(ids)
        sizes.add(len(ids))
    assert sizes == {3, 4, 5}


def test_guidance_ignores_query_relevance_for_learning_goals(tmp_path):
    """The sample is deterministic given a fixed version and pool, whatever the query says."""
    world = _seeded_store(tmp_path, 7, goal_count=30)
    first = [item.id for item in world.guidance_bundle("s", [], "goal 3").items]
    second = [item.id for item in world.guidance_bundle("s", [], "something else").items]
    assert first == second


def test_guidance_sampling_still_respects_the_focus_filter(tmp_path):
    """Sampling replaces ranking, not the deterministic focus gate."""
    world = _seeded_store(tmp_path, 11, goal_count=0)
    world.add_manual_memory("s", ManualMemoryRequest(content="Alice is a friend",
                                                     people=["Alice"]))
    with world._connect() as connection:
        person_id = connection.execute(
            "SELECT id FROM people WHERE display_name = 'Alice'").fetchone()["id"]
        for index in range(10):
            connection.execute(
                "INSERT INTO learning_goals(id,space_id,prompt,rationale,entry_ids,"
                "focus_kind,focus_id,created_at,updated_at) "
                "VALUES (?,'s',?,'context','[]','person',?,'1','1')",
                (f"p{index:02d}", f"about Alice {index}", person_id),
            )
        connection.execute(
            "INSERT INTO learning_goals(id,space_id,prompt,rationale,entry_ids,status,"
            "created_at,updated_at) VALUES ('retired','s','old','context','[]','retired','1','1')")
    assert world.guidance_bundle("s").items == []
    activated = [item.id for item in world.guidance_bundle("s", [person_id]).items]
    assert activated and all(item.startswith("p") for item in activated)


def test_guidance_returns_the_whole_small_goal_pool(tmp_path):
    world = _seeded_store(tmp_path, 1, goal_count=2)
    assert sorted(item.id for item in world.guidance_bundle("s").items) == ["g00", "g01"]


def test_guidance_is_stable_across_repeated_reads_at_an_unchanged_version(tmp_path):
    """Guidance is derived from the version, so an unchanged version must
    reproduce the identical selection on every read -- this is what keeps
    the agent-side prompt prefix KV-cache friendly across turns."""
    world = _seeded_store(tmp_path, 3, goal_count=30)
    first = world.context_bundle("s")
    second = world.context_bundle("s")
    assert first.version == second.version
    assert [item.id for item in first.guidance.items] == [
        item.id for item in second.guidance.items]


def test_guidance_rotates_when_the_version_changes(tmp_path):
    """A durable context change moves the version, which rotates the sample."""
    world = _seeded_store(tmp_path, 3, goal_count=30)
    before = world.context_bundle("s")
    world.overwrite_user_model("s", {"note": "mutated"})
    after = world.context_bundle("s")
    assert after.version != before.version
    assert [item.id for item in after.guidance.items] != [
        item.id for item in before.guidance.items]


def test_guidance_route_is_protected_llm_free_supports_filters_and_caps_limit(store):
    pytest.importorskip("fastapi")
    httpx = pytest.importorskip("httpx")

    from gossipmemo.app import create_app
    from gossipmemo.config import Settings
    from gossipmemo.context_budget import ContextBudget
    from gossipmemo.transport import ChatCompletionRequest, ProviderGate, RetryPolicy
    from gossipmemo.world import SocialMemoryWorld

    class ExplodingModel:
        configured = True
        gate = ProviderGate()
        context_budget = ContextBudget()
        retry_policy = RetryPolicy(attempts=1, base_seconds=0.001, max_seconds=0.001)
        user_name = "CurrentUser"
        extraction_policy = "balanced"

        async def aclose(self):
            return None

        def prepare(self, messages, *, structured: bool) -> ChatCompletionRequest:
            raise AssertionError("guidance route must not call the LLM")

        async def complete(self, request: ChatCompletionRequest) -> str:
            raise AssertionError("guidance route must not call the LLM")

    store.ensure_space("personal")
    _insert_hypothesis(store, "h1", "personal", "user", None, "User likes tea.")
    store.add_manual_memory("personal", ManualMemoryRequest(content="x", people=["Alice"]))
    alice_id = store.list_people("personal")[0].id
    _insert_hypothesis(store, "ha", "personal", "person", alice_id, "About Alice.")
    with store._connect() as connection:
        connection.execute(
            "INSERT INTO learning_goals(id, space_id, prompt, rationale, entry_ids, "
            "created_at, updated_at) VALUES ('g1', 'personal', 'a goal', 'context', '[]', '1', '1')"
        )

    async def scenario():
        world = SocialMemoryWorld(store, ExplodingModel())
        app = create_app(
            settings=Settings(
                database_path=store.path,
                llm_base_url="http://llm.test/v1",
                llm_api_key="test-key",
                llm_model="test-model",
                api_key="secret-token",
            ),
            world=world,
        )
        transport = httpx.ASGITransport(app=app)
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=transport, base_url="http://testserver"
            ) as client:
                unauthorized = await client.get("/v1/spaces/personal/guidance")
                assert unauthorized.status_code == 401

                headers = {"Authorization": "Bearer secret-token"}
                global_scope = await client.get(
                    "/v1/spaces/personal/guidance", headers=headers)
                assert global_scope.status_code == 200
                assert {item["id"] for item in global_scope.json()["items"]} == {"h1", "g1"}

                focused = await client.get(
                    "/v1/spaces/personal/guidance",
                    params={"person_id": alice_id},
                    headers=headers,
                )
                assert {item["id"] for item in focused.json()["items"]} == {"h1", "ha", "g1"}

                kind_filtered = await client.get(
                    "/v1/spaces/personal/guidance",
                    params={"kind": "hypothesis"},
                    headers=headers,
                )
                assert {item["id"] for item in kind_filtered.json()["items"]} == {"h1"}

                bad_kind = await client.get(
                    "/v1/spaces/personal/guidance",
                    params={"kind": "nonsense"},
                    headers=headers,
                )
                assert bad_kind.status_code == 422

                capped = await client.get(
                    "/v1/spaces/personal/guidance",
                    params={"limit": "1"},
                    headers=headers,
                )
                assert capped.status_code == 200
                assert len(capped.json()["items"]) == 1

    asyncio.run(scenario())


def _insert_hypothesis(
    store: SqliteWorldStore, hypothesis_id: str, space_id: str,
    owner_kind: str, owner_id: str | None, content: str, status: str = "open",
) -> None:
    with store._connect() as connection:
        connection.execute(
            "INSERT INTO hypotheses(id, space_id, owner_kind, owner_id, content, kind, "
            "confidence, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 'impression', 'low', ?, '1', '1')",
            (hypothesis_id, space_id, owner_kind, owner_id, content, status),
        )


def _insert_relationship(store: SqliteWorldStore, relationship_id: str, space_id: str,
                         person_a_id: str, person_b_id: str) -> None:
    ordered = sorted([person_a_id, person_b_id])
    with store._connect() as connection:
        connection.execute(
            "INSERT INTO relationships(id, space_id, person_a_id, person_b_id, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, '1', '1')",
            (relationship_id, space_id, ordered[0], ordered[1]),
        )


def test_list_guidance_includes_user_scoped_items_by_default(store):
    """User-scoped hypotheses and goals always come back, no focus needed."""
    store.ensure_space("personal")
    _insert_hypothesis(store, "h1", "personal", "user", None, "User likes tea.")
    with store._connect() as connection:
        connection.execute(
            "INSERT INTO learning_goals(id, space_id, prompt, rationale, entry_ids, "
            "created_at, updated_at) VALUES ('g1', 'personal', 'What do they do for fun?', "
            "'context', '[]', '1', '1')"
        )
    items = store.list_guidance("personal")
    assert {(item.id, item.kind, item.status) for item in items} == {
        ("h1", "hypothesis", "open"),
        ("g1", "learning_goal", "open"),
    }


def test_list_guidance_person_focus_filter_excludes_unrelated_people(store):
    store.add_manual_memory(
        "personal", ManualMemoryRequest(content="x", people=["Alice", "Bob"]))
    people = {p.display_name: p.id for p in store.list_people("personal")}
    _insert_hypothesis(store, "ha", "personal", "person", people["Alice"], "About Alice.")
    _insert_hypothesis(store, "hb", "personal", "person", people["Bob"], "About Bob.")

    alice_only = store.list_guidance("personal", person_ids=[people["Alice"]])
    assert [item.id for item in alice_only] == ["ha"]

    neither = store.list_guidance("personal")
    assert neither == []


def test_list_guidance_reaches_relationship_owned_items_via_member_people(store):
    store.add_manual_memory(
        "personal", ManualMemoryRequest(content="x", people=["Alice", "Bob"]))
    people = {p.display_name: p.id for p in store.list_people("personal")}
    _insert_relationship(store, "rel1", "personal", people["Alice"], people["Bob"])
    _insert_hypothesis(store, "hr", "personal", "relationship", "rel1", "Close friends.")

    assert store.list_guidance("personal") == []
    reached = store.list_guidance("personal", person_ids=[people["Alice"]])
    assert [item.id for item in reached] == ["hr"]


def test_list_guidance_includes_partial_goals(store):
    store.ensure_space("personal")
    with store._connect() as connection:
        connection.execute(
            "INSERT INTO learning_goals(id, space_id, prompt, rationale, entry_ids, status, "
            "created_at, updated_at) VALUES ('gp', 'personal', 'Half answered', 'context', "
            "'[]', 'partial', '1', '1')"
        )
    items = store.list_guidance("personal")
    assert [(item.id, item.status) for item in items] == [("gp", "partial")]


def test_list_guidance_excludes_closed_and_resolved_items(store):
    store.ensure_space("personal")
    _insert_hypothesis(store, "h_open", "personal", "user", None, "open one")
    _insert_hypothesis(store, "h_promoted", "personal", "user",
                       None, "promoted one", status="promoted")
    _insert_hypothesis(store, "h_rejected", "personal", "user",
                       None, "rejected one", status="rejected")
    with store._connect() as connection:
        connection.execute(
            "INSERT INTO learning_goals(id, space_id, prompt, rationale, entry_ids, status, "
            "created_at, updated_at) VALUES ('g_answered', 'personal', 'answered', 'context', "
            "'[]', 'answered', '1', '1')"
        )
        connection.execute(
            "INSERT INTO learning_goals(id, space_id, prompt, rationale, entry_ids, status, "
            "created_at, updated_at) VALUES ('g_retired', 'personal', 'retired', 'context', "
            "'[]', 'retired', '1', '1')"
        )
    items = store.list_guidance("personal")
    assert [item.id for item in items] == ["h_open"]


def test_list_guidance_limit_is_honored(store):
    store.ensure_space("personal")
    for index in range(5):
        _insert_hypothesis(store, f"h{index}", "personal", "user", None, f"claim {index}")
    assert len(store.list_guidance("personal", limit=2)) == 2
    assert len(store.list_guidance("personal", limit=50)) == 5


def test_list_guidance_kind_filter_restricts_to_one_kind(store):
    store.ensure_space("personal")
    _insert_hypothesis(store, "h1", "personal", "user", None, "a hypothesis")
    with store._connect() as connection:
        connection.execute(
            "INSERT INTO learning_goals(id, space_id, prompt, rationale, entry_ids, "
            "created_at, updated_at) VALUES ('g1', 'personal', 'a goal', 'context', '[]', '1', '1')"
        )
    only_hypotheses = store.list_guidance("personal", kind="hypothesis")
    assert [item.id for item in only_hypotheses] == ["h1"]
    only_goals = store.list_guidance("personal", kind="learning_goal")
    assert [item.id for item in only_goals] == ["g1"]
    both = store.list_guidance("personal")
    assert {item.id for item in both} == {"h1", "g1"}


def test_guidance_bundle_sampling_is_unaffected_by_the_list_guidance_refactor(tmp_path):
    """`_guidance` must keep sampling; `list_guidance` must not, after sharing the filter SQL."""
    world = _seeded_store(tmp_path, 99, goal_count=30)
    sampled = world.guidance_bundle("s").items
    assert 3 <= len(sampled) <= 5
    full = world.list_guidance("s", kind="learning_goal", limit=1000)
    assert len(full) == 30


def test_initialize_enables_wal_and_a_short_busy_timeout(store):
    with store._connect() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 1000
        assert connection.execute("PRAGMA synchronous").fetchone()[0] == 1


def test_initialize_fails_loudly_when_wal_is_refused(tmp_path, monkeypatch):
    """A filesystem that cannot do WAL reports a mode instead of raising."""

    import gossipmemo.store as store_module

    real_connect = store_module.sqlite3.connect

    class _FakeWalResult:
        def fetchone(self):
            return ("delete",)

    class RefusesWal:
        """A real connection that lies about `PRAGMA journal_mode = WAL`,
        standing in for a filesystem without WAL support. Migration also
        opens a connection through `sqlite3.connect` before WAL is ever
        enabled, so this proxies everything else straight through to a real
        connection rather than faking the whole SQLite surface."""

        def __init__(self, *args, **kwargs):
            self._real = real_connect(*args, **kwargs)

        def execute(self, sql, *params):
            if "journal_mode = WAL" in sql:
                return _FakeWalResult()
            return self._real.execute(sql, *params)

        def __getattr__(self, name):
            return getattr(self._real, name)

        @property
        def isolation_level(self):
            return self._real.isolation_level

        @isolation_level.setter
        def isolation_level(self, value):
            self._real.isolation_level = value

        def __enter__(self):
            return self._real.__enter__()

        def __exit__(self, *args):
            return self._real.__exit__(*args)

    monkeypatch.setattr(store_module.sqlite3, "connect", lambda *a, **k: RefusesWal(*a, **k))
    with pytest.raises(RuntimeError, match="refused WAL mode"):
        SqliteWorldStore(tmp_path / "world.db").initialize()
