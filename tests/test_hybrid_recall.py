"""Tests for RRF-fused hybrid (FTS + vector) recall at the three retrieval
call points wired in this slice:

- `SqliteWorldStore._recall_memories` (turn-path `about_user` recall)
- `SqliteWorldStore.read` (`/v1/spaces/{id}/query` candidate pruning)
- `SqliteWorldStore.load_extraction_comparisons` (extraction dedup set)

Store-level tests use `deterministic_unit_vector` the same way
`tests/test_store_vectors.py` does: embedding a memory with the vector
derived from a *different* string than its own content lets a test build an
unambiguous semantic winner regardless of what free text is used for the
FTS side, without needing a real embedding model.

World-level tests use `tests/fakes_embedding.py`'s `FakeEmbeddingClient` to
check the plumbing: which instruction each call site asks for, that
storage-side embedding never sees one, and every foreground degrade path
(no client, a raised exception, a timeout).
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path

import pytest

from gossipmemo.context_budget import ContextBudget
from gossipmemo.embedding import embed_query_vector
from gossipmemo.models import (
    ManualMemoryRequest,
    MessageInput,
    QueryRequest,
    TurnRequest,
)
from gossipmemo.prompts import PromptLibrary
from gossipmemo.reasoners.extraction import build_extraction_reasoner
from gossipmemo.reasoners.settings import ReasoningSettings
from gossipmemo.store import SqliteWorldStore
from gossipmemo.store._vectors import EmbeddingUpsert
from gossipmemo.transport import ChatCompletionRequest, ProviderGate, RetryPolicy
from gossipmemo.world import SocialMemoryWorld
from tests.fakes_embedding import FakeEmbeddingClient, deterministic_unit_vector

MODEL = "fake-embedding"
DIM = 8
DEFAULT_PROMPTS = PromptLibrary()


class _NoopModel:
    """Minimal `LlmTransport` double -- mirrors `tests/test_embedding_worker.py`."""

    configured = False
    gate = ProviderGate()
    context_budget = ContextBudget()
    retry_policy = RetryPolicy(attempts=1, base_seconds=0.001, max_seconds=0.001)

    async def aclose(self):
        return None

    def prepare(self, messages, *, structured: bool) -> ChatCompletionRequest:
        return ChatCompletionRequest(
            model="fake", messages=list(messages),
            response_format={"type": "json_object"} if structured else None,
        )

    async def complete(self, request: ChatCompletionRequest) -> str:
        return "{}"


@pytest.fixture
def store(tmp_path: Path) -> SqliteWorldStore:
    world_store = SqliteWorldStore(tmp_path / "world.db")
    world_store.initialize()
    world_store.ensure_space("s1")
    return world_store


def _insert_memory(
    store: SqliteWorldStore, memory_id: str, content: str, *,
    about_user: int = 1, status: str = "active", basis: str = "stated",
) -> None:
    now = "2026-01-01T00:00:00Z"
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "INSERT INTO memories(id, space_id, content, kind, basis, about_user, status, "
            "created_by, created_at, updated_at) VALUES (?, 's1', ?, 'fact', ?, ?, ?, "
            "'extractor', ?, ?)",
            (memory_id, content, basis, about_user, status, now, now),
        )


def _link_person(store: SqliteWorldStore, memory_id: str, person_id: str) -> None:
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "INSERT OR IGNORE INTO memory_people(memory_id, person_id) VALUES (?, ?)",
            (memory_id, person_id),
        )


def _embed(
    store: SqliteWorldStore, owner_kind: str, owner_id: str, vector, text: str = "x",
) -> None:
    store.upsert_embeddings(
        "s1", MODEL, DIM,
        [EmbeddingUpsert(owner_kind=owner_kind, owner_id=owner_id, text=text, vector=vector)],
    )


def _vector_for(text: str):
    return deterministic_unit_vector(text, DIM)


def _rows(store: SqliteWorldStore, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    with store._connect() as connection:
        return connection.execute(sql, params).fetchall()


def _batch(store: SqliteWorldStore, *message_ids: str) -> str:
    batch_id = store.create_extraction_batch("s1", list(message_ids))
    assert batch_id is not None
    return batch_id


# =====================================================================
# embed_query_vector: the shared foreground degrade helper
# =====================================================================


class _RaisingClient:
    model = MODEL
    dim = DIM

    async def embed(self, texts, *, instruction=None):
        raise RuntimeError("embedding provider unavailable")


class _SleepyClient:
    model = MODEL
    dim = DIM

    async def embed(self, texts, *, instruction=None):
        await asyncio.sleep(10)
        return [[0.1] * DIM for _ in texts]


def test_embed_query_vector_returns_none_without_a_client():
    assert asyncio.run(embed_query_vector(None, "hello", instruction="find x")) is None


def test_embed_query_vector_degrades_on_request_exception(caplog):
    with caplog.at_level(logging.WARNING, logger="gossipmemo.embedding"):
        result = asyncio.run(
            embed_query_vector(_RaisingClient(), "hello", instruction="find x", timeout=1.0)
        )
    assert result is None
    assert any("embedding_query_failed" in record.message for record in caplog.records)


def test_embed_query_vector_degrades_on_timeout():
    result = asyncio.run(
        embed_query_vector(_SleepyClient(), "hello", instruction="find x", timeout=0.05)
    )
    assert result is None


def test_embed_query_vector_returns_the_first_vector_on_success():
    client = FakeEmbeddingClient(model=MODEL, dim=DIM)
    result = asyncio.run(
        embed_query_vector(client, "hello", instruction="find x", timeout=1.0)
    )
    assert result == deterministic_unit_vector("Instruct: find x\nQuery: hello", DIM)
    assert client.calls == [(("hello",), "find x")]


def test_recall_memories_query_vector_none_is_byte_identical_to_pure_fts(store):
    _insert_memory(store, "m1", "Alice enjoys long distance running")
    with store._connect() as connection:
        without_vector = store._recall_memories(
            connection, "s1", "long distance running", about_user=True, limit=5,
        )
        with_explicit_none = store._recall_memories(
            connection, "s1", "long distance running", about_user=True, limit=5,
            query_vector=None,
        )
    assert [m.id for m in without_vector] == ["m1"]
    assert [m.model_dump() for m in without_vector] == [m.model_dump() for m in with_explicit_none]


def test_recall_memories_semantic_hit_without_fts_overlap(store):
    _insert_memory(store, "close", "Zhang Wei just switched jobs")
    _insert_memory(store, "far", "the weather has been cold lately")
    _embed(store, "memory", "close", _vector_for("Zhang Wei just switched jobs"))
    _embed(store, "memory", "far", _vector_for("the weather has been cold lately"))

    # No lexical overlap at all with either memory, but the query vector is
    # built to be identical to "close"'s own embedding -- simulating a
    # semantically-equivalent reworded query a real embedding model would
    # recognize.
    with store._connect() as connection:
        results = store._recall_memories(
            connection, "s1", "totally different unrelated phrasing", about_user=True, limit=5,
            query_vector=_vector_for("Zhang Wei just switched jobs"),
        )
    # "close" has zero lexical overlap with the query, so a pure-FTS recall
    # would have found nothing; the vector path alone must surface it, and
    # rank it ahead of the much-less-similar "far".
    assert [m.id for m in results][0] == "close"


def test_recall_memories_fts_hit_survives_hybrid_fusion(store):
    _insert_memory(store, "lexical", "Bob mentioned a distinctive hobby")
    _embed(store, "memory", "lexical", _vector_for("something else entirely"))

    with store._connect() as connection:
        results = store._recall_memories(
            connection, "s1", "a distinctive hobby", about_user=True, limit=5,
            # Vector side points nowhere near this memory -- it must still
            # surface purely on the FTS side of the fusion.
            query_vector=_vector_for("nothing at all like the memory"),
        )
    assert [m.id for m in results] == ["lexical"]


def test_recall_memories_about_user_filter_applies_to_vector_hits(store):
    _insert_memory(store, "about_user", "shared distinctive phrase", about_user=1)
    _insert_memory(store, "about_other", "shared distinctive phrase", about_user=0)
    vector = _vector_for("shared distinctive phrase")
    _embed(store, "memory", "about_user", vector, text="about_user")
    _embed(store, "memory", "about_other", vector, text="about_other")

    with store._connect() as connection:
        results = store._recall_memories(
            connection, "s1", "unrelated fts phrasing xyz", about_user=True, limit=5,
            query_vector=vector,
        )
    assert [m.id for m in results] == ["about_user"]


def test_recall_memories_person_ids_filter_applies_to_vector_hits(store):
    _insert_memory(store, "alice_mem", "shared distinctive phrase", about_user=0)
    _insert_memory(store, "bob_mem", "shared distinctive phrase", about_user=0)
    _link_person(store, "alice_mem", "person_alice")
    _link_person(store, "bob_mem", "person_bob")
    vector = _vector_for("shared distinctive phrase")
    _embed(store, "memory", "alice_mem", vector, text="alice_mem")
    _embed(store, "memory", "bob_mem", vector, text="bob_mem")

    with store._connect() as connection:
        results = store._recall_memories(
            connection, "s1", "unrelated fts phrasing xyz", about_user=None,
            person_ids=["person_alice"], limit=5, query_vector=vector,
        )
    assert [m.id for m in results] == ["alice_mem"]


def test_recall_memories_degrades_when_search_vectors_raises(store, monkeypatch, caplog):
    _insert_memory(store, "lexical", "a distinctive hobby mentioned here")

    def _boom(*args, **kwargs):
        raise RuntimeError("sidecar exploded")

    monkeypatch.setattr(store, "search_vectors", _boom)
    with caplog.at_level(logging.ERROR, logger="gossipmemo.store._memories"):
        with store._connect() as connection:
            results = store._recall_memories(
                connection, "s1", "a distinctive hobby", about_user=True, limit=5,
                query_vector=_vector_for("anything"),
            )
    assert [m.id for m in results] == ["lexical"]
    assert any("vector recall failed" in record.message for record in caplog.records)


# =====================================================================
# store.read -- /query candidate pruning
# =====================================================================


def test_read_query_vector_none_is_byte_identical_to_pure_fts(store):
    _insert_memory(store, "m1", "distinctive travel plans")
    request = QueryRequest(question="distinctive travel plans")
    without_vector = store.read("s1", request)
    with_explicit_none = store.read("s1", request, None)
    assert [m.id for m in without_vector.memories] == ["m1"]
    assert without_vector.model_dump() == with_explicit_none.model_dump()


def test_read_semantic_hit_without_fts_overlap(store):
    _insert_memory(store, "close", "Zhang Wei just switched jobs")
    _insert_memory(store, "far", "the weather has been cold lately")
    _embed(store, "memory", "close", _vector_for("Zhang Wei just switched jobs"))
    _embed(store, "memory", "far", _vector_for("the weather has been cold lately"))

    context = store.read(
        "s1", QueryRequest(question="totally different unrelated phrasing"),
        _vector_for("Zhang Wei just switched jobs"),
    )
    # "close" has zero lexical overlap with the question, so a pure-FTS
    # recall would have found nothing; the vector path alone must surface
    # it, and rank it ahead of the much-less-similar "far".
    assert [m.id for m in context.memories][0] == "close"


def test_read_fts_hit_survives_hybrid_fusion(store):
    _insert_memory(store, "lexical", "a rare distinctive hobby")

    context = store.read(
        "s1", QueryRequest(question="a rare distinctive hobby"),
        _vector_for("nothing like the memory at all"),
    )
    assert [m.id for m in context.memories] == ["lexical"]


def test_read_person_scoping_still_applies_to_vector_hits(store):
    alice_memory = store.add_manual_memory(
        "s1", ManualMemoryRequest(content="Alice's distinctive fact", people=["Alice"]),
    )
    bob_memory = store.add_manual_memory(
        "s1", ManualMemoryRequest(content="Bob's distinctive fact", people=["Bob"]),
    )
    vector = _vector_for("shared vector")
    _embed(store, "memory", alice_memory, vector, text="alice")
    _embed(store, "memory", bob_memory, vector, text="bob")

    context = store.read(
        "s1", QueryRequest(question="unrelated fts phrasing", people=["Alice"]),
        vector,
    )
    memory_ids = [m.id for m in context.memories]
    assert memory_ids == [alice_memory]
    assert bob_memory not in memory_ids


def test_read_degrades_when_search_vectors_raises(store, monkeypatch, caplog):
    _insert_memory(store, "lexical", "a rare distinctive hobby")

    def _boom(*args, **kwargs):
        raise RuntimeError("sidecar exploded")

    monkeypatch.setattr(store, "search_vectors", _boom)
    with caplog.at_level(logging.ERROR, logger="gossipmemo.store._memories"):
        context = store.read(
            "s1", QueryRequest(question="a rare distinctive hobby"),
            _vector_for("anything"),
        )
    assert [m.id for m in context.memories] == ["lexical"]
    assert any("vector search failed" in record.message for record in caplog.records)


# =====================================================================
# store.load_extraction_comparisons -- extraction dedup comparison set
# =====================================================================


def test_load_extraction_comparisons_query_vectors_none_is_byte_identical(store):
    _insert_memory(store, "m1", "distinctive travel plans", basis="stated")
    message_id = store.record_messages(
        "s1", [MessageInput(author="user", content="distinctive travel plans")],
    )[0]
    batch_id = _batch(store, message_id)

    without_vectors = store.load_extraction_comparisons("s1", batch_id)
    with_explicit_none = store.load_extraction_comparisons("s1", batch_id, query_vectors=None)
    assert [m.id for m in without_vectors] == ["m1"]
    assert (
        [m.model_dump() for m in without_vectors]
        == [m.model_dump() for m in with_explicit_none]
    )


def test_load_extraction_comparisons_semantic_hit_without_fts_overlap(store):
    _insert_memory(store, "old", "Zhang Wei just switched jobs", basis="stated")
    new_text = "Xiao Zhang moved to a different company"
    message_id = store.record_messages(
        "s1", [MessageInput(author="user", content=new_text)],
    )[0]
    batch_id = _batch(store, message_id)

    comparisons = store.load_extraction_comparisons(
        "s1", batch_id, query_vectors={new_text: _vector_for("Zhang Wei just switched jobs")},
    )
    assert [m.id for m in comparisons] == ["old"]


def test_load_extraction_comparisons_fts_hit_survives_hybrid_fusion(store):
    _insert_memory(store, "lexical", "a rare distinctive hobby", basis="stated")
    new_text = "a rare distinctive hobby indeed"
    message_id = store.record_messages(
        "s1", [MessageInput(author="user", content=new_text)],
    )[0]
    batch_id = _batch(store, message_id)

    comparisons = store.load_extraction_comparisons(
        "s1", batch_id, query_vectors={new_text: _vector_for("nothing alike")},
    )
    assert [m.id for m in comparisons] == ["lexical"]


def test_load_extraction_comparisons_excludes_inferred_from_vector_side_too(store):
    _insert_memory(store, "inferred", "Zhang Wei just switched jobs", basis="inferred")
    new_text = "Xiao Zhang moved to a different company"
    message_id = store.record_messages(
        "s1", [MessageInput(author="user", content=new_text)],
    )[0]
    batch_id = _batch(store, message_id)

    comparisons = store.load_extraction_comparisons(
        "s1", batch_id, query_vectors={new_text: _vector_for("Zhang Wei just switched jobs")},
    )
    assert comparisons == []


def test_load_extraction_comparisons_text_with_no_query_vector_falls_back_to_fts_only(store):
    _insert_memory(store, "lexical", "a rare distinctive hobby", basis="stated")
    new_text = "a rare distinctive hobby indeed"
    message_id = store.record_messages(
        "s1", [MessageInput(author="user", content=new_text)],
    )[0]
    batch_id = _batch(store, message_id)

    # query_vectors present but has no entry for this exact text.
    comparisons = store.load_extraction_comparisons(
        "s1", batch_id, query_vectors={"some other text": _vector_for("x")},
    )
    assert [m.id for m in comparisons] == ["lexical"]


# =====================================================================
# extraction reasoner: per-text embedding wiring
# =====================================================================


def test_extraction_reasoner_comparison_query_vectors_dedupes_and_uses_its_own_instruction():
    client = FakeEmbeddingClient(model=MODEL, dim=DIM)
    reasoner = build_extraction_reasoner(
        store=None, model=_NoopModel(), settings=ReasoningSettings(),
        embedding_client_getter=lambda: client,
    )

    result = asyncio.run(reasoner._comparison_query_vectors(["same text", "same text", "other"]))

    assert client.calls == [
        (("same text",), DEFAULT_PROMPTS.embedding_extraction_comparison_instruction),
        (("other",), DEFAULT_PROMPTS.embedding_extraction_comparison_instruction),
    ]
    assert set(result) == {"same text", "other"}


def test_extraction_reasoner_comparison_query_vectors_none_without_a_client():
    reasoner = build_extraction_reasoner(
        store=None, model=_NoopModel(), settings=ReasoningSettings(),
    )
    assert asyncio.run(reasoner._comparison_query_vectors(["text"])) is None


# =====================================================================
# world-level wiring: instructions, storage-side discipline, degrade paths
# =====================================================================


def _turn_expected_vector(
    text: str, instruction: str = DEFAULT_PROMPTS.embedding_turn_recall_instruction,
):
    return deterministic_unit_vector(f"Instruct: {instruction}\nQuery: {text}", DIM)


def test_turn_recall_uses_turn_instruction_and_recalls_a_semantic_only_match(tmp_path):
    store = SqliteWorldStore(tmp_path / "world.db")
    store.initialize()
    text = "Completely unrelated filler wording that shares no words"
    _insert_memory(store, "m1", "Zhang Wei just switched jobs", about_user=1)
    _embed(store, "memory", "m1", _turn_expected_vector(text))
    client = FakeEmbeddingClient(model=MODEL, dim=DIM)
    world = SocialMemoryWorld(store, _NoopModel(), embedding_client=client)

    async def scenario():
        await world.start()
        try:
            return await world.turn(
                "s1", TurnRequest(messages=[MessageInput(author="user", content=text)]),
            )
        finally:
            await world.stop()

    response = asyncio.run(scenario())

    assert [m.id for m in response.memory_recall] == ["m1"]
    assert client.calls[0] == ((text,), DEFAULT_PROMPTS.embedding_turn_recall_instruction)


def test_query_uses_query_instruction_and_recalls_a_semantic_only_match(tmp_path):
    store = SqliteWorldStore(tmp_path / "world.db")
    store.initialize()
    question = "Completely unrelated filler question wording"
    _insert_memory(store, "m1", "Zhang Wei just switched jobs", about_user=0)
    _embed(
        store, "memory", "m1",
        deterministic_unit_vector(
            f"Instruct: {DEFAULT_PROMPTS.embedding_query_instruction}\nQuery: {question}", DIM,
        ),
    )
    client = FakeEmbeddingClient(model=MODEL, dim=DIM)
    world = SocialMemoryWorld(store, _NoopModel(), embedding_client=client)

    async def scenario():
        await world.start()
        try:
            return await world.query("s1", QueryRequest(question=question))
        finally:
            await world.stop()

    response = asyncio.run(scenario())

    assert [m.id for m in response.memories] == ["m1"]
    assert client.calls[0] == ((question,), DEFAULT_PROMPTS.embedding_query_instruction)


def test_turn_degrades_to_pure_fts_without_an_embedding_client(tmp_path, monkeypatch):
    """The other degradation paths (raise, timeout) are covered where they
    happen, on `embed_query_vector`: each yields `query_vector=None`, and
    `None` is byte-identical to pure FTS by the test at the top of the
    `_recall_memories` section."""
    store = SqliteWorldStore(tmp_path / "world.db")
    store.initialize()
    store.ensure_space("s1")
    _insert_memory(store, "m1", "a rare distinctive hobby", about_user=1)

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("search_vectors must not be called without an embedding client")

    monkeypatch.setattr(store, "search_vectors", _fail_if_called)
    world = SocialMemoryWorld(store, _NoopModel())

    async def scenario():
        await world.start()
        try:
            return await world.turn(
                "s1",
                TurnRequest(
                    messages=[MessageInput(author="user", content="a rare distinctive hobby")],
                ),
            )
        finally:
            await world.stop()

    response = asyncio.run(scenario())
    assert [m.id for m in response.memory_recall] == ["m1"]


def test_storage_side_embedding_worker_never_carries_a_query_instruction(tmp_path):
    """Cross-check against the query-side tests above: the background
    worker's `embed()` calls (see `world._process_embedding_batch`) must
    never carry `instruction=`, the discipline `tests/test_embedding_worker.py`
    already covers in depth -- this just confirms it still holds once the
    hybrid-retrieval call sites are wired in.
    """
    store = SqliteWorldStore(tmp_path / "world.db")
    store.initialize()
    store.ensure_space("s1")
    store.add_manual_memory("s1", ManualMemoryRequest(content="a memory", about_user=True))
    client = FakeEmbeddingClient(model=MODEL, dim=DIM)
    world = SocialMemoryWorld(store, _NoopModel(), embedding_client=client)

    async def scenario():
        await world.start()
        for _ in range(300):
            if store.pending_embedding_count(model=client.model, dim=client.dim) == 0:
                break
            await asyncio.sleep(0.01)
        await world.stop()

    asyncio.run(scenario())
    assert client.calls
    for _texts, instruction in client.calls:
        assert instruction is None
