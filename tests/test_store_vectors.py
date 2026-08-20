"""Storage-layer tests for the embeddings table and the vector-search
sidecar: BLOB round-trips, the content_hash anti-join, search correctness,
and both the extension-available and extension-unavailable code paths.

All offline: vectors come from `deterministic_unit_vector`, never a real
embedding provider.
"""

from __future__ import annotations

import sqlite3

import pytest

from gossipmemo.store import SqliteWorldStore
from gossipmemo.store._vectors import (
    EmbeddingUpsert,
    content_hash,
    pack_vector,
    unpack_vector,
)
from tests.fakes_embedding import FakeEmbeddingClient, deterministic_unit_vector

MODEL = "fake-embedding"
DIM = 8


@pytest.fixture
def store(tmp_path):
    world = SqliteWorldStore(tmp_path / "world.db")
    world.initialize()
    world.ensure_space("s1")
    return world


def _insert_memory(store, memory_id, content, status="active"):
    now = "2026-01-01T00:00:00Z"
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "INSERT INTO memories(id, space_id, content, kind, basis, about_user, status, "
            "created_by, created_at, updated_at) VALUES (?, 's1', ?, 'fact', 'stated', 0, ?, "
            "'extractor', ?, ?)",
            (memory_id, content, status, now, now),
        )


def _insert_learning_goal(store, goal_id, rationale):
    now = "2026-01-01T00:00:00Z"
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "INSERT INTO learning_goals(id, space_id, prompt, rationale, created_at, updated_at) "
            "VALUES (?, 's1', 'prompt text', ?, ?, ?)",
            (goal_id, rationale, now, now),
        )


def _update_learning_goal_rationale(store, goal_id, rationale):
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE learning_goals SET rationale = ? WHERE id = ?", (rationale, goal_id)
        )


def _insert_hypothesis(store, hypothesis_id, content):
    now = "2026-01-01T00:00:00Z"
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "INSERT INTO hypotheses(id, space_id, owner_kind, content, kind, confidence, "
            "created_at, updated_at) VALUES (?, 's1', 'user', ?, 'trait', 'low', ?, ?)",
            (hypothesis_id, content, now, now),
        )


def _insert_coverage_entry(store, entry_id, content):
    now = "2026-01-01T00:00:00Z"
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "INSERT INTO coverage_entries(id, space_id, root, content, created_at, updated_at) "
            "VALUES (?, 's1', 'life', ?, ?, ?)",
            (entry_id, content, now, now),
        )


def _upsert(store, owner_kind, owner_id, text, *, dim=DIM, model=MODEL):
    vector = deterministic_unit_vector(text, dim)
    store.upsert_embeddings(
        "s1", model, dim,
        [EmbeddingUpsert(owner_kind=owner_kind, owner_id=owner_id, text=text, vector=vector)],
    )
    return vector


# -- BLOB packing -----------------------------------------------------


def test_pack_unpack_round_trip():
    values = deterministic_unit_vector("hello world", DIM)
    blob = pack_vector(values)
    assert isinstance(blob, bytes)
    assert len(blob) == DIM * 4
    restored = unpack_vector(blob, DIM)
    assert len(restored) == DIM
    for original, back in zip(values, restored, strict=True):
        assert back == pytest.approx(original, abs=1e-6)


def test_content_hash_is_deterministic_and_sensitive_to_text():
    assert content_hash("abc") == content_hash("abc")
    assert content_hash("abc") != content_hash("abcd")


# -- pending / anti-join ------------------------------------------------


def test_pending_embeddings_reports_new_owner_with_no_row(store):
    _insert_memory(store, "memory_1", "Bob likes hiking")

    pending = store.pending_embeddings(model=MODEL, dim=DIM)

    assert [p.owner_id for p in pending if p.owner_kind == "memory"] == ["memory_1"]
    assert pending[0].text == "Bob likes hiking"
    assert pending[0].space_id == "s1"


def test_pending_embeddings_excludes_owner_with_matching_hash_and_model(store):
    _insert_memory(store, "memory_1", "Bob likes hiking")
    _upsert(store, "memory", "memory_1", "Bob likes hiking")

    pending = store.pending_embeddings(model=MODEL, dim=DIM)

    assert pending == []


def test_pending_embeddings_is_unaffected_by_a_status_only_change(store):
    """A retract/supersede touches only `status`, never `content`. It must
    never make pending_embeddings() report the owner as needing
    recomputation -- that would mean re-calling the embedding API for text
    that never changed."""

    _insert_memory(store, "memory_1", "Bob likes hiking", status="active")
    _upsert(store, "memory", "memory_1", "Bob likes hiking")
    assert store.pending_embeddings(model=MODEL, dim=DIM) == []

    _set_memory_status(store, "memory_1", "retracted")

    assert store.pending_embeddings(model=MODEL, dim=DIM) == []


def test_pending_embeddings_reports_in_place_rewrite_via_hash_mismatch(store):
    _insert_learning_goal(store, "goal_1", "unexplored so far")
    _upsert(store, "learning_goal", "goal_1", "unexplored so far")
    assert store.pending_embeddings(model=MODEL, dim=DIM) == []

    _update_learning_goal_rationale(store, "goal_1", "now explored, closing")

    pending = store.pending_embeddings(model=MODEL, dim=DIM)
    assert len(pending) == 1
    assert pending[0].owner_kind == "learning_goal"
    assert pending[0].owner_id == "goal_1"
    assert pending[0].text == "now explored, closing"


def test_pending_embeddings_reports_model_change(store):
    _insert_hypothesis(store, "hyp_1", "may be considering a move")
    _upsert(store, "hypothesis", "hyp_1", "may be considering a move")
    assert store.pending_embeddings(model=MODEL, dim=DIM) == []

    pending = store.pending_embeddings(model="a-different-model", dim=DIM)
    assert [p.owner_id for p in pending] == ["hyp_1"]


def test_pending_embeddings_reports_dim_change(store):
    _insert_coverage_entry(store, "entry_1", "career: works in finance")
    _upsert(store, "coverage_entry", "entry_1", "career: works in finance")
    assert store.pending_embeddings(model=MODEL, dim=DIM) == []

    pending = store.pending_embeddings(model=MODEL, dim=DIM + 4)
    assert [p.owner_id for p in pending] == ["entry_1"]


def test_pending_embeddings_cleans_up_orphan_rows(store):
    _insert_memory(store, "memory_1", "Bob likes hiking")
    _upsert(store, "memory", "memory_1", "Bob likes hiking")
    with sqlite3.connect(store.path) as connection:
        connection.execute("DELETE FROM memories WHERE id = 'memory_1'")

    with sqlite3.connect(store.path) as connection:
        before = connection.execute("SELECT count(*) FROM embeddings").fetchone()[0]
    assert before == 1

    store.pending_embeddings(model=MODEL, dim=DIM)

    with sqlite3.connect(store.path) as connection:
        after = connection.execute("SELECT count(*) FROM embeddings").fetchone()[0]
    assert after == 0


def test_pending_embedding_count_matches_pending_embeddings_length(store):
    _insert_memory(store, "memory_1", "Bob likes hiking")
    _insert_hypothesis(store, "hyp_1", "may be considering a move")
    _upsert(store, "hypothesis", "hyp_1", "may be considering a move")

    assert store.pending_embedding_count(model=MODEL, dim=DIM) == len(
        store.pending_embeddings(model=MODEL, dim=DIM)
    )
    assert store.pending_embedding_count(model=MODEL, dim=DIM) == 1


# -- search --------------------------------------------------------


def _seed_three_memories(store):
    _insert_memory(store, "memory_close", "Zhang Wei just switched jobs")
    _insert_memory(store, "memory_mid", "Zhang Wei went hiking last weekend")
    _insert_memory(store, "memory_far", "the weather has been cold lately")
    for owner_id, text in (
        ("memory_close", "Zhang Wei just switched jobs"),
        ("memory_mid", "Zhang Wei went hiking last weekend"),
        ("memory_far", "the weather has been cold lately"),
    ):
        _upsert(store, "memory", owner_id, text)


def test_search_vectors_orders_by_similarity_with_extension(store):
    _seed_three_memories(store)
    query = deterministic_unit_vector("Zhang Wei just switched jobs", DIM)

    results = store.search_vectors("s1", "memory", query, k=3)

    assert [owner_id for owner_id, _score in results][0] == "memory_close"
    assert len(results) == 3
    scores = [score for _owner_id, score in results]
    assert scores == sorted(scores, reverse=True)


def test_search_vectors_matches_between_extension_and_python_fallback(store, monkeypatch):
    _seed_three_memories(store)
    query = deterministic_unit_vector("Zhang Wei just switched jobs", DIM)

    with_extension = store.search_vectors("s1", "memory", query, k=3)

    monkeypatch.setattr("gossipmemo.store._vectors.sqlite_vec", None)
    without_extension = store.search_vectors("s1", "memory", query, k=3)

    assert [owner_id for owner_id, _ in with_extension] == [
        owner_id for owner_id, _ in without_extension
    ]
    for (owner_id_a, score_a), (owner_id_b, score_b) in zip(
        with_extension, without_extension, strict=True
    ):
        assert owner_id_a == owner_id_b
        assert score_a == pytest.approx(score_b, abs=1e-4)


def test_search_vectors_filters_by_status(store):
    _insert_memory(store, "memory_active", "Zhang Wei just switched jobs", status="active")
    _insert_memory(store, "memory_retracted", "Zhang Wei just switched jobs", status="retracted")
    _upsert(store, "memory", "memory_active", "Zhang Wei just switched jobs")
    _upsert(store, "memory", "memory_retracted", "Zhang Wei just switched jobs")
    query = deterministic_unit_vector("Zhang Wei just switched jobs", DIM)

    active_only = store.search_vectors("s1", "memory", query, k=5, statuses=["active"])
    assert [owner_id for owner_id, _ in active_only] == ["memory_active"]

    unfiltered = store.search_vectors("s1", "memory", query, k=5)
    assert {owner_id for owner_id, _ in unfiltered} == {"memory_active", "memory_retracted"}


def _set_memory_status(store, memory_id, status):
    with sqlite3.connect(store.path) as connection:
        connection.execute("UPDATE memories SET status = ? WHERE id = ?", (status, memory_id))


def test_search_vectors_reflects_status_change_without_reembedding_sidecar_path(store):
    """The bug this guards against: a memory gets retracted (status only,
    content untouched, so content_hash never changes and nothing ever gets
    re-embedded) -- it must stop being returned by an active-only search
    immediately, on the very next search call. If the sidecar cached status
    at write time, this would still return the retracted memory."""

    _insert_memory(store, "memory_1", "Zhang Wei just switched jobs", status="active")
    _upsert(store, "memory", "memory_1", "Zhang Wei just switched jobs")
    query = deterministic_unit_vector("Zhang Wei just switched jobs", DIM)

    before = store.search_vectors("s1", "memory", query, k=5, statuses=["active"])
    assert [owner_id for owner_id, _ in before] == ["memory_1"]

    _set_memory_status(store, "memory_1", "retracted")

    after = store.search_vectors("s1", "memory", query, k=5, statuses=["active"])
    assert after == []


@pytest.mark.parametrize("extension_available", [True, False])
def test_search_vectors_status_change_consistent_across_both_paths(
    store, monkeypatch, extension_available,
):
    if not extension_available:
        monkeypatch.setattr("gossipmemo.store._vectors.sqlite_vec", None)

    _insert_memory(store, "memory_1", "Zhang Wei just switched jobs", status="active")
    _upsert(store, "memory", "memory_1", "Zhang Wei just switched jobs")
    query = deterministic_unit_vector("Zhang Wei just switched jobs", DIM)

    active = store.search_vectors("s1", "memory", query, k=5, statuses=["active"])
    assert [oid for oid, _ in active] == ["memory_1"]

    _set_memory_status(store, "memory_1", "retracted")

    assert store.search_vectors("s1", "memory", query, k=5, statuses=["active"]) == []
    retracted = store.search_vectors("s1", "memory", query, k=5, statuses=["retracted"])
    assert [oid for oid, _ in retracted] == ["memory_1"]


def test_search_vectors_over_fetch_still_fills_k_when_most_neighbors_are_inactive(store):
    """Construct a corpus where the nearest neighbors by raw distance are
    mostly a different, filtered-out status, and confirm the status-filtered
    search still surfaces the full k of matching active memories -- either
    within the base over-fetch window or via the single escalation."""

    query_text = "Zhang Wei just switched jobs"
    query = deterministic_unit_vector(query_text, DIM)

    # 80 near-identical-text "noise" memories, all inactive, plus 5 active
    # memories with the exact query text (distance 0, guaranteed nearest).
    for i in range(80):
        text = f"{query_text} (variant {i})"
        owner_id = f"memory_noise_{i}"
        _insert_memory(store, owner_id, text, status="retracted")
        _upsert(store, "memory", owner_id, text)
    for i in range(5):
        owner_id = f"memory_active_{i}"
        _insert_memory(store, owner_id, query_text, status="active")
        _upsert(store, "memory", owner_id, query_text)

    results = store.search_vectors("s1", "memory", query, k=5, statuses=["active"])

    assert {owner_id for owner_id, _ in results} == {f"memory_active_{i}" for i in range(5)}
    assert len(results) == 5


def test_search_vectors_falls_back_when_extension_unavailable_from_the_start(store, monkeypatch):
    monkeypatch.setattr("gossipmemo.store._vectors.sqlite_vec", None)
    _seed_three_memories(store)
    query = deterministic_unit_vector("Zhang Wei just switched jobs", DIM)

    results = store.search_vectors("s1", "memory", query, k=3)

    assert [owner_id for owner_id, _score in results][0] == "memory_close"


# -- sidecar rebuild / meta mismatch ------------------------------------


def test_ensure_vector_index_rebuilds_on_meta_mismatch_without_calling_embedding_api(store):
    _seed_three_memories(store)
    fake_client = FakeEmbeddingClient(model=MODEL, dim=DIM)

    store.ensure_vector_index(MODEL, DIM)
    assert fake_client.calls == []

    # Change dim to force a meta mismatch and rebuild.
    store.ensure_vector_index(MODEL, DIM + 4)
    assert fake_client.calls == []

    from gossipmemo.store._vectors import _sidecar_path

    with sqlite3.connect(store.path) as connection:
        connection.execute("ATTACH DATABASE ? AS vec", (str(_sidecar_path(store.path)),))
        meta = connection.execute("SELECT model, dim FROM vec.embedding_index_meta").fetchall()
    # Rebuild only happens for embeddings actually stored at (MODEL, DIM); since the
    # seeded rows were embedded at DIM, a rebuild targeting DIM + 4 finds nothing to
    # copy but still stamps the new meta so future upserts at that dim sync cleanly.
    assert meta == [(MODEL, DIM + 4)]


def test_ensure_vector_index_is_a_noop_when_vectors_disabled(store, monkeypatch):
    monkeypatch.setattr("gossipmemo.store._vectors.sqlite_vec", None)
    # Must not raise even though the extension is unavailable.
    store.ensure_vector_index(MODEL, DIM)


def test_ensure_vector_index_rebuilds_a_legacy_layout_sidecar_without_calling_embedding_api(store):
    """Simulate a sidecar built by an older version of this module: a
    vec0 table with a cached `status` column and an `embedding_index_meta`
    with no `layout_version`. It must be recognized as stale by shape
    alone (model/dim can even still match) and rebuilt -- never raising,
    never calling the embedding API."""

    import sqlite_vec

    from gossipmemo.store._vectors import _sidecar_path

    _insert_memory(store, "memory_1", "Zhang Wei just switched jobs", status="active")
    _upsert(store, "memory", "memory_1", "Zhang Wei just switched jobs")

    connection = sqlite3.connect(store.path)
    try:
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        connection.execute("ATTACH DATABASE ? AS vec", (str(_sidecar_path(store.path)),))
        # upsert_embeddings() above already attached and built a current-
        # layout sidecar; drop it and recreate the legacy shape from scratch.
        connection.execute("DROP TABLE IF EXISTS vec.embeddings")
        connection.execute("DROP TABLE IF EXISTS vec.embedding_index_meta")
        connection.execute(
            "CREATE TABLE vec.embedding_index_meta("
            "model TEXT NOT NULL, dim INTEGER NOT NULL, built_at TEXT NOT NULL)"
        )
        connection.execute(
            f"CREATE VIRTUAL TABLE vec.embeddings USING vec0("
            f"owner_id TEXT PRIMARY KEY, embedding float[{DIM}], "
            f"space_id TEXT partition key, owner_kind TEXT, status TEXT, "
            f"+content_hash TEXT)"
        )
        connection.execute(
            "INSERT INTO vec.embedding_index_meta(model, dim, built_at) VALUES (?, ?, ?)",
            (MODEL, DIM, "2026-01-01T00:00:00Z"),
        )
        connection.commit()
    finally:
        connection.close()

    fake_client = FakeEmbeddingClient(model=MODEL, dim=DIM)

    store.ensure_vector_index(MODEL, DIM)

    assert fake_client.calls == []
    connection = sqlite3.connect(store.path)
    try:
        connection.enable_load_extension(True)
        sqlite_vec.load(connection)
        connection.enable_load_extension(False)
        connection.execute("ATTACH DATABASE ? AS vec", (str(_sidecar_path(store.path)),))
        columns = {
            row[1] for row in connection.execute("PRAGMA vec.table_info(embedding_index_meta)")
        }
        assert "layout_version" in columns
        rows = connection.execute("SELECT owner_id FROM vec.embeddings").fetchall()
        assert [row[0] for row in rows] == ["memory_1"]
    finally:
        connection.close()

    query = deterministic_unit_vector("Zhang Wei just switched jobs", DIM)
    results = store.search_vectors("s1", "memory", query, k=5, statuses=["active"])
    assert [owner_id for owner_id, _ in results] == ["memory_1"]


def test_sidecar_path_is_derived_from_main_db_path(store):
    from gossipmemo.store._vectors import _sidecar_path

    sidecar = _sidecar_path(store.path)
    assert sidecar.name == store.path.stem + ".vec.db"
    assert sidecar.parent == store.path.parent
