from __future__ import annotations

import sqlite3

import pytest

from gossipmemo import migrations
from gossipmemo.migrations import (
    CURRENT_VERSION,
    MigrationError,
    migrate_database,
)
from gossipmemo.store import SqliteWorldStore

# Captured verbatim from `git show main:src/gossipmemo/schema.sql` -- the
# schema of the deployed version-1 database. Do NOT shell out to git at
# test time; this is a frozen historical artifact, not something that
# should track the current schema.sql.
V1_SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS spaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_models (
    space_id TEXT PRIMARY KEY REFERENCES spaces(id) ON DELETE CASCADE,
    profile_card TEXT NOT NULL DEFAULT '{}',
    profile_source_updated_at TEXT,
    profile_updated_at TEXT
);

CREATE TABLE IF NOT EXISTS coverage_maps (
    space_id TEXT PRIMARY KEY REFERENCES spaces(id) ON DELETE CASCADE,
    revision INTEGER NOT NULL DEFAULT 0,
    source_watermark TEXT,
    source_cursor_id TEXT,
    criteria TEXT NOT NULL,
    boundaries TEXT NOT NULL DEFAULT '[]',
    life_periods TEXT NOT NULL DEFAULT '[]',
    relationship_arcs TEXT NOT NULL DEFAULT '[]',
    behavioral_contexts TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS learning_goals (
    id TEXT PRIMARY KEY,
    space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
    prompt TEXT NOT NULL,
    rationale TEXT NOT NULL,
    criteria_refs TEXT NOT NULL,
    boundary_ids TEXT NOT NULL,
    focus_kind TEXT NOT NULL DEFAULT 'user' CHECK(focus_kind IN ('user', 'person', 'relationship')),
    focus_id TEXT,
    status TEXT NOT NULL DEFAULT 'open'
        CHECK(status IN ('open', 'partial', 'answered', 'deferred', 'retired')),
    status_reason TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK((focus_kind = 'user' AND focus_id IS NULL) OR
          (focus_kind IN ('person', 'relationship') AND focus_id IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS learning_goals_by_space_status
ON learning_goals(space_id, status, updated_at DESC);

CREATE TABLE IF NOT EXISTS continuities (
    space_id TEXT PRIMARY KEY REFERENCES spaces(id) ON DELETE CASCADE,
    text TEXT NOT NULL DEFAULT '',
    related_person_ids TEXT NOT NULL DEFAULT '[]',
    through_message_id TEXT,
    through_message_rowid INTEGER,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS people (
    id TEXT PRIMARY KEY,
    space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
    display_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    merged_into_person_id TEXT REFERENCES people(id),
    profile_card TEXT NOT NULL DEFAULT '{}',
    profile_source_updated_at TEXT,
    profile_updated_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS person_aliases (
    id TEXT PRIMARY KEY,
    space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
    person_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    value TEXT NOT NULL,
    normalized_value TEXT NOT NULL,
    UNIQUE(person_id, normalized_value)
);

CREATE INDEX IF NOT EXISTS person_alias_lookup
ON person_aliases(space_id, normalized_value);

CREATE TABLE IF NOT EXISTS relationships (
    id TEXT PRIMARY KEY,
    space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
    person_a_id TEXT NOT NULL REFERENCES people(id),
    person_b_id TEXT NOT NULL REFERENCES people(id),
    facets TEXT NOT NULL DEFAULT '[]',
    closeness TEXT,
    tone TEXT,
    summary TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'unknown',
    profile_source_updated_at TEXT,
    profile_updated_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK(person_a_id < person_b_id),
    UNIQUE(space_id, person_a_id, person_b_id)
);

CREATE TABLE IF NOT EXISTS extraction_batches (
    id TEXT PRIMARY KEY,
    space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
    author TEXT NOT NULL CHECK(author IN ('user', 'assistant')),
    content TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    source_provider TEXT NOT NULL,
    source_conversation_key TEXT,
    source_item_id TEXT,
    source_metadata TEXT NOT NULL DEFAULT '{}',
    idempotency_key TEXT,
    extraction_batch_id TEXT REFERENCES extraction_batches(id),
    extraction_state TEXT NOT NULL DEFAULT 'pending',
    extraction_attempts INTEGER NOT NULL DEFAULT 0,
    extracted_at TEXT,
    last_extraction_error TEXT,
    UNIQUE(space_id, idempotency_key)
);

CREATE UNIQUE INDEX IF NOT EXISTS messages_source_identity
ON messages(
    space_id,
    source_provider,
    COALESCE(source_conversation_key, ''),
    source_item_id
)
WHERE source_item_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    kind TEXT NOT NULL,
    basis TEXT NOT NULL,
    about_user INTEGER NOT NULL DEFAULT 0 CHECK(about_user IN (0, 1)),
    status TEXT NOT NULL DEFAULT 'active',
    valid_from TEXT,
    valid_to TEXT,
    supersedes_memory_id TEXT REFERENCES memories(id),
    invalidated_at TEXT,
    invalidation_reason TEXT,
    source_batch_id TEXT REFERENCES extraction_batches(id),
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS memories_active_by_space
ON memories(space_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS memory_people (
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    person_id TEXT NOT NULL REFERENCES people(id),
    PRIMARY KEY(memory_id, person_id)
);

CREATE INDEX IF NOT EXISTS memory_people_by_person
ON memory_people(person_id, memory_id);

CREATE TABLE IF NOT EXISTS memory_relationships (
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relationship_id TEXT NOT NULL REFERENCES relationships(id),
    role TEXT NOT NULL DEFAULT 'about',
    PRIMARY KEY(memory_id, relationship_id, role)
);

CREATE INDEX IF NOT EXISTS memory_relationships_by_relationship
ON memory_relationships(relationship_id, memory_id);

CREATE TABLE IF NOT EXISTS memory_derivations (
    derived_memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    source_memory_id TEXT NOT NULL REFERENCES memories(id),
    derivation_role TEXT NOT NULL DEFAULT 'support',
    PRIMARY KEY(derived_memory_id, source_memory_id, derivation_role)
);

CREATE TABLE IF NOT EXISTS hypotheses (
    id TEXT PRIMARY KEY,
    space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
    owner_kind TEXT NOT NULL CHECK(owner_kind IN ('user', 'person', 'relationship')),
    owner_id TEXT,
    content TEXT NOT NULL,
    kind TEXT NOT NULL,
    confidence TEXT NOT NULL CHECK(confidence IN ('low', 'medium', 'high')),
    status TEXT NOT NULL DEFAULT 'open'
        CHECK(status IN ('open', 'promoted', 'rejected', 'superseded', 'retired')),
    status_reason TEXT,
    promoted_memory_id TEXT REFERENCES memories(id),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK((owner_kind = 'user' AND owner_id IS NULL) OR
          (owner_kind IN ('person', 'relationship') AND owner_id IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS hypotheses_by_owner
ON hypotheses(space_id, owner_kind, owner_id, status);

CREATE TABLE IF NOT EXISTS hypothesis_evidence (
    hypothesis_id TEXT NOT NULL REFERENCES hypotheses(id) ON DELETE CASCADE,
    memory_id TEXT NOT NULL REFERENCES memories(id),
    role TEXT NOT NULL DEFAULT 'support' CHECK(role IN ('support', 'counter')),
    PRIMARY KEY(hypothesis_id, memory_id, role)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
    content,
    content='memories',
    content_rowid='rowid',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS memories_fts_insert AFTER INSERT ON memories BEGIN
    INSERT INTO memory_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_delete AFTER DELETE ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS memories_fts_update AFTER UPDATE OF content ON memories BEGIN
    INSERT INTO memory_fts(memory_fts, rowid, content)
    VALUES ('delete', old.rowid, old.content);
    INSERT INTO memory_fts(rowid, content) VALUES (new.rowid, new.content);
END;
"""


def _build_v1_database(path) -> None:
    """Create a real v1 database with representative rows across spaces:
    messages, memories (exercising the FTS triggers), people, aliases,
    relationships, learning_goals with the old criteria_refs/boundary_ids
    columns, and coverage_maps."""

    connection = sqlite3.connect(path)
    try:
        connection.executescript(V1_SCHEMA)
        now = "2026-01-01T00:00:00Z"
        connection.executemany(
            "INSERT INTO spaces(id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            [("personal", "Personal", now, now), ("work", "Work", now, now)],
        )
        connection.executemany(
            "INSERT INTO user_models(space_id) VALUES (?)",
            [("personal",), ("work",)],
        )
        connection.executemany(
            """INSERT INTO coverage_maps(
                space_id, revision, criteria, updated_at
            ) VALUES (?, ?, ?, ?)""",
            [("personal", 3, '["life"]', now), ("work", 1, '["career"]', now)],
        )
        connection.execute(
            """INSERT INTO people(id, space_id, display_name, created_at, updated_at)
               VALUES ('person_bob', 'personal', 'Bob', ?, ?)""",
            (now, now),
        )
        connection.execute(
            """INSERT INTO person_aliases(id, space_id, person_id, value, normalized_value)
               VALUES ('alias_bob', 'personal', 'person_bob', 'Bob', 'bob')"""
        )
        connection.execute(
            """INSERT INTO messages(
                id, space_id, author, content, occurred_at, ingested_at, source_provider
            ) VALUES ('message_1', 'personal', 'user', 'Bob likes hiking', ?, ?, 'test')""",
            (now, now),
        )
        connection.execute(
            """INSERT INTO memories(
                id, space_id, content, kind, basis, about_user, created_by, created_at, updated_at
            ) VALUES ('memory_1', 'personal', 'Bob likes hiking on weekends', 'fact',
                      'stated', 0, 'extractor', ?, ?)""",
            (now, now),
        )
        connection.execute(
            "INSERT INTO memory_people(memory_id, person_id) VALUES ('memory_1', 'person_bob')"
        )
        connection.execute(
            """INSERT INTO learning_goals(
                id, space_id, prompt, rationale, criteria_refs, boundary_ids,
                focus_kind, status, created_at, updated_at
            ) VALUES (
                'goal_1', 'personal', 'What does Bob do for work?', 'unexplored',
                '["criteria_a"]', '["boundary_a"]', 'user', 'open', ?, ?
            )""",
            (now, now),
        )
        connection.commit()
    finally:
        connection.close()


def _table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    }


def _structural_fingerprint(connection: sqlite3.Connection):
    """A normalized structural fingerprint: table/index/trigger names, and
    each table's columns as an order-independent set. Raw `sqlite_master.sql`
    text is not compared directly because SQLite's ALTER TABLE machinery can
    legitimately reorder columns or drop incidental "IF NOT EXISTS" wording
    without changing the actual schema."""

    tables = {}
    for name in _table_names(connection):
        columns = connection.execute(f"PRAGMA table_info({name})").fetchall()
        tables[name] = frozenset(
            (column[1], column[2], column[3], column[4], column[5]) for column in columns
        )
    indexes = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'index' "
            "AND name NOT LIKE 'sqlite_autoindex_%'"
        ).fetchall()
    }
    triggers = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'trigger'"
        ).fetchall()
    }
    return tables, indexes, triggers


def test_migrate_stamps_version_and_preserves_rows(tmp_path):
    path = tmp_path / "world.db"
    _build_v1_database(path)

    migrate_database(path)

    connection = sqlite3.connect(path)
    try:
        history = connection.execute(
            "SELECT version, checksum FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [row[0] for row in history] == [1, 2, 3]
        assert history[-1][0] == CURRENT_VERSION

        spaces = {row[0] for row in connection.execute("SELECT id FROM spaces")}
        assert spaces == {"personal", "work"}
        assert connection.execute("SELECT content FROM messages WHERE id = 'message_1'").fetchone()[
            0
        ] == "Bob likes hiking"
        assert connection.execute(
            "SELECT content FROM memories WHERE id = 'memory_1'"
        ).fetchone()[0] == "Bob likes hiking on weekends"
        assert connection.execute(
            "SELECT person_id FROM memory_people WHERE memory_id = 'memory_1'"
        ).fetchone()[0] == "person_bob"
        assert connection.execute(
            "SELECT display_name FROM people WHERE id = 'person_bob'"
        ).fetchone()[0] == "Bob"
    finally:
        connection.close()


def test_learning_goals_kept_with_entry_ids_default_and_old_columns_dropped(tmp_path):
    path = tmp_path / "world.db"
    _build_v1_database(path)

    migrate_database(path)

    connection = sqlite3.connect(path)
    try:
        row = connection.execute(
            "SELECT prompt, rationale, entry_ids, focus_kind, status "
            "FROM learning_goals WHERE id = 'goal_1'"
        ).fetchone()
        assert row == ("What does Bob do for work?", "unexplored", "[]", "user", "open")
        columns = {
            c[1] for c in connection.execute("PRAGMA table_info(learning_goals)").fetchall()
        }
        assert "criteria_refs" not in columns
        assert "boundary_ids" not in columns
        assert "entry_ids" in columns
    finally:
        connection.close()


def test_coverage_roots_seeded_and_coverage_maps_dropped(tmp_path):
    path = tmp_path / "world.db"
    _build_v1_database(path)

    migrate_database(path)
    # Coverage-root seeding happens in SqliteWorldStore.initialize(), which
    # runs migrate_database() first and then seeds every space's roots.
    SqliteWorldStore(path).initialize()

    connection = sqlite3.connect(path)
    try:
        assert "coverage_maps" not in _table_names(connection)
        from gossipmemo.models import COVERAGE_ROOTS

        for space_id in ("personal", "work"):
            rows = connection.execute(
                "SELECT root, source_watermark, source_cursor_id "
                "FROM coverage_roots WHERE space_id = ?",
                (space_id,),
            ).fetchall()
            roots = {row[0] for row in rows}
            assert roots == set(COVERAGE_ROOTS)
            assert all(row[1] is None and row[2] is None for row in rows)
    finally:
        connection.close()


def test_memory_fts_works_after_migration(tmp_path):
    path = tmp_path / "world.db"
    _build_v1_database(path)

    migrate_database(path)

    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT m.id FROM memory_fts JOIN memories m ON m.rowid = memory_fts.rowid "
            "WHERE memory_fts MATCH 'hiking'"
        ).fetchall()
        assert [row[0] for row in rows] == ["memory_1"]
    finally:
        connection.close()


def test_embeddings_table_created_empty_by_v2_to_v3_migration(tmp_path):
    path = tmp_path / "world.db"
    _build_v1_database(path)

    migrate_database(path)

    connection = sqlite3.connect(path)
    try:
        assert "embeddings" in _table_names(connection)
        assert connection.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0
        columns = {c[1] for c in connection.execute("PRAGMA table_info(embeddings)").fetchall()}
        assert columns == {
            "space_id", "owner_kind", "owner_id", "model", "dim",
            "vector", "content_hash", "created_at",
        }
    finally:
        connection.close()


def test_structural_equivalence_with_fresh_database(tmp_path):
    migrated_path = tmp_path / "migrated.db"
    _build_v1_database(migrated_path)
    migrate_database(migrated_path)
    SqliteWorldStore(migrated_path).initialize()

    fresh_path = tmp_path / "fresh.db"
    SqliteWorldStore(fresh_path).initialize()

    migrated_connection = sqlite3.connect(migrated_path)
    fresh_connection = sqlite3.connect(fresh_path)
    try:
        migrated_fingerprint = _structural_fingerprint(migrated_connection)
        fresh_fingerprint = _structural_fingerprint(fresh_connection)
        # schema_migrations rows legitimately differ (a migrated database's
        # history has 2 rows, a fresh one has 1); everything else must match.
        assert migrated_fingerprint == fresh_fingerprint
    finally:
        migrated_connection.close()
        fresh_connection.close()


def test_migration_is_idempotent_on_restart(tmp_path):
    path = tmp_path / "world.db"
    _build_v1_database(path)

    migrate_database(path)
    migrate_database(path)  # simulates a container restart on the new image

    connection = sqlite3.connect(path)
    try:
        history = connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert [row[0] for row in history] == [1, 2, 3]
    finally:
        connection.close()


def test_backup_file_created_before_migrating(tmp_path):
    path = tmp_path / "world.db"
    _build_v1_database(path)

    migrate_database(path)

    backups = list(tmp_path.glob(".world.db.pre-migration-v1.*.bak"))
    assert len(backups) == 1
    backup_connection = sqlite3.connect(backups[0])
    try:
        # The backup is a full pre-migration snapshot: it still has the old
        # coverage_maps table and the old learning_goals columns.
        assert "coverage_maps" in _table_names(backup_connection)
        assert backup_connection.execute(
            "SELECT content FROM memories WHERE id = 'memory_1'"
        ).fetchone()[0] == "Bob likes hiking on weekends"
    finally:
        backup_connection.close()


def test_failed_migration_rolls_back_and_leaves_old_version(tmp_path, monkeypatch):
    path = tmp_path / "world.db"
    _build_v1_database(path)

    def _boom(connection: sqlite3.Connection) -> None:
        connection.execute("DROP TABLE coverage_maps")
        raise RuntimeError("simulated failure mid-migration")

    monkeypatch.setitem(
        migrations.MIGRATIONS,
        2,
        migrations.Migration(version=2, description="broken", upgrade=_boom),
    )

    with pytest.raises(RuntimeError, match="simulated failure"):
        migrate_database(path)

    connection = sqlite3.connect(path)
    try:
        # No migration history was committed at all: the whole batch,
        # including the version-1 baseline row, rolled back together.
        assert _table_names(connection) & {"schema_migrations"} == set() or not connection.execute(
            "SELECT 1 FROM schema_migrations"
        ).fetchall()
        assert "coverage_maps" in _table_names(connection)
        assert connection.execute(
            "SELECT content FROM memories WHERE id = 'memory_1'"
        ).fetchone()[0] == "Bob likes hiking on weekends"
    finally:
        connection.close()


def test_refuses_database_newer_than_the_program(tmp_path):
    path = tmp_path / "world.db"
    SqliteWorldStore(path).initialize()  # stamps CURRENT_VERSION

    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "INSERT INTO schema_migrations(version, applied_at, description, checksum) "
            "VALUES (?, ?, ?, ?)",
            (CURRENT_VERSION + 1, "2026-01-01T00:00:00Z", "from the future", "not-a-real-checksum"),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(MigrationError, match="newer"):
        migrate_database(path)


def test_refuses_tampered_history(tmp_path):
    path = tmp_path / "world.db"
    SqliteWorldStore(path).initialize()

    connection = sqlite3.connect(path)
    try:
        connection.execute("UPDATE schema_migrations SET checksum = 'tampered'")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(MigrationError):
        migrate_database(path)


def test_fresh_database_stamped_at_current_version_without_replaying_history(tmp_path):
    path = tmp_path / "world.db"

    migrate_database(path)

    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT version, checksum FROM schema_migrations ORDER BY version"
        ).fetchall()
        assert rows == [(CURRENT_VERSION, migrations._FRESH_CREATE_MARKER)]
    finally:
        connection.close()


def test_adopts_a_v2_shaped_dev_database_without_history(tmp_path):
    path = tmp_path / "world.db"
    # A development checkout that ran the current schema.sql directly,
    # before this migration runner existed: it has the v2 tables and data,
    # but no schema_migrations table at all.
    store = SqliteWorldStore(path)
    store.initialize()
    store.ensure_space("personal")
    connection = sqlite3.connect(path)
    try:
        # Real pre-migration-runner checkouts never had this table at all
        # (it ships as part of this feature); dropping it, not emptying it,
        # is what reproduces that database shape.
        connection.execute("DROP TABLE schema_migrations")
        connection.commit()
    finally:
        connection.close()

    migrate_database(path)

    connection = sqlite3.connect(path)
    try:
        rows = connection.execute(
            "SELECT version, checksum FROM schema_migrations"
        ).fetchall()
        assert rows == [(CURRENT_VERSION, migrations._ADOPTED_MARKER)]
        assert connection.execute(
            "SELECT id FROM spaces WHERE id = 'personal'"
        ).fetchone() is not None
    finally:
        connection.close()


def test_refuses_unrecognized_database_shape(tmp_path):
    path = tmp_path / "world.db"
    connection = sqlite3.connect(path)
    try:
        connection.execute("CREATE TABLE mystery_app_data(id TEXT PRIMARY KEY)")
        connection.execute("INSERT INTO mystery_app_data(id) VALUES ('x')")
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(MigrationError, match="refusing to guess"):
        migrate_database(path)


def test_initialize_runs_migration_before_applying_schema(tmp_path):
    """SqliteWorldStore.initialize() must migrate a v1 database in place,
    not error out or silently drop its data, matching the deployment
    contract in README.md / CLAUDE.md."""

    path = tmp_path / "world.db"
    _build_v1_database(path)

    SqliteWorldStore(path).initialize()

    connection = sqlite3.connect(path)
    try:
        assert connection.execute(
            "SELECT display_name FROM people WHERE id = 'person_bob'"
        ).fetchone()[0] == "Bob"
        assert "coverage_maps" not in _table_names(connection)
    finally:
        connection.close()
