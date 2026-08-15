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
    status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open', 'partial', 'answered', 'deferred', 'retired')),
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
    status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open', 'promoted', 'rejected', 'superseded', 'retired')),
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
