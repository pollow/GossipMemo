PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS spaces (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    ego_person_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS people (
    id TEXT PRIMARY KEY,
    space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
    display_name TEXT NOT NULL,
    normalized_name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    merged_into_person_id TEXT REFERENCES people(id),
    profile_card TEXT NOT NULL DEFAULT '{}',
    memory_revision INTEGER NOT NULL DEFAULT 0,
    profile_memory_revision INTEGER NOT NULL DEFAULT 0,
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
    context_key TEXT,
    valid_from TEXT,
    valid_to TEXT,
    UNIQUE(person_id, normalized_value, context_key)
);

CREATE TABLE IF NOT EXISTS person_external_identities (
    id TEXT PRIMARY KEY,
    space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
    person_id TEXT NOT NULL REFERENCES people(id) ON DELETE CASCADE,
    provider TEXT NOT NULL,
    external_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(space_id, provider, external_id)
);

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
    memory_revision INTEGER NOT NULL DEFAULT 0,
    profile_memory_revision INTEGER NOT NULL DEFAULT 0,
    profile_updated_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    CHECK(person_a_id < person_b_id),
    UNIQUE(space_id, person_a_id, person_b_id)
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
    author_person_id TEXT REFERENCES people(id),
    author_raw TEXT NOT NULL,
    content TEXT NOT NULL,
    occurred_at TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    source_provider TEXT NOT NULL,
    source_conversation_key TEXT,
    source_item_id TEXT,
    source_metadata TEXT NOT NULL DEFAULT '{}',
    idempotency_key TEXT,
    extraction_policy TEXT NOT NULL DEFAULT 'balanced',
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
    status TEXT NOT NULL DEFAULT 'active',
    valid_from TEXT,
    valid_to TEXT,
    supersedes_memory_id TEXT REFERENCES memories(id),
    invalidated_at TEXT,
    invalidation_reason TEXT,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS memories_active_by_space
ON memories(space_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS memory_people (
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    person_id TEXT NOT NULL REFERENCES people(id),
    role TEXT NOT NULL,
    PRIMARY KEY(memory_id, person_id, role)
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

CREATE TABLE IF NOT EXISTS memory_sources (
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    message_id TEXT NOT NULL REFERENCES messages(id) ON DELETE CASCADE,
    source_role TEXT NOT NULL DEFAULT 'support',
    evidence_text TEXT,
    PRIMARY KEY(memory_id, message_id, source_role)
);

CREATE TABLE IF NOT EXISTS memory_derivations (
    derived_memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    source_memory_id TEXT NOT NULL REFERENCES memories(id),
    derivation_role TEXT NOT NULL DEFAULT 'support',
    PRIMARY KEY(derived_memory_id, source_memory_id, derivation_role)
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
