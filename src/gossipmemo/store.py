from __future__ import annotations

import json
import re
import sqlite3
import uuid
import hashlib
import unicodedata
from collections import Counter
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Protocol

from .models import (
    ExtractionResult,
    ManualMemoryRequest,
    MemoryView,
    MessageInput,
    ModelMessage,
    PersonReasoningResult,
    PersonView,
    QueryContext,
    QueryRequest,
    RelationshipReasoningResult,
    RelationshipView,
    SupersedeRequest,
    UserModelReasoningResult,
    UserModelView,
    ContinuityReasoningResult,
    ContinuityView,
    ContextBundle,
    utc_now,
)


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _now() -> str:
    return utc_now().isoformat()


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _loads(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def _fts_query(question: str, excluded: Iterable[str] = ()) -> str | None:
    """Build a conservative FTS5 OR query from natural-language input."""

    if question.casefold().strip() in {"dossier", "reason"}:
        return None
    excluded_terms = {_normalized(value) for value in excluded}
    terms: list[str] = []
    for token in re.findall(r"[^\W_]+", question.casefold(), flags=re.UNICODE):
        if len(token) < 3 or _normalized(token) in excluded_terms:
            continue
        if any("\u4e00" <= char <= "\u9fff" for char in token) and len(token) > 3:
            terms.extend(token[index : index + 3] for index in range(len(token) - 2))
        else:
            terms.append(token)
    unique = list(dict.fromkeys(terms))[:16]
    return " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in unique) or None


class WorldStore(Protocol):
    """Internal persistence seam expressed in social-memory operations."""

    def initialize(self) -> None: ...

    def ensure_space(self, space_id: str, name: str | None = None) -> str: ...

    def record_messages(
        self, space_id: str, messages: list[MessageInput]
    ) -> list[str]: ...

    def create_extraction_batch(
        self, space_id: str, message_ids: list[str]
    ) -> str | None: ...

    def unbatched_messages(self, space_id: str) -> list[tuple[str, str]]: ...

    def load_batch(self, space_id: str, batch_id: str) -> list[ModelMessage]: ...

    def load_extraction_context(
        self, space_id: str, batch_id: str, limit: int = 2
    ) -> list[ModelMessage]: ...

    def load_known_people(
        self, space_id: str, messages: list[ModelMessage]
    ) -> list[dict[str, Any]]: ...

    def apply_extraction(
        self, space_id: str, batch_id: str, result: ExtractionResult
    ) -> tuple[set[str], set[str]]: ...

    def read(self, space_id: str, request: QueryRequest) -> QueryContext: ...

    def user_model_context(
        self, space_id: str
    ) -> tuple[UserModelView, list[MemoryView], str | None] | None: ...

    def apply_user_model_reasoning(
        self, space_id: str, expected_watermark: str | None,
        result: UserModelReasoningResult,
    ) -> bool: ...

    def continuity_context(self, space_id: str) -> tuple[ContinuityView | None, list[ModelMessage]] | None: ...

    def pending_continuities(self, threshold: int = 20) -> list[str]: ...

    def apply_continuity_reasoning(self, space_id: str, expected_through_message_id: str | None, result: ContinuityReasoningResult) -> bool: ...

    def context_bundle(self, space_id: str) -> ContextBundle: ...

    def match_people_in_text(self, space_id: str, text: str) -> list[PersonView]: ...

    def recall_user_memories(self, space_id: str, text: str, limit: int = 5) -> list[MemoryView]: ...

    def stale_entities(
        self,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]: ...

    def overwrite_user_model(self, space_id: str, profile_card: dict[str, Any]) -> None: ...


class AmbiguousPersonError(ValueError):
    def __init__(self, reference: str) -> None:
        super().__init__(f"person reference is ambiguous: {reference}")
        self.reference = reference


class SqliteWorldStore:
    """SQLite Adapter. Each method owns its short atomic write internally."""

    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        schema = Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
        with self._connect() as connection:
            connection.executescript(schema)

    def ensure_space(self, space_id: str, name: str | None = None) -> str:
        with self._connect() as connection:
            now = _now()
            connection.execute(
                "INSERT OR IGNORE INTO spaces(id, name, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (space_id, name or space_id, now, now),
            )
            connection.execute(
                "INSERT OR IGNORE INTO user_models(space_id) VALUES (?)", (space_id,)
            )
            connection.execute(
                "INSERT OR IGNORE INTO continuities(space_id, updated_at) VALUES (?, ?)",
                (space_id, now),
            )
            return space_id

    def _find_person(
        self, connection: sqlite3.Connection, space_id: str, reference: str
    ) -> sqlite3.Row | None:
        row = connection.execute(
            "SELECT * FROM people WHERE space_id = ? AND id = ? AND status = 'active'",
            (space_id, reference),
        ).fetchone()
        if row:
            return row
        normalized = _normalized(reference)
        alias_rows = connection.execute(
            """
            SELECT p.* FROM people p
            JOIN person_aliases a ON a.person_id = p.id
            WHERE p.space_id = ? AND a.normalized_value = ? AND p.status = 'active'
            LIMIT 2
            """,
            (space_id, normalized),
        ).fetchall()
        if len(alias_rows) == 1:
            return alias_rows[0]
        if len(alias_rows) > 1:
            raise AmbiguousPersonError(reference)
        return None

    def _create_person(
        self,
        connection: sqlite3.Connection,
        space_id: str,
        display_name: str,
        *,
        reuse_unique_name: bool = True,
    ) -> str:
        if reuse_unique_name:
            existing = self._find_person(connection, space_id, display_name)
            if existing:
                return existing["id"]
        person_id = _id("person")
        now = _now()
        connection.execute(
            """
            INSERT INTO people(
                id, space_id, display_name, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (person_id, space_id, display_name, now, now),
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO person_aliases(
                id, space_id, person_id, value, normalized_value
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (_id("alias"), space_id, person_id, display_name, _normalized(display_name)),
        )
        return person_id

    def record_messages(
        self, space_id: str, messages: list[MessageInput]
    ) -> list[str]:
        self.ensure_space(space_id)
        message_ids: list[str] = []
        with self._connect() as connection:
            for message in messages:
                duplicate = None
                if message.idempotency_key:
                    duplicate = connection.execute(
                        """
                        SELECT id, extraction_state FROM messages
                        WHERE space_id = ? AND idempotency_key = ?
                        """,
                        (space_id, message.idempotency_key),
                    ).fetchone()
                if not duplicate and message.source.item_id:
                    duplicate = connection.execute(
                        """
                        SELECT id, extraction_state FROM messages
                        WHERE space_id = ? AND source_provider = ?
                          AND source_conversation_key IS ? AND source_item_id = ?
                        """,
                        (
                            space_id,
                            message.source.provider,
                            message.source.conversation_key,
                            message.source.item_id,
                        ),
                    ).fetchone()
                if duplicate:
                    message_ids.append(duplicate["id"])
                    continue

                message_id = _id("message")
                try:
                    connection.execute(
                        """
                        INSERT INTO messages(
                            id, space_id, author, content,
                            occurred_at, ingested_at, source_provider,
                            source_conversation_key, source_item_id, source_metadata,
                            idempotency_key
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            message_id,
                            space_id,
                            message.author,
                            message.content,
                            message.occurred_at.isoformat(),
                            _now(),
                            message.source.provider,
                            message.source.conversation_key,
                            message.source.item_id,
                            _json(message.source.metadata),
                            message.idempotency_key,
                        ),
                    )
                except sqlite3.IntegrityError:
                    # A concurrent caller may have won either unique identity
                    # after our preflight read. Resolve the durable winner.
                    if message.idempotency_key:
                        duplicate = connection.execute(
                            """
                            SELECT id, extraction_state FROM messages
                            WHERE space_id = ? AND idempotency_key = ?
                            """,
                            (space_id, message.idempotency_key),
                        ).fetchone()
                    if not duplicate and message.source.item_id:
                        duplicate = connection.execute(
                            """
                            SELECT id, extraction_state FROM messages
                            WHERE space_id = ? AND source_provider = ?
                              AND source_conversation_key IS ? AND source_item_id = ?
                            """,
                            (
                                space_id,
                                message.source.provider,
                                message.source.conversation_key,
                                message.source.item_id,
                            ),
                        ).fetchone()
                    if not duplicate:
                        raise
                    message_ids.append(duplicate["id"])
                    continue
                message_ids.append(message_id)
        return message_ids

    def create_extraction_batch(
        self, space_id: str, message_ids: list[str]
    ) -> str | None:
        if not message_ids:
            return None
        placeholders = ",".join("?" for _ in message_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT id FROM messages
                WHERE space_id = ? AND id IN ({placeholders})
                  AND extraction_batch_id IS NULL
                  AND extraction_state != 'completed'""",
                (space_id, *message_ids),
            ).fetchall()
            pending_ids = [row["id"] for row in rows]
            if not pending_ids:
                return None
            batch_id = _id("batch")
            connection.execute(
                "INSERT INTO extraction_batches(id, space_id, created_at) VALUES (?, ?, ?)",
                (batch_id, space_id, _now()),
            )
            pending_placeholders = ",".join("?" for _ in pending_ids)
            connection.execute(
                f"UPDATE messages SET extraction_batch_id = ? "
                f"WHERE id IN ({pending_placeholders})",
                (batch_id, *pending_ids),
            )
            return batch_id

    def unbatched_messages(self, space_id: str) -> list[tuple[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT id, ingested_at FROM messages
                WHERE space_id = ? AND extraction_batch_id IS NULL
                  AND extraction_state IN ('pending', 'failed')
                ORDER BY ingested_at, id""",
                (space_id,),
            ).fetchall()
            return [(row["id"], row["ingested_at"]) for row in rows]

    def load_batch(self, space_id: str, batch_id: str) -> list[ModelMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT * FROM messages
                WHERE space_id = ? AND extraction_batch_id = ?
                  AND extraction_state != 'completed'
                ORDER BY ingested_at""",
                (space_id, batch_id),
            ).fetchall()
            return [
                ModelMessage(
                    id=row["id"],
                    space_id=row["space_id"],
                    author=row["author"],
                    content=row["content"],
                    occurred_at=row["occurred_at"],
                    source_provider=row["source_provider"],
                    source_conversation_key=row["source_conversation_key"],
                    source_item_id=row["source_item_id"],
                    source_metadata=_loads(row["source_metadata"], {}),
                )
                for row in rows
            ]

    def load_extraction_context(
        self, space_id: str, batch_id: str, limit: int = 2
    ) -> list[ModelMessage]:
        """Load recent same-conversation messages preceding an extraction batch."""
        if limit <= 0:
            return []
        with self._connect() as connection:
            batch_rows = connection.execute(
                """SELECT source_provider, source_conversation_key
                   FROM messages
                   WHERE space_id = ? AND extraction_batch_id = ?
                     AND extraction_state != 'completed'
                     AND source_provider != ''
                     AND source_conversation_key IS NOT NULL
                     AND source_conversation_key != ''
                   GROUP BY source_provider, source_conversation_key""",
                (space_id, batch_id),
            ).fetchall()
            selected: list[sqlite3.Row] = []
            for batch_row in batch_rows:
                first_batch_row = connection.execute(
                    """SELECT MIN(rowid) AS first_rowid FROM messages
                       WHERE space_id = ? AND extraction_batch_id = ?
                         AND source_provider = ?
                         AND source_conversation_key = ?""",
                    (
                        space_id,
                        batch_id,
                        batch_row["source_provider"],
                        batch_row["source_conversation_key"],
                    ),
                ).fetchone()
                if first_batch_row is None or first_batch_row["first_rowid"] is None:
                    continue
                selected.extend(
                    connection.execute(
                        """SELECT rowid AS message_rowid, * FROM messages
                       WHERE space_id = ? AND source_provider = ?
                         AND source_conversation_key = ?
                         AND rowid < ?
                       ORDER BY rowid DESC LIMIT ?""",
                        (
                            space_id,
                            batch_row["source_provider"],
                            batch_row["source_conversation_key"],
                            first_batch_row["first_rowid"],
                            limit,
                        ),
                    ).fetchall()
                )
            selected.sort(key=lambda row: row["message_rowid"])
            return [
                ModelMessage(
                    id=row["id"],
                    space_id=row["space_id"],
                    author=row["author"],
                    content=row["content"],
                    occurred_at=row["occurred_at"],
                    source_provider=row["source_provider"],
                    source_conversation_key=row["source_conversation_key"],
                    source_item_id=row["source_item_id"],
                    source_metadata=_loads(row["source_metadata"], {}),
                )
                for row in selected
            ]

    def load_known_people(
        self, space_id: str, messages: list[ModelMessage]
    ) -> list[dict[str, Any]]:
        """Return matching active people with their complete alias catalogs."""
        if not messages:
            return []
        matched = self.match_people_in_text(
            space_id, "\n".join(message.content for message in messages)
        )
        if not matched:
            return []
        person_ids = [person.id for person in matched]
        placeholders = ",".join("?" for _ in person_ids)
        with self._connect() as connection:
            rows = connection.execute(
                f"""SELECT p.id, p.display_name, a.value AS alias
                   FROM people p JOIN person_aliases a ON a.person_id = p.id
                   WHERE p.space_id = ? AND p.status = 'active'
                     AND p.id IN ({placeholders})
                   ORDER BY p.display_name, a.value""",
                (space_id, *person_ids),
            ).fetchall()
        catalog: dict[str, dict[str, Any]] = {}
        for row in rows:
            item = catalog.setdefault(
                row["id"],
                {"id": row["id"], "display_name": row["display_name"], "aliases": []},
            )
            if row["alias"] not in item["aliases"]:
                item["aliases"].append(row["alias"])
        return list(catalog.values())

    def mark_extraction_attempt(self, space_id: str, batch_id: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE messages
                SET extraction_attempts = extraction_attempts + 1,
                    extraction_state = 'pending', last_extraction_error = NULL
                WHERE space_id = ? AND extraction_batch_id = ?
                  AND extraction_state != 'completed'
                """,
                (space_id, batch_id),
            )

    def fail_extraction(self, space_id: str, batch_id: str, error: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE messages SET extraction_state = 'failed', last_extraction_error = ?
                WHERE space_id = ? AND extraction_batch_id = ?
                  AND extraction_state != 'completed'
                """,
                (error[:2000], space_id, batch_id),
            )

    def _ensure_relationship(
        self,
        connection: sqlite3.Connection,
        space_id: str,
        person_a_id: str,
        person_b_id: str,
        facets: list[dict[str, Any]],
    ) -> str:
        if person_a_id == person_b_id:
            raise ValueError("a relationship requires two different people")
        person_a_id, person_b_id = sorted((person_a_id, person_b_id))
        row = connection.execute(
            """
            SELECT id FROM relationships
            WHERE space_id = ? AND person_a_id = ? AND person_b_id = ?
            """,
            (space_id, person_a_id, person_b_id),
        ).fetchone()
        if row:
            if facets:
                connection.execute(
                    "UPDATE relationships SET facets = ?, updated_at = ? WHERE id = ?",
                    (_json(facets), _now(), row["id"]),
                )
            return row["id"]
        relationship_id = _id("relationship")
        now = _now()
        connection.execute(
            """
            INSERT INTO relationships(
                id, space_id, person_a_id, person_b_id, facets, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                relationship_id,
                space_id,
                person_a_id,
                person_b_id,
                _json(facets),
                now,
                now,
            ),
        )
        return relationship_id

    def apply_extraction(
        self, space_id: str, batch_id: str, result: ExtractionResult
    ) -> tuple[set[str], set[str]]:
        affected_people: set[str] = set()
        affected_relationships: set[str] = set()
        with self._connect() as connection:
            batch = connection.execute(
                "SELECT id FROM extraction_batches WHERE space_id = ? AND id = ?",
                (space_id, batch_id),
            ).fetchone()
            if not batch:
                raise KeyError(batch_id)
            pending = connection.execute(
                """SELECT 1 FROM messages
                WHERE extraction_batch_id = ? AND extraction_state != 'completed'
                LIMIT 1""",
                (batch_id,),
            ).fetchone()
            if not pending:
                return affected_people, affected_relationships

            people_by_ref: dict[str, str] = {}
            extracted_name_counts = Counter(
                _normalized(person.display_name) for person in result.people
            )
            for person in result.people:
                person_id = self._create_person(
                    connection,
                    space_id,
                    person.display_name,
                    reuse_unique_name=(
                        extracted_name_counts[_normalized(person.display_name)] == 1
                    ),
                )
                people_by_ref[person.ref] = person_id
                for alias in person.aliases:
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO person_aliases(
                            id, space_id, person_id, value, normalized_value
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            _id("alias"),
                            space_id,
                            person_id,
                            alias,
                            _normalized(alias),
                        ),
                    )

            now = _now()
            for candidate in result.memories:
                memory_id = _id("memory")
                connection.execute(
                    """
                    INSERT INTO memories(
                        id, space_id, content, kind, basis, about_user, valid_from, valid_to,
                        source_batch_id, created_by, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'extractor', ?, ?)
                    """,
                    (
                        memory_id,
                        space_id,
                        candidate.content,
                        candidate.kind,
                        candidate.basis,
                        int(candidate.about_user),
                        candidate.valid_from,
                        candidate.valid_to,
                        batch_id,
                        now,
                        now,
                    ),
                )
                for reference in candidate.people:
                    person_id = people_by_ref.get(reference)
                    if not person_id:
                        found = self._find_person(connection, space_id, reference)
                        person_id = found["id"] if found else None
                    if not person_id:
                        continue
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO memory_people(memory_id, person_id)
                        VALUES (?, ?)
                        """,
                        (memory_id, person_id),
                    )
                    affected_people.add(person_id)

                for relationship in candidate.relationships:
                    person_a_id = people_by_ref.get(relationship.person_a_ref)
                    person_b_id = people_by_ref.get(relationship.person_b_ref)
                    if not person_a_id or not person_b_id:
                        continue
                    relationship_id = self._ensure_relationship(
                        connection,
                        space_id,
                        person_a_id,
                        person_b_id,
                        relationship.facets,
                    )
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO memory_relationships(
                            memory_id, relationship_id
                        ) VALUES (?, ?)
                        """,
                        (memory_id, relationship_id),
                    )
                    affected_relationships.add(relationship_id)

            connection.execute(
                """
                UPDATE messages
                SET extraction_state = 'completed', extracted_at = ?,
                    last_extraction_error = NULL
                WHERE extraction_batch_id = ?
                """,
                (now, batch_id),
            )
            connection.execute(
                "UPDATE extraction_batches SET completed_at = ? WHERE id = ?",
                (now, batch_id),
            )
        return affected_people, affected_relationships

    def pending_extractions(self) -> list[tuple[str, str | None, list[str]]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT space_id, id, extraction_batch_id FROM messages
                WHERE extraction_state IN ('pending', 'failed')
                ORDER BY ingested_at
                """
            ).fetchall()
        grouped: dict[tuple[str, str | None], list[str]] = {}
        for row in rows:
            grouped.setdefault(
                (row["space_id"], row["extraction_batch_id"]), []
            ).append(row["id"])
        return [
            (space_id, batch_id, message_ids)
            for (space_id, batch_id), message_ids in grouped.items()
        ]

    def extraction_states(self, space_id: str, message_ids: list[str]) -> list[str]:
        if not message_ids:
            return []
        states: list[str] = []
        for offset in range(0, len(message_ids), 500):
            chunk = message_ids[offset : offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            with self._connect() as connection:
                rows = connection.execute(
                    f"SELECT extraction_state FROM messages "
                    f"WHERE space_id = ? AND id IN ({placeholders})",
                    (space_id, *chunk),
                ).fetchall()
            states.extend(row["extraction_state"] for row in rows)
        return states

    def _memory_view(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
        include_evidence: bool,
    ) -> MemoryView:
        people = [
            {"id": item["person_id"], "name": item["display_name"]}
            for item in connection.execute(
                """
                SELECT mp.person_id, p.display_name
                FROM memory_people mp JOIN people p ON p.id = mp.person_id
                WHERE mp.memory_id = ? ORDER BY p.display_name
                """,
                (row["id"],),
            ).fetchall()
        ]
        evidence: list[dict[str, Any]] = []
        if include_evidence:
            evidence = [
                {
                    "batch_id": item["batch_id"],
                    "message_id": item["message_id"],
                    "text": item["content"],
                    "author": item["author"],
                    "occurred_at": item["occurred_at"],
                    "source_provider": item["source_provider"],
                }
                for item in connection.execute(
                    """
                    SELECT m.extraction_batch_id AS batch_id, m.id AS message_id,
                           m.content, m.author,
                           m.occurred_at, m.source_provider
                    FROM messages m
                    WHERE m.extraction_batch_id = ?
                    ORDER BY m.ingested_at
                    """,
                    (row["source_batch_id"],),
                ).fetchall()
            ]
        return MemoryView(
            id=row["id"],
            content=row["content"],
            kind=row["kind"],
            basis=row["basis"],
            status=row["status"],
            people=people,
            evidence=evidence,
            created_at=row["created_at"],
            about_user=bool(row["about_user"]),
            valid_from=row["valid_from"],
            valid_to=row["valid_to"],
        )

    def _person_watermark(self, connection: sqlite3.Connection, space_id: str, person_id: str) -> str | None:
        row = connection.execute(
            """SELECT MAX(m.updated_at) AS watermark FROM memories m
               JOIN memory_people mp ON mp.memory_id = m.id
               WHERE m.space_id = ? AND mp.person_id = ?""",
            (space_id, person_id),
        ).fetchone()
        return row["watermark"] if row else None

    def _relationship_watermark(self, connection: sqlite3.Connection, space_id: str, relationship_id: str) -> str | None:
        row = connection.execute(
            """SELECT MAX(m.updated_at) AS watermark FROM memories m
               JOIN relationships r ON r.id = ? AND r.space_id = m.space_id
               WHERE m.space_id = ? AND (
                 m.id IN (SELECT memory_id FROM memory_relationships WHERE relationship_id = r.id)
                 OR m.id IN (SELECT a.memory_id FROM memory_people a JOIN memory_people b ON b.memory_id = a.memory_id
                             WHERE a.person_id = r.person_a_id AND b.person_id = r.person_b_id))""",
            (relationship_id, space_id),
        ).fetchone()
        return row["watermark"] if row else None

    def _person_view(self, connection: sqlite3.Connection, row: sqlite3.Row) -> PersonView:
        watermark = self._person_watermark(connection, row["space_id"], row["id"])
        return PersonView(
            id=row["id"],
            display_name=row["display_name"],
            profile_card=_loads(row["profile_card"], {}),
            profile_source_updated_at=row["profile_source_updated_at"],
            profile_updated_at=row["profile_updated_at"],
            stale=watermark is not None and row["profile_source_updated_at"] != watermark,
        )

    def _relationship_view(self, connection: sqlite3.Connection, row: sqlite3.Row) -> RelationshipView:
        watermark = self._relationship_watermark(connection, row["space_id"], row["id"])
        return RelationshipView(
            id=row["id"],
            person_a_id=row["person_a_id"],
            person_b_id=row["person_b_id"],
            facets=_loads(row["facets"], []),
            closeness=row["closeness"],
            tone=row["tone"],
            status=row["status"],
            summary=row["summary"],
            profile_source_updated_at=row["profile_source_updated_at"],
            profile_updated_at=row["profile_updated_at"],
            stale=watermark is not None and row["profile_source_updated_at"] != watermark,
        )

    def read(self, space_id: str, request: QueryRequest) -> QueryContext:
        with self._connect() as connection:
            person_ids: list[str] = []
            people: list[PersonView] = []
            for reference in request.people:
                row = self._find_person(connection, space_id, reference)
                if row and row["id"] not in person_ids:
                    person_ids.append(row["id"])
                    people.append(self._person_view(connection, row))

            relationships: list[RelationshipView] = []
            relationship_ids: list[str] = []
            if request.include_relationships and person_ids:
                placeholders = ",".join("?" for _ in person_ids)
                if request.expand_relationships:
                    sql = f"""
                        SELECT * FROM relationships WHERE space_id = ? AND (
                            person_a_id IN ({placeholders}) OR person_b_id IN ({placeholders})
                        ) ORDER BY updated_at DESC
                    """
                    params: list[Any] = [space_id, *person_ids, *person_ids]
                else:
                    sql = f"""
                        SELECT * FROM relationships WHERE space_id = ?
                          AND person_a_id IN ({placeholders})
                          AND person_b_id IN ({placeholders})
                        ORDER BY updated_at DESC
                    """
                    params = [space_id, *person_ids, *person_ids]
                rows = connection.execute(sql, params).fetchall()
                relationships = [self._relationship_view(connection, row) for row in rows]
                relationship_ids = [row["id"] for row in rows]
                if request.expand_relationships:
                    expanded_ids = {
                        endpoint
                        for row in rows
                        for endpoint in (row["person_a_id"], row["person_b_id"])
                    }
                    new_ids = expanded_ids.difference(person_ids)
                    if new_ids:
                        placeholders = ",".join("?" for _ in new_ids)
                        expanded_rows = connection.execute(
                            f"SELECT * FROM people WHERE space_id = ? AND id IN ({placeholders})",
                            [space_id, *new_ids],
                        ).fetchall()
                        people.extend(self._person_view(connection, row) for row in expanded_rows)
                        person_ids.extend(row["id"] for row in expanded_rows)

            if request.people and not person_ids:
                memory_rows = []
            elif person_ids:
                person_placeholders = ",".join("?" for _ in person_ids)
                relation_clause = ""
                params = [space_id, *person_ids]
                if relationship_ids:
                    relation_placeholders = ",".join("?" for _ in relationship_ids)
                    relation_clause = (
                        " OR m.id IN (SELECT memory_id FROM memory_relationships "
                        f"WHERE relationship_id IN ({relation_placeholders}))"
                    )
                    params.extend(relationship_ids)
                memory_rows = connection.execute(
                    f"""
                    SELECT DISTINCT m.* FROM memories m
                    WHERE m.space_id = ? AND m.status = 'active' AND (
                        m.id IN (SELECT memory_id FROM memory_people
                                 WHERE person_id IN ({person_placeholders}))
                        {relation_clause}
                    )
                    ORDER BY m.created_at DESC
                    """,
                    params,
                ).fetchall()
            else:
                memory_rows = connection.execute(
                    """
                    SELECT * FROM memories
                    WHERE space_id = ? AND status = 'active'
                    ORDER BY created_at DESC
                    """,
                    (space_id,),
                ).fetchall()

            fts_query = _fts_query(request.question, request.people)
            if fts_query and memory_rows:
                memory_ids = [row["id"] for row in memory_rows]
                placeholders = ",".join("?" for _ in memory_ids)
                matched = connection.execute(
                    f"""
                    SELECT m.* FROM memory_fts
                    JOIN memories m ON m.rowid = memory_fts.rowid
                    WHERE memory_fts MATCH ? AND m.id IN ({placeholders})
                    ORDER BY bm25(memory_fts), m.created_at DESC LIMIT ?
                    """,
                    [fts_query, *memory_ids, request.limit],
                ).fetchall()
                # Natural-language wording can have no lexical overlap. In
                # that case the latest structurally scoped memories remain a
                # useful fallback for synthesis.
                if matched:
                    memory_rows = matched
            memory_rows = memory_rows[: request.limit]
            memories = [
                self._memory_view(connection, row, request.include_evidence)
                for row in memory_rows
            ]
            return QueryContext(
                people=people,
                relationships=relationships,
                memories=memories,
            )

    def person_context(
        self, space_id: str, person_id: str
    ) -> tuple[PersonView, list[MemoryView]] | None:
        request = QueryRequest(question="reason", people=[person_id], limit=100)
        context = self.read(space_id, request)
        if not context.people:
            return None
        with self._connect() as connection:
            watermark = self._person_watermark(connection, space_id, person_id)
        return context.people[0], context.memories, watermark

    def relationship_context(
        self, space_id: str, relationship_id: str
    ) -> tuple[RelationshipView, list[MemoryView]] | None:
        with self._connect() as connection:
            relationship = connection.execute(
                "SELECT * FROM relationships WHERE space_id = ? AND id = ?",
                (space_id, relationship_id),
            ).fetchone()
            if not relationship:
                return None
            rows = connection.execute(
                """
                SELECT DISTINCT m.* FROM memories m
                LEFT JOIN memory_relationships mr ON mr.memory_id = m.id
                LEFT JOIN memory_people a ON a.memory_id = m.id
                LEFT JOIN memory_people b ON b.memory_id = m.id
                WHERE m.space_id = ? AND m.status = 'active' AND (
                    mr.relationship_id = ? OR
                    (a.person_id = ? AND b.person_id = ?)
                ) ORDER BY m.created_at DESC LIMIT 100
                """,
                (
                    space_id,
                    relationship_id,
                    relationship["person_a_id"],
                    relationship["person_b_id"],
                ),
            ).fetchall()
            return (
                self._relationship_view(connection, relationship),
                [self._memory_view(connection, row, True) for row in rows],
                self._relationship_watermark(connection, space_id, relationship_id),
            )

    def _insert_inferred_memories(
        self,
        connection: sqlite3.Connection,
        space_id: str,
        inferred: Iterable[Any],
        person_id: str | None = None,
        relationship_id: str | None = None,
    ) -> bool:
        inserted = False
        for item in inferred:
            existing = connection.execute(
                """
                SELECT id FROM memories
                WHERE space_id = ? AND status = 'active' AND basis = 'inferred'
                  AND content = ?
                """,
                (space_id, item.content),
            ).fetchone()
            if existing:
                continue
            valid_sources = [
                row["id"]
                for row in connection.execute(
                    f"""
                    SELECT id FROM memories
                    WHERE space_id = ? AND id IN ({','.join('?' for _ in item.source_memory_ids)})
                    """,
                    [space_id, *item.source_memory_ids],
                ).fetchall()
            ]
            if not valid_sources:
                continue
            memory_id = _id("memory")
            now = _now()
            connection.execute(
                """
                INSERT INTO memories(
                    id, space_id, content, kind, basis, created_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'inferred', 'reasoner', ?, ?)
                """,
                (memory_id, space_id, item.content, item.kind, now, now),
            )
            if person_id:
                connection.execute(
                    "INSERT INTO memory_people(memory_id, person_id) VALUES (?, ?)",
                    (memory_id, person_id),
                )
            if relationship_id:
                connection.execute(
                    "INSERT INTO memory_relationships(memory_id, relationship_id) VALUES (?, ?)",
                    (memory_id, relationship_id),
                )
            connection.executemany(
                """
                INSERT INTO memory_derivations(derived_memory_id, source_memory_id)
                VALUES (?, ?)
                """,
                [(memory_id, source_id) for source_id in valid_sources],
            )
            inserted = True
        return inserted

    def apply_person_reasoning(
        self,
        space_id: str,
        person_id: str,
        expected_watermark: str | None,
        result: PersonReasoningResult,
    ) -> bool:
        with self._connect() as connection:
            claimed = connection.execute(
                """UPDATE people SET profile_card = ?, profile_source_updated_at = ?,
                    profile_updated_at = ?, updated_at = ?
                WHERE space_id = ? AND id = ?
                  AND ? IS (SELECT MAX(m.updated_at) FROM memories m
                            JOIN memory_people mp ON mp.memory_id = m.id
                            WHERE m.space_id = ? AND mp.person_id = ?)
                  AND (profile_source_updated_at IS NULL OR profile_source_updated_at < ?)""",
                (
                    _json(result.profile_card),
                    expected_watermark, _now(), _now(),
                    space_id,
                    person_id,
                    expected_watermark, space_id, person_id, expected_watermark,
                ),
            )
            if claimed.rowcount != 1:
                return False
            inserted = self._insert_inferred_memories(
                connection, space_id, result.inferred_memories, person_id=person_id
            )
            final_watermark = self._person_watermark(connection, space_id, person_id)
            connection.execute(
                """UPDATE people SET profile_card = ?, profile_source_updated_at = ?,
                    profile_updated_at = ?, updated_at = ? WHERE id = ?""",
                (
                    _json(result.profile_card),
                    final_watermark, _now(), _now(),
                    person_id,
                ),
            )
            return True

    def apply_relationship_reasoning(
        self,
        space_id: str,
        relationship_id: str,
        expected_watermark: str | None,
        result: RelationshipReasoningResult,
    ) -> bool:
        with self._connect() as connection:
            if self._relationship_watermark(connection, space_id, relationship_id) != expected_watermark:
                return False
            claimed = connection.execute(
                """UPDATE relationships SET facets = ?, closeness = ?, tone = ?,
                    status = ?, summary = ?, profile_source_updated_at = ?,
                    profile_updated_at = ?, updated_at = ?
                WHERE space_id = ? AND id = ?""",
                (
                    _json(result.facets),
                    result.closeness,
                    result.tone,
                    result.status,
                    result.summary,
                    expected_watermark, _now(), _now(),
                    space_id,
                    relationship_id,
                ),
            )
            if claimed.rowcount != 1:
                return False
            inserted = self._insert_inferred_memories(
                connection,
                space_id,
                result.inferred_memories,
                relationship_id=relationship_id,
            )
            final_watermark = self._relationship_watermark(connection, space_id, relationship_id)
            now = _now()
            connection.execute(
                """UPDATE relationships SET facets = ?, closeness = ?, tone = ?,
                    status = ?, summary = ?, profile_source_updated_at = ?,
                    profile_updated_at = ?, updated_at = ? WHERE id = ?""",
                (
                    _json(result.facets),
                    result.closeness,
                    result.tone,
                    result.status,
                    result.summary,
                    final_watermark,
                    now,
                    now,
                    relationship_id,
                ),
            )
            return True

    def user_model_context(
        self, space_id: str
    ) -> tuple[UserModelView, list[MemoryView], str | None] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM user_models WHERE space_id = ?", (space_id,)
            ).fetchone()
            if not row:
                return None
            memories = connection.execute(
                """SELECT * FROM memories WHERE space_id = ? AND status = 'active'
                   AND about_user = 1 ORDER BY created_at DESC LIMIT 100""",
                (space_id,),
            ).fetchall()
            watermark = self._user_model_watermark(connection, space_id)
            view = UserModelView(
                space_id=space_id,
                profile_card=_loads(row["profile_card"], {}),
                profile_source_updated_at=row["profile_source_updated_at"],
                profile_updated_at=row["profile_updated_at"],
                stale=watermark is not None and row["profile_source_updated_at"] != watermark,
            )
            return view, [self._memory_view(connection, item, True) for item in memories], watermark

    def continuity_context(
        self, space_id: str
    ) -> tuple[ContinuityView | None, list[ModelMessage]] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM continuities WHERE space_id = ?", (space_id,)
            ).fetchone()
            if not row:
                return None
            continuity = ContinuityView(
                text=row["text"],
                related_person_ids=_loads(row["related_person_ids"], []),
                through_message_id=row["through_message_id"],
            )
            if row["through_message_rowid"] is None:
                after = 0
            else:
                after = row["through_message_rowid"]
            messages = connection.execute(
                "SELECT * FROM messages WHERE space_id = ? AND rowid > ? "
                "ORDER BY rowid", (space_id, after)
            ).fetchall()
            return continuity, [ModelMessage(
                id=item["id"], space_id=item["space_id"], author=item["author"],
                content=item["content"], occurred_at=item["occurred_at"],
                source_provider=item["source_provider"],
                source_conversation_key=item["source_conversation_key"],
                source_item_id=item["source_item_id"],
                source_metadata=_loads(item["source_metadata"], {}),
            ) for item in messages]

    def pending_continuities(self, threshold: int = 20) -> list[str]:
        with self._connect() as connection:
            return [row["space_id"] for row in connection.execute(
                "SELECT c.space_id FROM continuities c LEFT JOIN messages m "
                "ON m.space_id = c.space_id AND m.rowid > COALESCE(c.through_message_rowid, 0) "
                "GROUP BY c.space_id HAVING COUNT(m.rowid) >= ?", (threshold,)
            ).fetchall()]

    def apply_continuity_reasoning(
        self, space_id: str, expected_through_message_id: str | None,
        result: ContinuityReasoningResult,
    ) -> bool:
        with self._connect() as connection:
            current = connection.execute(
                "SELECT through_message_id FROM continuities WHERE space_id = ?", (space_id,)
            ).fetchone()
            if not current or current["through_message_id"] != expected_through_message_id:
                return False
            message = connection.execute(
                "SELECT rowid FROM messages WHERE space_id = ? AND id = ?",
                (space_id, result.through_message_id),
            ).fetchone()
            if not message:
                return False
            valid_people = {
                row["id"] for row in connection.execute(
                    "SELECT id FROM people WHERE space_id = ? AND status = 'active'", (space_id,)
                ).fetchall()
            }
            people = [item for item in result.related_person_ids if item in valid_people]
            connection.execute(
                "UPDATE continuities SET text = ?, related_person_ids = ?, "
                "through_message_id = ?, through_message_rowid = ?, updated_at = ? "
                "WHERE space_id = ?",
                (result.text, _json(people), result.through_message_id,
                 message["rowid"], _now(), space_id),
            )
            return True

    def context_bundle(self, space_id: str) -> ContextBundle:
        with self._connect() as connection:
            user = connection.execute("SELECT * FROM user_models WHERE space_id = ?", (space_id,)).fetchone()
            cont = connection.execute("SELECT * FROM continuities WHERE space_id = ?", (space_id,)).fetchone()
            user_view = UserModelView(space_id=space_id, profile_card=_loads(user["profile_card"], {}),
                                      profile_source_updated_at=user["profile_source_updated_at"],
                                      profile_updated_at=user["profile_updated_at"]) if user else None
            continuity = ContinuityView(text=cont["text"], related_person_ids=_loads(cont["related_person_ids"], []),
                                        through_message_id=cont["through_message_id"]) if cont else None
            ids = continuity.related_person_ids if continuity else []
            people = []
            for person_id in ids:
                row = connection.execute("SELECT * FROM people WHERE id = ? AND space_id = ? AND status = 'active'", (person_id, space_id)).fetchone()
                if row:
                    people.append(PersonView(id=row["id"], display_name=row["display_name"], profile_card=_loads(row["profile_card"], {}),
                                             profile_source_updated_at=row["profile_source_updated_at"], profile_updated_at=row["profile_updated_at"]))
            payload = {"user_model": user_view.model_dump(mode="json") if user_view else None,
                       "continuity": continuity.model_dump(mode="json") if continuity else None,
                       "people": [item.model_dump(mode="json") for item in people]}
            version = hashlib.sha256(_json(payload).encode()).hexdigest()[:16]
            return ContextBundle(version=version, user_model=user_view, continuity=continuity, people=people)

    def match_people_in_text(self, space_id: str, text: str) -> list[PersonView]:
        """Resolve explicit aliases without creating people or invoking an LLM."""
        folded = unicodedata.normalize("NFKC", text).casefold()
        with self._connect() as connection:
            aliases = connection.execute(
                """SELECT a.normalized_value, a.value, a.person_id, p.*
                   FROM person_aliases a JOIN people p ON p.id = a.person_id
                   WHERE a.space_id = ? AND p.status = 'active'""", (space_id,)
            ).fetchall()
            by_alias: dict[str, set[str]] = {}
            for row in aliases:
                by_alias.setdefault(unicodedata.normalize("NFKC", row["normalized_value"]).casefold(), set()).add(row["person_id"])

            candidates: list[tuple[int, int, str]] = []
            for alias, person_ids in by_alias.items():
                if not alias or len(person_ids) != 1:
                    continue  # An ambiguous alias is deliberately ignored.
                escaped = re.escape(alias).replace(r"\ ", r"\s+")
                latin = bool(re.search(r"[a-z]", alias))
                boundary_left = r"(?<![a-z])" if latin else ""
                boundary_right = r"(?![a-z])" if latin else ""
                for match in re.finditer(boundary_left + escaped + boundary_right, folded):
                    candidates.append((len(alias), match.start(), next(iter(person_ids))))
            selected: list[tuple[int, str]] = []
            occupied: list[tuple[int, int]] = []
            for length, start, person_id in sorted(candidates, key=lambda item: (-item[0], item[1])):
                end = start + length
                if any(start < right and end > left for left, right in occupied):
                    continue
                occupied.append((start, end))
                selected.append((start, person_id))
            people: list[PersonView] = []
            seen: set[str] = set()
            rows_by_id = {row["id"]: row for row in aliases}
            for _, person_id in sorted(selected):
                if person_id not in seen:
                    people.append(self._person_view(connection, rows_by_id[person_id]))
                    seen.add(person_id)
            return people

    def recall_user_memories(self, space_id: str, text: str, limit: int = 5) -> list[MemoryView]:
        query = _fts_query(text)
        if not query:
            return []
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT m.* FROM memory_fts JOIN memories m ON m.rowid = memory_fts.rowid
                   WHERE memory_fts MATCH ? AND m.space_id = ? AND m.status = 'active'
                     AND m.about_user = 1 ORDER BY bm25(memory_fts), m.created_at DESC LIMIT ?""",
                (query, space_id, limit),
            ).fetchall()
            return [self._memory_view(connection, row, False) for row in rows]

    def _user_model_watermark(self, connection: sqlite3.Connection, space_id: str) -> str | None:
        row = connection.execute(
            "SELECT MAX(updated_at) AS watermark FROM memories "
            "WHERE space_id = ? AND status = 'active' AND about_user = 1",
            (space_id,),
        ).fetchone()
        return row["watermark"] if row else None

    def apply_user_model_reasoning(
        self, space_id: str, expected_watermark: str | None,
        result: UserModelReasoningResult,
    ) -> bool:
        with self._connect() as connection:
            if self._user_model_watermark(connection, space_id) != expected_watermark:
                return False
            now = _now()
            updated = connection.execute(
                """UPDATE user_models SET profile_card = ?, profile_source_updated_at = ?,
                   profile_updated_at = ? WHERE space_id = ?
                   AND (profile_source_updated_at IS NULL OR profile_source_updated_at < ?)""",
                (_json(result.profile_card), expected_watermark, now, space_id,
                 expected_watermark),
            )
            return updated.rowcount == 1

    def overwrite_user_model(self, space_id: str, profile_card: dict[str, Any]) -> None:
        """Explicitly replace the rebuildable card (used by USER.md import)."""
        self.ensure_space(space_id)
        with self._connect() as connection:
            watermark = self._user_model_watermark(connection, space_id)
            connection.execute(
                """UPDATE user_models SET profile_card = ?,
                   profile_source_updated_at = ?, profile_updated_at = ?
                   WHERE space_id = ?""",
                (_json(profile_card), watermark, _now(), space_id),
            )

    def stale_entities(self) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]:
        with self._connect() as connection:
            people = [
                (row["space_id"], row["id"])
                for row in connection.execute(
                    """SELECT p.space_id, p.id FROM people p
                    WHERE p.status = 'active' AND (
                      p.profile_source_updated_at IS NULL OR p.profile_source_updated_at <
                      (SELECT MAX(m.updated_at) FROM memories m JOIN memory_people mp ON mp.memory_id = m.id
                       WHERE m.space_id = p.space_id AND mp.person_id = p.id)
                    ) AND EXISTS (SELECT 1 FROM memories m JOIN memory_people mp ON mp.memory_id = m.id
                       WHERE m.space_id = p.space_id AND mp.person_id = p.id)"""
                ).fetchall()
            ]
            relationships = []
            for row in connection.execute("SELECT * FROM relationships").fetchall():
                watermark = self._relationship_watermark(connection, row["space_id"], row["id"])
                if watermark is not None and row["profile_source_updated_at"] != watermark:
                    relationships.append((row["space_id"], row["id"]))
            user_models = []
            for row in connection.execute("SELECT * FROM user_models").fetchall():
                watermark = self._user_model_watermark(connection, row["space_id"])
                if watermark is not None and row["profile_source_updated_at"] != watermark:
                    user_models.append(row["space_id"])
            return people, relationships, user_models

    def add_manual_memory(
        self, space_id: str, request: ManualMemoryRequest
    ) -> str:
        self.ensure_space(space_id)
        with self._connect() as connection:
            memory_id = _id("memory")
            now = _now()
            connection.execute(
                """
                INSERT INTO memories(
                    id, space_id, content, kind, basis, about_user, valid_from, valid_to,
                    created_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'manual', ?, ?, ?, 'human', ?, ?)
                """,
                (
                    memory_id,
                    space_id,
                    request.content,
                    request.kind,
                    int(request.about_user),
                    request.valid_from,
                    request.valid_to,
                    now,
                    now,
                ),
            )
            affected: set[str] = set()
            for reference in request.people:
                person = self._find_person(connection, space_id, reference)
                if not person:
                    person_id = self._create_person(connection, space_id, reference)
                else:
                    person_id = person["id"]
                connection.execute(
                    "INSERT OR IGNORE INTO memory_people(memory_id, person_id) VALUES (?, ?)",
                    (memory_id, person_id),
                )
                affected.add(person_id)
            return memory_id

    def supersede_memory(
        self,
        space_id: str,
        memory_id: str,
        request: SupersedeRequest,
    ) -> str | None:
        with self._connect() as connection:
            original = connection.execute(
                "SELECT * FROM memories WHERE space_id = ? AND id = ? AND status = 'active'",
                (space_id, memory_id),
            ).fetchone()
            if not original:
                return None
            now = _now()
            replacement_id = _id("memory")
            connection.execute(
                """
                INSERT INTO memories(
                    id, space_id, content, kind, basis, about_user, valid_from, valid_to,
                    supersedes_memory_id, created_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'manual', ?, ?, ?, ?, 'human', ?, ?)
                """,
                (
                    replacement_id,
                    space_id,
                    request.content,
                    request.kind or original["kind"],
                    int(original["about_user"] if request.about_user is None else request.about_user),
                    request.valid_from if request.valid_from is not None else original["valid_from"],
                    request.valid_to if request.valid_to is not None else original["valid_to"],
                    memory_id,
                    now,
                    now,
                ),
            )
            connection.execute(
                """
                INSERT INTO memory_people(memory_id, person_id)
                SELECT ?, person_id FROM memory_people WHERE memory_id = ?
                """,
                (replacement_id, memory_id),
            )
            connection.execute(
                """
                INSERT INTO memory_relationships(memory_id, relationship_id, role)
                SELECT ?, relationship_id, role
                FROM memory_relationships WHERE memory_id = ?
                """,
                (replacement_id, memory_id),
            )
            connection.execute(
                """
                UPDATE memories SET status = 'superseded', invalidated_at = ?,
                    invalidation_reason = ?, updated_at = ? WHERE id = ?
                """,
                (now, request.reason, now, memory_id),
            )
            return replacement_id

    def retract_memory(
        self, space_id: str, memory_id: str, reason: str | None = None
    ) -> bool:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT status FROM memories WHERE space_id = ? AND id = ?",
                (space_id, memory_id),
            ).fetchone()
            if not row:
                return False
            if row["status"] == "retracted":
                return True
            people = connection.execute(
                "SELECT DISTINCT person_id FROM memory_people WHERE memory_id = ?",
                (memory_id,),
            ).fetchall()
            relationships = connection.execute(
                """
                SELECT DISTINCT relationship_id FROM memory_relationships
                WHERE memory_id = ?
                """,
                (memory_id,),
            ).fetchall()
            now = _now()
            connection.execute(
                """
                UPDATE memories SET status = 'retracted', invalidated_at = ?,
                    invalidation_reason = ?, updated_at = ?
                WHERE id = ?
                """,
                (now, reason, now, memory_id),
            )
            return True
