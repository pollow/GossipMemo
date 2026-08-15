from __future__ import annotations

import json
import re
import sqlite3
import uuid
import hashlib
import unicodedata
from difflib import SequenceMatcher
from collections import Counter
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Protocol

from .models import (
    ExtractionResult,
    HypothesisActions,
    HypothesisView,
    HypothesisEvidence,
    InferredMemoryActions,
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
    CoverageAuditPatch,
    CoverageBoundary,
    CoverageBoundaryUpsert,
    CoverageMapView,
    GoalPlanningResult,
    LearningGoalView,
    GuidanceItem,
    GuidanceBundle,
    coverage_skeleton,
    ContextBundle,
    utc_now,
)


DEFAULT_EXTRACTION_COMPARISON_LIMIT = 12


def _id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


def _now() -> str:
    return utc_now().isoformat()


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _similar_memory_content(left: str, right: str) -> bool:
    """Conservatively match near-identical Memory wording."""

    def canonical(value: str) -> str:
        value = unicodedata.normalize("NFKC", value).casefold()
        value = re.sub(r"[^\w\u4e00-\u9fff]+", " ", value, flags=re.UNICODE)
        return " ".join(value.split())

    left, right = canonical(left), canonical(right)
    if left == right:
        return True
    # Preserve polarity and numeric/date changes even when wording is close.
    if re.findall(r"\d+(?:\.\d+)?", left) != re.findall(r"\d+(?:\.\d+)?", right):
        return False
    def has_negation(value: str) -> bool:
        english = re.search(r"\b(?:not|never|no)\b", value) is not None
        return english or any(token in value for token in ("不", "没", "未"))

    if has_negation(left) != has_negation(right):
        return False
    return SequenceMatcher(None, left, right).ratio() >= 0.97


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

    def load_extraction_comparisons(
        self, space_id: str, batch_id: str,
        limit: int = DEFAULT_EXTRACTION_COMPARISON_LIMIT,
    ) -> list[MemoryView]: ...

    def apply_extraction(
        self, space_id: str, batch_id: str, result: ExtractionResult,
        comparison_memory_ids: set[str] | None = None,
    ) -> tuple[set[str], set[str]]: ...

    def read(self, space_id: str, request: QueryRequest) -> QueryContext: ...

    def user_model_context(
        self, space_id: str
    ) -> tuple[UserModelView, list[MemoryView], str | None] | None: ...

    def apply_user_model_reasoning(
        self, space_id: str, expected_watermark: str | None,
        result: UserModelReasoningResult,
    ) -> bool: ...

    def coverage_context(self, space_id: str, limit: int = 24) -> tuple[CoverageMapView, list[MemoryView], list[HypothesisView], bool] | None: ...

    def apply_coverage_audit(self, space_id: str, expected_watermark: str | None, expected_cursor_id: str | None, patch: CoverageAuditPatch, evidence_ids: set[str], context_boundary_ids: set[str], context_hypothesis_ids: set[str]) -> bool: ...

    def learning_goal_context(self, space_id: str) -> tuple[CoverageMapView, list[HypothesisView], list[LearningGoalView], list[LearningGoalView]] | None: ...

    def apply_goal_planning(self, space_id: str, expected_revision: int, result: GoalPlanningResult, context_goal_ids: set[str]) -> bool: ...

    def apply_inferred_memory_actions(
        self, space_id: str, owner_kind: str, owner_id: str | None,
        source_memory_ids: set[str], context_inferred_memory_ids: set[str],
        actions: InferredMemoryActions,
    ) -> None: ...

    def apply_hypothesis_actions(
        self, space_id: str, owner_kind: str, owner_id: str | None,
        source_memory_ids: set[str], context_hypothesis_ids: set[str],
        actions: HypothesisActions,
    ) -> None: ...

    def continuity_context(self, space_id: str) -> tuple[ContinuityView | None, list[ModelMessage]] | None: ...

    def pending_continuities(self, threshold: int = 20) -> list[str]: ...

    def apply_continuity_reasoning(self, space_id: str, expected_through_message_id: str | None, result: ContinuityReasoningResult) -> bool: ...

    def context_bundle(self, space_id: str) -> ContextBundle: ...
    def guidance_bundle(self, space_id: str, activated_person_ids: Iterable[str] = (), query: str = "") -> GuidanceBundle: ...

    def match_people_in_text(self, space_id: str, text: str) -> list[PersonView]: ...

    def recall_user_memories(self, space_id: str, text: str, limit: int = 5) -> list[MemoryView]: ...

    def stale_entities(
        self,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]: ...

    def stale_coverage_spaces(self) -> list[str]: ...

    def overwrite_user_model(self, space_id: str, profile_card: dict[str, Any]) -> None: ...

    def merge_person(
        self, space_id: str, source_person_id: str, target_person_id: str
    ) -> dict[str, Any]: ...


class AmbiguousPersonError(ValueError):
    def __init__(self, reference: str) -> None:
        super().__init__(f"person reference is ambiguous: {reference}")
        self.reference = reference


class PersonMergeError(ValueError):
    """Raised when an explicit person merge cannot be applied safely."""

    def __init__(self, message: str, *, conflict: bool = False) -> None:
        super().__init__(message)
        self.conflict = conflict


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
                "INSERT OR IGNORE INTO coverage_maps(space_id, criteria, updated_at) VALUES (?, ?, ?)",
                (space_id, _json(coverage_skeleton()), now),
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

    def load_extraction_comparisons(
        self, space_id: str, batch_id: str,
        limit: int = DEFAULT_EXTRACTION_COMPARISON_LIMIT,
    ) -> list[MemoryView]:
        """Return a small comparison set; these rows are never new evidence."""
        if limit <= 0:
            return []
        with self._connect() as connection:
            messages = connection.execute(
                """SELECT content FROM messages WHERE space_id = ?
                   AND extraction_batch_id = ? AND author = 'user'
                   AND extraction_state != 'completed'""", (space_id, batch_id)
            ).fetchall()
            texts = [row["content"] for row in messages]
            context = self.load_extraction_context(space_id, batch_id)
            person_ids = [person.id for person in self.match_people_in_text(
                space_id, "\n".join(texts + [message.content for message in context])
            )]
            candidates: dict[str, sqlite3.Row] = {}
            for text in texts:
                fts = _fts_query(text)
                if not fts:
                    continue
                for row in connection.execute(
                    """SELECT m.* FROM memory_fts JOIN memories m ON m.rowid = memory_fts.rowid
                       WHERE memory_fts MATCH ? AND m.space_id = ? AND m.status = 'active'
                         AND m.basis <> 'inferred'
                       ORDER BY bm25(memory_fts), m.created_at DESC LIMIT ?""",
                    (fts, space_id, limit),
                ).fetchall():
                    candidates[row["id"]] = row
            if person_ids:
                placeholders = ",".join("?" for _ in person_ids)
                for row in connection.execute(
                    f"""SELECT DISTINCT m.* FROM memories m JOIN memory_people mp ON mp.memory_id = m.id
                       WHERE m.space_id = ? AND m.status = 'active' AND m.basis <> 'inferred'
                         AND mp.person_id IN ({placeholders}) ORDER BY m.updated_at DESC LIMIT ?""",
                    (space_id, *person_ids, limit),
                ).fetchall():
                    candidates[row["id"]] = row
            rows = sorted(candidates.values(), key=lambda row: row["updated_at"], reverse=True)[:limit]
            return [self._memory_view(connection, row, False) for row in rows]

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
        self, space_id: str, batch_id: str, result: ExtractionResult,
        comparison_memory_ids: set[str] | None = None,
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
                try:
                    person_id = self._create_person(
                        connection, space_id, person.display_name,
                        reuse_unique_name=(extracted_name_counts[_normalized(person.display_name)] == 1),
                    )
                except AmbiguousPersonError:
                    # Do not abort the batch or create another guessed same-name
                    # person; the memory itself is still durable evidence.
                    continue
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

            def resolve_extracted_person(reference: str) -> str | None:
                """Resolve model refs without letting ambiguous aliases abort extraction."""
                person_id = people_by_ref.get(reference)
                if person_id:
                    return person_id
                try:
                    found = self._find_person(connection, space_id, reference)
                except AmbiguousPersonError:
                    return None
                return found["id"] if found else None

            now = _now()
            comparison_rows: dict[str, sqlite3.Row] = {}
            comparison_people: dict[str, set[str]] = {}
            comparison_relationships: dict[str, set[str]] = {}
            if comparison_memory_ids:
                placeholders = ",".join("?" for _ in comparison_memory_ids)
                rows = connection.execute(
                    f"""SELECT * FROM memories WHERE space_id = ?
                        AND status = 'active' AND basis <> 'inferred'
                        AND id IN ({placeholders})""",
                    (space_id, *comparison_memory_ids),
                ).fetchall()
                comparison_rows = {row["id"]: row for row in rows}
                for memory_id in comparison_rows:
                    comparison_people[memory_id] = {
                        row["person_id"]
                        for row in connection.execute(
                            "SELECT person_id FROM memory_people WHERE memory_id = ?",
                            (memory_id,),
                        ).fetchall()
                    }
                    comparison_relationships[memory_id] = {
                        row["relationship_id"]
                        for row in connection.execute(
                            "SELECT relationship_id FROM memory_relationships "
                            "WHERE memory_id = ?",
                            (memory_id,),
                        ).fetchall()
                    }
            inserted_signatures: list[tuple[ExtractedMemory, set[str], set[str]]] = []
            for candidate in result.memories:
                people_ids: set[str] = set()
                for reference in candidate.people:
                    person_id = resolve_extracted_person(reference)
                    if person_id:
                        people_ids.add(person_id)
                relationship_ids: set[str] = set()
                for relationship in candidate.relationships:
                    a = resolve_extracted_person(relationship.person_a_ref)
                    b = resolve_extracted_person(relationship.person_b_ref)
                    if a and b:
                        existing = connection.execute(
                            """SELECT id FROM relationships WHERE space_id = ?
                               AND ((person_a_id = ? AND person_b_id = ?) OR
                                    (person_a_id = ? AND person_b_id = ?))""",
                            (space_id, a, b, b, a),
                        ).fetchone()
                        if existing:
                            relationship_ids.add(existing["id"])
                comparison = comparison_rows.get(candidate.supersedes_memory_id or "")

                def same_shape(row: sqlite3.Row) -> bool:
                    memory_id = row["id"]
                    return (
                        row["kind"] == candidate.kind
                        and row["basis"] == candidate.basis
                        and bool(row["about_user"]) == candidate.about_user
                        and row["valid_from"] == candidate.valid_from
                        and row["valid_to"] == candidate.valid_to
                        and comparison_people[memory_id] == people_ids
                        and comparison_relationships[memory_id] == relationship_ids
                    )

                if comparison is None and any(
                    same_shape(row)
                    and _similar_memory_content(row["content"], candidate.content)
                    for row in comparison_rows.values()
                ):
                    continue
                if any(
                    item.kind == candidate.kind and item.basis == candidate.basis
                    and item.about_user == candidate.about_user
                    and item.valid_from == candidate.valid_from and item.valid_to == candidate.valid_to
                    and prior_people == people_ids and prior_relationships == relationship_ids
                    and _similar_memory_content(item.content, candidate.content)
                    for item, prior_people, prior_relationships in inserted_signatures
                ):
                    continue
                memory_id = _id("memory")
                connection.execute(
                    """
                    INSERT INTO memories(
                        id, space_id, content, kind, basis, about_user, valid_from, valid_to,
                        supersedes_memory_id, source_batch_id, created_by, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'extractor', ?, ?)
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
                        candidate.supersedes_memory_id if comparison is not None else None,
                        batch_id,
                        now,
                        now,
                    ),
                )
                for reference in candidate.people:
                    person_id = resolve_extracted_person(reference)
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
                    person_a_id = resolve_extracted_person(relationship.person_a_ref)
                    person_b_id = resolve_extracted_person(relationship.person_b_ref)
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
                    relationship_ids.add(relationship_id)
                inserted_signatures.append((candidate, people_ids, relationship_ids))
                if comparison is not None:
                    for row in connection.execute(
                        "SELECT person_id FROM memory_people WHERE memory_id = ?",
                        (comparison["id"],),
                    ).fetchall():
                        affected_people.add(row["person_id"])
                    for row in connection.execute(
                        "SELECT relationship_id FROM memory_relationships WHERE memory_id = ?",
                        (comparison["id"],),
                    ).fetchall():
                        affected_relationships.add(row["relationship_id"])
                    connection.execute(
                        """UPDATE memories SET status = 'superseded',
                           invalidated_at = ?, invalidation_reason = ?, updated_at = ? WHERE id = ?""",
                        (now, "superseded by extraction update", now, comparison["id"]),
                    )

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
               WHERE m.space_id = ? AND mp.person_id = ? AND m.basis <> 'inferred'""",
            (space_id, person_id),
        ).fetchone()
        return row["watermark"] if row else None

    def _relationship_watermark(self, connection: sqlite3.Connection, space_id: str, relationship_id: str) -> str | None:
        row = connection.execute(
            """SELECT MAX(m.updated_at) AS watermark FROM memories m
               JOIN relationships r ON r.id = ? AND r.space_id = m.space_id
               WHERE m.space_id = ? AND m.basis <> 'inferred' AND (
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
    ) -> tuple[PersonView, list[MemoryView], str | None] | None:
        with self._connect() as connection:
            person = connection.execute("SELECT * FROM people WHERE space_id = ? AND id = ?", (space_id, person_id)).fetchone()
            if not person:
                return None
            rows = connection.execute("""SELECT DISTINCT m.* FROM memories m JOIN memory_people mp ON mp.memory_id = m.id
                WHERE m.space_id = ? AND m.status = 'active' AND m.basis <> 'inferred' AND mp.person_id = ?
                ORDER BY m.created_at DESC, m.id DESC""", (space_id, person_id)).fetchall()
            watermark = self._person_watermark(connection, space_id, person_id)
            view = self._person_view(connection, person)
            return view, [self._memory_view(connection, row, True) for row in rows], watermark

    def owner_review_context(self, space_id: str, owner_kind: str, owner_id: str | None) -> tuple[list[MemoryView], list[HypothesisView]]:
        """Comparison-only state captured with an owner reasoning snapshot."""
        with self._connect() as connection:
            if owner_kind == "person":
                clause, params = "EXISTS (SELECT 1 FROM memory_people mp WHERE mp.memory_id = m.id AND mp.person_id = ?)", [owner_id]
            elif owner_kind == "relationship":
                clause, params = "EXISTS (SELECT 1 FROM memory_relationships mr WHERE mr.memory_id = m.id AND mr.relationship_id = ?)", [owner_id]
            else:
                clause, params = "m.about_user = 1", []
            inferred = connection.execute(f"SELECT m.* FROM memories m WHERE m.space_id = ? AND m.status = 'active' AND m.basis = 'inferred' AND {clause} ORDER BY m.created_at DESC LIMIT 100", [space_id, *params]).fetchall()
            hypotheses = connection.execute("SELECT * FROM hypotheses WHERE space_id = ? AND owner_kind = ? AND owner_id IS ? AND status = 'open' ORDER BY created_at DESC LIMIT 100", (space_id, owner_kind, owner_id)).fetchall()
            return ([self._memory_view(connection, row, True) for row in inferred], [HypothesisView(id=row['id'], space_id=row['space_id'], owner_kind=row['owner_kind'], owner_id=row['owner_id'], content=row['content'], kind=row['kind'], confidence=row['confidence'], status=row['status'], promoted_memory_id=row['promoted_memory_id'], evidence=[HypothesisEvidence(memory_id=e['memory_id'], role=e['role']) for e in connection.execute('SELECT memory_id, role FROM hypothesis_evidence WHERE hypothesis_id = ?', (row['id'],)).fetchall()], created_at=row['created_at'], updated_at=row['updated_at']) for row in hypotheses])

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
                WHERE m.space_id = ? AND m.status = 'active' AND m.basis <> 'inferred' AND (
                    mr.relationship_id = ? OR
                    (a.person_id = ? AND b.person_id = ?)
                ) ORDER BY m.created_at DESC, m.id DESC
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
                [self._memory_view(connection, row, True) for row in rows
                 if row["basis"] != "inferred"],
                self._relationship_watermark(connection, space_id, relationship_id),
            )

    def _reconcile_inferred_memories(
        self,
        connection: sqlite3.Connection,
        space_id: str,
        inferred: Iterable[Any],
        person_id: str | None = None,
        relationship_id: str | None = None,
        source_ids: set[str] | None = None,
        owner_kind: str | None = None,
    ) -> None:
        """Add or deduplicate inferences; lifecycle retraction is explicit elsewhere."""
        owner_kind = owner_kind or ("person" if person_id else "relationship")
        owner_id = person_id or relationship_id
        if person_id:
            target_join = "JOIN memory_people mt ON mt.memory_id = m.id AND mt.person_id = ?"
            target_params: list[Any] = [person_id]
        elif relationship_id:
            target_join = "JOIN memory_relationships mt ON mt.memory_id = m.id AND mt.relationship_id = ?"
            target_params = [relationship_id]
        else:
            target_join = ""
            target_params = []
        target_filter = "AND m.about_user = 1" if owner_kind == "user" else ""
        existing_rows = connection.execute(
            f"""SELECT m.id, m.content, m.kind FROM memories m {target_join}
                WHERE m.space_id = ? AND m.status = 'active'
                  AND m.basis = 'inferred' AND m.created_by = 'reasoner' {target_filter}""",
            [*target_params, space_id],
        ).fetchall()
        matched: set[str] = set()
        for item in inferred:
            existing = next((row for row in existing_rows
                             if row["kind"] == item.kind
                             and _similar_memory_content(row["content"], item.content)), None)
            valid_sources = [
                row["id"]
                for row in connection.execute(
                    f"""
                    SELECT DISTINCT m.id FROM memories m
                    WHERE m.space_id = ? AND m.status = 'active'
                      AND m.basis <> 'inferred'
                      AND m.id IN ({','.join('?' for _ in item.source_memory_ids)})
                    """,
                    [space_id, *item.source_memory_ids],
                ).fetchall()
            ]
            if source_ids is not None:
                valid_sources = [source_id for source_id in valid_sources if source_id in source_ids]
            if not valid_sources:
                continue
            if existing:
                matched.add(existing["id"])
                connection.executemany(
                    """
                    INSERT OR IGNORE INTO memory_derivations(
                        derived_memory_id, source_memory_id
                    )
                    VALUES (?, ?)
                    """,
                    [(existing["id"], source_id) for source_id in valid_sources],
                )
                continue
            memory_id = _id("memory")
            now = _now()
            connection.execute(
                """
                INSERT INTO memories(
                    id, space_id, content, kind, basis, about_user, created_by, created_at, updated_at
                ) VALUES (?, ?, ?, ?, 'inferred', ?, 'reasoner', ?, ?)
                """,
                (memory_id, space_id, item.content, item.kind, int(owner_kind == "user"), now, now),
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
            matched.add(memory_id)

    def apply_inferred_memory_actions(
        self,
        space_id: str,
        owner_kind: str,
        owner_id: str | None,
        source_memory_ids: set[str],
        context_inferred_memory_ids: set[str],
        actions: InferredMemoryActions,
    ) -> None:
        """Apply explicitly scoped inferred-Memory lifecycle actions.

        ``source_memory_ids`` is the non-inferred evidence shown to the
        reasoner. Retractions additionally require the target-owned inference
        to have been supplied by the trusted caller in context.
        """
        if owner_kind not in {"user", "person", "relationship"}:
            raise ValueError("owner_kind must be user, person, or relationship")
        if (owner_kind == "user") != (owner_id is None):
            raise ValueError("user hypotheses have no owner_id; other hypotheses require one")
        with self._connect() as connection:
            self._apply_inferred_memory_actions(
                connection, space_id, owner_kind, owner_id, source_memory_ids,
                context_inferred_memory_ids, actions,
            )

    def _apply_inferred_memory_actions(
        self, connection: sqlite3.Connection, space_id: str, owner_kind: str,
        owner_id: str | None, source_memory_ids: set[str],
        context_inferred_memory_ids: set[str], actions: InferredMemoryActions,
    ) -> None:
        if actions.upserts:
            target = {"person": "person_id", "relationship": "relationship_id"}.get(owner_kind)
            target_kwargs = {target: owner_id} if target else {}
            self._reconcile_inferred_memories(
                connection, space_id, actions.upserts, **target_kwargs,
                source_ids=source_memory_ids, owner_kind=owner_kind,
            )
        for retraction in actions.retractions:
            if retraction.memory_id not in context_inferred_memory_ids:
                continue
            if owner_kind == "person":
                owner_clause = "EXISTS (SELECT 1 FROM memory_people mp WHERE mp.memory_id = m.id AND mp.person_id = ?)"
                owner_params: list[Any] = [owner_id]
            elif owner_kind == "relationship":
                owner_clause = "EXISTS (SELECT 1 FROM memory_relationships mr WHERE mr.memory_id = m.id AND mr.relationship_id = ?)"
                owner_params = [owner_id]
            else:
                owner_clause = "m.about_user = 1"
                owner_params = []
            row = connection.execute(
                f"""SELECT m.id FROM memories m WHERE m.space_id = ? AND m.id = ?
                    AND m.basis = 'inferred' AND m.status = 'active' AND {owner_clause}""",
                [space_id, retraction.memory_id, *owner_params],
            ).fetchone()
            if not row:
                continue
            now = _now()
            connection.execute(
                    """UPDATE memories SET status = 'retracted', invalidated_at = ?,
                       invalidation_reason = ?, updated_at = ? WHERE id = ?""",
                (now, retraction.reason, now, retraction.memory_id),
            )

    def apply_hypothesis_actions(
        self,
        space_id: str,
        owner_kind: str,
        owner_id: str | None,
        source_memory_ids: set[str],
        context_hypothesis_ids: set[str],
        actions: HypothesisActions,
    ) -> None:
        """Persist standalone hypotheses and explicit, context-scoped transitions."""
        if owner_kind not in {"user", "person", "relationship"}:
            raise ValueError("owner_kind must be user, person, or relationship")
        if (owner_kind == "user") != (owner_id is None):
            raise ValueError("user hypotheses have no owner_id; other hypotheses require one")
        with self._connect() as connection:
            for item in actions.upserts:
                if item.hypothesis_id and item.hypothesis_id not in context_hypothesis_ids:
                    continue
                evidence = [
                    row for row in connection.execute(
                        f"""SELECT m.id FROM memories m WHERE m.space_id = ? AND m.status = 'active'
                            AND m.basis <> 'inferred' AND m.id IN ({','.join('?' for _ in item.evidence)})""",
                        [space_id, *(e.memory_id for e in item.evidence)],
                    ).fetchall()
                    if row["id"] in source_memory_ids
                ]
                evidence_ids = {row["id"] for row in evidence}
                if not evidence_ids:
                    continue
                hypothesis_id = item.hypothesis_id
                if hypothesis_id:
                    existing = connection.execute(
                        """SELECT id FROM hypotheses WHERE id = ? AND space_id = ?
                           AND owner_kind = ? AND owner_id IS ? AND status = 'open'""",
                        (hypothesis_id, space_id, owner_kind, owner_id),
                    ).fetchone()
                    if not existing and connection.execute(
                        "SELECT 1 FROM hypotheses WHERE id = ?", (hypothesis_id,)
                    ).fetchone():
                        continue
                else:
                    existing = connection.execute(
                        """SELECT id FROM hypotheses WHERE space_id = ? AND owner_kind = ? AND owner_id IS ?
                           AND status = 'open' AND kind = ? AND content = ?""",
                        (space_id, owner_kind, owner_id, item.kind, item.content),
                    ).fetchone()
                now = _now()
                if existing:
                    hypothesis_id = existing["id"]
                    connection.execute(
                        "UPDATE hypotheses SET content = ?, kind = ?, confidence = ?, updated_at = ? WHERE id = ?",
                        (item.content, item.kind, item.confidence, now, hypothesis_id),
                    )
                else:
                    hypothesis_id = hypothesis_id or _id("hypothesis")
                    connection.execute(
                        """INSERT INTO hypotheses(
                            id, space_id, owner_kind, owner_id, content, kind, confidence,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (hypothesis_id, space_id, owner_kind, owner_id, item.content, item.kind,
                         item.confidence, now, now),
                    )
                connection.executemany(
                    "INSERT OR IGNORE INTO hypothesis_evidence(hypothesis_id, memory_id, role) VALUES (?, ?, ?)",
                    [(hypothesis_id, e.memory_id, e.role) for e in item.evidence if e.memory_id in evidence_ids],
                )
            for transition in actions.transitions:
                if transition.hypothesis_id not in context_hypothesis_ids:
                    continue
                row = connection.execute(
                    """SELECT id FROM hypotheses WHERE id = ? AND space_id = ?
                       AND owner_kind = ? AND owner_id IS ? AND status = 'open'""",
                    (transition.hypothesis_id, space_id, owner_kind, owner_id),
                ).fetchone()
                if not row:
                    continue
                if transition.status == "promoted" and not transition.promoted_memory_id:
                    continue
                if transition.status != "promoted" and transition.promoted_memory_id:
                    continue
                if transition.promoted_memory_id:
                    if owner_kind == "person":
                        owner_clause = "EXISTS (SELECT 1 FROM memory_people mp WHERE mp.memory_id = m.id AND mp.person_id = ?)"
                        owner_params: list[Any] = [owner_id]
                    elif owner_kind == "relationship":
                        owner_clause = "EXISTS (SELECT 1 FROM memory_relationships mr WHERE mr.memory_id = m.id AND mr.relationship_id = ?)"
                        owner_params = [owner_id]
                    else:
                        owner_clause = "m.about_user = 1"
                        owner_params = []
                    memory = connection.execute(
                        f"SELECT m.id FROM memories m WHERE m.id = ? AND m.space_id = ? AND m.status = 'active' AND {owner_clause}",
                        [transition.promoted_memory_id, space_id, *owner_params],
                    ).fetchone()
                    if not memory:
                        continue
                now = _now()
                connection.execute(
                    """UPDATE hypotheses SET status = ?, status_reason = ?, promoted_memory_id = ?,
                       updated_at = ? WHERE id = ?""",
                    (transition.status, transition.reason, transition.promoted_memory_id, now, row["id"]),
                )

    def _person_reasoning_source_ids(
        self, connection: sqlite3.Connection, space_id: str, person_id: str
    ) -> set[str]:
        rows = connection.execute(
            """SELECT DISTINCT m.id FROM memories m JOIN memory_people mp ON mp.memory_id = m.id
               WHERE m.space_id = ? AND m.status = 'active' AND m.basis <> 'inferred'
                 AND mp.person_id = ? ORDER BY m.created_at DESC, m.id DESC""",
            (space_id, person_id),
        ).fetchall()
        return {row["id"] for row in rows}

    def _relationship_reasoning_source_ids(
        self, connection: sqlite3.Connection, space_id: str, relationship_id: str
    ) -> set[str]:
        relationship = connection.execute(
            "SELECT person_a_id, person_b_id FROM relationships WHERE space_id = ? AND id = ?",
            (space_id, relationship_id),
        ).fetchone()
        if not relationship:
            return set()
        rows = connection.execute(
            """SELECT DISTINCT m.id FROM memories m
               LEFT JOIN memory_relationships mr ON mr.memory_id = m.id
               LEFT JOIN memory_people a ON a.memory_id = m.id
               LEFT JOIN memory_people b ON b.memory_id = m.id
               WHERE m.space_id = ? AND m.status = 'active' AND m.basis <> 'inferred'
                 AND (mr.relationship_id = ? OR
                      (a.person_id = ? AND b.person_id = ?))
               ORDER BY m.created_at DESC, m.id DESC""",
            (space_id, relationship_id, relationship["person_a_id"], relationship["person_b_id"]),
        ).fetchall()
        return {row["id"] for row in rows}

    def apply_person_reasoning(
        self,
        space_id: str,
        person_id: str,
        expected_watermark: str | None,
        result: PersonReasoningResult,
        context_inferred_memory_ids: set[str] | None = None,
        context_hypothesis_ids: set[str] | None = None,
    ) -> bool:
        with self._connect() as connection:
            claimed = connection.execute(
                """UPDATE people SET profile_card = ?, profile_source_updated_at = ?,
                    profile_updated_at = ?, updated_at = ?
                WHERE space_id = ? AND id = ?
                  AND ? IS (SELECT MAX(m.updated_at) FROM memories m
                            JOIN memory_people mp ON mp.memory_id = m.id
                            WHERE m.space_id = ? AND mp.person_id = ?
                              AND m.basis <> 'inferred')
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
            actions = result.inferred_memory_actions or InferredMemoryActions(upserts=result.inferred_memories)
            self._apply_inferred_memory_actions(
                connection, space_id, "person", person_id,
                self._person_reasoning_source_ids(connection, space_id, person_id), context_inferred_memory_ids or set(), actions,
            )
            if result.hypothesis_actions:
                self._apply_hypothesis_actions(connection, space_id, "person", person_id, self._person_reasoning_source_ids(connection, space_id, person_id), context_hypothesis_ids or set(), result.hypothesis_actions)
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
        context_inferred_memory_ids: set[str] | None = None,
        context_hypothesis_ids: set[str] | None = None,
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
            actions = result.inferred_memory_actions or InferredMemoryActions(upserts=result.inferred_memories)
            self._apply_inferred_memory_actions(
                connection, space_id, "relationship", relationship_id,
                self._relationship_reasoning_source_ids(connection, space_id, relationship_id), context_inferred_memory_ids or set(), actions,
            )
            if result.hypothesis_actions:
                self._apply_hypothesis_actions(connection, space_id, "relationship", relationship_id, self._relationship_reasoning_source_ids(connection, space_id, relationship_id), context_hypothesis_ids or set(), result.hypothesis_actions)
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
                   AND about_user = 1 ORDER BY created_at DESC, id DESC""",
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

    def _coverage_view(self, row: sqlite3.Row) -> CoverageMapView:
        criteria = _loads(row["criteria"], coverage_skeleton())
        # Existing development DBs may have a partial hand-written map; keep
        # the static skeleton complete at every read.
        for criterion_id, value in coverage_skeleton().items():
            criteria.setdefault(criterion_id, value)
        return CoverageMapView(
            space_id=row["space_id"], revision=row["revision"],
            source_watermark=row["source_watermark"], criteria=criteria,
            source_cursor_id=row["source_cursor_id"],
            boundaries=[CoverageBoundary.model_validate(value) for value in _loads(row["boundaries"], [])],
            life_periods=_loads(row["life_periods"], []),
            relationship_arcs=_loads(row["relationship_arcs"], []),
            behavioral_contexts=_loads(row["behavioral_contexts"], []),
        )

    def _hypothesis_view(self, connection: sqlite3.Connection, row: sqlite3.Row) -> HypothesisView:
        evidence = connection.execute(
            "SELECT memory_id, role FROM hypothesis_evidence WHERE hypothesis_id = ?", (row["id"],)
        ).fetchall()
        return HypothesisView(id=row["id"], space_id=row["space_id"], owner_kind=row["owner_kind"], owner_id=row["owner_id"], content=row["content"], kind=row["kind"], confidence=row["confidence"], status=row["status"], promoted_memory_id=row["promoted_memory_id"], evidence=[HypothesisEvidence(memory_id=item["memory_id"], role=item["role"]) for item in evidence], created_at=row["created_at"], updated_at=row["updated_at"])

    def coverage_context(self, space_id: str, limit: int = 24) -> tuple[CoverageMapView, list[MemoryView], list[HypothesisView], bool] | None:
        """Return a bounded changed-evidence chunk and whether more remains."""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM coverage_maps WHERE space_id = ?", (space_id,)).fetchone()
            if not row:
                return None
            coverage = self._coverage_view(row)
            watermark, cursor_id = coverage.source_watermark or "", coverage.source_cursor_id or ""
            changed = connection.execute(
                """SELECT * FROM memories WHERE space_id = ? AND basis <> 'inferred'
                   AND (updated_at > ? OR (updated_at = ? AND id > ?))
                   ORDER BY updated_at, id LIMIT ?""",
                (space_id, watermark, watermark, cursor_id, limit + 1),
            ).fetchall()
            chunk, pending = changed[:limit], len(changed) > limit
            hypotheses = connection.execute(
                "SELECT * FROM hypotheses WHERE space_id = ? AND status = 'open' ORDER BY updated_at DESC LIMIT 100", (space_id,)
            ).fetchall()
            return coverage, [self._memory_view(connection, item, True) for item in chunk], [self._hypothesis_view(connection, item) for item in hypotheses], pending

    def apply_coverage_audit(self, space_id: str, expected_watermark: str | None, expected_cursor_id: str | None, patch: CoverageAuditPatch, evidence_ids: set[str], context_boundary_ids: set[str], context_hypothesis_ids: set[str]) -> bool:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM coverage_maps WHERE space_id = ?", (space_id,)).fetchone()
            if not row or row["source_watermark"] != expected_watermark or row["source_cursor_id"] != expected_cursor_id:
                return False
            coverage = self._coverage_view(row)
            active_ids = {item["id"] for item in connection.execute("SELECT id FROM memories WHERE space_id = ? AND status = 'active' AND basis <> 'inferred'", (space_id,)).fetchall()}
            for item in patch.criteria:
                if item.criterion_id not in coverage.criteria:
                    continue
                evidence = [memory_id for memory_id in item.evidence_memory_ids if memory_id in evidence_ids and memory_id in active_ids]
                # A patch with no current evidence cannot raise coverage. This
                # is the persistence backstop for the prompt's hypothesis rule.
                old = coverage.criteria[item.criterion_id]
                levels = ["unknown", "fragmentary", "grounded", "rich"]
                level = item.level if evidence or levels.index(item.level) <= levels.index(old.get("level", "unknown")) else old.get("level", "unknown")
                coverage.criteria[item.criterion_id] = {"level": level, "known_state": item.known_state, "evidence_memory_ids": sorted(set(old.get("evidence_memory_ids", []) + evidence))}
            # Evidence can be retracted/superseded after a prior audit. Prune
            # it before applying this patch, and do not let an empty criterion
            # retain a misleading grounded/rich level.
            for criterion in coverage.criteria.values():
                criterion["evidence_memory_ids"] = [memory_id for memory_id in criterion.get("evidence_memory_ids", []) if memory_id in active_ids]
                if not criterion["evidence_memory_ids"]:
                    criterion["level"], criterion["known_state"] = "unknown", ""
            boundaries = {item.id: item for item in coverage.boundaries}
            for item in boundaries.values():
                item.evidence_memory_ids = [memory_id for memory_id in item.evidence_memory_ids if memory_id in active_ids]
            for item in patch.boundary_upserts:
                if all(ref in coverage.criteria for ref in item.criterion_refs):
                    evidence = [memory_id for memory_id in item.evidence_memory_ids if memory_id in evidence_ids]
                    if item.hypothesis_id not in context_hypothesis_ids:
                        item.hypothesis_id = None
                    duplicate = next((boundary for boundary in boundaries.values() if boundary.status == "open" and boundary.kind == item.kind and boundary.summary == item.summary and boundary.criterion_refs == item.criterion_refs), None)
                    if duplicate:
                        continue
                    boundary_id = _id("boundary")
                    boundaries[boundary_id] = CoverageBoundary(id=boundary_id, kind=item.kind, summary=item.summary, criterion_refs=item.criterion_refs, evidence_memory_ids=evidence, hypothesis_id=item.hypothesis_id)
            for transition in patch.boundary_transitions:
                if transition.boundary_id in context_boundary_ids and transition.boundary_id in boundaries:
                    boundaries[transition.boundary_id].status = transition.status
                    boundaries[transition.boundary_id].status_reason = transition.reason
            cursor_row = connection.execute("SELECT updated_at, id FROM memories WHERE id IN (%s) ORDER BY updated_at DESC, id DESC LIMIT 1" % ",".join("?" for _ in evidence_ids), list(evidence_ids)).fetchone()
            next_watermark = cursor_row["updated_at"] if cursor_row else expected_watermark
            next_cursor_id = cursor_row["id"] if cursor_row else expected_cursor_id
            connection.execute("""UPDATE coverage_maps SET revision = revision + 1, source_watermark = ?, source_cursor_id = ?, criteria = ?, boundaries = ?, life_periods = ?, relationship_arcs = ?, behavioral_contexts = ?, updated_at = ? WHERE space_id = ?""", (next_watermark, next_cursor_id, _json(coverage.criteria), _json([item.model_dump() for item in boundaries.values()]), _json(list(dict.fromkeys(coverage.life_periods + patch.life_periods))[:30]), _json(list(dict.fromkeys(coverage.relationship_arcs + patch.relationship_arcs))[:30]), _json(list(dict.fromkeys(coverage.behavioral_contexts + patch.behavioral_contexts))[:30]), _now(), space_id))
            return True

    def _learning_goal_view(self, row: sqlite3.Row) -> LearningGoalView:
        return LearningGoalView(id=row["id"], space_id=row["space_id"], prompt=row["prompt"], rationale=row["rationale"], criteria_refs=_loads(row["criteria_refs"], []), boundary_ids=_loads(row["boundary_ids"], []), focus_kind=row["focus_kind"], focus_id=row["focus_id"], status=row["status"], status_reason=row["status_reason"], created_at=row["created_at"], updated_at=row["updated_at"])

    def learning_goal_context(self, space_id: str) -> tuple[CoverageMapView, list[HypothesisView], list[LearningGoalView], list[LearningGoalView]] | None:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM coverage_maps WHERE space_id = ?", (space_id,)).fetchone()
            if not row:
                return None
            hypotheses = connection.execute("SELECT * FROM hypotheses WHERE space_id = ? AND status = 'open' ORDER BY updated_at DESC LIMIT 100", (space_id,)).fetchall()
            goals = connection.execute("SELECT * FROM learning_goals WHERE space_id = ? ORDER BY updated_at DESC", (space_id,)).fetchall()
            return self._coverage_view(row), [self._hypothesis_view(connection, item) for item in hypotheses], [self._learning_goal_view(item) for item in goals if item["status"] == "open"], [self._learning_goal_view(item) for item in goals if item["status"] != "open"][:20]

    def apply_goal_planning(self, space_id: str, expected_revision: int, result: GoalPlanningResult, context_goal_ids: set[str]) -> bool:
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM coverage_maps WHERE space_id = ?", (space_id,)).fetchone()
            if not row or row["revision"] != expected_revision:
                return False
            coverage, boundary_ids = self._coverage_view(row), {item.id for item in self._coverage_view(row).boundaries}
            open_boundary_ids = {item.id for item in self._coverage_view(row).boundaries if item.status == "open"}
            now = _now()
            for item in result.upserts:
                if not set(item.criteria_refs).issubset(coverage.criteria) or not set(item.boundary_ids).issubset(open_boundary_ids):
                    continue
                if item.focus_kind != "user":
                    table = "people" if item.focus_kind == "person" else "relationships"
                    if not item.focus_id or not connection.execute(f"SELECT 1 FROM {table} WHERE id = ? AND space_id = ?", (item.focus_id, space_id)).fetchone():
                        continue
                if item.goal_id and item.goal_id in context_goal_ids:
                    connection.execute("UPDATE learning_goals SET prompt = ?, rationale = ?, criteria_refs = ?, boundary_ids = ?, focus_kind = ?, focus_id = ?, updated_at = ? WHERE id = ? AND space_id = ?", (item.prompt, item.rationale, _json(item.criteria_refs), _json(item.boundary_ids), item.focus_kind, item.focus_id, now, item.goal_id, space_id))
                elif not item.goal_id:
                    duplicate = connection.execute("SELECT 1 FROM learning_goals WHERE space_id = ? AND prompt = ? AND criteria_refs = ? AND boundary_ids = ? AND status = 'open'", (space_id, item.prompt, _json(item.criteria_refs), _json(item.boundary_ids))).fetchone()
                    if not duplicate:
                        connection.execute("INSERT INTO learning_goals(id, space_id, prompt, rationale, criteria_refs, boundary_ids, focus_kind, focus_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (_id("goal"), space_id, item.prompt, item.rationale, _json(item.criteria_refs), _json(item.boundary_ids), item.focus_kind, item.focus_id, now, now))
            for item in result.transitions:
                if item.goal_id in context_goal_ids:
                    connection.execute("UPDATE learning_goals SET status = ?, status_reason = ?, updated_at = ? WHERE id = ? AND space_id = ?", (item.status, item.reason, now, item.goal_id, space_id))
            return True

    def continuity_context(
        self, space_id: str, limit: int = 32,
        max_message_chars: int = 8000, max_total_chars: int = 48000,
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
            rows = connection.execute(
                "SELECT * FROM messages WHERE space_id = ? AND rowid > ? "
                "ORDER BY rowid LIMIT ?", (space_id, after, limit)
            ).fetchall()
            messages: list[ModelMessage] = []
            total_chars = 0
            for item in rows:
                remaining = max_total_chars - total_chars
                if remaining <= 0:
                    break
                content = item["content"][:min(max_message_chars, remaining)]
                messages.append(ModelMessage(
                    id=item["id"], space_id=item["space_id"], author=item["author"],
                    content=content, occurred_at=item["occurred_at"],
                    source_provider=item["source_provider"],
                    source_conversation_key=item["source_conversation_key"],
                    source_item_id=item["source_item_id"],
                    source_metadata=_loads(item["source_metadata"], {}),
                ))
                total_chars += len(content)
            return continuity, messages

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

    def _guidance(self, connection: sqlite3.Connection, space_id: str, activated_person_ids: Iterable[str] = (), query: str = "") -> GuidanceBundle:
        """Return a bounded, deterministic set; coverage maps never leave this method."""
        person_ids = tuple(dict.fromkeys(activated_person_ids))
        placeholders = ','.join('?' for _ in person_ids) or 'NULL'
        rows = connection.execute(
            f"SELECT id, owner_kind, owner_id, content, confidence, status, updated_at FROM hypotheses "
            f"WHERE space_id = ? AND status = 'open' AND (owner_kind = 'user' OR (owner_kind = 'person' AND owner_id IN ({placeholders})) OR "
            f"(owner_kind = 'relationship' AND owner_id IN (SELECT id FROM relationships WHERE space_id = ? AND (person_a_id IN ({placeholders}) OR person_b_id IN ({placeholders}))))) "
            "ORDER BY updated_at DESC, id",
            (space_id, *person_ids, space_id, *person_ids, *person_ids),
        ).fetchall()
        updated: dict[str, str] = {}
        items = [GuidanceItem(id=row['id'], kind='hypothesis', content=row['content'], owner_kind=row['owner_kind'], owner_id=row['owner_id'], status=row['status'], confidence=row['confidence']) for row in rows]
        updated.update({row['id']: row['updated_at'] for row in rows})
        goals = connection.execute(
            f"SELECT id, focus_kind, focus_id, prompt, status, updated_at FROM learning_goals "
            f"WHERE space_id = ? AND status IN ('open', 'partial') AND (focus_kind = 'user' OR (focus_kind = 'person' AND focus_id IN ({placeholders})) OR "
            f"(focus_kind = 'relationship' AND focus_id IN (SELECT id FROM relationships WHERE space_id = ? AND (person_a_id IN ({placeholders}) OR person_b_id IN ({placeholders}))))) "
            "ORDER BY updated_at DESC, id",
            (space_id, *person_ids, space_id, *person_ids, *person_ids),
        ).fetchall()
        items.extend(GuidanceItem(id=row['id'], kind='learning_goal', content=row['prompt'], owner_kind=row['focus_kind'], owner_id=row['focus_id'], status=row['status']) for row in goals)
        updated.update({row['id']: row['updated_at'] for row in goals})
        # Lexical relevance is only a tie-breaker over recency. Character
        # bigrams keep short/non-space-separated text useful (including CJK).
        folded = query.casefold()
        grams = {folded[i:i + 2] for i in range(max(0, len(folded) - 1))}
        def rank(item: GuidanceItem) -> tuple[int, str, str]:
            text = item.content.casefold()
            relevance = (1000 if folded and folded in text else 0) + sum(text.count(g) for g in grams)
            return (relevance, updated[item.id], item.id)
        by_kind: dict[str, list[GuidanceItem]] = {"hypothesis": [], "learning_goal": []}
        for item in items:
            by_kind[item.kind].append(item)
        selected = [sorted(by_kind[k], key=rank, reverse=True)[0] for k in ("hypothesis", "learning_goal") if by_kind[k]]
        selected.sort(key=lambda item: (item.kind, item.id))
        return GuidanceBundle(items=selected)

    def guidance_bundle(self, space_id: str, activated_person_ids: Iterable[str] = (), query: str = "") -> GuidanceBundle:
        with self._connect() as connection:
            return self._guidance(connection, space_id, activated_person_ids, query)

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
            guidance = self._guidance(connection, space_id)
            payload = {"user_model": user_view.model_dump(mode="json") if user_view else None,
                       "continuity": continuity.model_dump(mode="json") if continuity else None,
                       "people": [item.model_dump(mode="json") for item in people],
                       "guidance": guidance.model_dump(mode="json")}
            version = hashlib.sha256(_json(payload).encode()).hexdigest()[:16]
            return ContextBundle(version=version, user_model=user_view, continuity=continuity, people=people, guidance=guidance)

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

    def merge_person(
        self, space_id: str, source_person_id: str, target_person_id: str
    ) -> dict[str, Any]:
        """Atomically merge ``source_person_id`` into ``target_person_id``.

        This is deliberately a domain operation, not an LLM operation.  The
        caller is responsible for obtaining user confirmation before invoking
        it.  Raw messages and memory text remain immutable.
        """
        if source_person_id == target_person_id:
            raise PersonMergeError("source and target person must differ", conflict=True)
        with self._connect() as connection:
            source = connection.execute(
                "SELECT * FROM people WHERE space_id = ? AND id = ?",
                (space_id, source_person_id),
            ).fetchone()
            target = connection.execute(
                "SELECT * FROM people WHERE space_id = ? AND id = ?",
                (space_id, target_person_id),
            ).fetchone()
            if source is None or target is None:
                raise PersonMergeError("source or target person not found")
            if source["status"] == "merged":
                if source["merged_into_person_id"] == target_person_id:
                    return {
                        "source_person_id": source_person_id,
                        "target_person_id": target_person_id,
                        "status": "merged",
                        "affected_relationship_ids": [],
                    }
                raise PersonMergeError(
                    "source person was already merged into another person",
                    conflict=True,
                )
            if source["status"] != "active" or target["status"] != "active":
                raise PersonMergeError("source and target persons must be active", conflict=True)

            now = _now()
            aliases = connection.execute(
                "SELECT value, normalized_value FROM person_aliases WHERE person_id = ?",
                (source_person_id,),
            ).fetchall()
            for alias in aliases:
                connection.execute(
                    "INSERT OR IGNORE INTO person_aliases("
                    "id, space_id, person_id, value, normalized_value"
                    ") VALUES (?, ?, ?, ?, ?)",
                    (
                        _id("alias"),
                        space_id,
                        target_person_id,
                        alias["value"],
                        alias["normalized_value"],
                    ),
                )
            connection.execute(
                "DELETE FROM person_aliases WHERE person_id = ?",
                (source_person_id,),
            )

            connection.execute(
                "INSERT OR IGNORE INTO memory_people(memory_id, person_id) "
                "SELECT memory_id, ? FROM memory_people WHERE person_id = ?",
                (target_person_id, source_person_id),
            )
            connection.execute(
                "DELETE FROM memory_people WHERE person_id = ?", (source_person_id,)
            )

            affected_relationships: set[str] = set()
            relationships = connection.execute(
                "SELECT * FROM relationships "
                "WHERE space_id = ? AND (person_a_id = ? OR person_b_id = ?)",
                (space_id, source_person_id, source_person_id),
            ).fetchall()
            for relationship in relationships:
                other_id = (
                    relationship["person_b_id"]
                    if relationship["person_a_id"] == source_person_id
                    else relationship["person_a_id"]
                )
                if other_id == target_person_id:
                    connection.execute(
                        "DELETE FROM memory_relationships WHERE relationship_id = ?",
                        (relationship["id"],),
                    )
                    connection.execute(
                        "DELETE FROM relationships WHERE id = ?",
                        (relationship["id"],),
                    )
                    continue
                a_id, b_id = sorted((target_person_id, other_id))
                existing = connection.execute(
                    "SELECT id FROM relationships "
                    "WHERE space_id = ? AND person_a_id = ? AND person_b_id = ?",
                    (space_id, a_id, b_id),
                ).fetchone()
                if existing:
                    affected_relationships.add(existing["id"])
                    connection.execute(
                        "INSERT OR IGNORE INTO memory_relationships("
                        "memory_id, relationship_id, role"
                        ") SELECT memory_id, ?, role FROM memory_relationships "
                        "WHERE relationship_id = ?",
                        (existing["id"], relationship["id"]),
                    )
                    connection.execute(
                        "DELETE FROM memory_relationships WHERE relationship_id = ?",
                        (relationship["id"],),
                    )
                    connection.execute(
                        "DELETE FROM relationships WHERE id = ?",
                        (relationship["id"],),
                    )
                else:
                    connection.execute(
                        "UPDATE relationships SET person_a_id = ?, person_b_id = ?, "
                        "facets = '[]', closeness = NULL, tone = NULL, "
                        "status = 'unknown', summary = '', "
                        "profile_source_updated_at = NULL, profile_updated_at = NULL, "
                        "updated_at = ? WHERE id = ?",
                        (a_id, b_id, now, relationship["id"]),
                    )
                    affected_relationships.add(relationship["id"])
            for relationship_id in affected_relationships:
                connection.execute(
                    "UPDATE relationships SET facets = '[]', closeness = NULL, "
                    "tone = NULL, status = 'unknown', summary = '', "
                    "profile_source_updated_at = NULL, profile_updated_at = NULL, "
                    "updated_at = ? WHERE id = ?",
                    (now, relationship_id),
                )

            continuity = connection.execute(
                "SELECT related_person_ids FROM continuities WHERE space_id = ?", (space_id,)
            ).fetchone()
            if continuity:
                ids = _loads(continuity["related_person_ids"], [])
                rewritten = list(
                    dict.fromkeys(
                        target_person_id if item == source_person_id else item
                        for item in ids
                    )
                )
                connection.execute(
                    "UPDATE continuities SET related_person_ids = ?, updated_at = ? "
                    "WHERE space_id = ?",
                    (_json(rewritten), now, space_id),
                )
            connection.execute(
                "UPDATE people SET status = 'merged', merged_into_person_id = ?, "
                "updated_at = ? WHERE id = ?",
                (target_person_id, now, source_person_id),
            )
            connection.execute(
                "UPDATE people SET profile_source_updated_at = NULL, "
                "profile_updated_at = NULL, updated_at = ? WHERE id = ?",
                (now, target_person_id),
            )
            return {
                "source_person_id": source_person_id,
                "target_person_id": target_person_id,
                "status": "merged",
                "affected_relationship_ids": sorted(affected_relationships),
            }

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
        context_hypothesis_ids: set[str] | None = None,
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
            if updated.rowcount != 1:
                return False
            if result.hypothesis_actions:
                source_ids = {row["id"] for row in connection.execute("SELECT id FROM memories WHERE space_id = ? AND status = 'active' AND about_user = 1 AND basis <> 'inferred'", (space_id,)).fetchall()}
                self._apply_hypothesis_actions(connection, space_id, "user", None, source_ids, context_hypothesis_ids or set(), result.hypothesis_actions)
            return True

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
                       WHERE m.space_id = p.space_id AND mp.person_id = p.id
                         AND m.basis <> 'inferred')
                    ) AND EXISTS (SELECT 1 FROM memories m JOIN memory_people mp ON mp.memory_id = m.id
                       WHERE m.space_id = p.space_id AND mp.person_id = p.id
                         AND m.basis <> 'inferred')"""
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

    def stale_coverage_spaces(self) -> list[str]:
        with self._connect() as connection:
            return [
                row["space_id"] for row in connection.execute(
                    """SELECT cm.space_id FROM coverage_maps cm WHERE EXISTS (
                       SELECT 1 FROM memories m WHERE m.space_id = cm.space_id
                       AND m.basis <> 'inferred' AND (
                         m.updated_at > COALESCE(cm.source_watermark, '') OR
                         (m.updated_at = COALESCE(cm.source_watermark, '')
                          AND m.id > COALESCE(cm.source_cursor_id, ''))
                       )
                    )"""
                ).fetchall()
            ]

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
