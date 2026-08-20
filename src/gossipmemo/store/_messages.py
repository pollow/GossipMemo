"""Raw messages and the extraction-batch lifecycle."""

from __future__ import annotations

import logging
import sqlite3
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ..models import (
    ExtractedMemory,
    ExtractionResult,
    MemoryView,
    MessageInput,
    ModelMessage,
)
from ._errors import AmbiguousPersonError
from ._memories import _MemoriesMixin
from .policy import (
    RRF_CANDIDATE_K,
    dump_json,
    fts_query,
    load_json,
    new_id,
    normalized,
    now_iso,
    reciprocal_rank_fusion,
    similar_memory_content,
)

logger = logging.getLogger(__name__)

DEFAULT_EXTRACTION_COMPARISON_LIMIT = 12


@dataclass
class PendingExtraction:
    """One batch of messages still awaiting extraction.

    `batch_id` is None for messages that have not been batched yet, in
    which case `attempts` and `state` describe nothing useful.
    """

    space_id: str
    batch_id: str | None
    message_ids: list[str]
    attempts: int
    state: str


class _MessagesMixin(_MemoriesMixin):
    """Records messages and drives extraction batches through their states."""

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

                message_id = new_id("message")
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
                            now_iso(),
                            message.source.provider,
                            message.source.conversation_key,
                            message.source.item_id,
                            dump_json(message.source.metadata),
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
            batch_id = new_id("batch")
            connection.execute(
                "INSERT INTO extraction_batches(id, space_id, created_at) VALUES (?, ?, ?)",
                (batch_id, space_id, now_iso()),
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
                    source_metadata=load_json(row["source_metadata"], {}),
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
                    source_metadata=load_json(row["source_metadata"], {}),
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
        query_vectors: Mapping[str, Sequence[float]] | None = None,
    ) -> list[MemoryView]:
        """Return a small comparison set; these rows are never new evidence.

        `query_vectors` is optional: a new-fact text -> already-embedded
        query vector map (the extraction reasoner embeds the batch's
        user-authored texts with the "same fact?" instruction before
        calling this). Keyed by the exact text rather than position, so a
        text this method's own SQL selection doesn't happen to carry a key
        for just falls back to FTS-only for that text -- this stays
        resilient to any drift between the reasoner's view of the batch
        and the `texts` selected below, rather than depending on both
        sides agreeing on an implicit row order.

        `query_vectors=None` (or empty) is the exact pre-existing
        FTS + person-match candidate pool, sorted by recency,
        byte-for-byte. When given, the FTS and vector hits are RRF-fused
        into a ranked prefix of the result; any person-matched candidate
        that neither path surfaced is still appended after it, most recent
        first, so the "known people" signal is never silently dropped.
        """
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
            fts_ranked_ids: list[str] = []
            for text in texts:
                fts = fts_query(text)
                if not fts:
                    continue
                for row in connection.execute(
                    """SELECT m.* FROM memory_fts JOIN memories m ON m.rowid = memory_fts.rowid
                       WHERE memory_fts MATCH ? AND m.space_id = ? AND m.status = 'active'
                         AND m.basis <> 'inferred'
                       ORDER BY bm25(memory_fts), m.created_at DESC LIMIT ?""",
                    (fts, space_id, limit),
                ).fetchall():
                    if row["id"] not in candidates:
                        fts_ranked_ids.append(row["id"])
                    candidates[row["id"]] = row

            vector_ranked_ids: list[str] = []
            if query_vectors:
                seen: set[str] = set()
                for text in texts:
                    vector = query_vectors.get(text)
                    if vector is None:
                        continue
                    try:
                        hits = self.search_vectors(
                            space_id, "memory", vector, RRF_CANDIDATE_K, statuses=["active"],
                        )
                    except Exception:
                        logger.exception(
                            "vector comparison search failed for batch %s; falling "
                            "back to FTS-only ranking for this text", batch_id,
                        )
                        continue
                    for owner_id, _ in hits:
                        if owner_id not in seen:
                            seen.add(owner_id)
                            vector_ranked_ids.append(owner_id)
                missing_ids = [
                    item_id for item_id in vector_ranked_ids if item_id not in candidates
                ]
                if missing_ids:
                    placeholders = ",".join("?" for _ in missing_ids)
                    for row in connection.execute(
                        f"""SELECT m.* FROM memories m WHERE m.space_id = ?
                           AND m.status = 'active' AND m.basis <> 'inferred'
                           AND m.id IN ({placeholders})""",
                        (space_id, *missing_ids),
                    ).fetchall():
                        candidates[row["id"]] = row
                # A vector hit whose owner row didn't match (retracted/superseded
                # since the search, or a different basis) never entered
                # `candidates` -- drop it from the ranking too.
                vector_ranked_ids = [
                    item_id for item_id in vector_ranked_ids if item_id in candidates
                ]

            if person_ids:
                placeholders = ",".join("?" for _ in person_ids)
                for row in connection.execute(
                    f"""SELECT DISTINCT m.* FROM memories m
                       JOIN memory_people mp ON mp.memory_id = m.id
                       WHERE m.space_id = ? AND m.status = 'active' AND m.basis <> 'inferred'
                         AND mp.person_id IN ({placeholders}) ORDER BY m.updated_at DESC LIMIT ?""",
                    (space_id, *person_ids, limit),
                ).fetchall():
                    candidates[row["id"]] = row

            if query_vectors and (fts_ranked_ids or vector_ranked_ids):
                fused_ids = reciprocal_rank_fusion([fts_ranked_ids, vector_ranked_ids])
                remaining = sorted(
                    (row_id for row_id in candidates if row_id not in fused_ids),
                    key=lambda row_id: candidates[row_id]["updated_at"], reverse=True,
                )
                ordered_ids = (fused_ids + remaining)[:limit]
                rows = [candidates[row_id] for row_id in ordered_ids]
            else:
                rows = sorted(candidates.values(),
                              key=lambda row: row["updated_at"], reverse=True)[:limit]
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
                normalized(person.display_name) for person in result.people
            )
            for person in result.people:
                try:
                    person_id = self._create_person(
                        connection, space_id, person.display_name,
                        reuse_unique_name=(
                            extracted_name_counts[normalized(person.display_name)] == 1),
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
                            new_id("alias"),
                            space_id,
                            person_id,
                            alias,
                            normalized(alias),
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

            now = now_iso()
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
                    resolved_id = resolve_extracted_person(reference)
                    if resolved_id:
                        people_ids.add(resolved_id)
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

                def same_shape(
                    row: sqlite3.Row,
                    candidate: ExtractedMemory = candidate,
                    people_ids: set[str] = people_ids,
                    relationship_ids: set[str] = relationship_ids,
                ) -> bool:
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
                    and similar_memory_content(row["content"], candidate.content)
                    for row in comparison_rows.values()
                ):
                    continue
                if any(
                    item.kind == candidate.kind and item.basis == candidate.basis
                    and item.about_user == candidate.about_user
                    and item.valid_from == candidate.valid_from
                    and item.valid_to == candidate.valid_to
                    and prior_people == people_ids and prior_relationships == relationship_ids
                    and similar_memory_content(item.content, candidate.content)
                    for item, prior_people, prior_relationships in inserted_signatures
                ):
                    continue
                memory_id = new_id("memory")
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
                    resolved_id = resolve_extracted_person(reference)
                    if not resolved_id:
                        continue
                    connection.execute(
                        """
                        INSERT OR IGNORE INTO memory_people(memory_id, person_id)
                        VALUES (?, ?)
                        """,
                        (memory_id, resolved_id),
                    )
                    affected_people.add(resolved_id)

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
                           invalidated_at = ?, invalidation_reason = ?, updated_at = ?
                           WHERE id = ?""",
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

    def pending_extractions(self) -> list[PendingExtraction]:
        """Every batch still awaiting extraction, neediest first.

        Carries each batch's attempt count and state so the drain can decide
        per batch whether to keep trying it without a follow-up query per
        pending batch.
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT space_id, id, extraction_batch_id,
                       extraction_attempts, extraction_state
                FROM messages
                WHERE extraction_state IN ('pending', 'failed')
                ORDER BY extraction_attempts, ingested_at
                """
            ).fetchall()
        grouped: dict[tuple[str, str | None], PendingExtraction] = {}
        for row in rows:
            key = (row["space_id"], row["extraction_batch_id"])
            pending = grouped.get(key)
            if pending is None:
                # Every message in a batch shares the batch's attempts and
                # state; the first row settles both.
                grouped[key] = PendingExtraction(
                    space_id=row["space_id"],
                    batch_id=row["extraction_batch_id"],
                    message_ids=[row["id"]],
                    attempts=row["extraction_attempts"],
                    state=row["extraction_state"],
                )
            else:
                pending.message_ids.append(row["id"])
        return list(grouped.values())

    def extraction_states(self, space_id: str, message_ids: list[str]) -> list[str]:
        if not message_ids:
            return []
        states: list[str] = []
        for offset in range(0, len(message_ids), 500):
            chunk = message_ids[offset: offset + 500]
            placeholders = ",".join("?" for _ in chunk)
            with self._connect() as connection:
                rows = connection.execute(
                    f"SELECT extraction_state FROM messages "
                    f"WHERE space_id = ? AND id IN ({placeholders})",
                    (space_id, *chunk),
                ).fetchall()
            states.extend(row["extraction_state"] for row in rows)
        return states
