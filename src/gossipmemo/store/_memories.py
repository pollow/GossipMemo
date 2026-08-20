"""Memory reads, recall, and the manual memory lifecycle."""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Iterable, Sequence
from typing import Any

from ..models import (
    ManualMemoryRequest,
    MemoryView,
    PersonView,
    QueryContext,
    QueryRequest,
    RelationshipView,
    SupersedeRequest,
)
from ._people import _PeopleMixin
from .policy import (
    RRF_CANDIDATE_K,
    fts_query,
    new_id,
    now_iso,
    reciprocal_rank_fusion,
)

logger = logging.getLogger(__name__)


class _MemoriesMixin(_PeopleMixin):
    """Reads, recalls, adds, supersedes, and retracts Memories."""

    def read(
        self, space_id: str, request: QueryRequest,
        query_vector: Sequence[float] | None = None,
    ) -> QueryContext:
        """Candidate pruning for `/v1/spaces/{id}/query`.

        `query_vector` is optional and computed by the caller (see
        `store/_context.py`'s module docstring on why the store never
        awaits): `None` means "run exactly the pure-FTS path this always
        ran", byte-for-byte. When present, it adds a second, RRF-fused
        recall path over `memories` embeddings, scoped to the same
        structural candidate pool (`memory_rows`) this method already
        computed -- so person/relationship scoping keeps applying to
        vector hits exactly as it does to FTS hits.
        """
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

            rows_by_id = {row["id"]: row for row in memory_rows}
            fts_ranked_ids: list[str] = []
            query = fts_query(request.question, request.people)
            if query and memory_rows:
                memory_ids = [row["id"] for row in memory_rows]
                placeholders = ",".join("?" for _ in memory_ids)
                fts_limit = RRF_CANDIDATE_K if query_vector is not None else request.limit
                matched = connection.execute(
                    f"""
                    SELECT m.* FROM memory_fts
                    JOIN memories m ON m.rowid = memory_fts.rowid
                    WHERE memory_fts MATCH ? AND m.id IN ({placeholders})
                    ORDER BY bm25(memory_fts), m.created_at DESC LIMIT ?
                    """,
                    [query, *memory_ids, fts_limit],
                ).fetchall()
                # Natural-language wording can have no lexical overlap. In
                # that case the latest structurally scoped memories remain a
                # useful fallback for synthesis.
                if matched:
                    fts_ranked_ids = [row["id"] for row in matched]
                    if query_vector is None:
                        memory_rows = matched
            if query_vector is not None and memory_rows:
                allowed_ids = set(rows_by_id)
                try:
                    vector_hits = self.search_vectors(
                        space_id, "memory", query_vector, RRF_CANDIDATE_K,
                        statuses=["active"],
                    )
                except Exception:
                    logger.exception(
                        "vector search failed for %s; falling back to FTS-only ranking",
                        space_id,
                    )
                    vector_hits = []
                vector_ranked_ids = [
                    owner_id for owner_id, _ in vector_hits if owner_id in allowed_ids
                ]
                if fts_ranked_ids or vector_ranked_ids:
                    fused_ids = reciprocal_rank_fusion([fts_ranked_ids, vector_ranked_ids])
                    memory_rows = [rows_by_id[item_id] for item_id in fused_ids]
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

    def recall_user_memories(self, space_id: str, text: str, limit: int = 5) -> list[MemoryView]:
        return self.recall_memories(space_id, text, about_user=True, limit=limit)

    def recall_memories(
        self, space_id: str, text: str, about_user: bool | None = None,
        person_ids: Iterable[str] | None = None, limit: int = 5,
    ) -> list[MemoryView]:
        with self._connect() as connection:
            return self._recall_memories(
                connection, space_id, text, about_user=about_user,
                person_ids=person_ids, limit=limit,
            )

    def _recall_memories(
        self, connection: sqlite3.Connection, space_id: str, text: str,
        about_user: bool | None = None, person_ids: Iterable[str] | None = None,
        limit: int = 5, query_vector: Sequence[float] | None = None,
    ) -> list[MemoryView]:
        """FTS (optionally RRF-fused with vector) recall on a caller's
        connection, so `turn_view` can share it.

        `query_vector=None` is the exact pre-existing pure-FTS path,
        byte-for-byte. When given, a second recall path searches `memory`
        embeddings and is fused with the FTS ranking by RRF; the
        `about_user`/`person_ids` filters are re-applied in SQL to whatever
        vector hits aren't already among the FTS rows, so both paths honor
        the same scoping.
        """
        query = fts_query(text)
        if not query:
            return []
        person_ids = list(person_ids) if person_ids is not None else None
        base_clauses = ["m.space_id = ?", "m.status = 'active'"]
        base_params: list[Any] = [space_id]
        if about_user is not None:
            base_clauses.append("m.about_user = ?")
            base_params.append(1 if about_user else 0)
        if person_ids:
            placeholders = ",".join("?" for _ in person_ids)
            base_clauses.append(
                f"m.id IN (SELECT memory_id FROM memory_people WHERE person_id IN ({placeholders}))"
            )
            base_params.extend(person_ids)

        fts_limit = RRF_CANDIDATE_K if query_vector is not None else limit
        fts_params: list[Any] = [query, *base_params, fts_limit]
        rows = connection.execute(
            f"""SELECT m.* FROM memory_fts JOIN memories m ON m.rowid = memory_fts.rowid
               WHERE memory_fts MATCH ? AND {' AND '.join(base_clauses)}
               ORDER BY bm25(memory_fts), m.created_at DESC LIMIT ?""",
            fts_params,
        ).fetchall()
        if query_vector is None:
            return [self._memory_view(connection, row, False) for row in rows]

        rows_by_id: dict[str, sqlite3.Row] = {row["id"]: row for row in rows}
        fts_ranked_ids = [row["id"] for row in rows]

        try:
            vector_hits = self.search_vectors(
                space_id, "memory", query_vector, RRF_CANDIDATE_K, statuses=["active"],
            )
        except Exception:
            logger.exception(
                "vector recall failed for %s; falling back to FTS-only ranking", space_id,
            )
            vector_hits = []

        vector_ranked_ids: list[str] = []
        if vector_hits:
            missing_ids = [
                owner_id for owner_id, _ in vector_hits if owner_id not in rows_by_id
            ]
            if missing_ids:
                placeholders = ",".join("?" for _ in missing_ids)
                extra_clauses = [f"m.id IN ({placeholders})", *base_clauses]
                extra_rows = connection.execute(
                    f"SELECT m.* FROM memories m WHERE {' AND '.join(extra_clauses)}",
                    [*missing_ids, *base_params],
                ).fetchall()
                for row in extra_rows:
                    rows_by_id[row["id"]] = row
            vector_ranked_ids = [
                owner_id for owner_id, _ in vector_hits if owner_id in rows_by_id
            ]

        fused_ids = reciprocal_rank_fusion([fts_ranked_ids, vector_ranked_ids])[:limit]
        return [self._memory_view(connection, rows_by_id[item_id], False) for item_id in fused_ids]

    def add_manual_memory(
        self, space_id: str, request: ManualMemoryRequest
    ) -> str:
        self.ensure_space(space_id)
        with self._connect() as connection:
            memory_id = new_id("memory")
            now = now_iso()
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
            now = now_iso()
            replacement_id = new_id("memory")
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
                    int(original["about_user"]
                        if request.about_user is None else request.about_user),
                    (request.valid_from if request.valid_from is not None
                     else original["valid_from"]),
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
            now = now_iso()
            connection.execute(
                """
                UPDATE memories SET status = 'retracted', invalidated_at = ?,
                    invalidation_reason = ?, updated_at = ?
                WHERE id = ?
                """,
                (now, reason, now, memory_id),
            )
            return True
