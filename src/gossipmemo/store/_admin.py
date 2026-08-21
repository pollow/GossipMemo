"""Read-only aggregate queries for the admin UI.

Deliberately **not** part of the `WorldStore` protocol in `world.py`: the
admin views are SQLite-only, plain-SQL, and read-only, and that coupling to
the concrete adapter is intentional and honest rather than something the
reasoners should ever depend on. Every method here only reads; nothing in
this module writes.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

from .policy import load_json
from ._base import _BaseStore

#: Space-wide message/memory/people counts used on both the space list and
#: a single space's overview page.
_COUNTS_SQL = """
    SELECT
        (SELECT COUNT(*) FROM messages WHERE space_id = s.id) AS message_count,
        (SELECT COUNT(*) FROM memories WHERE space_id = s.id) AS memory_count,
        (SELECT COUNT(*) FROM people WHERE space_id = s.id AND status = 'active') AS people_count
    FROM spaces s WHERE s.id = ?
"""


@dataclass
class SpaceSummary:
    id: str
    name: str
    created_at: str
    message_count: int
    memory_count: int
    people_count: int


@dataclass
class SpaceOverview:
    id: str
    name: str
    message_count: int
    memory_count: int
    people_count: int
    user_model_profile_card: str
    user_model_updated_at: str | None
    continuity_text: str
    continuity_updated_at: str | None
    continuity_related_person_ids: list[str]


@dataclass
class MessageRow:
    id: str
    author: str
    content: str
    occurred_at: str
    ingested_at: str
    extraction_batch_id: str | None
    extraction_state: str


@dataclass
class MemoryRow:
    id: str
    content: str
    kind: str
    status: str
    about_user: bool
    created_at: str


@dataclass
class LinkedPerson:
    id: str
    display_name: str


@dataclass
class LinkedRelationship:
    id: str
    person_a_id: str
    person_b_id: str
    summary: str


@dataclass
class DerivationLink:
    memory_id: str
    content: str
    role: str


@dataclass
class MemoryDetail:
    id: str
    content: str
    kind: str
    basis: str
    status: str
    about_user: bool
    valid_from: str | None
    valid_to: str | None
    created_at: str
    updated_at: str
    supersedes_memory_id: str | None
    invalidated_at: str | None
    invalidation_reason: str | None
    source_batch_id: str | None
    created_by: str
    people: list[LinkedPerson] = field(default_factory=list)
    relationships: list[LinkedRelationship] = field(default_factory=list)
    derived_from: list[DerivationLink] = field(default_factory=list)
    derives: list[DerivationLink] = field(default_factory=list)
    source_messages: list[MessageRow] = field(default_factory=list)


def _message_row(row: sqlite3.Row) -> MessageRow:
    return MessageRow(
        id=row["id"],
        author=row["author"],
        content=row["content"],
        occurred_at=row["occurred_at"],
        ingested_at=row["ingested_at"],
        extraction_batch_id=row["extraction_batch_id"],
        extraction_state=row["extraction_state"],
    )


class _AdminReadMixin(_BaseStore):
    """Read-only queries backing the admin UI. See module docstring."""

    def admin_list_spaces(self) -> list[SpaceSummary]:
        # One query per space for its counts, rather than one clever
        # correlated-subquery join: this only runs against a landing page
        # with a handful of spaces, and plain code stays easy to trust.
        with self._connect() as connection:
            space_rows = connection.execute(
                "SELECT id, name, created_at FROM spaces ORDER BY name, id"
            ).fetchall()
            summaries: list[SpaceSummary] = []
            for space in space_rows:
                counts = connection.execute(_COUNTS_SQL, (space["id"],)).fetchone()
                summaries.append(
                    SpaceSummary(
                        id=space["id"],
                        name=space["name"],
                        created_at=space["created_at"],
                        message_count=counts["message_count"],
                        memory_count=counts["memory_count"],
                        people_count=counts["people_count"],
                    )
                )
            return summaries

    def admin_space_overview(self, space_id: str) -> SpaceOverview | None:
        with self._connect() as connection:
            space = connection.execute(
                "SELECT id, name FROM spaces WHERE id = ?", (space_id,)
            ).fetchone()
            if not space:
                return None
            counts = connection.execute(_COUNTS_SQL, (space_id,)).fetchone()
            user_model = connection.execute(
                "SELECT profile_card, profile_updated_at FROM user_models WHERE space_id = ?",
                (space_id,),
            ).fetchone()
            continuity = connection.execute(
                "SELECT text, related_person_ids, updated_at FROM continuities "
                "WHERE space_id = ?",
                (space_id,),
            ).fetchone()
            return SpaceOverview(
                id=space["id"],
                name=space["name"],
                message_count=counts["message_count"],
                memory_count=counts["memory_count"],
                people_count=counts["people_count"],
                user_model_profile_card=(user_model["profile_card"] if user_model else "{}"),
                user_model_updated_at=(user_model["profile_updated_at"] if user_model else None),
                continuity_text=(continuity["text"] if continuity else ""),
                continuity_updated_at=(continuity["updated_at"] if continuity else None),
                continuity_related_person_ids=(
                    load_json(continuity["related_person_ids"], []) if continuity else []
                ),
            )

    def admin_count_messages(self, space_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS n FROM messages WHERE space_id = ?", (space_id,)
            ).fetchone()
            return row["n"]

    def admin_list_messages(
        self, space_id: str, offset: int, limit: int
    ) -> list[MessageRow]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM messages WHERE space_id = ?
                ORDER BY occurred_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                (space_id, limit, offset),
            ).fetchall()
            return [_message_row(row) for row in rows]

    def admin_count_memories(
        self,
        space_id: str,
        *,
        state: str | None = None,
        kind: str | None = None,
        about_user: bool | None = None,
    ) -> int:
        clauses, params = _memory_filter_clauses(space_id, state, kind, about_user)
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) AS n FROM memories WHERE {' AND '.join(clauses)}",
                params,
            ).fetchone()
            return row["n"]

    def admin_list_memories(
        self,
        space_id: str,
        offset: int,
        limit: int,
        *,
        state: str | None = None,
        kind: str | None = None,
        about_user: bool | None = None,
    ) -> list[MemoryRow]:
        clauses, params = _memory_filter_clauses(space_id, state, kind, about_user)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, content, kind, status, about_user, created_at FROM memories
                WHERE {' AND '.join(clauses)}
                ORDER BY created_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
            return [
                MemoryRow(
                    id=row["id"],
                    content=row["content"],
                    kind=row["kind"],
                    status=row["status"],
                    about_user=bool(row["about_user"]),
                    created_at=row["created_at"],
                )
                for row in rows
            ]

    def admin_memory_detail(self, space_id: str, memory_id: str) -> MemoryDetail | None:
        with self._connect() as connection:
            memory = connection.execute(
                "SELECT * FROM memories WHERE space_id = ? AND id = ?",
                (space_id, memory_id),
            ).fetchone()
            if not memory:
                return None

            people_rows = connection.execute(
                """
                SELECT p.id, p.display_name FROM people p
                JOIN memory_people mp ON mp.person_id = p.id
                WHERE mp.memory_id = ?
                ORDER BY p.display_name, p.id
                """,
                (memory_id,),
            ).fetchall()
            people = [LinkedPerson(id=row["id"], display_name=row["display_name"])
                      for row in people_rows]

            relationship_rows = connection.execute(
                """
                SELECT r.id, r.person_a_id, r.person_b_id, r.summary FROM relationships r
                JOIN memory_relationships mr ON mr.relationship_id = r.id
                WHERE mr.memory_id = ?
                ORDER BY r.id
                """,
                (memory_id,),
            ).fetchall()
            relationships = [
                LinkedRelationship(
                    id=row["id"],
                    person_a_id=row["person_a_id"],
                    person_b_id=row["person_b_id"],
                    summary=row["summary"],
                )
                for row in relationship_rows
            ]

            derived_from_rows = connection.execute(
                """
                SELECT m.id, m.content, d.derivation_role FROM memory_derivations d
                JOIN memories m ON m.id = d.source_memory_id
                WHERE d.derived_memory_id = ?
                ORDER BY m.created_at DESC, m.id
                """,
                (memory_id,),
            ).fetchall()
            derived_from = [
                DerivationLink(memory_id=row["id"],
                               content=row["content"], role=row["derivation_role"])
                for row in derived_from_rows
            ]

            derives_rows = connection.execute(
                """
                SELECT m.id, m.content, d.derivation_role FROM memory_derivations d
                JOIN memories m ON m.id = d.derived_memory_id
                WHERE d.source_memory_id = ?
                ORDER BY m.created_at DESC, m.id
                """,
                (memory_id,),
            ).fetchall()
            derives = [
                DerivationLink(memory_id=row["id"],
                               content=row["content"], role=row["derivation_role"])
                for row in derives_rows
            ]

            source_messages: list[MessageRow] = []
            if memory["source_batch_id"]:
                message_rows = connection.execute(
                    """
                    SELECT * FROM messages WHERE space_id = ? AND extraction_batch_id = ?
                    ORDER BY occurred_at
                    """,
                    (space_id, memory["source_batch_id"]),
                ).fetchall()
                source_messages = [_message_row(row) for row in message_rows]

            return MemoryDetail(
                id=memory["id"],
                content=memory["content"],
                kind=memory["kind"],
                basis=memory["basis"],
                status=memory["status"],
                about_user=bool(memory["about_user"]),
                valid_from=memory["valid_from"],
                valid_to=memory["valid_to"],
                created_at=memory["created_at"],
                updated_at=memory["updated_at"],
                supersedes_memory_id=memory["supersedes_memory_id"],
                invalidated_at=memory["invalidated_at"],
                invalidation_reason=memory["invalidation_reason"],
                source_batch_id=memory["source_batch_id"],
                created_by=memory["created_by"],
                people=people,
                relationships=relationships,
                derived_from=derived_from,
                derives=derives,
                source_messages=source_messages,
            )


def _memory_filter_clauses(
    space_id: str, state: str | None, kind: str | None, about_user: bool | None
) -> tuple[list[str], list[object]]:
    clauses = ["space_id = ?"]
    params: list[object] = [space_id]
    if state is not None:
        clauses.append("status = ?")
        params.append(state)
    if kind is not None:
        clauses.append("kind = ?")
        params.append(kind)
    if about_user is not None:
        clauses.append("about_user = ?")
        params.append(1 if about_user else 0)
    return clauses, params


__all__ = [
    "DerivationLink",
    "LinkedPerson",
    "LinkedRelationship",
    "MemoryDetail",
    "MemoryRow",
    "MessageRow",
    "SpaceOverview",
    "SpaceSummary",
    "_AdminReadMixin",
]
