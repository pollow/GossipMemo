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
    person_a_name: str
    person_b_id: str
    person_b_name: str
    summary: str


@dataclass
class DerivationLink:
    memory_id: str
    content: str
    role: str


@dataclass
class PersonListRow:
    id: str
    display_name: str
    status: str
    aliases: list[str] = field(default_factory=list)


@dataclass
class RelationshipListRow:
    id: str
    person_a_id: str
    person_a_name: str
    person_b_id: str
    person_b_name: str
    status: str
    summary: str


@dataclass
class LearningGoalRow:
    id: str
    prompt: str
    rationale: str
    focus_kind: str
    focus_id: str | None
    status: str
    status_reason: str | None
    created_at: str
    updated_at: str


@dataclass
class HypothesisEvidenceRef:
    memory_id: str
    content: str


@dataclass
class HypothesisRow:
    id: str
    owner_kind: str
    owner_id: str | None
    content: str
    kind: str
    confidence: str
    status: str
    created_at: str
    updated_at: str
    support_evidence: list[HypothesisEvidenceRef] = field(default_factory=list)
    counter_evidence: list[HypothesisEvidenceRef] = field(default_factory=list)


@dataclass
class CoverageRootRow:
    root: str
    revision: int
    entry_count: int
    source_watermark: str | None
    source_cursor_id: str | None


@dataclass
class CoverageEntryRow:
    id: str
    root: str
    path: str
    content: str
    status: str
    created_at: str
    updated_at: str


#: Every kind is capped at this many hits; one extra row is fetched to
#: detect (without a COUNT) whether the cap actually truncated something.
_SEARCH_LIMIT = 20


@dataclass
class MemorySearchHit:
    id: str
    content: str


@dataclass
class MessageSearchHit:
    id: str
    content: str
    occurred_at: str


@dataclass
class PersonSearchHit:
    id: str
    display_name: str


@dataclass
class LearningGoalSearchHit:
    id: str
    prompt: str


@dataclass
class HypothesisSearchHit:
    id: str
    content: str


@dataclass
class CoverageEntrySearchHit:
    id: str
    root: str
    path: str
    content: str


@dataclass
class ContinuitySearchHit:
    text: str


@dataclass
class SearchGroup:
    hits: list
    truncated: bool


@dataclass
class SearchResults:
    memories: SearchGroup
    messages: SearchGroup
    people: SearchGroup
    learning_goals: SearchGroup
    hypotheses: SearchGroup
    coverage_entries: SearchGroup
    continuities: SearchGroup


def _escape_like(keyword: str) -> str:
    """Escape `%`, `_`, and the escape character itself for a `LIKE ...
    ESCAPE '\\'` clause. Callers must always pair this with `ESCAPE '\\'`
    in the SQL and bind the result as a parameter -- never interpolate the
    keyword into SQL text."""

    return keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


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
                SELECT r.id, r.person_a_id, pa.display_name AS person_a_name,
                       r.person_b_id, pb.display_name AS person_b_name, r.summary
                FROM relationships r
                JOIN memory_relationships mr ON mr.relationship_id = r.id
                JOIN people pa ON pa.id = r.person_a_id
                JOIN people pb ON pb.id = r.person_b_id
                WHERE mr.memory_id = ?
                ORDER BY r.id
                """,
                (memory_id,),
            ).fetchall()
            relationships = [
                LinkedRelationship(
                    id=row["id"],
                    person_a_id=row["person_a_id"],
                    person_a_name=row["person_a_name"],
                    person_b_id=row["person_b_id"],
                    person_b_name=row["person_b_name"],
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

    def admin_count_people(self, space_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS n FROM people WHERE space_id = ? AND status = 'active'",
                (space_id,),
            ).fetchone()
            return row["n"]

    def admin_list_people(
        self, space_id: str, offset: int, limit: int
    ) -> list[PersonListRow]:
        with self._connect() as connection:
            page_rows = connection.execute(
                """
                SELECT id, display_name, status FROM people
                WHERE space_id = ? AND status = 'active'
                ORDER BY display_name, id
                LIMIT ? OFFSET ?
                """,
                (space_id, limit, offset),
            ).fetchall()
            people: list[PersonListRow] = []
            for row in page_rows:
                alias_rows = connection.execute(
                    "SELECT value FROM person_aliases WHERE person_id = ? ORDER BY value",
                    (row["id"],),
                ).fetchall()
                people.append(
                    PersonListRow(
                        id=row["id"],
                        display_name=row["display_name"],
                        status=row["status"],
                        aliases=[alias["value"] for alias in alias_rows],
                    )
                )
            return people

    def admin_person_display_name(self, space_id: str, person_id: str) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT display_name FROM people WHERE space_id = ? AND id = ?",
                (space_id, person_id),
            ).fetchone()
            return row["display_name"] if row else None

    def admin_person_aliases(self, space_id: str, person_id: str) -> list[str]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT a.value FROM person_aliases a
                WHERE a.space_id = ? AND a.person_id = ?
                ORDER BY a.value
                """,
                (space_id, person_id),
            ).fetchall()
            return [row["value"] for row in rows]

    def admin_count_relationships(self, space_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS n FROM relationships WHERE space_id = ?", (space_id,)
            ).fetchone()
            return row["n"]

    def admin_list_relationships(
        self, space_id: str, offset: int, limit: int
    ) -> list[RelationshipListRow]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT r.id, r.person_a_id, pa.display_name AS person_a_name,
                       r.person_b_id, pb.display_name AS person_b_name,
                       r.status, r.summary
                FROM relationships r
                JOIN people pa ON pa.id = r.person_a_id
                JOIN people pb ON pb.id = r.person_b_id
                WHERE r.space_id = ?
                ORDER BY r.updated_at DESC, r.id DESC
                LIMIT ? OFFSET ?
                """,
                (space_id, limit, offset),
            ).fetchall()
            return [
                RelationshipListRow(
                    id=row["id"],
                    person_a_id=row["person_a_id"],
                    person_a_name=row["person_a_name"],
                    person_b_id=row["person_b_id"],
                    person_b_name=row["person_b_name"],
                    status=row["status"],
                    summary=row["summary"],
                )
                for row in rows
            ]

    def admin_count_learning_goals(
        self, space_id: str, *, root: str | None = None, status: str | None = None
    ) -> int:
        clauses, params = _learning_goal_filter_clauses(space_id, root, status)
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) AS n FROM learning_goals lg WHERE {' AND '.join(clauses)}",
                params,
            ).fetchone()
            return row["n"]

    def admin_list_learning_goals(
        self,
        space_id: str,
        offset: int,
        limit: int,
        *,
        root: str | None = None,
        status: str | None = None,
    ) -> list[LearningGoalRow]:
        clauses, params = _learning_goal_filter_clauses(space_id, root, status)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT lg.id, lg.prompt, lg.rationale, lg.focus_kind, lg.focus_id,
                       lg.status, lg.status_reason, lg.created_at, lg.updated_at
                FROM learning_goals lg
                WHERE {' AND '.join(clauses)}
                ORDER BY lg.updated_at DESC, lg.id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
            return [
                LearningGoalRow(
                    id=row["id"],
                    prompt=row["prompt"],
                    rationale=row["rationale"],
                    focus_kind=row["focus_kind"],
                    focus_id=row["focus_id"],
                    status=row["status"],
                    status_reason=row["status_reason"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]

    def admin_count_hypotheses(
        self, space_id: str, *, owner_kind: str | None = None
    ) -> int:
        clauses, params = _hypothesis_filter_clauses(space_id, owner_kind)
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) AS n FROM hypotheses WHERE {' AND '.join(clauses)}",
                params,
            ).fetchone()
            return row["n"]

    def admin_list_hypotheses(
        self,
        space_id: str,
        offset: int,
        limit: int,
        *,
        owner_kind: str | None = None,
    ) -> list[HypothesisRow]:
        clauses, params = _hypothesis_filter_clauses(space_id, owner_kind)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT id, owner_kind, owner_id, content, kind, confidence, status,
                       created_at, updated_at
                FROM hypotheses
                WHERE {' AND '.join(clauses)}
                ORDER BY updated_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                [*params, limit, offset],
            ).fetchall()
            hypotheses: list[HypothesisRow] = []
            for row in rows:
                evidence_rows = connection.execute(
                    """
                    SELECT he.role, m.id AS memory_id, m.content
                    FROM hypothesis_evidence he
                    JOIN memories m ON m.id = he.memory_id
                    WHERE he.hypothesis_id = ?
                    ORDER BY m.created_at DESC, m.id
                    """,
                    (row["id"],),
                ).fetchall()
                support = [
                    HypothesisEvidenceRef(memory_id=item["memory_id"], content=item["content"])
                    for item in evidence_rows
                    if item["role"] == "support"
                ]
                counter = [
                    HypothesisEvidenceRef(memory_id=item["memory_id"], content=item["content"])
                    for item in evidence_rows
                    if item["role"] == "counter"
                ]
                hypotheses.append(
                    HypothesisRow(
                        id=row["id"],
                        owner_kind=row["owner_kind"],
                        owner_id=row["owner_id"],
                        content=row["content"],
                        kind=row["kind"],
                        confidence=row["confidence"],
                        status=row["status"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        support_evidence=support,
                        counter_evidence=counter,
                    )
                )
            return hypotheses

    def admin_list_coverage_roots(self, space_id: str) -> list[CoverageRootRow]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT cr.root, cr.revision, cr.source_watermark, cr.source_cursor_id,
                       (SELECT COUNT(*) FROM coverage_entries ce
                        WHERE ce.space_id = cr.space_id AND ce.root = cr.root
                          AND ce.status = 'active') AS entry_count
                FROM coverage_roots cr
                WHERE cr.space_id = ?
                ORDER BY cr.root
                """,
                (space_id,),
            ).fetchall()
            return [
                CoverageRootRow(
                    root=row["root"],
                    revision=row["revision"],
                    entry_count=row["entry_count"],
                    source_watermark=row["source_watermark"],
                    source_cursor_id=row["source_cursor_id"],
                )
                for row in rows
            ]

    def admin_get_coverage_root(self, space_id: str, root: str) -> CoverageRootRow | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT cr.root, cr.revision, cr.source_watermark, cr.source_cursor_id,
                       (SELECT COUNT(*) FROM coverage_entries ce
                        WHERE ce.space_id = cr.space_id AND ce.root = cr.root
                          AND ce.status = 'active') AS entry_count
                FROM coverage_roots cr
                WHERE cr.space_id = ? AND cr.root = ?
                """,
                (space_id, root),
            ).fetchone()
            if not row:
                return None
            return CoverageRootRow(
                root=row["root"],
                revision=row["revision"],
                entry_count=row["entry_count"],
                source_watermark=row["source_watermark"],
                source_cursor_id=row["source_cursor_id"],
            )

    def admin_count_coverage_entries(self, space_id: str, root: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS n FROM coverage_entries
                WHERE space_id = ? AND root = ? AND status = 'active'
                """,
                (space_id, root),
            ).fetchone()
            return row["n"]

    def admin_list_coverage_entries(
        self, space_id: str, root: str, offset: int, limit: int
    ) -> list[CoverageEntryRow]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, root, path, content, status, created_at, updated_at
                FROM coverage_entries
                WHERE space_id = ? AND root = ? AND status = 'active'
                ORDER BY (path = '') DESC, path, id
                LIMIT ? OFFSET ?
                """,
                (space_id, root, limit, offset),
            ).fetchall()
            return [
                CoverageEntryRow(
                    id=row["id"],
                    root=row["root"],
                    path=row["path"],
                    content=row["content"],
                    status=row["status"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]

    def admin_search(self, space_id: str, keyword: str) -> SearchResults:
        """One keyword, scanned across seven kinds within one space.

        Each kind is a separate plain `LIKE ... ESCAPE '\\' COLLATE
        NOCASE` query, bound-parameter only, capped at `_SEARCH_LIMIT`
        rows (one extra row is fetched per kind to detect truncation
        without a second COUNT query).
        """

        escaped = _escape_like(keyword)
        pattern = f"%{escaped}%"
        fetch_limit = _SEARCH_LIMIT + 1
        with self._connect() as connection:
            memory_rows = connection.execute(
                """
                SELECT id, content FROM memories
                WHERE space_id = ? AND content LIKE ? ESCAPE '\\' COLLATE NOCASE
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (space_id, pattern, fetch_limit),
            ).fetchall()
            memories = [MemorySearchHit(id=row["id"], content=row["content"])
                        for row in memory_rows]

            message_rows = connection.execute(
                """
                SELECT id, content, occurred_at FROM messages
                WHERE space_id = ? AND content LIKE ? ESCAPE '\\' COLLATE NOCASE
                ORDER BY occurred_at DESC, id DESC
                LIMIT ?
                """,
                (space_id, pattern, fetch_limit),
            ).fetchall()
            messages = [
                MessageSearchHit(id=row["id"], content=row["content"],
                                 occurred_at=row["occurred_at"])
                for row in message_rows
            ]

            person_rows = connection.execute(
                """
                SELECT DISTINCT p.id, p.display_name
                FROM people p
                LEFT JOIN person_aliases a ON a.person_id = p.id
                WHERE p.space_id = ? AND p.status = 'active'
                  AND (p.display_name LIKE ? ESCAPE '\\' COLLATE NOCASE
                       OR a.value LIKE ? ESCAPE '\\' COLLATE NOCASE)
                ORDER BY p.display_name, p.id
                LIMIT ?
                """,
                (space_id, pattern, pattern, fetch_limit),
            ).fetchall()
            people = [PersonSearchHit(id=row["id"], display_name=row["display_name"])
                      for row in person_rows]

            goal_rows = connection.execute(
                """
                SELECT id, prompt FROM learning_goals
                WHERE space_id = ? AND prompt LIKE ? ESCAPE '\\' COLLATE NOCASE
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (space_id, pattern, fetch_limit),
            ).fetchall()
            learning_goals = [
                LearningGoalSearchHit(id=row["id"], prompt=row["prompt"])
                for row in goal_rows
            ]

            hypothesis_rows = connection.execute(
                """
                SELECT id, content FROM hypotheses
                WHERE space_id = ? AND content LIKE ? ESCAPE '\\' COLLATE NOCASE
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (space_id, pattern, fetch_limit),
            ).fetchall()
            hypotheses = [
                HypothesisSearchHit(id=row["id"], content=row["content"])
                for row in hypothesis_rows
            ]

            coverage_rows = connection.execute(
                """
                SELECT id, root, path, content FROM coverage_entries
                WHERE space_id = ? AND status = 'active'
                  AND content LIKE ? ESCAPE '\\' COLLATE NOCASE
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                (space_id, pattern, fetch_limit),
            ).fetchall()
            coverage_entries = [
                CoverageEntrySearchHit(
                    id=row["id"], root=row["root"], path=row["path"], content=row["content"]
                )
                for row in coverage_rows
            ]

            continuity_rows = connection.execute(
                """
                SELECT text FROM continuities
                WHERE space_id = ? AND text LIKE ? ESCAPE '\\' COLLATE NOCASE
                LIMIT ?
                """,
                (space_id, pattern, fetch_limit),
            ).fetchall()
            continuities = [ContinuitySearchHit(text=row["text"]) for row in continuity_rows]

        def _cap(items: list) -> SearchGroup:
            truncated = len(items) > _SEARCH_LIMIT
            return SearchGroup(hits=items[:_SEARCH_LIMIT], truncated=truncated)

        return SearchResults(
            memories=_cap(memories),
            messages=_cap(messages),
            people=_cap(people),
            learning_goals=_cap(learning_goals),
            hypotheses=_cap(hypotheses),
            coverage_entries=_cap(coverage_entries),
            continuities=_cap(continuities),
        )


def _learning_goal_filter_clauses(
    space_id: str, root: str | None, status: str | None
) -> tuple[list[str], list[object]]:
    clauses = ["lg.space_id = ?"]
    params: list[object] = [space_id]
    if root is not None:
        clauses.append(
            "EXISTS (SELECT 1 FROM json_each(lg.entry_ids) je "
            "JOIN coverage_entries ce ON ce.id = je.value AND ce.space_id = lg.space_id "
            "WHERE ce.root = ?)"
        )
        params.append(root)
    if status is not None:
        clauses.append("lg.status = ?")
        params.append(status)
    return clauses, params


def _hypothesis_filter_clauses(
    space_id: str, owner_kind: str | None
) -> tuple[list[str], list[object]]:
    clauses = ["space_id = ?"]
    params: list[object] = [space_id]
    if owner_kind is not None:
        clauses.append("owner_kind = ?")
        params.append(owner_kind)
    return clauses, params


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
    "CoverageEntryRow",
    "CoverageEntrySearchHit",
    "CoverageRootRow",
    "ContinuitySearchHit",
    "DerivationLink",
    "HypothesisEvidenceRef",
    "HypothesisRow",
    "HypothesisSearchHit",
    "LearningGoalRow",
    "LearningGoalSearchHit",
    "LinkedPerson",
    "LinkedRelationship",
    "MemoryDetail",
    "MemoryRow",
    "MemorySearchHit",
    "MessageRow",
    "MessageSearchHit",
    "PersonListRow",
    "PersonSearchHit",
    "RelationshipListRow",
    "SearchGroup",
    "SearchResults",
    "SpaceOverview",
    "SpaceSummary",
    "_AdminReadMixin",
]
