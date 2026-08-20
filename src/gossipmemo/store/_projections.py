"""Row-to-view mappers and the freshness watermarks they share."""

from __future__ import annotations

import sqlite3
from typing import Any

from ..models import (
    CoverageEntryView,
    CoverageRootView,
    HypothesisEvidence,
    HypothesisView,
    LearningGoalView,
    MemoryView,
    PersonView,
    RelationshipView,
)
from ._vectors import _VectorsMixin
from .policy import (
    is_profile_stale,
    load_json,
)


class _ProjectionsMixin(_VectorsMixin):
    """Maps SQLite rows onto view models and reads projection watermarks."""

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

    def _person_watermark(
        self, connection: sqlite3.Connection, space_id: str, person_id: str,
    ) -> str | None:
        row = connection.execute(
            """SELECT MAX(m.updated_at) AS watermark FROM memories m
               JOIN memory_people mp ON mp.memory_id = m.id
               WHERE m.space_id = ? AND mp.person_id = ? AND m.basis <> 'inferred'""",
            (space_id, person_id),
        ).fetchone()
        return row["watermark"] if row else None

    def _relationship_watermark(
        self, connection: sqlite3.Connection, space_id: str, relationship_id: str,
    ) -> str | None:
        row = connection.execute(
            """SELECT MAX(m.updated_at) AS watermark FROM memories m
               JOIN relationships r ON r.id = ? AND r.space_id = m.space_id
               WHERE m.space_id = ? AND m.basis <> 'inferred' AND (
                 m.id IN (SELECT memory_id FROM memory_relationships WHERE relationship_id = r.id)
                 OR m.id IN (SELECT a.memory_id FROM memory_people a
                             JOIN memory_people b ON b.memory_id = a.memory_id
                             WHERE a.person_id = r.person_a_id AND b.person_id = r.person_b_id))""",
            (relationship_id, space_id),
        ).fetchone()
        return row["watermark"] if row else None

    def _person_view(self, connection: sqlite3.Connection, row: sqlite3.Row) -> PersonView:
        watermark = self._person_watermark(connection, row["space_id"], row["id"])
        return PersonView(
            id=row["id"],
            display_name=row["display_name"],
            profile_card=load_json(row["profile_card"], {}),
            profile_source_updated_at=row["profile_source_updated_at"],
            profile_updated_at=row["profile_updated_at"],
            stale=is_profile_stale(row["profile_source_updated_at"], watermark),
        )

    def _relationship_view(
        self, connection: sqlite3.Connection, row: sqlite3.Row,
    ) -> RelationshipView:
        watermark = self._relationship_watermark(connection, row["space_id"], row["id"])
        return RelationshipView(
            id=row["id"],
            person_a_id=row["person_a_id"],
            person_b_id=row["person_b_id"],
            facets=load_json(row["facets"], []),
            closeness=row["closeness"],
            tone=row["tone"],
            status=row["status"],
            summary=row["summary"],
            profile_source_updated_at=row["profile_source_updated_at"],
            profile_updated_at=row["profile_updated_at"],
            stale=is_profile_stale(row["profile_source_updated_at"], watermark),
        )

    def _coverage_root_view(self, row: sqlite3.Row) -> CoverageRootView:
        return CoverageRootView(
            space_id=row["space_id"], root=row["root"], revision=row["revision"],
            source_watermark=row["source_watermark"], source_cursor_id=row["source_cursor_id"],
        )

    def _coverage_entry_view(self, row: sqlite3.Row) -> CoverageEntryView:
        return CoverageEntryView(
            id=row["id"], space_id=row["space_id"], root=row["root"], path=row["path"],
            content=row["content"], status=row["status"],
            created_at=row["created_at"], updated_at=row["updated_at"],
        )

    def _hypothesis_view(self, connection: sqlite3.Connection, row: sqlite3.Row) -> HypothesisView:
        evidence = connection.execute(
            "SELECT memory_id, role FROM hypothesis_evidence WHERE hypothesis_id = ?", (row["id"],)
        ).fetchall()
        return HypothesisView(
            id=row["id"], space_id=row["space_id"], owner_kind=row["owner_kind"],
            owner_id=row["owner_id"], content=row["content"], kind=row["kind"],
            confidence=row["confidence"], status=row["status"],
            promoted_memory_id=row["promoted_memory_id"],
            evidence=[HypothesisEvidence(memory_id=item["memory_id"], role=item["role"])
                      for item in evidence],
            created_at=row["created_at"], updated_at=row["updated_at"])

    def _learning_goal_view(self, row: sqlite3.Row) -> LearningGoalView:
        return LearningGoalView(
            id=row["id"], space_id=row["space_id"], prompt=row["prompt"],
            rationale=row["rationale"], entry_ids=load_json(row["entry_ids"], []),
            focus_kind=row["focus_kind"], focus_id=row["focus_id"], status=row["status"],
            status_reason=row["status_reason"], created_at=row["created_at"],
            updated_at=row["updated_at"])

    def _user_model_watermark(self, connection: sqlite3.Connection, space_id: str) -> str | None:
        row = connection.execute(
            "SELECT MAX(updated_at) AS watermark FROM memories "
            "WHERE space_id = ? AND status = 'active' AND about_user = 1",
            (space_id,),
        ).fetchone()
        return row["watermark"] if row else None
