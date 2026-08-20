"""The rolling per-space continuity projection."""

from __future__ import annotations

from typing import Any

from ..models import (
    ContinuityReasoningResult,
    ContinuityView,
    ModelMessage,
)
from ._coverage import _CoverageMixin
from .policy import (
    dump_json,
    load_json,
    now_iso,
)


class _ContinuityMixin(_CoverageMixin):
    """Reads, schedules, and applies the rolling continuity projection."""

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
                related_person_ids=load_json(row["related_person_ids"], []),
                through_message_id=row["through_message_id"],
            )
            if row["through_message_rowid"] is None:
                after = 0
            else:
                after = row["through_message_rowid"]
            rows = connection.execute(
                "SELECT * FROM messages WHERE space_id = ? AND rowid > ? ORDER BY rowid",
                (space_id, after),
            ).fetchall()
            messages = [
                ModelMessage(
                    id=item["id"], space_id=item["space_id"], author=item["author"],
                    content=item["content"], occurred_at=item["occurred_at"],
                    source_provider=item["source_provider"],
                    source_conversation_key=item["source_conversation_key"],
                    source_item_id=item["source_item_id"],
                    source_metadata=load_json(item["source_metadata"], {}),
                )
                for item in rows
            ]
            return continuity, messages

    def pending_continuities(self, threshold: int = 20, space_id: str | None = None) -> list[str]:
        with self._connect() as connection:
            query = (
                "SELECT c.space_id FROM continuities c LEFT JOIN messages m "
                "ON m.space_id = c.space_id AND m.rowid > COALESCE(c.through_message_rowid, 0) "
            )
            params: tuple[Any, ...] = ()
            if space_id is not None:
                query += "WHERE c.space_id = ? "
                params += (space_id,)
            query += "GROUP BY c.space_id HAVING COUNT(m.rowid) >= ?"
            params += (threshold,)
            return [row["space_id"] for row in connection.execute(query, params).fetchall()]

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
                (result.text, dump_json(people), result.through_message_id,
                 message["rowid"], now_iso(), space_id),
            )
            return True
