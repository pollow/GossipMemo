"""Coverage roots and entries, and the learning goals planned from them."""

from __future__ import annotations

import sqlite3
from typing import Any

from ..models import (
    COVERAGE_ROOTS,
    CoverageEntryView,
    CoverageRootView,
    ExtractedCoverageAudit,
    GoalPlanningResult,
    LearningGoalView,
    MemoryView,
)
from ._reasoning import _ReasoningMixin
from .policy import (
    dump_json,
    new_id,
    now_iso,
)


class _CoverageMixin(_ReasoningMixin):
    """Coverage audits and learning-goal planning over the coverage roots."""

    def _coverage_entries(
        self, connection: sqlite3.Connection, space_id: str, root: str | None = None,
    ) -> list[CoverageEntryView]:
        query = "SELECT * FROM coverage_entries WHERE space_id = ? AND status = 'active'"
        params: tuple[Any, ...] = (space_id,)
        if root is not None:
            query += " AND root = ?"
            params += (root,)
        return [
            self._coverage_entry_view(row)
            for row in connection.execute(query + " ORDER BY root, path, id", params).fetchall()
        ]

    def coverage_context(
        self, space_id: str, limit: int | None = 400,
    ) -> tuple[CoverageRootView, list[CoverageEntryView], list[MemoryView]] | None:
        """Return one root that is behind, its active entries, and its backlog.

        Roots are audited one at a time, in their declared order, because
        each root owns its own cursor: the caller audits as much of the
        returned backlog as one request holds and commits that root alone.
        `limit` bounds how much backlog a single attempt reads, never how
        far the cursor may advance; the caller loops until nothing is
        behind. Returns None when every root in the space is caught up.
        """
        with self._connect() as connection:
            rows = {row["root"]: row for row in connection.execute(
                "SELECT * FROM coverage_roots WHERE space_id = ?", (space_id,)).fetchall()}
            if not rows:
                return None
            for root in COVERAGE_ROOTS:
                row = rows.get(root)
                if row is None:
                    continue
                watermark, cursor_id = row["source_watermark"] or "", row["source_cursor_id"] or ""
                query = """SELECT * FROM memories WHERE space_id = ? AND basis <> 'inferred'
                           AND (updated_at > ? OR (updated_at = ? AND id > ?))
                           ORDER BY updated_at, id"""
                params: tuple[Any, ...] = (space_id, watermark, watermark, cursor_id)
                if limit is not None:
                    query += " LIMIT ?"
                    params += (limit,)
                backlog = connection.execute(query, params).fetchall()
                if not backlog:
                    continue
                return (
                    self._coverage_root_view(row),
                    self._coverage_entries(connection, space_id, root),
                    [self._memory_view(connection, item, True) for item in backlog],
                )
            return None

    def apply_coverage_audit(
        self, space_id: str, root: str, expected_watermark: str | None,
        expected_cursor_id: str | None, audit: ExtractedCoverageAudit,
        chunk: list[MemoryView], context_entry_ids: set[str],
    ) -> bool:
        """Commit one root's audit and advance that root's cursor alone.

        `chunk` is the evidence the audit actually read; the cursor advances
        to whichever of those Memories has the latest `(updated_at, id)`,
        matching the order `coverage_context` read them in.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM coverage_roots WHERE space_id = ? AND root = ?",
                (space_id, root)).fetchone()
            if (not row or row["source_watermark"] != expected_watermark
                    or row["source_cursor_id"] != expected_cursor_id):
                return False
            chunk_ids = [item.id for item in chunk]
            now = now_iso()
            entries = {item.id: item for item in self._coverage_entries(
                connection, space_id, root)}
            for edit in audit.modifications:
                entry = entries.get(edit.entry_id)
                if edit.entry_id not in context_entry_ids or entry is None:
                    continue
                connection.execute(
                    "UPDATE coverage_entries SET path = ?, content = ?, status = ?, updated_at = ? "
                    "WHERE id = ? AND space_id = ?",
                    (entry.path if edit.path is None else edit.path, edit.content, edit.status,
                     now, edit.entry_id, space_id),
                )
                entries.pop(edit.entry_id)
            existing = {(item.path, item.content) for item in entries.values()}
            for addition in audit.additions:
                # Redundant entries are acceptable and mergeable later; an
                # exact repeat of a stored entry is only noise, so skip it.
                if (addition.path, addition.content) in existing:
                    continue
                existing.add((addition.path, addition.content))
                connection.execute(
                    "INSERT INTO coverage_entries"
                    "(id, space_id, root, path, content, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (new_id("entry"), space_id, root, addition.path, addition.content, now, now),
                )
            placeholders = ",".join("?" for _ in chunk_ids)
            cursor_row = connection.execute(
                f"SELECT updated_at, id FROM memories WHERE id IN ({placeholders}) "
                "ORDER BY updated_at DESC, id DESC LIMIT 1",
                chunk_ids).fetchone() if chunk_ids else None
            next_watermark = cursor_row["updated_at"] if cursor_row else expected_watermark
            next_cursor_id = cursor_row["id"] if cursor_row else expected_cursor_id
            connection.execute(
                "UPDATE coverage_roots SET revision = revision + 1, source_watermark = ?, "
                "source_cursor_id = ?, updated_at = ? WHERE space_id = ? AND root = ?",
                (next_watermark, next_cursor_id, now, space_id, root),
            )
            return True

    def _coverage_revision(self, connection: sqlite3.Connection, space_id: str) -> int | None:
        """Sum every root revision as one space-level planning CAS token.

        Each root revision only ever increases, so the sum changes whenever
        any root's coverage did -- which is exactly when a plan built on the
        old entries is stale.
        """
        row = connection.execute(
            "SELECT COUNT(*) AS roots, COALESCE(SUM(revision), 0) AS revision "
            "FROM coverage_roots WHERE space_id = ?",
            (space_id,),
        ).fetchone()
        return int(row["revision"]) if row and row["roots"] else None

    def learning_goal_context(self, space_id: str) -> tuple[
        int, list[CoverageEntryView], list[LearningGoalView], list[LearningGoalView],
    ] | None:
        """Coverage entries and goal lifecycles -- the planner's whole world.

        Memories are the auditor's input, entries are the planner's: a
        planning pass reads only what coverage already summarized, plus the
        goals it may transition.
        """
        with self._connect() as connection:
            revision = self._coverage_revision(connection, space_id)
            if revision is None:
                return None
            goals = connection.execute(
                "SELECT * FROM learning_goals WHERE space_id = ? ORDER BY updated_at DESC",
                (space_id,)).fetchall()
            # `partial` is still served to the agent by list_guidance
            # (status IN ('open', 'partial')), so it belongs with the open
            # bucket here too -- otherwise the planner treats a goal the
            # agent is still actively pursuing as settled history.
            open_statuses = {"open", "partial"}
            return (
                revision,
                self._coverage_entries(connection, space_id),
                [self._learning_goal_view(item) for item in goals
                 if item["status"] in open_statuses],
                [self._learning_goal_view(item) for item in goals
                 if item["status"] not in open_statuses][:20])

    def _goal_focus(
        self, connection: sqlite3.Connection, space_id: str, text: str
    ) -> tuple[str, str | None]:
        """Derive a goal's focus from its own words, never from the model.

        The planner is never given person IDs, so the anchor is the person
        the goal already names: deterministic alias matching resolves
        exactly-one-person text to that person, and anything else -- nobody
        named, or several people -- stays a user-focused goal.
        """
        people = self._match_people(connection, space_id, text)
        return ("person", people[0].id) if len(people) == 1 else ("user", None)

    def apply_goal_planning(
        self, space_id: str, expected_revision: int, result: GoalPlanningResult,
        context_goal_ids: set[str],
    ) -> bool:
        """Commit one planning pass; a goal that files nowhere is still kept.

        `entry_ids` is best effort: an ID that does not name an active entry
        in this space is dropped, and the goal is stored regardless. Nothing
        a planner proposes is discarded for failing to cite something.
        """
        with self._connect() as connection:
            if self._coverage_revision(connection, space_id) != expected_revision:
                return False
            now = now_iso()
            known_entry_ids = {item.id for item in self._coverage_entries(connection, space_id)}
            for item in result.upserts:
                entry_ids = dump_json(
                    [ref for ref in dict.fromkeys(item.entry_ids) if ref in known_entry_ids])
                focus_kind, focus_id = self._goal_focus(
                    connection, space_id, item.prompt + "\n" + item.rationale)
                if item.goal_id and item.goal_id in context_goal_ids:
                    connection.execute(
                        "UPDATE learning_goals SET prompt = ?, rationale = ?, entry_ids = ?, "
                        "focus_kind = ?, focus_id = ?, updated_at = ? "
                        "WHERE id = ? AND space_id = ?",
                        (item.prompt, item.rationale, entry_ids, focus_kind, focus_id, now,
                         item.goal_id, space_id))
                elif not item.goal_id:
                    duplicate = connection.execute(
                        "SELECT 1 FROM learning_goals "
                        "WHERE space_id = ? AND prompt = ? AND status = 'open'",
                        (space_id, item.prompt)).fetchone()
                    if not duplicate:
                        connection.execute(
                            "INSERT INTO learning_goals(id, space_id, prompt, rationale, "
                            "entry_ids, focus_kind, focus_id, created_at, updated_at) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            (new_id("goal"), space_id, item.prompt, item.rationale, entry_ids,
                             focus_kind, focus_id, now, now))
            for transition in result.transitions:
                if transition.goal_id in context_goal_ids:
                    connection.execute(
                        "UPDATE learning_goals SET status = ?, status_reason = ?, updated_at = ? "
                        "WHERE id = ? AND space_id = ?",
                        (transition.status, transition.reason, now, transition.goal_id, space_id))
            return True

    def stale_coverage_spaces(self) -> list[str]:
        with self._connect() as connection:
            return [
                row["space_id"] for row in connection.execute(
                    """SELECT DISTINCT cr.space_id FROM coverage_roots cr WHERE EXISTS (
                       SELECT 1 FROM memories m WHERE m.space_id = cr.space_id
                       AND m.basis <> 'inferred' AND (
                         m.updated_at > COALESCE(cr.source_watermark, '') OR
                         (m.updated_at = COALESCE(cr.source_watermark, '')
                          AND m.id > COALESCE(cr.source_cursor_id, ''))
                       )
                    )"""
                ).fetchall()
            ]
