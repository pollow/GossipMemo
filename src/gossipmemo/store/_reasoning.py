"""Projection rebuilds and the inferred-memory/hypothesis lifecycle."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from typing import Any

from ..models import (
    HypothesisActions,
    HypothesisEvidence,
    HypothesisView,
    InferredMemoryActions,
    MemoryView,
    PersonReasoningResult,
    RelationshipReasoningResult,
    UserModelReasoningResult,
    UserModelView,
)
from ._messages import _MessagesMixin
from .policy import (
    dump_json,
    is_profile_stale,
    load_json,
    new_id,
    now_iso,
    similar_memory_content,
)


class _ReasoningMixin(_MessagesMixin):
    """Applies reasoning results to Person, Relationship, and UserModel cards."""

    def owner_review_context(
        self, space_id: str, owner_kind: str, owner_id: str | None,
    ) -> tuple[list[MemoryView], list[HypothesisView]]:
        """Comparison-only state captured with an owner reasoning snapshot."""
        with self._connect() as connection:
            if owner_kind == "person":
                clause, params = (
                    "EXISTS (SELECT 1 FROM memory_people mp "
                    "WHERE mp.memory_id = m.id AND mp.person_id = ?)", [owner_id])
            elif owner_kind == "relationship":
                clause, params = (
                    "EXISTS (SELECT 1 FROM memory_relationships mr "
                    "WHERE mr.memory_id = m.id AND mr.relationship_id = ?)", [owner_id])
            else:
                clause, params = "m.about_user = 1", []
            inferred = connection.execute(
                f"SELECT m.* FROM memories m WHERE m.space_id = ? AND m.status = 'active' "
                f"AND m.basis = 'inferred' AND {clause} ORDER BY m.created_at DESC LIMIT 100",
                [space_id, *params]).fetchall()
            hypotheses = connection.execute(
                "SELECT * FROM hypotheses WHERE space_id = ? AND owner_kind = ? AND owner_id IS ? "
                "AND status = 'open' ORDER BY created_at DESC LIMIT 100",
                (space_id, owner_kind, owner_id)).fetchall()
            return (
                [self._memory_view(connection, row, True) for row in inferred],
                [HypothesisView(
                    id=row['id'], space_id=row['space_id'], owner_kind=row['owner_kind'],
                    owner_id=row['owner_id'], content=row['content'], kind=row['kind'],
                    confidence=row['confidence'], status=row['status'],
                    promoted_memory_id=row['promoted_memory_id'],
                    evidence=[
                        HypothesisEvidence(memory_id=e['memory_id'], role=e['role'])
                        for e in connection.execute(
                            'SELECT memory_id, role FROM hypothesis_evidence '
                            'WHERE hypothesis_id = ?', (row['id'],)).fetchall()],
                    created_at=row['created_at'], updated_at=row['updated_at'])
                 for row in hypotheses])

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
        if person_id:
            target_join = "JOIN memory_people mt ON mt.memory_id = m.id AND mt.person_id = ?"
            target_params: list[Any] = [person_id]
        elif relationship_id:
            target_join = ("JOIN memory_relationships mt "
                           "ON mt.memory_id = m.id AND mt.relationship_id = ?")
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
                             and similar_memory_content(row["content"], item.content)), None)
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
                valid_sources = [
                    source_id for source_id in valid_sources if source_id in source_ids]
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
            memory_id = new_id("memory")
            now = now_iso()
            connection.execute(
                """
                INSERT INTO memories(
                    id, space_id, content, kind, basis, about_user, created_by,
                    created_at, updated_at
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

    def _apply_inferred_memory_actions(
        self, connection: sqlite3.Connection, space_id: str, owner_kind: str,
        owner_id: str | None, source_memory_ids: set[str],
        context_inferred_memory_ids: set[str], actions: InferredMemoryActions,
    ) -> None:
        """Apply explicitly scoped inferred-Memory lifecycle actions.

        ``source_memory_ids`` is the non-inferred evidence shown to the
        reasoner. Retractions additionally require the target-owned inference
        to have been supplied by the trusted caller in context.
        """
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
                owner_clause = ("EXISTS (SELECT 1 FROM memory_people mp "
                                "WHERE mp.memory_id = m.id AND mp.person_id = ?)")
                owner_params: list[Any] = [owner_id]
            elif owner_kind == "relationship":
                owner_clause = ("EXISTS (SELECT 1 FROM memory_relationships mr "
                                "WHERE mr.memory_id = m.id AND mr.relationship_id = ?)")
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
            now = now_iso()
            connection.execute(
                """UPDATE memories SET status = 'retracted', invalidated_at = ?,
                       invalidation_reason = ?, updated_at = ? WHERE id = ?""",
                (now, retraction.reason, now, retraction.memory_id),
            )

    def _apply_hypothesis_actions(
        self, connection: sqlite3.Connection, space_id: str, owner_kind: str,
        owner_id: str | None, source_memory_ids: set[str],
        context_hypothesis_ids: set[str], actions: HypothesisActions,
    ) -> None:
        """Persist standalone hypotheses and explicit, context-scoped transitions.

        Runs inside an existing atomic reasoning write.
        """
        for item in actions.upserts:
            if item.hypothesis_id and item.hypothesis_id not in context_hypothesis_ids:
                continue
            evidence = [
                row for row in connection.execute(
                    f"""SELECT m.id FROM memories m WHERE m.space_id = ? AND m.status = 'active'
                        AND m.basis <> 'inferred'
                        AND m.id IN ({','.join('?' for _ in item.evidence)})""",
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
                    """SELECT id FROM hypotheses
                       WHERE space_id = ? AND owner_kind = ? AND owner_id IS ?
                         AND status = 'open' AND kind = ? AND content = ?""",
                    (space_id, owner_kind, owner_id, item.kind, item.content),
                ).fetchone()
            now = now_iso()
            if existing:
                hypothesis_id = existing["id"]
                connection.execute(
                    "UPDATE hypotheses SET content = ?, kind = ?, confidence = ?, updated_at = ? "
                    "WHERE id = ?",
                    (item.content, item.kind, item.confidence, now, hypothesis_id),
                )
            else:
                hypothesis_id = hypothesis_id or new_id("hypothesis")
                connection.execute(
                    """INSERT INTO hypotheses(
                        id, space_id, owner_kind, owner_id, content, kind, confidence,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (hypothesis_id, space_id, owner_kind, owner_id, item.content, item.kind,
                     item.confidence, now, now),
                )
            connection.executemany(
                "INSERT OR IGNORE INTO hypothesis_evidence(hypothesis_id, memory_id, role) "
                "VALUES (?, ?, ?)",
                [(hypothesis_id, evidence.memory_id, evidence.role)
                 for evidence in item.evidence if evidence.memory_id in evidence_ids],
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
                    owner_clause = ("EXISTS (SELECT 1 FROM memory_people mp "
                                    "WHERE mp.memory_id = m.id AND mp.person_id = ?)")
                    owner_params: list[Any] = [owner_id]
                elif owner_kind == "relationship":
                    owner_clause = ("EXISTS (SELECT 1 FROM memory_relationships mr "
                                    "WHERE mr.memory_id = m.id AND mr.relationship_id = ?)")
                    owner_params = [owner_id]
                else:
                    owner_clause = "m.about_user = 1"
                    owner_params = []
                memory = connection.execute(
                    "SELECT m.id FROM memories m WHERE m.id = ? AND m.space_id = ? "
                    f"AND m.status = 'active' AND {owner_clause}",
                    [transition.promoted_memory_id, space_id, *owner_params],
                ).fetchone()
                if not memory:
                    continue
            now = now_iso()
            connection.execute(
                """UPDATE hypotheses SET status = ?, status_reason = ?, promoted_memory_id = ?,
                   updated_at = ? WHERE id = ?""",
                (transition.status, transition.reason,
                 transition.promoted_memory_id, now, row["id"]),
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
                    dump_json(result.profile_card),
                    expected_watermark, now_iso(), now_iso(),
                    space_id,
                    person_id,
                    expected_watermark, space_id, person_id, expected_watermark,
                ),
            )
            if claimed.rowcount != 1:
                return False
            actions = result.inferred_memory_actions or InferredMemoryActions(
                upserts=result.inferred_memories)
            self._apply_inferred_memory_actions(
                connection, space_id, "person", person_id,
                self._person_reasoning_source_ids(
                    connection, space_id, person_id), context_inferred_memory_ids or set(), actions,
            )
            if result.hypothesis_actions:
                self._apply_hypothesis_actions(
                    connection, space_id, "person", person_id,
                    self._person_reasoning_source_ids(connection, space_id, person_id),
                    context_hypothesis_ids or set(), result.hypothesis_actions)
            final_watermark = self._person_watermark(connection, space_id, person_id)
            connection.execute(
                """UPDATE people SET profile_card = ?, profile_source_updated_at = ?,
                    profile_updated_at = ?, updated_at = ? WHERE id = ?""",
                (
                    dump_json(result.profile_card),
                    final_watermark, now_iso(), now_iso(),
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
            current = self._relationship_watermark(connection, space_id, relationship_id)
            if current != expected_watermark:
                return False
            claimed = connection.execute(
                """UPDATE relationships SET facets = ?, closeness = ?, tone = ?,
                    status = ?, summary = ?, profile_source_updated_at = ?,
                    profile_updated_at = ?, updated_at = ?
                WHERE space_id = ? AND id = ?""",
                (
                    dump_json(result.facets),
                    result.closeness,
                    result.tone,
                    result.status,
                    result.summary,
                    expected_watermark, now_iso(), now_iso(),
                    space_id,
                    relationship_id,
                ),
            )
            if claimed.rowcount != 1:
                return False
            actions = result.inferred_memory_actions or InferredMemoryActions(
                upserts=result.inferred_memories)
            self._apply_inferred_memory_actions(
                connection, space_id, "relationship", relationship_id,
                self._relationship_reasoning_source_ids(
                    connection, space_id, relationship_id),
                context_inferred_memory_ids or set(), actions,
            )
            if result.hypothesis_actions:
                self._apply_hypothesis_actions(
                    connection, space_id, "relationship", relationship_id,
                    self._relationship_reasoning_source_ids(
                        connection, space_id, relationship_id),
                    context_hypothesis_ids or set(), result.hypothesis_actions)
            final_watermark = self._relationship_watermark(connection, space_id, relationship_id)
            now = now_iso()
            connection.execute(
                """UPDATE relationships SET facets = ?, closeness = ?, tone = ?,
                    status = ?, summary = ?, profile_source_updated_at = ?,
                    profile_updated_at = ?, updated_at = ? WHERE id = ?""",
                (
                    dump_json(result.facets),
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
        self, space_id: str, *, delta_only: bool = False
    ) -> tuple[UserModelView, list[MemoryView], str | None] | None:
        """Read the space's UserModel, its `about_user` memories and watermark.

        `delta_only` has the same meaning as in `person_context`: the fold
        read, bounded below by the card's own watermark and including the
        rows that have since been retracted or superseded.
        """
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM user_models WHERE space_id = ?", (space_id,)
            ).fetchone()
            if not row:
                return None
            if delta_only:
                since = row["profile_source_updated_at"]
                memories = connection.execute(
                    """SELECT * FROM memories WHERE space_id = ? AND about_user = 1
                       AND (? IS NULL OR updated_at > ?)
                       ORDER BY updated_at ASC, id ASC""",
                    (space_id, since, since),
                ).fetchall()
            else:
                memories = connection.execute(
                    """SELECT * FROM memories WHERE space_id = ? AND status = 'active'
                       AND about_user = 1 ORDER BY created_at DESC, id DESC""",
                    (space_id,),
                ).fetchall()
            watermark = self._user_model_watermark(connection, space_id)
            view = UserModelView(
                space_id=space_id,
                profile_card=load_json(row["profile_card"], {}),
                profile_source_updated_at=row["profile_source_updated_at"],
                profile_updated_at=row["profile_updated_at"],
                stale=is_profile_stale(row["profile_source_updated_at"], watermark),
            )
            return view, [self._memory_view(connection, item, True) for item in memories], watermark

    def apply_user_model_reasoning(
        self, space_id: str, expected_watermark: str | None,
        result: UserModelReasoningResult,
        context_hypothesis_ids: set[str] | None = None,
    ) -> bool:
        with self._connect() as connection:
            if self._user_model_watermark(connection, space_id) != expected_watermark:
                return False
            now = now_iso()
            updated = connection.execute(
                """UPDATE user_models SET profile_card = ?, profile_source_updated_at = ?,
                   profile_updated_at = ? WHERE space_id = ?
                   AND (profile_source_updated_at IS NULL OR profile_source_updated_at < ?)""",
                (dump_json(result.profile_card), expected_watermark, now, space_id,
                 expected_watermark),
            )
            if updated.rowcount != 1:
                return False
            if result.hypothesis_actions:
                source_ids = {row["id"] for row in connection.execute(
                    "SELECT id FROM memories WHERE space_id = ? AND status = 'active' "
                    "AND about_user = 1 AND basis <> 'inferred'", (space_id,)).fetchall()}
                self._apply_hypothesis_actions(
                    connection, space_id, "user", None, source_ids,
                    context_hypothesis_ids or set(), result.hypothesis_actions)
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
                (dump_json(profile_card), watermark, now_iso(), space_id),
            )

    def stale_entities(self) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[str]]:
        with self._connect() as connection:
            people = [
                (row["space_id"], row["id"])
                for row in connection.execute(
                    """SELECT p.space_id, p.id FROM people p
                    WHERE p.status = 'active' AND (
                      p.profile_source_updated_at IS NULL OR p.profile_source_updated_at <
                       (SELECT MAX(m.updated_at) FROM memories m
                        JOIN memory_people mp ON mp.memory_id = m.id
                       WHERE m.space_id = p.space_id AND mp.person_id = p.id
                         AND m.basis <> 'inferred')
                    ) AND EXISTS (SELECT 1 FROM memories m
                       JOIN memory_people mp ON mp.memory_id = m.id
                       WHERE m.space_id = p.space_id AND mp.person_id = p.id
                         AND m.basis <> 'inferred')"""
                ).fetchall()
            ]
            relationships = []
            for row in connection.execute("SELECT * FROM relationships").fetchall():
                watermark = self._relationship_watermark(connection, row["space_id"], row["id"])
                if is_profile_stale(row["profile_source_updated_at"], watermark):
                    relationships.append((row["space_id"], row["id"]))
            user_models = []
            for row in connection.execute("SELECT * FROM user_models").fetchall():
                watermark = self._user_model_watermark(connection, row["space_id"])
                if is_profile_stale(row["profile_source_updated_at"], watermark):
                    user_models.append(row["space_id"])
            return people, relationships, user_models
