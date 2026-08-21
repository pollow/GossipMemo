"""The LLM-free read path serving the context, guidance, and turn APIs."""

from __future__ import annotations

import hashlib
import logging
import sqlite3
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pydantic import ValidationError

from ..models import (
    ContextBundle,
    ContinuityView,
    GuidanceBundle,
    GuidanceItem,
    MemoryView,
    PersonView,
    UserModelView,
)
from ._continuity import _ContinuityMixin
from .policy import (
    dump_json,
    load_json,
    rank_guidance,
    sample_learning_goals,
)

logger = logging.getLogger(__name__)

# What the turn read path can genuinely fail with: SQLite trouble (a busy or
# locked database, and FTS5 `MATCH` syntax rejected as an `OperationalError`)
# and pydantic rejecting a row while a view is built. Every other exception
# is a bug and must propagate rather than degrade into `"unavailable"`.
TURN_READ_ERRORS = (sqlite3.Error, ValidationError)


@dataclass(frozen=True)
class TurnView:
    """Everything the turn read path enriches one persisted batch with.

    `context_status` reports whether that enrichment is trustworthy, so it
    is derived once in `turn_view` from which steps succeeded, rather than
    assigned from inside the individual guards.
    """

    known_people: list[PersonView] = field(default_factory=list)
    guidance: GuidanceBundle = field(default_factory=GuidanceBundle)
    memory_recall: list[MemoryView] = field(default_factory=list)
    context_update: ContextBundle | None = None
    context_status: Literal["available", "unavailable"] = "available"


class _ContextMixin(_ContinuityMixin):
    """Assembles context bundles, guidance, and turn views."""

    GUIDANCE_GOAL_MIN = 3
    GUIDANCE_GOAL_MAX = 5

    def _open_hypothesis_rows(
        self, connection: sqlite3.Connection, space_id: str, person_ids: tuple[str, ...],
    ) -> list[sqlite3.Row]:
        """Open hypotheses reachable from `person_ids`, user-scoped ones always included.

        Shared by `_guidance` (sampled) and `list_guidance` (full list) so the
        owner-filtering shape cannot drift between the two paths.
        """
        placeholders = ','.join('?' for _ in person_ids) or 'NULL'
        return connection.execute(
            "SELECT id, owner_kind, owner_id, content, confidence, status, updated_at "
            "FROM hypotheses "
            "WHERE space_id = ? AND status = 'open' AND (owner_kind = 'user' "
            f"OR (owner_kind = 'person' AND owner_id IN ({placeholders})) "
            "OR (owner_kind = 'relationship' AND owner_id IN "
            f"(SELECT id FROM relationships WHERE space_id = ? "
            f"AND (person_a_id IN ({placeholders}) OR person_b_id IN ({placeholders}))))) "
            "ORDER BY updated_at DESC, id",
            (space_id, *person_ids, space_id, *person_ids, *person_ids),
        ).fetchall()

    def _open_learning_goal_rows(
        self, connection: sqlite3.Connection, space_id: str, person_ids: tuple[str, ...],
    ) -> list[sqlite3.Row]:
        """Open/partial learning goals reachable from `person_ids`.

        Mirrors `_open_hypothesis_rows`.
        """
        placeholders = ','.join('?' for _ in person_ids) or 'NULL'
        return connection.execute(
            "SELECT id, focus_kind, focus_id, prompt, status, updated_at FROM learning_goals "
            "WHERE space_id = ? AND status IN ('open', 'partial') AND (focus_kind = 'user' "
            f"OR (focus_kind = 'person' AND focus_id IN ({placeholders})) "
            "OR (focus_kind = 'relationship' AND focus_id IN "
            f"(SELECT id FROM relationships WHERE space_id = ? "
            f"AND (person_a_id IN ({placeholders}) OR person_b_id IN ({placeholders}))))) "
            "ORDER BY updated_at DESC, id",
            (space_id, *person_ids, space_id, *person_ids, *person_ids),
        ).fetchall()

    def _guidance(
        self,
        connection: sqlite3.Connection,
        space_id: str,
        activated_person_ids: Iterable[str] = (),
        query: str = "",
        *,
        version: str,
        goals: int | None = None,
        goal_seed: str | None = None,
    ) -> GuidanceBundle:
        """Return a bounded set; coverage itself never leaves this method.

        Hypotheses and learning goals are selected differently on purpose.
        A hypothesis is a claim about a specific owner, so the activated
        people already narrow it to the current subject and the single best
        match is a useful thing to confirm. A learning goal is a direction
        rather than a claim: only the consuming agent knows what the
        conversation is currently about, and this method has nothing but a
        query string to judge with. So goals are not ranked at all — 3-5 are
        sampled and the agent decides which, if any, fit. Diversity of the
        pool is what makes the sample useful, and that is the planner's job
        (it fans out per coverage root).

        The sample is a deterministic function of a seed string and the
        candidate pool; see `policy.sample_learning_goals`. The seed
        defaults to the context bundle `version`, which is stable for as
        long as the durable context is -- but no longer: a version bump
        reshuffles the whole draw, so with a pool much larger than the
        sample the goals in the prompt churn for reasons the conversation
        cannot explain, and the agent's prompt prefix churns with them.

        Only the caller knows the conversation's actual needs, so it can
        override both halves. `goal_seed` replaces `version` in the seed
        (pin it per conversation or per day to hold the sample still across
        version bumps, or rotate it deliberately); the pool ids stay in the
        seed either way, so adding or answering a goal still changes the
        draw. `goals` fixes the sample size: `0` for hypotheses only, `n`
        for exactly `min(n, len(pool))`. Both default to `None`, which is
        exactly the behavior described above. This is also the seam a
        continuity-based selection would replace the seeded draw at.
        """
        person_ids = tuple(dict.fromkeys(activated_person_ids))
        rows = self._open_hypothesis_rows(connection, space_id, person_ids)
        hypotheses = [GuidanceItem(
            id=row['id'], kind='hypothesis', content=row['content'],
            owner_kind=row['owner_kind'], owner_id=row['owner_id'], status=row['status'],
            confidence=row['confidence']) for row in rows]
        updated = {row['id']: row['updated_at'] for row in rows}
        goal_rows = self._open_learning_goal_rows(connection, space_id, person_ids)
        learning_goals = [GuidanceItem(id=row['id'], kind='learning_goal', content=row['prompt'],
                                       owner_kind=row['focus_kind'], owner_id=row['focus_id'],
                                       status=row['status']) for row in goal_rows]
        selected = rank_guidance(hypotheses, updated, query)[:1]
        selected.extend(sample_learning_goals(
            learning_goals, version if goal_seed is None else goal_seed,
            self.GUIDANCE_GOAL_MIN, self.GUIDANCE_GOAL_MAX, goals))
        selected.sort(key=lambda item: (item.kind, item.id))
        return GuidanceBundle(items=selected)

    def _context_state(
        self, connection: sqlite3.Connection, space_id: str
    ) -> tuple[UserModelView | None, ContinuityView | None, list[PersonView], str]:
        """Compute the durable context payload and its version.

        Shared by `context_bundle` and `guidance_bundle` so the turn path
        derives guidance from the exact same version string the context
        path would compute for the same space.
        """
        user = connection.execute(
            "SELECT * FROM user_models WHERE space_id = ?", (space_id,)).fetchone()
        cont = connection.execute(
            "SELECT * FROM continuities WHERE space_id = ?", (space_id,)).fetchone()
        user_view = UserModelView(
            space_id=space_id, profile_card=load_json(user["profile_card"], {}),
            profile_source_updated_at=user["profile_source_updated_at"],
            profile_updated_at=user["profile_updated_at"]) if user else None
        continuity = ContinuityView(
            text=cont["text"], related_person_ids=load_json(cont["related_person_ids"], []),
            through_message_id=cont["through_message_id"]) if cont else None
        ids = continuity.related_person_ids if continuity else []
        people = []
        for person_id in ids:
            row = connection.execute(
                "SELECT * FROM people WHERE id = ? AND space_id = ? AND status = 'active'",
                (person_id, space_id)).fetchone()
            if row:
                people.append(PersonView(
                    id=row["id"], display_name=row["display_name"],
                    profile_card=load_json(row["profile_card"], {}),
                    profile_source_updated_at=row["profile_source_updated_at"],
                    profile_updated_at=row["profile_updated_at"]))
        # Guidance is deliberately outside this payload: the learning-goal
        # sample is *derived from* the version (see `_guidance`), so hashing
        # it here would be circular -- the version could never settle.
        # Excluding it keeps that derivation acyclic while still making
        # guidance stable for as long as the version itself is unchanged.
        payload = {"user_model": user_view.model_dump(mode="json") if user_view else None,
                   "continuity": continuity.model_dump(mode="json") if continuity else None,
                   "people": [item.model_dump(mode="json") for item in people]}
        version = hashlib.sha256(dump_json(payload).encode()).hexdigest()[:16]
        return user_view, continuity, people, version

    def guidance_bundle(
        self, space_id: str, activated_person_ids: Iterable[str] = (), query: str = "",
    ) -> GuidanceBundle:
        with self._connect() as connection:
            _, _, _, version = self._context_state(connection, space_id)
            return self._guidance(
                connection, space_id, activated_person_ids, query, version=version)

    def list_guidance(
        self, space_id: str, person_ids: Iterable[str] = (),
        kind: str | None = None, limit: int = 50,
    ) -> list[GuidanceItem]:
        """Return the full set of open hypotheses and open/partial learning goals for a focus.

        Unlike `_guidance` (which samples a small, version-stable set for the
        passive, KV-cache-friendly context bundle), this is for an agent that
        explicitly asks: it returns everything matching the owner/focus
        filter, unsampled and not ranked by anything but recency. It shares
        the exact filtering SQL with `_guidance` via `_open_hypothesis_rows`
        and `_open_learning_goal_rows` so the two paths cannot drift apart.
        """
        ids = tuple(dict.fromkeys(person_ids))
        capped = max(0, limit)
        items: list[GuidanceItem] = []
        with self._connect() as connection:
            if kind in (None, 'hypothesis'):
                rows = self._open_hypothesis_rows(connection, space_id, ids)
                items.extend(GuidanceItem(
                    id=row['id'], kind='hypothesis', content=row['content'],
                    owner_kind=row['owner_kind'], owner_id=row['owner_id'],
                    status=row['status'], confidence=row['confidence'],
                ) for row in rows)
            if kind in (None, 'learning_goal'):
                rows = self._open_learning_goal_rows(connection, space_id, ids)
                items.extend(GuidanceItem(
                    id=row['id'], kind='learning_goal', content=row['prompt'],
                    owner_kind=row['focus_kind'], owner_id=row['focus_id'], status=row['status'],
                ) for row in rows)
        return items[:capped]

    def context_bundle(
        self, space_id: str, *, goals: int | None = None, goal_seed: str | None = None,
    ) -> ContextBundle:
        """The passive bundle.

        `goals` and `goal_seed` steer the learning-goal draw; see `_guidance`.
        """
        with self._connect() as connection:
            user_view, continuity, people, version = self._context_state(connection, space_id)
            guidance = self._guidance(
                connection, space_id, version=version, goals=goals, goal_seed=goal_seed)
            return ContextBundle(
                version=version, user_model=user_view, continuity=continuity,
                people=people, guidance=guidance)

    def turn_view(
        self, space_id: str, text: str, memory_limit: int, known_version: str | None,
        query_vector: Sequence[float] | None = None,
        *,
        goals: int | None = None,
        goal_seed: str | None = None,
    ) -> TurnView:
        """One connection, one `_context_state` call, for the whole turn read path.

        Alias matching, guidance, memory recall, and the context bundle all
        read from one connection, and `_context_state` (the
        `user_models`/`continuities`/related-people read plus the version
        hash) runs exactly once. The `ContextBundle` is built only when
        `known_version` is stale.

        Each step degrades on its own and records whether it succeeded;
        `context_status` is then derived once, at the end, from those
        outcomes. Memory recall is deliberately absent from that derivation:
        an empty recall is an ordinary result, so a failed recall is logged
        but does not make the context unavailable. Only the expected
        failures of these reads are caught (`TURN_READ_ERRORS`); anything
        else -- a programming error in particular -- propagates, and cannot
        lose the write, because `world.turn` records the messages and
        schedules intake before calling this.

        Guidance sampling is deliberately derived from the exact same
        `version` the (possibly returned) `ContextBundle` uses -- see
        `_guidance`'s docstring -- which holds because both come from the
        single `_context_state` call below. `goals` and `goal_seed` are
        forwarded to both of those `_guidance` calls for the same reason:
        a turn-derived bundle must be sampled exactly like the one
        `context_bundle` would return for the same knobs, or the two read
        paths disagree about what the caller pinned.
        """
        known_people: list[PersonView] = []
        memory_recall: list[MemoryView] = []
        context_update: ContextBundle | None = None
        guidance = GuidanceBundle()
        matched = False
        guided = False
        contextualized = False
        with self._connect() as connection:
            try:
                known_people = self._match_people(connection, space_id, text)
                matched = True
            except TURN_READ_ERRORS:
                logger.exception("turn person matching failed for %s", space_id)
            state: tuple[UserModelView | None, ContinuityView |
                         None, list[PersonView], str] | None = None
            try:
                state = self._context_state(connection, space_id)
                _, _, _, version = state
                guidance = self._guidance(
                    connection, space_id,
                    [person.id for person in known_people], text, version=version,
                    goals=goals, goal_seed=goal_seed,
                )
                guided = True
            except TURN_READ_ERRORS:
                state = None
                logger.exception("turn guidance preparation failed for %s", space_id)
            try:
                memory_recall = self._recall_memories(
                    connection, space_id, text, about_user=True, limit=memory_limit,
                    query_vector=query_vector,
                )
            except TURN_READ_ERRORS:
                logger.exception("turn memory recall failed for %s", space_id)
            if state is not None:
                try:
                    user_view, continuity, people, version = state
                    if version != known_version:
                        generic_guidance = self._guidance(
                            connection, space_id, version=version,
                            goals=goals, goal_seed=goal_seed)
                        context_update = ContextBundle(
                            version=version, user_model=user_view, continuity=continuity,
                            people=people, guidance=generic_guidance,
                        )
                    contextualized = True
                except TURN_READ_ERRORS:
                    logger.exception("turn context preparation failed for %s", space_id)
        return TurnView(
            known_people=known_people,
            guidance=guidance,
            memory_recall=memory_recall,
            context_update=context_update,
            context_status=(
                "available" if matched and guided and contextualized else "unavailable"),
        )
