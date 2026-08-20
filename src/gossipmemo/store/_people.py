"""People, aliases, relationships, and the person/relationship cards."""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from typing import Any

from ..models import (
    MemoryView,
    PersonSummaryView,
    PersonView,
    RelationshipView,
)
from ._errors import AmbiguousPersonError, PersonMergeError
from ._projections import _ProjectionsMixin
from .policy import (
    dump_json,
    load_json,
    new_id,
    normalized,
    now_iso,
)


class _PeopleMixin(_ProjectionsMixin):
    """Person lookup and creation, alias matching, merges, and cards."""

    def _find_person(
        self, connection: sqlite3.Connection, space_id: str, reference: str
    ) -> sqlite3.Row | None:
        row = connection.execute(
            "SELECT * FROM people WHERE space_id = ? AND id = ? AND status = 'active'",
            (space_id, reference),
        ).fetchone()
        if row:
            return row
        normalized_reference = normalized(reference)
        alias_rows = connection.execute(
            """
            SELECT p.* FROM people p
            JOIN person_aliases a ON a.person_id = p.id
            WHERE p.space_id = ? AND a.normalized_value = ? AND p.status = 'active'
            LIMIT 2
            """,
            (space_id, normalized_reference),
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
        person_id = new_id("person")
        now = now_iso()
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
            (new_id("alias"), space_id, person_id, display_name, normalized(display_name)),
        )
        return person_id

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
                    (dump_json(facets), now_iso(), row["id"]),
                )
            return row["id"]
        relationship_id = new_id("relationship")
        now = now_iso()
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
                dump_json(facets),
                now,
                now,
            ),
        )
        return relationship_id

    def person_context(
        self, space_id: str, person_id: str
    ) -> tuple[PersonView, list[MemoryView], str | None] | None:
        with self._connect() as connection:
            person = connection.execute(
                "SELECT * FROM people WHERE space_id = ? AND id = ?",
                (space_id, person_id)).fetchone()
            if not person:
                return None
            rows = connection.execute(
                """SELECT DISTINCT m.* FROM memories m
                JOIN memory_people mp ON mp.memory_id = m.id
                WHERE m.space_id = ? AND m.status = 'active'
                  AND m.basis <> 'inferred' AND mp.person_id = ?
                ORDER BY m.created_at DESC, m.id DESC""", (space_id, person_id)).fetchall()
            watermark = self._person_watermark(connection, space_id, person_id)
            view = self._person_view(connection, person)
            return view, [self._memory_view(connection, row, True) for row in rows], watermark

    def relationship_context(
        self, space_id: str, relationship_id: str
    ) -> tuple[RelationshipView, list[MemoryView], str | None] | None:
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

    def match_people_in_text(self, space_id: str, text: str) -> list[PersonView]:
        """Resolve explicit aliases without creating people or invoking an LLM."""
        with self._connect() as connection:
            return self._match_people(connection, space_id, text)

    def list_people(
        self, space_id: str, query: str = "", limit: int = 50
    ) -> list[PersonSummaryView]:
        """List/search active people, aliases included, for merge discovery.

        Unlike `_match_people`, an alias shared by several people is
        deliberately *not* dropped here -- surfacing that ambiguity is the
        entire point of this listing (it is the precondition for
        `merge_person`). With no query every active person is returned,
        ordered by display name, bounded by `limit`.
        """
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT p.id AS person_id, p.display_name AS display_name, a.value AS alias_value
                FROM people p
                LEFT JOIN person_aliases a ON a.person_id = p.id
                WHERE p.space_id = ? AND p.status = 'active'
                ORDER BY p.display_name, p.id
                """,
                (space_id,),
            ).fetchall()
            order: list[str] = []
            display_names: dict[str, str] = {}
            aliases: dict[str, list[str]] = {}
            for row in rows:
                person_id = row["person_id"]
                if person_id not in display_names:
                    order.append(person_id)
                    display_names[person_id] = row["display_name"]
                    aliases[person_id] = []
                alias_value = row["alias_value"]
                if alias_value and alias_value not in aliases[person_id]:
                    aliases[person_id].append(alias_value)

            folded_query = unicodedata.normalize("NFKC", query).casefold().strip()
            matched_ids: list[str] = []
            for person_id in order:
                if not folded_query:
                    matched_ids.append(person_id)
                    continue
                folded_name = unicodedata.normalize(
                    "NFKC", display_names[person_id]).casefold()
                if folded_query in folded_name:
                    matched_ids.append(person_id)
                    continue
                for alias in aliases[person_id]:
                    if folded_query in unicodedata.normalize("NFKC", alias).casefold():
                        matched_ids.append(person_id)
                        break

            capped = max(0, limit)
            return [
                PersonSummaryView(
                    id=person_id,
                    display_name=display_names[person_id],
                    aliases=aliases[person_id],
                )
                for person_id in matched_ids[:capped]
            ]

    def _match_people(
        self, connection: sqlite3.Connection, space_id: str, text: str
    ) -> list[PersonView]:
        """Alias matching on a caller's connection, so a writer can reuse it."""
        folded = unicodedata.normalize("NFKC", text).casefold()
        aliases = connection.execute(
            """SELECT a.normalized_value, a.value, a.person_id, p.*
               FROM person_aliases a JOIN people p ON p.id = a.person_id
               WHERE a.space_id = ? AND p.status = 'active'""", (space_id,)
        ).fetchall()
        by_alias: dict[str, set[str]] = {}
        for row in aliases:
            by_alias.setdefault(unicodedata.normalize(
                "NFKC", row["normalized_value"]).casefold(), set()).add(row["person_id"])

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

            now = now_iso()
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
                        new_id("alias"),
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
                ids = load_json(continuity["related_person_ids"], [])
                rewritten = list(
                    dict.fromkeys(
                        target_person_id if item == source_person_id else item
                        for item in ids
                    )
                )
                connection.execute(
                    "UPDATE continuities SET related_person_ids = ?, updated_at = ? "
                    "WHERE space_id = ?",
                    (dump_json(rewritten), now, space_id),
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
