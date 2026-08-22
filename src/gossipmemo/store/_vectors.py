"""Embedding storage: durable BLOB facts in the main database, plus a
rebuildable vector-search sidecar (`<db>.vec.db`) backed by the `sqlite-vec`
extension.

Two layers, deliberately kept apart:

- The main-database `embeddings` table (schema.sql, migrated) is the source
  of truth: one row per (owner_kind, owner_id), a packed float32 vector, and
  the `content_hash` of the text it was computed from.
- The sidecar is a `vec0` virtual-table index over the same vectors. A `vec0`
  table produces about ten shadow tables in `sqlite_master`, which would
  pollute this store's structural-fingerprint / table-shape detection if it
  lived in the main database -- see `migrations.py`. So it lives in a
  separate attached file instead, is never part of a migration, and is
  always rebuildable from the main table without calling the embedding API
  again.

Every entry point here degrades silently to a pure-Python dot-product scan
when the `sqlite-vec` extension cannot be loaded or the sidecar cannot be
opened -- never raises to the caller, never fails startup. Both paths return
the same shape: `[(owner_id, score), ...]` ordered best-match first, where
`score` is the cosine similarity between the (already L2-normalized) query
vector and the stored vector -- computed directly as a dot product in the
Python path, and converted from `vec0`'s default L2 distance in the sidecar
path (`cos = 1 - distance**2 / 2`, exact for unit vectors).
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ._base import _BaseStore
from .policy import now_iso

logger = logging.getLogger(__name__)

try:
    import sqlite_vec
except ImportError:  # pragma: no cover - sqlite-vec is a declared dependency
    sqlite_vec = None  # type: ignore[assignment]

# The four polymorphic owner kinds an embedding row can point at, and where
# each one's indexed text and lifecycle status column live. Deliberately
# excludes messages, people, relationships, and continuities -- those are
# projections that get rewritten wholesale, which would churn vectors
# endlessly; see the design brief for the full rationale.
OWNER_KINDS: tuple[str, ...] = ("memory", "learning_goal", "hypothesis", "coverage_entry")

_OWNER_TEXT_SOURCES: dict[str, tuple[str, str, str]] = {
    "memory": ("memories", "content", "status"),
    "learning_goal": ("learning_goals", "rationale", "status"),
    "hypothesis": ("hypotheses", "content", "status"),
    "coverage_entry": ("coverage_entries", "content", "status"),
}

# Bumped whenever the sidecar's vec0/meta table *shape* changes (not the
# embedded model/dim, which is tracked separately in embedding_index_meta).
# A sidecar built under an older layout is treated as absent -- see
# _read_sidecar_meta -- and rebuilt from the main table, never touching the
# embedding API.
_SIDECAR_LAYOUT_VERSION = 2

# search_vectors() over-fetches from vec0 when a status filter is present:
# vec0's `k` truncates the KNN search *before* the live JOIN against the
# owner table's status column runs (status is not a vec0 metadata column --
# deliberately, see the module docstring -- so vec0 has no way to filter on
# it itself). Asking for more candidates than k, then filtering and cutting
# back down to k, makes it very likely the true top-k *active* (or whatever
# statuses were asked for) matches are among them. _SIDECAR_OVER_FETCH_MIN
# keeps small k from over-fetching too little to matter.
_SIDECAR_OVER_FETCH_MULTIPLIER = 5
_SIDECAR_OVER_FETCH_MIN = 50
# If even the over-fetched, filtered result is short of k *and* the raw
# (unfiltered) KNN scan came back exactly at the over-fetch cap -- meaning
# there could be more matches beyond what was fetched -- escalate once to a
# much wider k and accept whatever that yields. Never loop: one escalation
# is a pragmatic bound on cost, not a correctness guarantee for pathological
# status distributions.
_SIDECAR_OVER_FETCH_ESCALATION_MULTIPLIER = 20


def content_hash(text: str) -> str:
    """The single hash algorithm used to detect stale embeddings. Centralized
    here so nothing else in the codebase reimplements it differently."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pack_vector(values: Sequence[float]) -> bytes:
    """Float32, little-endian, tightly packed -- the format both the main
    table's BLOB column and the `sqlite-vec` extension expect."""

    return struct.pack(f"<{len(values)}f", *values)


def unpack_vector(blob: bytes, dim: int) -> list[float]:
    return list(struct.unpack(f"<{dim}f", blob))


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _l2_distance_to_cosine(distance: float) -> float:
    """`vec0`'s default metric is L2. For unit vectors, L2**2 = 2 - 2*cos,
    so this is exact (up to float error), not an approximation."""

    return 1.0 - (distance * distance) / 2.0


def _sidecar_path(main_path: Path) -> Path:
    return main_path.with_name(main_path.stem + ".vec.db")


@dataclass(frozen=True)
class PendingEmbedding:
    """One (owner_kind, owner_id) whose text has no matching embedding row
    yet -- new, hash-stale, or computed under a different model/dim."""

    space_id: str
    owner_kind: str
    owner_id: str
    text: str


@dataclass(frozen=True)
class EmbeddingUpsert:
    """One vector ready to be written, alongside the text it was computed
    from (to re-derive `content_hash`). Lifecycle status is deliberately
    NOT carried here: it lives only on the owner row and is read live at
    search time (via a JOIN, both in the sidecar and the Python fallback),
    never cached alongside the vector -- a cached status column would go
    stale the moment an owner's status changes without its text changing
    (e.g. a memory being retracted), which is exactly the kind of
    state-flag drift this design avoids by using content_hash instead."""

    owner_kind: str
    owner_id: str
    text: str
    vector: Sequence[float]


class _VectorsMixin(_BaseStore):
    """Store/retrieve/search embedding vectors. No RRF, no FTS fusion, no
    recall policy -- that belongs to the hybrid-retrieval slice built on top
    of `search_vectors`."""

    # -- extension / sidecar plumbing -----------------------------------

    def _load_vec_extension(self, connection: sqlite3.Connection) -> bool:
        if sqlite_vec is None:
            logger.warning(
                "sqlite-vec package is not importable; vector search falls "
                "back to a Python dot-product scan"
            )
            return False
        try:
            connection.enable_load_extension(True)
            try:
                sqlite_vec.load(connection)
            finally:
                connection.enable_load_extension(False)
        except sqlite3.Error as error:
            logger.warning(
                "sqlite-vec extension failed to load; vector search falls "
                "back to a Python dot-product scan: %s", error,
            )
            return False
        return True

    def _attach_sidecar(self, connection: sqlite3.Connection) -> bool:
        sidecar = _sidecar_path(self.path)
        try:
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            connection.execute("ATTACH DATABASE ? AS vec", (str(sidecar),))
            connection.execute(
                "CREATE TABLE IF NOT EXISTS vec.embedding_index_meta("
                "model TEXT NOT NULL, dim INTEGER NOT NULL, "
                "layout_version INTEGER NOT NULL, built_at TEXT NOT NULL)"
            )
        except (sqlite3.Error, OSError) as error:
            logger.warning(
                "failed to attach vector sidecar database %s; vector search "
                "falls back to a Python dot-product scan: %s", sidecar, error,
            )
            return False
        return True

    def _enable_vectors(self, connection: sqlite3.Connection) -> bool:
        """Load the extension and attach the sidecar on `connection`.
        Returns whether both succeeded -- callers must treat False as
        "fall back to Python", never as an error."""

        return self._load_vec_extension(connection) and self._attach_sidecar(connection)

    def _read_sidecar_meta(
        self, connection: sqlite3.Connection,
    ) -> tuple[str, int, int] | None:
        """The sidecar's recorded (model, dim, layout_version), or None if
        it has never been built, or was built under a shape old enough that
        `layout_version` (or the whole meta table) doesn't exist yet --
        both are treated identically to "needs a rebuild"."""

        try:
            row = connection.execute(
                "SELECT model, dim, layout_version FROM vec.embedding_index_meta LIMIT 1"
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        if row is None:
            return None
        return (row["model"], row["dim"], row["layout_version"])

    def ensure_vector_index(self, model: str, dim: int) -> None:
        """Idempotent maintenance entry point: call at startup (after
        induction/backfill scheduling) and whenever the active embedding
        model/dim might have changed. Rebuilds the sidecar's `vec0` table
        from the main `embeddings` BLOBs -- never calling the embedding API
        -- exactly when its recorded (model, dim, layout_version) meta
        disagrees with the current one. A no-op, quietly, if vectors are
        unavailable."""

        with self._connect() as connection:
            if not self._enable_vectors(connection):
                return
            if self._read_sidecar_meta(connection) == (model, dim, _SIDECAR_LAYOUT_VERSION):
                return
            self._rebuild_sidecar(connection, model, dim)

    def _rebuild_sidecar(self, connection: sqlite3.Connection, model: str, dim: int) -> None:
        """Drop and recreate both the `vec0` table and its meta row from
        scratch. Dropping (not just clearing) `embedding_index_meta` is
        what lets this also fix up a meta table built under an older
        layout (missing `layout_version`) -- it always ends up with the
        current column shape. No owner-table status is read or cached here
        -- status is never stored in the sidecar; see EmbeddingUpsert."""

        connection.execute("DROP TABLE IF EXISTS vec.embeddings")
        connection.execute("DROP TABLE IF EXISTS vec.embedding_index_meta")
        connection.execute(
            "CREATE TABLE vec.embedding_index_meta("
            "model TEXT NOT NULL, dim INTEGER NOT NULL, "
            "layout_version INTEGER NOT NULL, built_at TEXT NOT NULL)"
        )
        connection.execute(
            f"CREATE VIRTUAL TABLE vec.embeddings USING vec0("
            f"owner_id TEXT PRIMARY KEY, embedding float[{dim}], "
            f"space_id TEXT partition key, owner_kind TEXT, "
            f"+content_hash TEXT)"
        )
        rows = connection.execute(
            "SELECT space_id, owner_kind, owner_id, vector, content_hash "
            "FROM embeddings WHERE model = ? AND dim = ?",
            (model, dim),
        ).fetchall()
        connection.executemany(
            "INSERT INTO vec.embeddings(owner_id, embedding, space_id, owner_kind, content_hash) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (row["owner_id"], row["vector"], row["space_id"], row["owner_kind"],
                 row["content_hash"])
                for row in rows
            ],
        )
        connection.execute(
            "INSERT INTO vec.embedding_index_meta(model, dim, layout_version, built_at) "
            "VALUES (?, ?, ?, ?)",
            (model, dim, _SIDECAR_LAYOUT_VERSION, now_iso()),
        )

    # -- writes ------------------------------------------------------------

    def upsert_embeddings(
        self, space_id: str, model: str, dim: int, items: Sequence[EmbeddingUpsert],
    ) -> None:
        """Batch upsert into the main `embeddings` table (always durable)
        and, best-effort, into the sidecar index if it is currently
        reachable and already built for this exact (model, dim). If the
        sidecar is unavailable or its meta does not match, the write is
        skipped there -- `ensure_vector_index` reconciles it later from the
        main table, without ever calling the embedding API again."""

        if not items:
            return
        now = now_iso()
        with self._connect() as connection:
            for item in items:
                connection.execute(
                    """
                    INSERT INTO embeddings(
                        space_id, owner_kind, owner_id, model, dim, vector, content_hash, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(owner_kind, owner_id) DO UPDATE SET
                        space_id = excluded.space_id,
                        model = excluded.model,
                        dim = excluded.dim,
                        vector = excluded.vector,
                        content_hash = excluded.content_hash,
                        created_at = excluded.created_at
                    """,
                    (
                        space_id, item.owner_kind, item.owner_id, model, dim,
                        pack_vector(item.vector), content_hash(item.text), now,
                    ),
                )

            if not self._enable_vectors(connection):
                return
            if self._read_sidecar_meta(connection) != (model, dim, _SIDECAR_LAYOUT_VERSION):
                # Sidecar is unbuilt or stale for this (model, dim, layout);
                # leave it alone here -- ensure_vector_index() does a full
                # rebuild that will pick up the rows just written above.
                return
            for item in items:
                blob = pack_vector(item.vector)
                h = content_hash(item.text)
                # `space_id` is the vec0 partition key and stays out of the SET
                # list: sqlite-vec rejects an UPDATE that assigns one at all
                # ("UPDATE on partition key columns are not supported yet"),
                # even when the value is unchanged. Nothing needs it there --
                # `owner_id` is globally unique and no code path ever moves a
                # row between spaces, so a row's partition is fixed the moment
                # it is inserted below.
                cursor = connection.execute(
                    "UPDATE vec.embeddings SET embedding = ?, "
                    "owner_kind = ?, content_hash = ? WHERE owner_id = ?",
                    (blob, item.owner_kind, h, item.owner_id),
                )
                if cursor.rowcount == 0:
                    connection.execute(
                        "INSERT INTO vec.embeddings"
                        "(owner_id, embedding, space_id, owner_kind, content_hash) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (item.owner_id, blob, space_id, item.owner_kind, h),
                    )

    # -- pending / maintenance ----------------------------------------------

    def pending_embeddings(
        self, *, model: str, dim: int, owner_kinds: Sequence[str] | None = None,
    ) -> list[PendingEmbedding]:
        """Owners with no embedding row, a stale `content_hash` (their text
        was rewritten in place), or a row computed under a different
        model/dim. Also sweeps up orphaned embedding rows (owner deleted)
        along the way -- see `_cleanup_orphan_embeddings`."""

        kinds = tuple(owner_kinds) if owner_kinds else OWNER_KINDS
        pending: list[PendingEmbedding] = []
        with self._connect() as connection:
            self._cleanup_orphan_embeddings(connection, kinds)
            for kind in kinds:
                table, text_col, _status_col = _OWNER_TEXT_SOURCES[kind]
                rows = connection.execute(
                    f"""
                    SELECT o.space_id AS space_id, o.id AS owner_id, o.{text_col} AS text,
                           e.content_hash AS existing_hash, e.model AS existing_model,
                           e.dim AS existing_dim
                    FROM {table} o
                    LEFT JOIN embeddings e ON e.owner_kind = ? AND e.owner_id = o.id
                    """,
                    (kind,),
                ).fetchall()
                for row in rows:
                    text = row["text"] or ""
                    if (
                        row["existing_hash"] is None
                        or row["existing_hash"] != content_hash(text)
                        or row["existing_model"] != model
                        or row["existing_dim"] != dim
                    ):
                        pending.append(
                            PendingEmbedding(row["space_id"], kind, row["owner_id"], text)
                        )
        return pending

    def pending_embedding_count(self, *, model: str, dim: int) -> int:
        """Cheap count for `health()` -- does not materialize owner text."""

        with self._connect() as connection:
            self._cleanup_orphan_embeddings(connection, OWNER_KINDS)
            total = 0
            for kind in OWNER_KINDS:
                table, text_col, _status_col = _OWNER_TEXT_SOURCES[kind]
                rows = connection.execute(
                    f"""
                    SELECT o.{text_col} AS text, e.content_hash AS existing_hash,
                           e.model AS existing_model, e.dim AS existing_dim
                    FROM {table} o
                    LEFT JOIN embeddings e ON e.owner_kind = ? AND e.owner_id = o.id
                    """,
                    (kind,),
                ).fetchall()
                for row in rows:
                    text = row["text"] or ""
                    if (
                        row["existing_hash"] is None
                        or row["existing_hash"] != content_hash(text)
                        or row["existing_model"] != model
                        or row["existing_dim"] != dim
                    ):
                        total += 1
        return total

    def _cleanup_orphan_embeddings(
        self, connection: sqlite3.Connection, kinds: Sequence[str],
    ) -> None:
        for kind in kinds:
            table, _text_col, _status_col = _OWNER_TEXT_SOURCES[kind]
            connection.execute(
                f"DELETE FROM embeddings WHERE owner_kind = ? "
                f"AND owner_id NOT IN (SELECT id FROM {table})",
                (kind,),
            )

    # -- search --------------------------------------------------------

    def search_vectors(
        self,
        space_id: str,
        owner_kind: str,
        query_vector: Sequence[float],
        k: int,
        *,
        statuses: Sequence[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Top-`k` nearest owners by cosine similarity, best match first.
        No RRF, no FTS fusion -- a single ranked list for one owner_kind."""

        with self._connect() as connection:
            if self._enable_vectors(connection):
                try:
                    return self._search_sidecar(
                        connection, space_id, owner_kind, query_vector, k, statuses,
                    )
                except sqlite3.Error as error:
                    logger.warning(
                        "vector sidecar search failed; falling back to a "
                        "Python dot-product scan: %s", error,
                    )
            return self._search_python(connection, space_id, owner_kind, query_vector, k, statuses)

    def _search_sidecar(
        self,
        connection: sqlite3.Connection,
        space_id: str,
        owner_kind: str,
        query_vector: Sequence[float],
        k: int,
        statuses: Sequence[str] | None,
    ) -> list[tuple[str, float]]:
        table, _text_col, status_col = _OWNER_TEXT_SOURCES[owner_kind]

        if not statuses:
            # No status filter: vec0's own k truncation is exactly the
            # answer, no over-fetch needed.
            rows = self._sidecar_knn_joined(
                connection, table, status_col, space_id, owner_kind, query_vector, k, None,
            )
            return [(row["owner_id"], _l2_distance_to_cosine(row["distance"])) for row in rows[:k]]

        over_fetch_k = max(k * _SIDECAR_OVER_FETCH_MULTIPLIER, _SIDECAR_OVER_FETCH_MIN)
        rows = self._sidecar_knn_joined(
            connection, table, status_col, space_id, owner_kind, query_vector,
            over_fetch_k, statuses,
        )
        if len(rows) < k:
            raw_count = self._sidecar_knn_raw_count(
                connection, space_id, owner_kind, query_vector, over_fetch_k,
            )
            if raw_count == over_fetch_k:
                # The unfiltered KNN scan was exactly capped at over_fetch_k,
                # so there may be more true candidates beyond it that the
                # live status filter would have kept. Escalate once, wide,
                # and accept whatever comes back -- see the constant's
                # comment for why this never loops.
                escalated_k = max(
                    k * _SIDECAR_OVER_FETCH_ESCALATION_MULTIPLIER, _SIDECAR_OVER_FETCH_MIN,
                )
                if escalated_k > over_fetch_k:
                    rows = self._sidecar_knn_joined(
                        connection, table, status_col, space_id, owner_kind, query_vector,
                        escalated_k, statuses,
                    )
        return [(row["owner_id"], _l2_distance_to_cosine(row["distance"])) for row in rows[:k]]

    def _sidecar_knn_joined(
        self,
        connection: sqlite3.Connection,
        table: str,
        status_col: str,
        space_id: str,
        owner_kind: str,
        query_vector: Sequence[float],
        k: int,
        statuses: Sequence[str] | None,
    ) -> list[sqlite3.Row]:
        """The verified cross-database KNN query: `owner_id`/`distance` come
        from the sidecar's `vec0` table, `status` is read live via a JOIN
        against the owner's row in the main database -- never cached."""

        sql = (
            "SELECT e.owner_id AS owner_id, e.distance AS distance "
            f"FROM vec.embeddings e JOIN main.{table} o ON o.id = e.owner_id "
            "WHERE e.embedding MATCH ? AND k = ? AND e.space_id = ? AND e.owner_kind = ?"
        )
        params: list[object] = [pack_vector(query_vector), k, space_id, owner_kind]
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            sql += f" AND o.{status_col} IN ({placeholders})"
            params.extend(statuses)
        sql += " ORDER BY e.distance"
        return connection.execute(sql, params).fetchall()

    def _sidecar_knn_raw_count(
        self,
        connection: sqlite3.Connection,
        space_id: str,
        owner_kind: str,
        query_vector: Sequence[float],
        k: int,
    ) -> int:
        """How many candidates vec0's own KNN returns before any live status
        filter -- used only to tell whether a deficient, filtered result was
        actually capped by `k` (worth escalating) or is simply everything
        that exists (not worth escalating)."""

        rows = connection.execute(
            "SELECT e.owner_id AS owner_id FROM vec.embeddings e "
            "WHERE e.embedding MATCH ? AND k = ? AND e.space_id = ? AND e.owner_kind = ?",
            (pack_vector(query_vector), k, space_id, owner_kind),
        ).fetchall()
        return len(rows)

    def _search_python(
        self,
        connection: sqlite3.Connection,
        space_id: str,
        owner_kind: str,
        query_vector: Sequence[float],
        k: int,
        statuses: Sequence[str] | None,
    ) -> list[tuple[str, float]]:
        table, _text_col, status_col = _OWNER_TEXT_SOURCES[owner_kind]
        sql = (
            f"SELECT e.owner_id AS owner_id, e.vector AS vector, e.dim AS dim "
            f"FROM embeddings e JOIN {table} o ON o.id = e.owner_id "
            f"WHERE e.space_id = ? AND e.owner_kind = ?"
        )
        params: list[object] = [space_id, owner_kind]
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            sql += f" AND o.{status_col} IN ({placeholders})"
            params.extend(statuses)
        rows = connection.execute(sql, params).fetchall()
        scored = [
            (row["owner_id"], _dot(unpack_vector(row["vector"], row["dim"]), query_vector))
            for row in rows
        ]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]


__all__ = [
    "EmbeddingUpsert",
    "OWNER_KINDS",
    "PendingEmbedding",
    "_VectorsMixin",
    "content_hash",
    "pack_vector",
    "unpack_vector",
]
