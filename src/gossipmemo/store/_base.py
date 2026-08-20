"""Connection handling, schema bootstrap, and space creation."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..migrations import migrate_database
from ..models import COVERAGE_ROOTS
from .policy import now_iso


class _BaseStore:
    """Shared base for the store mixins: connections, schema, spaces."""

    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        # Short on purpose: under WAL a reader never waits on a writer, so a
        # lock wait here means two writers overlapped, and every write in
        # this store is a sub-second atomic statement. Waiting ten seconds
        # for one only hides a stuck writer behind a stalled request.
        connection.execute("PRAGMA busy_timeout = 1000")
        # Under WAL (enforced by `_enable_wal` at startup) NORMAL still fsyncs
        # every checkpoint, so the database can never be corrupted by this --
        # only the last few committed transactions can be lost on a power
        # loss or OS crash before they reach the WAL file. That fsync was
        # measured at ~90% of a single-row write's latency; for a local,
        # single-user memory store, trading that narrow crash window for a
        # roughly 10x faster hot write path is worth it.
        connection.execute("PRAGMA synchronous = NORMAL")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        """Apply the schema, and seed the coverage roots of every space.

        `ensure_space` seeds roots too, but it only runs on an ingest path.
        A space that exists without its rows is audited by nobody and
        reports no staleness -- `stale_coverage_spaces` reads
        `coverage_roots`, so an empty table is indistinguishable from a
        space that is fully caught up. Seeding here keeps that silence
        impossible for a space whose roots are new to it.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        migrate_database(self.path)
        self._enable_wal()
        schema_path = Path(__file__).resolve().parent.parent / "schema.sql"
        schema = schema_path.read_text(encoding="utf-8")
        with self._connect() as connection:
            connection.executescript(schema)
            now = now_iso()
            connection.executemany(
                "INSERT OR IGNORE INTO coverage_roots(space_id, root, updated_at) "
                "SELECT id, ?, ? FROM spaces",
                [(root, now) for root in COVERAGE_ROOTS],
            )

    def _enable_wal(self) -> None:
        """Switch the database to WAL once, and confirm it took.

        The mode is a property of the database file, not the connection, so
        this runs at startup only. `PRAGMA journal_mode` reports the mode it
        ended up in instead of raising, so a filesystem that cannot support
        WAL -- a network mount, almost always -- leaves the database in
        rollback-journal mode with nothing in the logs. Readers would then
        block behind every writer while `busy_timeout` is deliberately
        short, turning a silent downgrade into intermittent "database is
        locked" errors much later. Fail at startup instead.

        Runs outside `_connect` because the journal mode cannot be changed
        from inside a transaction.
        """
        connection = sqlite3.connect(self.path, timeout=10.0)
        try:
            row = connection.execute("PRAGMA journal_mode = WAL").fetchone()
        finally:
            connection.close()
        mode = (row[0] if row else "").lower()
        if mode != "wal":
            raise RuntimeError(
                f"SQLite refused WAL mode for {self.path} (still {mode!r}). "
                "This usually means the database is on a filesystem that "
                "cannot support it, such as a network mount."
            )

    def ensure_space(self, space_id: str, name: str | None = None) -> str:
        with self._connect() as connection:
            now = now_iso()
            connection.execute(
                "INSERT OR IGNORE INTO spaces(id, name, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                (space_id, name or space_id, now, now),
            )
            connection.execute(
                "INSERT OR IGNORE INTO user_models(space_id) VALUES (?)", (space_id,)
            )
            connection.executemany(
                "INSERT OR IGNORE INTO coverage_roots(space_id, root, updated_at) VALUES (?, ?, ?)",
                [(space_id, root, now) for root in COVERAGE_ROOTS],
            )
            connection.execute(
                "INSERT OR IGNORE INTO continuities(space_id, updated_at) VALUES (?, ?)",
                (space_id, now),
            )
            return space_id
