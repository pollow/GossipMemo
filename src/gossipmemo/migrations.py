"""SQLite schema migration runner.

GossipMemo's first shipped release (pinned as commit
`0bc92208314bd685a63bd0b8415eda65c511cea0` on `main`) had no migration
history table at all -- its schema is treated as version 1 by definition.
Every schema change after that ships with an entry in `MIGRATIONS` below and
a bump of `CURRENT_VERSION`, so a deployed database is upgraded in place on
the next restart instead of being deleted and rebuilt.

Detection rule (see `migrate_database` for the implementation):

- A database file with **no user tables at all** is brand new. It is
  stamped directly at `CURRENT_VERSION` without replaying history -- there
  is nothing to preserve, so pretending it lived through every past
  migration would be fiction, not history.
- A database with a `schema_migrations` table has its history validated
  (ordered, contiguous, checksums matching known migrations) and any
  pending migrations above its recorded version are applied.
- A database with **no** `schema_migrations` table but real tables in it is
  either:
  - a legacy version-1 database (recognized by the presence of
    `coverage_maps`, the table version 2 removes) -- history is backfilled
    with a version-1 baseline row and the pending migrations are applied, or
  - a database that a development checkout already created directly from a
    post-migration `schema.sql` (recognized by having `coverage_roots` /
    `coverage_entries` and a `learning_goals.entry_ids` column, with no
    `coverage_maps` and no `criteria_refs`/`boundary_ids` columns) -- this
    is structurally already at `CURRENT_VERSION`, so it is adopted with a
    marker row rather than refused or re-migrated.
  Any other shape is refused: this runner never guesses at an unrecognized
  database, per the migration policy in CLAUDE.md.

Every marker/checksum recorded in `schema_migrations` either matches a
known `Migration.checksum` (a real, applied upgrade) or one of the sentinel
sentinels below (`_LEGACY_BASELINE_MARKER`, `_FRESH_CREATE_MARKER`,
`_ADOPTED_MARKER`), and only as the very first row. Anything else -- a
version number matching no known migration, a gap in the sequence, a
mismatched checksum -- makes the history untrustworthy, and
`migrate_database` refuses to run rather than silently trusting it.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


class MigrationError(RuntimeError):
    """Raised when a database cannot be safely brought to the running
    program's schema version. The caller (`SqliteWorldStore.initialize`)
    does not catch this: startup must fail loudly rather than serve a
    program against a database it cannot trust."""


# Sentinel checksums for `schema_migrations` rows that were not produced by
# running a real numbered migration's SQL. They are validated to appear
# only as the very first history row (see `_validate_history`). They can
# never collide with a real `Migration.checksum`, which is always a 64
# character sha256 hex digest.
_LEGACY_BASELINE_MARKER = "legacy-baseline"
_FRESH_CREATE_MARKER = "fresh-create"
_ADOPTED_MARKER = "adopted-without-history"
_SENTINEL_MARKERS = frozenset(
    {_LEGACY_BASELINE_MARKER, _FRESH_CREATE_MARKER, _ADOPTED_MARKER}
)

_SCHEMA_MIGRATIONS_DDL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL,
    description TEXT NOT NULL,
    checksum TEXT NOT NULL
)
"""


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass(frozen=True)
class Migration:
    version: int  # the version this migration upgrades the database TO
    description: str
    upgrade: Callable[[sqlite3.Connection], None]

    @property
    def checksum(self) -> str:
        return hashlib.sha256(
            f"{self.version}:{self.description}".encode("utf-8")
        ).hexdigest()


def _upgrade_v1_to_v2(connection: sqlite3.Connection) -> None:
    """Replace `coverage_maps` with per-root `coverage_roots` /
    `coverage_entries`, and migrate `learning_goals` to `entry_ids`.

    `coverage_maps` rows have no equivalent under the new coverage-entries
    model and are dropped outright -- callers re-derive coverage entries
    from existing Memories. `learning_goals` rows ARE preserved: their old
    `criteria_refs` / `boundary_ids` have no replacement, so they are
    dropped, and every row gets `entry_ids = '[]'`. Everything else
    (messages, memories, people, aliases, relationships, user_models,
    continuities, hypotheses, extraction_batches, the FTS tables and
    triggers) is untouched by this migration.

    `coverage_roots` is intentionally NOT seeded here: `SqliteWorldStore.
    initialize()` already seeds coverage roots (with null cursors) for
    every space that lacks them, and it runs immediately after this
    migration returns, so seeding here would just duplicate that logic.
    """

    connection.execute("DROP TABLE coverage_maps")
    connection.execute(
        """
        CREATE TABLE coverage_roots (
            space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
            root TEXT NOT NULL,
            revision INTEGER NOT NULL DEFAULT 0,
            source_watermark TEXT,
            source_cursor_id TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(space_id, root)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE coverage_entries (
            id TEXT PRIMARY KEY,
            space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
            root TEXT NOT NULL,
            path TEXT NOT NULL DEFAULT '',
            content TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active', 'superseded')),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE INDEX coverage_entries_by_root
        ON coverage_entries(space_id, root, status)
        """
    )
    # `criteria_refs` and `boundary_ids` have no replacement under the new
    # model and are dropped; every other column, and every row, survives.
    # This uses the classic rebuild-and-rename recipe (rather than ALTER
    # TABLE ... DROP COLUMN twice) so the resulting table's column order and
    # CREATE TABLE text match a fresh `schema.sql` apply exactly, which is
    # what the structural-equivalence test in tests/test_migrations.py
    # checks. `PRAGMA foreign_keys` is already OFF for this connection (see
    # `migrate_database`), which is required for a rebuild-and-rename to be
    # safe inside a single transaction.
    connection.execute(
        """
        CREATE TABLE learning_goals_v2 (
            id TEXT PRIMARY KEY,
            space_id TEXT NOT NULL REFERENCES spaces(id) ON DELETE CASCADE,
            prompt TEXT NOT NULL,
            rationale TEXT NOT NULL,
            entry_ids TEXT NOT NULL DEFAULT '[]',
            focus_kind TEXT NOT NULL DEFAULT 'user' CHECK(focus_kind IN ('user', 'person', 'relationship')),
            focus_id TEXT,
            status TEXT NOT NULL DEFAULT 'open' CHECK(status IN ('open', 'partial', 'answered', 'deferred', 'retired')),
            status_reason TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK((focus_kind = 'user' AND focus_id IS NULL) OR
                  (focus_kind IN ('person', 'relationship') AND focus_id IS NOT NULL))
        )
        """
    )
    connection.execute(
        """
        INSERT INTO learning_goals_v2 (
            id, space_id, prompt, rationale, entry_ids,
            focus_kind, focus_id, status, status_reason, created_at, updated_at
        )
        SELECT id, space_id, prompt, rationale, '[]',
               focus_kind, focus_id, status, status_reason, created_at, updated_at
        FROM learning_goals
        """
    )
    connection.execute("DROP TABLE learning_goals")
    connection.execute("ALTER TABLE learning_goals_v2 RENAME TO learning_goals")
    connection.execute(
        """
        CREATE INDEX learning_goals_by_space_status
        ON learning_goals(space_id, status, updated_at DESC)
        """
    )


MIGRATIONS: dict[int, Migration] = {
    2: Migration(
        version=2,
        description=(
            "coverage_maps -> coverage_roots/coverage_entries; "
            "learning_goals.criteria_refs/boundary_ids -> entry_ids"
        ),
        upgrade=_upgrade_v1_to_v2,
    ),
}

CURRENT_VERSION = max(MIGRATIONS, default=1)


def _user_tables(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
    ).fetchall()
    return {row[0] for row in rows}


def _column_names(connection: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}


def _looks_like_v1(tables: set[str]) -> bool:
    return "coverage_maps" in tables


def _looks_like_v2(connection: sqlite3.Connection, tables: set[str]) -> bool:
    if "coverage_maps" in tables:
        return False
    if not {"coverage_roots", "coverage_entries", "learning_goals"} <= tables:
        return False
    columns = _column_names(connection, "learning_goals")
    return "entry_ids" in columns and not ({"criteria_refs", "boundary_ids"} & columns)


def _read_history(connection: sqlite3.Connection) -> list[tuple[int, str]] | None:
    """Return `[(version, checksum), ...]` ordered ascending, or None if the
    `schema_migrations` table does not exist yet."""

    if "schema_migrations" not in _user_tables(connection):
        return None
    rows = connection.execute(
        "SELECT version, checksum FROM schema_migrations ORDER BY version"
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def _validate_history(path: Path, history: list[tuple[int, str]]) -> int:
    """Validate an existing history and return its highest version.

    Raises `MigrationError` for anything that is not a clean, trustworthy
    record: an empty table, a gap in the version sequence, a sentinel
    marker anywhere but the first row, or a checksum that does not match
    the migration registered for that version.
    """

    if not history:
        raise MigrationError(
            f"{path}: schema_migrations table exists but is empty; refusing "
            "to guess this database's schema version"
        )
    versions = [version for version, _ in history]
    if versions != list(range(versions[0], versions[0] + len(versions))):
        raise MigrationError(
            f"{path}: schema_migrations history has a gap or duplicate "
            f"({versions}); refusing to trust a tampered or corrupted history"
        )
    for index, (version, checksum) in enumerate(history):
        if checksum in _SENTINEL_MARKERS:
            if index != 0:
                raise MigrationError(
                    f"{path}: schema_migrations row for version {version} "
                    "carries a bookkeeping marker outside the first row; "
                    "refusing to trust a tampered history"
                )
            continue
        migration = MIGRATIONS.get(version)
        if migration is None or checksum != migration.checksum:
            raise MigrationError(
                f"{path}: schema_migrations row for version {version} does "
                "not match any known migration's checksum; refusing to "
                "trust a tampered or unrecognized history"
            )
    highest = versions[-1]
    if highest > CURRENT_VERSION:
        raise MigrationError(
            f"{path}: database is stamped at schema version {highest}, "
            f"newer than this program's version {CURRENT_VERSION}; refusing "
            "to run an older program against a newer database"
        )
    return highest


def _backup(connection: sqlite3.Connection, path: Path, from_version: int) -> Path:
    backup_path = path.parent / \
        f".{path.name}.pre-migration-v{from_version}.{_now().replace(':', '')}.bak"
    try:
        backup_connection = sqlite3.connect(backup_path)
        try:
            connection.backup(backup_connection)
        finally:
            backup_connection.close()
    except sqlite3.Error as error:
        raise MigrationError(
            f"{path}: could not create pre-migration backup at {backup_path}: {error}"
        ) from error
    if not backup_path.exists() or backup_path.stat().st_size == 0:
        raise MigrationError(
            f"{path}: pre-migration backup at {backup_path} was not created"
        )
    return backup_path


def _apply_pending(connection: sqlite3.Connection, path: Path, start_after: int) -> None:
    """Apply every migration above `start_after`, one write transaction per
    version, rolling back and raising on the first failure."""

    for version in range(start_after + 1, CURRENT_VERSION + 1):
        migration = MIGRATIONS.get(version)
        if migration is None:
            raise MigrationError(
                f"{path}: no migration registered for version {version}; "
                "cannot reach CURRENT_VERSION"
            )
        connection.execute("BEGIN IMMEDIATE")
        try:
            migration.upgrade(connection)
            connection.execute(
                "INSERT INTO schema_migrations(version, applied_at, description, checksum) "
                "VALUES (?, ?, ?, ?)",
                (migration.version, _now(), migration.description, migration.checksum),
            )
        except Exception:
            connection.execute("ROLLBACK")
            raise
        else:
            connection.execute("COMMIT")


def migrate_database(path: Path) -> None:
    """Bring the SQLite database at `path` to `CURRENT_VERSION`, in place.

    Called once at the top of `SqliteWorldStore.initialize()`, before the
    application schema (`schema.sql`) is applied. See the module docstring
    for the detection rule this implements.
    """

    connection = sqlite3.connect(path, timeout=10.0)
    connection.isolation_level = None  # manual transaction control below
    try:
        # Table rebuilds (DROP TABLE, ALTER TABLE ... DROP COLUMN) inside a
        # transaction are unsafe with `PRAGMA foreign_keys = ON`: SQLite
        # defers FK checks to COMMIT, and a transient DROP/CREATE cycle can
        # trip them even though the final state is consistent. The pragma
        # also cannot be changed while a transaction is open, so it must be
        # set here, before anything below opens one.
        connection.execute("PRAGMA foreign_keys = OFF")

        existing_tables = _user_tables(connection)
        if not existing_tables:
            # A brand-new, empty database file. Stamp it directly at
            # CURRENT_VERSION -- there is no history to preserve, and
            # replaying every past migration against nothing would just be
            # a slower way of creating the same schema.
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(_SCHEMA_MIGRATIONS_DDL)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at, description, checksum) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        CURRENT_VERSION,
                        _now(),
                        "database created directly at the current schema version",
                        _FRESH_CREATE_MARKER,
                    ),
                )
            except Exception:
                connection.execute("ROLLBACK")
                raise
            else:
                connection.execute("COMMIT")
            return

        history = _read_history(connection)
        if history is not None:
            highest = _validate_history(path, history)
            if highest == CURRENT_VERSION:
                return  # already up to date; restarting is a no-op
            _backup(connection, path, highest)
            _apply_pending(connection, path, highest)
            return

        # No schema_migrations table, but the database is not empty either:
        # recognize exactly the two known unstamped shapes, and refuse
        # anything else rather than guessing.
        if _looks_like_v1(existing_tables):
            _backup(connection, path, 1)
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(_SCHEMA_MIGRATIONS_DDL)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at, description, checksum) "
                    "VALUES (1, ?, ?, ?)",
                    (
                        _now(),
                        "legacy schema deployed before migration history existed",
                        _LEGACY_BASELINE_MARKER,
                    ),
                )
                for version in range(2, CURRENT_VERSION + 1):
                    migration = MIGRATIONS[version]
                    migration.upgrade(connection)
                    connection.execute(
                        "INSERT INTO schema_migrations(version, applied_at, description, checksum) "
                        "VALUES (?, ?, ?, ?)",
                        (migration.version, _now(), migration.description, migration.checksum),
                    )
            except Exception:
                connection.execute("ROLLBACK")
                raise
            else:
                connection.execute("COMMIT")
            return

        if _looks_like_v2(connection, existing_tables):
            # A development checkout that already ran `schema.sql` from
            # this branch before this migration runner existed. It is
            # structurally at CURRENT_VERSION already, so adopt it with a
            # marker row instead of refusing it or trying to "migrate" a
            # database that has nothing left to migrate.
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(_SCHEMA_MIGRATIONS_DDL)
                connection.execute(
                    "INSERT INTO schema_migrations(version, applied_at, description, checksum) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        CURRENT_VERSION,
                        _now(),
                        "adopted a pre-existing current-schema database with no migration history",
                        _ADOPTED_MARKER,
                    ),
                )
            except Exception:
                connection.execute("ROLLBACK")
                raise
            else:
                connection.execute("COMMIT")
            return

        raise MigrationError(
            f"{path}: database has no migration history and does not match "
            "a recognized schema shape (legacy version 1 or the current "
            "version); refusing to guess. If this database is known-safe, "
            "stamp it manually -- see README.md's migration section."
        )
    finally:
        connection.close()
