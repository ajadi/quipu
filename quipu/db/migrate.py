"""Migration runner for Quipu SQLite DB.

MIGRATIONS: ordered list of migration modules, each exporting VERSION/UP/DOWN.
init_db(conn): apply all pending migrations using PRAGMA user_version.
"""

import sqlite3

from quipu.db.migrations import _migration_0001, _migration_0002, _migration_0003, _migration_0004, _migration_0005, _migration_0006

# Ordered list of migration modules; add future migrations here in order.
MIGRATIONS = [
    _migration_0001,
    _migration_0002,
    _migration_0003,
    _migration_0004,
    _migration_0005,
    _migration_0006,
]


def init_db(conn: sqlite3.Connection) -> None:
    """Apply all pending migrations to *conn*.

    Uses PRAGMA user_version as the schema version counter.
    Migrations are idempotent (not atomic): all DDL uses IF NOT EXISTS, so a
    crash mid-migration is safely re-applied on next init.
    """
    current: int = conn.execute("PRAGMA user_version").fetchone()[0]
    for migration in MIGRATIONS:
        if migration.VERSION > current:
            # executescript() implicitly commits any open transaction before running.
            conn.executescript(migration.UP)
            conn.execute(f"PRAGMA user_version = {migration.VERSION}")
            conn.commit()
            current = migration.VERSION
