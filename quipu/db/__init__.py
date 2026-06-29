"""Quipu DB package — connection factory."""

import sqlite3
from pathlib import Path

from quipu.db.migrate import init_db


def get_connection(path: str | Path) -> sqlite3.Connection:
    """Return an initialized sqlite3.Connection for the DB at *path*.

    Opens (creates if absent) the SQLite file, applies pending migrations,
    and returns the connection ready for use.

    Settings applied:
    - row_factory = sqlite3.Row  (column-name access)
    - PRAGMA journal_mode = WAL  (concurrent readers)
    - PRAGMA foreign_keys = ON
    """
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    init_db(conn)
    return conn
