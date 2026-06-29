"""Quipu storage public API.

Usage:
    from quipu.storage import store
    s = store()          # opens ~/.quipu/quipu.db, inits schema
    s = store("/tmp/x.db")   # explicit path
"""

from quipu.db import get_connection
from quipu.storage.paths import resolve_db_path
from quipu.storage.store import Atom, Store, pack_embedding, unpack_embedding


def store(db_path: str | None = None) -> Store:
    """Open (and init-if-absent) the Quipu store.

    AC: ``from quipu.storage import store; s = store()``

    Args:
        db_path: Optional explicit path to the SQLite file. If None, path is
                 resolved via resolve_db_path() (env var / default).

    Returns:
        An initialized Store instance.
    """
    path = resolve_db_path(db_path)
    conn = get_connection(path)
    return Store(conn)


__all__ = ["store", "Store", "Atom", "pack_embedding", "unpack_embedding"]
