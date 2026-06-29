"""KNN query helpers for the atoms_vec virtual table.

query_ready(conn) -> bool
    True iff sqlite-vec is loaded AND build status is 'complete'.

knn(conn, query_vec, project_id, top_k) -> list[tuple[int, float]]
    Returns [(atoms.rowid, dot_score)] ordered by dot_score DESC
    (COSINE/dot scale — higher is better — compatible with R3 fusion).

    Vec0 uses L2 distance; for L2-normalized vectors:
        dot = 1 - dist² / 2
    so ordering by L2-distance ASC == ordering by dot DESC.
"""

from __future__ import annotations

import sqlite3
import struct

from quipu.vec._gate import is_loaded
from quipu.vec._meta import is_build_complete

_EMBED_DIMS = 384
_EMBED_FMT = f"<{_EMBED_DIMS}f"


def query_ready(conn: sqlite3.Connection) -> bool:
    """Return True iff the vec path is usable on this connection.

    Checks both:
    1. sqlite-vec extension is live (vec_version() callable).
    2. ember_vec_meta.key='build' status == 'complete'.
    """
    return is_loaded(conn) and is_build_complete(conn)


def _pack_query(query_vec: list[float]) -> bytes:
    """Pack a 384-dim float list to LE bytes for sqlite-vec."""
    return struct.pack(_EMBED_FMT, *query_vec)


def knn(
    conn: sqlite3.Connection,
    query_vec: list[float],
    project_id: str,
    top_k: int,
) -> list[tuple[int, float]]:
    """Return top_k (rowid, dot_score) pairs via atoms_vec KNN.

    Only returns rowids whose atoms row belongs to *project_id* and is not
    invalidated. Scores are in cosine/dot scale (higher == more relevant).

    Raises sqlite3.OperationalError if atoms_vec is not available.
    """
    blob = _pack_query(query_vec)

    # Fetch 10× more candidates than top_k because the project/invalidated
    # JOIN filter may drop rows.  The LIMIT is bound to fetch_k (the full
    # buffer), NOT top_k — binding LIMIT to top_k would under-deliver when
    # the filter discards rows before the Python slice.  The caller (this
    # function) slices to top_k after filtering.
    fetch_k = min(top_k * 10, 10_000)

    rows = conn.execute(
        """
        SELECT v.rowid, v.distance
        FROM   atoms_vec v
        JOIN   atoms a ON a.rowid = v.rowid
        WHERE  v.embedding MATCH ?
          AND  k = ?
          AND  a.project_id = ?
          AND  a.invalidated = 0
        ORDER  BY v.distance ASC
        LIMIT  ?
        """,
        (blob, fetch_k, project_id, fetch_k),
    ).fetchall()

    results: list[tuple[int, float]] = []
    for rowid, dist in rows:
        # Convert L2-distance² to dot score: dot = 1 - dist²/2
        # (valid for L2-normalized vectors; dist here is L2 distance, not dist²)
        dot_score = 1.0 - (dist * dist) / 2.0
        results.append((rowid, dot_score))

    return results[:top_k]
