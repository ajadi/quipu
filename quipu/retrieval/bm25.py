"""BM25 ranking via SQLite FTS5 for a list of Atoms.

FTS5's built-in bm25() function returns NEGATIVE scores (more relevant =
more negative). We negate them so higher == more relevant, consistent with
the R1 cosine signal.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quipu.storage.store import Atom


def bm25_rank(atoms: "list[Atom]", query: str) -> "dict[str, float]":
    """Rank atoms against query using FTS5 BM25.

    Builds an in-memory FTS5 virtual table from the provided atoms' content,
    runs a MATCH query, and returns per-atom scores (negated so higher is
    better).

    Args:
        atoms: Atoms scoped to the target project (non-invalidated).
        query: The search query string.

    Returns:
        Mapping of atom.id → positive BM25 score for matching atoms.
        Atoms that do not match the query are absent from the result.

    Raises:
        RuntimeError: If FTS5 is not available in the current sqlite3 build.
    """
    if not atoms:
        return {}

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row

    # Verify FTS5 is available
    try:
        conn.execute("CREATE VIRTUAL TABLE _fts5_check USING fts5(x)")
        conn.execute("DROP TABLE _fts5_check")
    except sqlite3.OperationalError as exc:
        conn.close()
        raise RuntimeError(
            "FTS5 is not available in this sqlite3 build. "
            "TASK-003 requires FTS5 bm25() for BM25 ranking."
        ) from exc

    # Create FTS5 table with a rowid→atom_id mapping table
    conn.execute(
        "CREATE TABLE atom_map (rowid INTEGER PRIMARY KEY, atom_id TEXT NOT NULL)"
    )
    conn.execute("CREATE VIRTUAL TABLE atom_fts USING fts5(content, content='')")

    # Populate
    for i, atom in enumerate(atoms, start=1):
        conn.execute(
            "INSERT INTO atom_map (rowid, atom_id) VALUES (?, ?)",
            (i, atom.id),
        )
        conn.execute(
            "INSERT INTO atom_fts (rowid, content) VALUES (?, ?)",
            (i, atom.content),
        )
    conn.commit()

    # Query FTS5; bm25() returns negative — negate so higher == more relevant
    try:
        rows = conn.execute(
            """
            SELECT m.atom_id, -bm25(atom_fts) AS score
            FROM atom_fts
            JOIN atom_map AS m ON m.rowid = atom_fts.rowid
            WHERE atom_fts MATCH ?
            ORDER BY score DESC
            """,
            (query,),
        ).fetchall()
    except sqlite3.OperationalError:
        # Query parse error (e.g. empty string, special chars) → no results
        conn.close()
        return {}

    conn.close()
    return {row["atom_id"]: row["score"] for row in rows}
