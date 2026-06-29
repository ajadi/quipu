"""Quipu storage: Atom dataclass, Store CRUD class, and embedding helpers."""

import json
import re
import sqlite3
import struct
import uuid
from dataclasses import dataclass, field
from typing import Any

# ISO-8601 UTC format accepted by insert(created_at=...)
_ISO8601_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")


# ---------------------------------------------------------------------------
# Embedding helpers (stdlib only — no numpy)
# ---------------------------------------------------------------------------

_EMBEDDING_DIMS = 384
_EMBEDDING_FORMAT = f"<{_EMBEDDING_DIMS}f"
_EMBEDDING_BYTES = _EMBEDDING_DIMS * 4  # 1536 bytes


def pack_embedding(vec: list[float]) -> bytes:
    """Pack a 384-dim float32 list to little-endian BLOB bytes."""
    if len(vec) != _EMBEDDING_DIMS:
        raise ValueError(f"expected {_EMBEDDING_DIMS} dims, got {len(vec)}")
    return struct.pack(_EMBEDDING_FORMAT, *vec)


def unpack_embedding(blob: bytes) -> list[float]:
    """Unpack a 1536-byte BLOB to a 384-dim float32 list."""
    if len(blob) != _EMBEDDING_BYTES:
        raise ValueError(f"expected {_EMBEDDING_BYTES} bytes, got {len(blob)}")
    return list(struct.unpack(_EMBEDDING_FORMAT, blob))


# ---------------------------------------------------------------------------
# Atom dataclass
# ---------------------------------------------------------------------------

@dataclass
class Atom:
    id: str
    content: str
    embedding: bytes | None
    project_id: str | None
    type: str
    scope: str
    metadata: dict
    refs: list
    invalidated: bool
    created_at: str
    updated_at: str
    session_id: str | None = None
    access_count: int = 0
    last_accessed: str | None = None
    tags: list[str] | None = None


def _row_to_atom(row: sqlite3.Row) -> Atom:
    keys = row.keys()
    tags_raw = row["tags"] if "tags" in keys else None
    tags = json.loads(tags_raw) if tags_raw else None
    return Atom(
        id=row["id"],
        content=row["content"],
        embedding=row["embedding"],
        project_id=row["project_id"],
        type=row["type"],
        scope=row["scope"],
        metadata=json.loads(row["metadata"]) if row["metadata"] else {},
        refs=json.loads(row["refs"]) if row["refs"] else [],
        invalidated=bool(row["invalidated"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        session_id=row["session_id"] if "session_id" in keys else None,
        access_count=row["access_count"] if "access_count" in keys else 0,
        last_accessed=row["last_accessed"] if "last_accessed" in keys else None,
        tags=tags,
    )


# ---------------------------------------------------------------------------
# Store class
# ---------------------------------------------------------------------------

class Store:
    """Thin repository layer over the Quipu SQLite DB.

    Caller owns the connection lifetime; use as a context manager or call
    close() explicitly.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    # --- context manager ---

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """Close the underlying connection."""
        self._conn.close()

    # --- CRUD ---

    def insert(
        self,
        *,
        content: str,
        embedding: bytes | None = None,
        project_id: str | None = None,
        type: str = "diary",
        scope: str = "project",
        metadata: dict | None = None,
        refs: list | None = None,
        id: str | None = None,
        created_at: "str | None" = None,
        session_id: "str | None" = None,
        tags: list[str] | None = None,
    ) -> "Atom":
        """Insert a new atom and return the populated Atom dataclass.

        Args:
            created_at: Optional ISO-8601 UTC timestamp to use for both
                        created_at and updated_at columns (preserves event-time
                        ordering when draining a capture queue). When None, both
                        columns use the SQL DEFAULT (strftime('now')).
            session_id: Optional session identifier for grouping atoms by
                        capture session. NULL = ungrouped (backward-compat).
            tags: Optional list of string tags stored as JSON array text.
                  None → NULL (backward-compat for pre-0005 call sites).
        """
        if created_at is not None and not _ISO8601_UTC_RE.match(created_at):
            raise ValueError(
                "created_at must be ISO-8601 UTC (YYYY-MM-DDTHH:MM:SS[.fff]Z)"
            )

        atom_id = id if id is not None else uuid.uuid4().hex
        meta_json = json.dumps(metadata) if metadata is not None else json.dumps({})
        refs_json = json.dumps(refs) if refs is not None else json.dumps([])
        tags_json = json.dumps(tags) if tags is not None else None

        if created_at is not None:
            self._conn.execute(
                """
                INSERT INTO atoms (id, type, scope, content, embedding, metadata,
                                   project_id, refs, invalidated,
                                   created_at, updated_at, session_id, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?)
                """,
                (atom_id, type, scope, content, embedding, meta_json,
                 project_id, refs_json, created_at, created_at, session_id, tags_json),
            )
        else:
            self._conn.execute(
                """
                INSERT INTO atoms (id, type, scope, content, embedding, metadata,
                                   project_id, refs, invalidated, session_id, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?, ?)
                """,
                (atom_id, type, scope, content, embedding, meta_json,
                 project_id, refs_json, session_id, tags_json),
            )
        self._conn.commit()
        return self.get(atom_id)  # type: ignore[return-value]  # just inserted

    def get(self, atom_id: str) -> "Atom | None":
        """Return the Atom with the given id, or None if not found."""
        row = self._conn.execute(
            "SELECT * FROM atoms WHERE id = ?", (atom_id,)
        ).fetchone()
        return _row_to_atom(row) if row else None

    def update_invalidated(self, atom_id: str, invalidated: bool = True) -> bool:
        """Set the invalidated flag on an atom.

        Returns True if the atom existed (was updated), False otherwise.
        """
        cur = self._conn.execute(
            "UPDATE atoms SET invalidated = ? WHERE id = ?",
            (1 if invalidated else 0, atom_id),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def delete(self, atom_id: str) -> bool:
        """Permanently delete an atom by id.

        Returns True if the atom existed, False otherwise.
        """
        cur = self._conn.execute("DELETE FROM atoms WHERE id = ?", (atom_id,))
        self._conn.commit()
        return cur.rowcount > 0

    def list_by_project(
        self,
        project_id: str,
        *,
        include_invalidated: bool = True,
        limit: int | None = None,
    ) -> list["Atom"]:
        """Return atoms for a given project_id, ordered by created_at DESC."""
        if include_invalidated:
            sql = (
                "SELECT * FROM atoms WHERE project_id = ? "
                "ORDER BY created_at DESC"
            )
            params: tuple = (project_id,)
        else:
            sql = (
                "SELECT * FROM atoms WHERE project_id = ? AND invalidated = 0 "
                "ORDER BY created_at DESC"
            )
            params = (project_id,)

        if limit is not None:
            if not isinstance(limit, int) or limit < 1:
                raise ValueError(f"limit must be a positive int, got {limit!r}")
            sql += f" LIMIT {int(limit)}"

        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_atom(r) for r in rows]

    def list_by_session(
        self,
        project_id: str,
        session_id: str,
        *,
        include_invalidated: bool = True,
        limit: int | None = None,
    ) -> list["Atom"]:
        """Return atoms for a given project_id and session_id, ordered by created_at DESC."""
        if include_invalidated:
            sql = (
                "SELECT * FROM atoms WHERE project_id = ? AND session_id = ? "
                "ORDER BY created_at DESC"
            )
        else:
            sql = (
                "SELECT * FROM atoms WHERE project_id = ? AND session_id = ? "
                "AND invalidated = 0 ORDER BY created_at DESC"
            )

        if limit is not None:
            if not isinstance(limit, int) or limit < 1:
                raise ValueError(f"limit must be a positive int, got {limit!r}")
            sql += f" LIMIT {int(limit)}"

        rows = self._conn.execute(sql, (project_id, session_id)).fetchall()
        return [_row_to_atom(r) for r in rows]

    def increment_access(self, atom_id: str) -> bool:
        """Increment access_count and set last_accessed to now.

        Returns True if the atom existed, False otherwise.
        """
        cur = self._conn.execute(
            "UPDATE atoms SET access_count = access_count + 1, "
            "last_accessed = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
            "WHERE id = ?",
            (atom_id,),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def list_stale(
        self,
        project_id: str,
        *,
        min_age_days: int = 90,
        min_access_count: int = 3,
    ) -> list["Atom"]:
        """Return non-invalidated atoms older than *min_age_days* and with
        access_count < *min_access_count*.

        GC candidates — soft-invalidation is opt-in, never hard-delete.
        """
        if not isinstance(min_age_days, int) or min_age_days < 0:
            raise ValueError(f"min_age_days must be a non-negative int, got {min_age_days!r}")
        if not isinstance(min_access_count, int) or min_access_count < 0:
            raise ValueError(f"min_access_count must be a non-negative int, got {min_access_count!r}")

        cutoff = f"strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-{int(min_age_days)} days')"
        sql = (
            "SELECT * FROM atoms WHERE project_id = ? AND invalidated = 0 "
            f"AND created_at <= {cutoff} "
            "AND access_count < ? "
            "ORDER BY access_count ASC, created_at ASC"
        )
        rows = self._conn.execute(sql, (project_id, int(min_access_count))).fetchall()
        return [_row_to_atom(r) for r in rows]

    # --- Knowledge Graph ---

    def insert_triple(
        self,
        *,
        subject: str,
        predicate: str,
        object: str,
        valid_from: "str | None" = None,
        valid_to: "str | None" = None,
        confidence: float = 1.0,
        source_ref: "str | None" = None,
        project_id: "str | None" = None,
    ) -> dict:
        """Insert a KG triple. Returns a dict with the row fields."""
        self._conn.execute(
            """
            INSERT INTO kg_triples (subject, predicate, object, valid_from, valid_to,
                                     confidence, source_ref, project_id)
            VALUES (?, ?, ?,
                    COALESCE(?, strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                    ?, ?, ?, ?)
            """,
            (subject, predicate, object, valid_from, valid_to,
             confidence, source_ref, project_id),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT * FROM kg_triples WHERE rowid = last_insert_rowid()"
        ).fetchone()
        return dict(row)

    def insert_edge(
        self,
        *,
        from_atom_id: str,
        to_atom_id: str,
        edge_type: str,
        project_id: "str | None" = None,
        metadata: "dict | None" = None,
    ) -> dict:
        """Insert a typed edge between two atoms. Returns a dict with the row fields."""
        meta_json = json.dumps(metadata) if metadata is not None else None
        self._conn.execute(
            """
            INSERT INTO kg_edges (from_atom_id, to_atom_id, edge_type, project_id, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            (from_atom_id, to_atom_id, edge_type, project_id, meta_json),
        )
        self._conn.commit()
        row = self._conn.execute(
            "SELECT * FROM kg_edges WHERE rowid = last_insert_rowid()"
        ).fetchone()
        d = dict(row)
        if d.get("metadata"):
            d["metadata"] = json.loads(d["metadata"])
        return d

    def get_connected_atoms(
        self,
        atom_id: str,
        *,
        project_id: "str | None" = None,
        max_depth: int = 2,
        edge_types: "list[str] | None" = None,
    ) -> "list[Atom]":
        """BFS from atom_id through kg_edges; return reachable atoms (excluding start)."""
        if max_depth < 1:
            return []

        visited: set[str] = {atom_id}
        frontier = {atom_id}
        all_connected: set[str] = set()

        for _ in range(max_depth):
            if not frontier:
                break
            next_frontier: set[str] = set()
            for fid in frontier:
                rows: list[sqlite3.Row] = []
                if edge_types:
                    placeholders = ",".join("?" * len(edge_types))
                    rows = self._conn.execute(
                        f"SELECT from_atom_id, to_atom_id FROM kg_edges "
                        f"WHERE (from_atom_id = ? OR to_atom_id = ?) "
                        f"AND edge_type IN ({placeholders})",
                        (fid, fid, *edge_types),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT from_atom_id, to_atom_id FROM kg_edges "
                        "WHERE from_atom_id = ? OR to_atom_id = ?",
                        (fid, fid),
                    ).fetchall()
                for r in rows:
                    other = r["to_atom_id"] if r["from_atom_id"] == fid else r["from_atom_id"]
                    if other not in visited:
                        visited.add(other)
                        next_frontier.add(other)
                        all_connected.add(other)
            frontier = next_frontier

        if not all_connected:
            return []

        placeholders = ",".join("?" * len(all_connected))
        rows = self._conn.execute(
            f"SELECT * FROM atoms WHERE id IN ({placeholders})",
            tuple(all_connected),
        ).fetchall()
        return [_row_to_atom(r) for r in rows]

    def traverse(
        self,
        atom_id: str,
        *,
        project_id: "str | None" = None,
        max_depth: int = 2,
        edge_types: "list[str] | None" = None,
        as_of: "str | None" = None,
    ) -> dict:
        """Return subgraph {nodes: [Atom dicts], edges: [edge dicts]} from BFS."""
        if max_depth < 1:
            return {"nodes": [], "edges": []}

        visited: set[str] = {atom_id}
        frontier = {atom_id}
        edge_ids: set[int] = set()
        edges_list: list[dict] = []
        connected_ids: set[str] = set()

        for _ in range(max_depth):
            if not frontier:
                break
            next_frontier: set[str] = set()
            for fid in frontier:
                rows: list[sqlite3.Row] = []
                if edge_types:
                    placeholders = ",".join("?" * len(edge_types))
                    rows = self._conn.execute(
                        f"SELECT * FROM kg_edges "
                        f"WHERE (from_atom_id = ? OR to_atom_id = ?) "
                        f"AND edge_type IN ({placeholders})",
                        (fid, fid, *edge_types),
                    ).fetchall()
                else:
                    rows = self._conn.execute(
                        "SELECT * FROM kg_edges "
                        "WHERE from_atom_id = ? OR to_atom_id = ?",
                        (fid, fid),
                    ).fetchall()
                for r in rows:
                    if r["id"] not in edge_ids:
                        edge_ids.add(r["id"])
                        edge_dict = dict(r)
                        if edge_dict.get("metadata"):
                            edge_dict["metadata"] = json.loads(edge_dict["metadata"])
                        edges_list.append(edge_dict)
                    other = r["to_atom_id"] if r["from_atom_id"] == fid else r["from_atom_id"]
                    if other not in visited:
                        visited.add(other)
                        next_frontier.add(other)
                        connected_ids.add(other)
            frontier = next_frontier

        nodes: list[dict] = []
        if connected_ids or atom_id:
            all_ids = connected_ids | {atom_id}
            placeholders = ",".join("?" * len(all_ids))
            rows = self._conn.execute(
                f"SELECT * FROM atoms WHERE id IN ({placeholders})",
                tuple(all_ids),
            ).fetchall()
            atoms = [_row_to_atom(r) for r in rows]
            nodes = [_atom_to_dict(a) for a in atoms]

        return {"nodes": nodes, "edges": edges_list}


def _atom_to_dict(atom: "Atom") -> dict:
    return {
        "id": atom.id,
        "content": atom.content,
        "project_id": atom.project_id,
        "type": atom.type,
        "scope": atom.scope,
        "metadata": atom.metadata,
        "refs": atom.refs,
        "invalidated": atom.invalidated,
        "created_at": atom.created_at,
        "updated_at": atom.updated_at,
        "session_id": atom.session_id,
        "tags": atom.tags,
    }
