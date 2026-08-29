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
MAX_GRAPH_DEPTH = 10
MAX_GRAPH_TRIPLE_NEIGHBOURS = 32


# ---------------------------------------------------------------------------
# Embedding helpers (stdlib only — no numpy)
# ---------------------------------------------------------------------------
# Dimension is derived from the active model at CALL time (not import time),
# so a QUIPU_EMBEDDING_MODEL change is honored without reimporting.


def pack_embedding(vec: list[float]) -> bytes:
    """Pack a float32 list (active model's dim) to little-endian BLOB bytes."""
    from quipu.models.cache import active_dim

    dim = active_dim()
    if len(vec) != dim:
        raise ValueError(f"expected {dim} dims, got {len(vec)}")
    return struct.pack(f"<{dim}f", *vec)


def unpack_embedding(blob: bytes) -> list[float]:
    """Unpack a BLOB to a float32 list of the active model's dim."""
    from quipu.models.cache import active_dim

    dim = active_dim()
    if len(blob) != dim * 4:
        raise ValueError(f"expected {dim * 4} bytes, got {len(blob)}")
    return list(struct.unpack(f"<{dim}f", blob))


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
                "ORDER BY created_at DESC, rowid DESC"
            )
            params: tuple = (project_id,)
        else:
            sql = (
                "SELECT * FROM atoms WHERE project_id = ? AND invalidated = 0 "
                "ORDER BY created_at DESC, rowid DESC"
            )
            params = (project_id,)

        if limit is not None:
            if not isinstance(limit, int) or limit < 1:
                raise ValueError(f"limit must be a positive int, got {limit!r}")
            sql += f" LIMIT {int(limit)}"

        rows = self._conn.execute(sql, params).fetchall()
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

    def increment_access_batch(self, atom_ids: list[str]) -> int:
        """Increment access_count and set last_accessed to now for all *atom_ids*
        in a single UPDATE + single commit.

        Returns the number of rows updated. No-op (returns 0) if *atom_ids* is
        empty.
        """
        if not atom_ids:
            return 0
        placeholders = ",".join("?" for _ in atom_ids)
        cur = self._conn.execute(
            "UPDATE atoms SET access_count = access_count + 1, "
            "last_accessed = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
            f"WHERE id IN ({placeholders})",
            atom_ids,
        )
        self._conn.commit()
        return cur.rowcount

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
        if project_id is not None:
            endpoint_rows = self._conn.execute(
                "SELECT id, project_id FROM atoms WHERE id IN (?, ?)",
                (from_atom_id, to_atom_id),
            ).fetchall()
            endpoints = {row["id"]: row["project_id"] for row in endpoint_rows}
            if (
                endpoints.get(from_atom_id) != project_id
                or endpoints.get(to_atom_id) != project_id
            ):
                raise ValueError(
                    "edge endpoints must exist and belong to the supplied project_id"
                )
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
        as_of: "str | None" = None,
    ) -> "list[Atom]":
        """BFS from atom_id through KG edges and triples, excluding the start."""
        if max_depth < 1:
            return []
        if max_depth > MAX_GRAPH_DEPTH:
            raise ValueError(f"max_depth must not exceed {MAX_GRAPH_DEPTH}")

        visited: set[str] = {atom_id}
        frontier = {atom_id}
        all_connected: set[str] = set()

        for _ in range(max_depth):
            if not frontier:
                break
            next_frontier: set[str] = set()
            for fid in frontier:
                rows = self._edge_rows(
                    fid, project_id=project_id, edge_types=edge_types, as_of=as_of
                )
                for r in rows:
                    other = r["to_atom_id"] if r["from_atom_id"] == fid else r["from_atom_id"]
                    if other not in visited:
                        visited.add(other)
                        next_frontier.add(other)
                        all_connected.add(other)
                if not edge_types:
                    _, triple_neighbours = self._triple_neighbours(
                        fid, project_id=project_id, as_of=as_of
                    )
                    for other in triple_neighbours:
                        if other not in visited:
                            visited.add(other)
                            next_frontier.add(other)
                            all_connected.add(other)
            frontier = next_frontier

        if not all_connected:
            return []

        placeholders = ",".join("?" * len(all_connected))
        project_filter = " AND project_id = ?" if project_id is not None else ""
        params = tuple(all_connected)
        if project_id is not None:
            params += (project_id,)
        rows = self._conn.execute(
            f"SELECT * FROM atoms WHERE id IN ({placeholders}){project_filter} ORDER BY id",
            params,
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
        if max_depth > MAX_GRAPH_DEPTH:
            raise ValueError(f"max_depth must not exceed {MAX_GRAPH_DEPTH}")

        visited: set[str] = {atom_id}
        frontier = {atom_id}
        edge_ids: set[int | str] = set()
        edges_list: list[dict] = []
        connected_ids: set[str] = set()

        for _ in range(max_depth):
            if not frontier:
                break
            next_frontier: set[str] = set()
            for fid in frontier:
                rows = self._edge_rows(
                    fid, project_id=project_id, edge_types=edge_types, as_of=as_of
                )
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
                if not edge_types:
                    triple_relations, triple_neighbours = self._triple_neighbours(
                        fid, project_id=project_id, as_of=as_of
                    )
                    for relation in triple_relations:
                        if relation["id"] not in edge_ids:
                            edge_ids.add(relation["id"])
                            edges_list.append(relation)
                    for other in triple_neighbours:
                        if other not in visited:
                            visited.add(other)
                            next_frontier.add(other)
                            connected_ids.add(other)
            frontier = next_frontier

        nodes: list[dict] = []
        if connected_ids or atom_id:
            all_ids = connected_ids | {atom_id}
            placeholders = ",".join("?" * len(all_ids))
            project_filter = " AND project_id = ?" if project_id is not None else ""
            params = tuple(all_ids)
            if project_id is not None:
                params += (project_id,)
            rows = self._conn.execute(
                f"SELECT * FROM atoms WHERE id IN ({placeholders}){project_filter} ORDER BY id",
                params,
            ).fetchall()
            atoms = [_row_to_atom(r) for r in rows]
            nodes = [_atom_to_dict(a) for a in atoms]

        return {
            "nodes": sorted(nodes, key=lambda node: node["id"]),
            "edges": sorted(edges_list, key=lambda edge: str(edge["id"])),
        }

    def _edge_rows(
        self,
        atom_id: str,
        *,
        project_id: "str | None",
        edge_types: "list[str] | None",
        as_of: "str | None",
    ) -> list[sqlite3.Row]:
        """Return KG edges incident to *atom_id*, respecting graph time/scope."""
        clauses = ["(from_atom_id = ? OR to_atom_id = ?)"]
        params: tuple = (atom_id, atom_id)
        if edge_types:
            clauses.append(f"edge_type IN ({','.join('?' * len(edge_types))})")
            params += tuple(edge_types)
        if as_of is not None:
            clauses.append(
                "COALESCE(CASE WHEN json_valid(metadata) "
                "THEN json_extract(metadata, '$.valid_from') END, created_at) <= ?"
            )
            clauses.append(
                "(CASE WHEN json_valid(metadata) "
                "THEN json_extract(metadata, '$.valid_to') END IS NULL "
                "OR CASE WHEN json_valid(metadata) "
                "THEN json_extract(metadata, '$.valid_to') END > ?)"
            )
            params += (as_of, as_of)
        if project_id is not None:
            clauses.append("project_id = ?")
            params += (project_id,)
        return self._conn.execute(
            "SELECT * FROM kg_edges WHERE " + " AND ".join(clauses) + " ORDER BY id", params
        ).fetchall()

    def _triple_neighbours(
        self,
        atom_id: str,
        *,
        project_id: "str | None",
        as_of: "str | None",
    ) -> tuple[list[dict], set[str]]:
        """Return provenance-correct triple relations plus reachable atom IDs."""
        project_filter = " AND project_id = ?" if project_id is not None else ""
        temporal_filter = ""
        temporal_params: tuple[str, ...] = ()
        if as_of is not None:
            temporal_filter = " AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)"
            temporal_params = (as_of, as_of)

        direct_params: tuple = (atom_id, atom_id, *temporal_params)
        if project_id is not None:
            direct_params += (project_id,)
        rows = self._conn.execute(
            "SELECT * FROM kg_triples WHERE (subject = ? OR object = ?)"
            f"{temporal_filter}{project_filter} ORDER BY id",
            direct_params,
        ).fetchall()

        entity_params: tuple = (atom_id, *temporal_params)
        if project_id is not None:
            entity_params += (project_id,)
        entity_rows = self._conn.execute(
            "SELECT DISTINCT subject FROM kg_triples WHERE object = ?"
            f"{temporal_filter}{project_filter} ORDER BY subject",
            entity_params,
        ).fetchall()
        subjects = [row["subject"] for row in entity_rows]
        if subjects:
            placeholders = ",".join("?" * len(subjects))
            shared_params: tuple = (*subjects, *temporal_params)
            if project_id is not None:
                shared_params += (project_id,)
            rows += self._conn.execute(
                f"SELECT * FROM kg_triples WHERE subject IN ({placeholders})"
                f"{temporal_filter}{project_filter} ORDER BY id",
                shared_params,
            ).fetchall()

        unique_rows = list({row["id"]: row for row in rows}.values())
        candidate_ids: set[str] = set()
        for row in unique_rows:
            if row["subject"] == atom_id:
                candidate_ids.add(row["object"])
            elif row["object"] == atom_id:
                candidate_ids.add(row["subject"])
            elif row["subject"] in subjects:
                candidate_ids.add(row["object"])
        candidate_ids.discard(atom_id)
        if not candidate_ids:
            return [], set()

        placeholders = ",".join("?" * len(candidate_ids))
        atom_params: tuple = tuple(candidate_ids)
        if project_id is not None:
            atom_params += (project_id,)
        atoms = self._conn.execute(
            f"SELECT id FROM atoms WHERE id IN ({placeholders})"
            f"{' AND project_id = ?' if project_id is not None else ''} ORDER BY id LIMIT ?",
            atom_params + (MAX_GRAPH_TRIPLE_NEIGHBOURS,),
        ).fetchall()
        atom_ids = {row["id"] for row in atoms}
        relations: list[dict] = []
        rows_by_subject: dict[str, list[sqlite3.Row]] = {}
        for row in unique_rows:
            rows_by_subject.setdefault(row["subject"], []).append(row)

        for source in unique_rows:
            if source["subject"] == atom_id and source["object"] in atom_ids:
                relations.append({
                    "id": f"triple:{source['id']}",
                    "kind": "triple",
                    "from_atom_id": atom_id,
                    "to_atom_id": source["object"],
                    "metadata": {"predicate": source["predicate"], "triple_id": source["id"]},
                })
                continue
            if source["object"] == atom_id and source["subject"] in atom_ids:
                relations.append({
                    "id": f"triple:{source['id']}",
                    "kind": "triple",
                    "from_atom_id": atom_id,
                    "to_atom_id": source["subject"],
                    "metadata": {"predicate": source["predicate"], "triple_id": source["id"]},
                })
                continue
            if source["object"] != atom_id:
                continue
            for target in rows_by_subject.get(source["subject"], []):
                other = target["object"]
                if other == atom_id or other not in atom_ids:
                    continue
                relations.append({
                    "id": f"triple:{source['id']}:{target['id']}",
                    "kind": "triple",
                    "from_atom_id": atom_id,
                    "to_atom_id": other,
                    "metadata": {
                        "subject": source["subject"],
                        "predicate": source["predicate"],
                        "source_triple_id": source["id"],
                        "target_triple_id": target["id"],
                        "source_ref": source["source_ref"],
                        "valid_from": source["valid_from"],
                        "valid_to": source["valid_to"],
                        "confidence": source["confidence"],
                    },
                })
        return relations, atom_ids


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
