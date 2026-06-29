"""Quipu MCP tool definitions and dispatch logic.

All 7 tool handlers are pure functions (store, default_project_id, arguments)
so unit tests can inject a temp Store directly without subprocess/asyncio.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from mcp.types import TextContent, Tool

from quipu.storage import Store
from quipu.write import flush, write

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal error types
# ---------------------------------------------------------------------------


class _ToolError(Exception):
    """Validation / project_id-missing errors — captured by dispatch."""


# ---------------------------------------------------------------------------
# atom → dict helper (embedding NEVER serialized)
# ---------------------------------------------------------------------------


def _atom_to_dict(atom: Any) -> dict:
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


# ---------------------------------------------------------------------------
# project_id resolution helper
# ---------------------------------------------------------------------------


def _resolve_project_id(arguments: dict, default_project_id: str | None) -> str | None:
    if "project_id" in arguments:
        supplied = arguments["project_id"]
        if default_project_id is not None:
            # Server is bound to a project — enforce scope.
            if supplied is not None and not isinstance(supplied, str):
                raise _ToolError("project_id must be a string")
            if supplied != default_project_id:
                raise _ToolError(
                    f"project_id {supplied!r} is not permitted: server is bound to"
                    f" project {default_project_id!r}"
                )
        return supplied
    return default_project_id


# ---------------------------------------------------------------------------
# Tool handlers
# ---------------------------------------------------------------------------


def _handle_quipu_write(
    store: Store, default_project_id: str | None, arguments: dict
) -> dict:
    content = arguments.get("content")
    if not isinstance(content, str) or not content:
        raise _ToolError("content is required and must be a non-empty string")
    if len(content) > 100_000:
        raise _ToolError("content exceeds maximum length of 100000 characters")
    metadata = arguments.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise _ToolError("metadata must be an object")
    project_id = _resolve_project_id(arguments, default_project_id)
    atom_id = write(content, metadata, project_id, store=store)

    conflicts: list[dict] = []
    if project_id is not None:
        from quipu.storage.store import unpack_embedding
        from quipu.invalidation import find_conflicts
        new_atom = store.get(atom_id)
        if new_atom is not None and new_atom.embedding is not None:
            new_vec = unpack_embedding(new_atom.embedding)
            existing = store.list_by_project(project_id, include_invalidated=False)
            conflicts = find_conflicts(new_vec, existing, exclude_id=atom_id)

    return {"id": atom_id, "conflicts": conflicts}


def _handle_quipu_search(
    store: Store, default_project_id: str | None, arguments: dict
) -> dict:
    from quipu.retrieval import search

    query = arguments.get("query")
    if not isinstance(query, str) or not query:
        raise _ToolError("query is required and must be a non-empty string")
    tier = arguments.get("tier", "R3")
    if not isinstance(tier, str):
        raise _ToolError("tier must be a string")
    top_k = arguments.get("top_k", 10)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise _ToolError("top_k must be an integer >= 1")
    if top_k > 1000:
        raise _ToolError("top_k must be <= 1000")
    project_id = _resolve_project_id(arguments, default_project_id)
    if project_id is None:
        raise _ToolError(
            "project_id is required for quipu_search — set project_id arg or QUIPU_PROJECT_ID env"
        )
    session_id = arguments.get("session_id")
    if session_id is not None and not isinstance(session_id, str):
        raise _ToolError("session_id must be a string")
    tags = arguments.get("tags")
    if tags is not None:
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise _ToolError("tags must be a list of strings")
    graph_expand = arguments.get("graph_expand", False)
    if not isinstance(graph_expand, bool):
        raise _ToolError("graph_expand must be a boolean")
    graph_depth = arguments.get("graph_depth", 1)
    if not isinstance(graph_depth, int) or isinstance(graph_depth, bool) or graph_depth < 1:
        raise _ToolError("graph_depth must be an integer >= 1")
    results = search(
        query, tier=tier, project_id=project_id, top_k=top_k, store=store,
        session_id=session_id, tags=tags,
        graph_expand=graph_expand, graph_depth=graph_depth,
    )
    return {
        "results": [
            {
                "id": r.atom.id,
                "content": r.atom.content,
                "score": r.score,
                "tier": r.tier,
                "type": r.atom.type,
                "scope": r.atom.scope,
                "invalidated": r.atom.invalidated,
                "metadata": r.atom.metadata,
            }
            for r in results
        ]
    }


def _handle_quipu_get(
    store: Store, default_project_id: str | None, arguments: dict
) -> dict:
    atom_id = arguments.get("id")
    if not isinstance(atom_id, str) or not atom_id:
        raise _ToolError("id is required and must be a non-empty string")
    project_id = _resolve_project_id(arguments, default_project_id)
    if project_id is None:
        raise _ToolError(
            "project_id is required for quipu_get — set project_id arg or QUIPU_PROJECT_ID env"
        )
    atom = store.get(atom_id)
    if atom is None or atom.project_id != project_id:
        return {"found": False}
    return _atom_to_dict(atom)


def _handle_quipu_list(
    store: Store, default_project_id: str | None, arguments: dict
) -> dict:
    project_id = _resolve_project_id(arguments, default_project_id)
    if project_id is None:
        raise _ToolError(
            "project_id is required for quipu_list — set project_id arg or QUIPU_PROJECT_ID env"
        )
    limit = arguments.get("limit")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise _ToolError("limit must be an integer >= 1")
    atoms = store.list_by_project(project_id, include_invalidated=True, limit=limit)
    return {"atoms": [_atom_to_dict(a) for a in atoms]}


def _handle_quipu_invalidate(
    store: Store, default_project_id: str | None, arguments: dict
) -> dict:
    atom_id = arguments.get("id")
    if not isinstance(atom_id, str) or not atom_id:
        raise _ToolError("id is required and must be a non-empty string")
    project_id = _resolve_project_id(arguments, default_project_id)
    if project_id is None:
        raise _ToolError(
            "project_id is required for quipu_invalidate — set project_id arg or QUIPU_PROJECT_ID env"
        )
    atom = store.get(atom_id)
    if atom is None or atom.project_id != project_id:
        return {"id": atom_id, "invalidated": False, "existed": False}
    store.update_invalidated(atom_id, True)
    # Re-fetch after update so the emit carries the post-invalidation updated_at
    # (the atoms_updated_at trigger bumps updated_at on every row update).
    updated_atom = store.get(atom_id)
    if updated_atom is None:
        # Concurrent delete raced our invalidate: row is gone.
        # Do NOT claim success — the oplog invalidate entry was never emitted,
        # so peers would never learn of this invalidation (sync divergence).
        return {"id": atom_id, "invalidated": False, "existed": False}
    from quipu.oplog.producer import emit
    emit(store, op="invalidate", atom=updated_atom, project_id=project_id)
    return {"id": atom_id, "invalidated": True, "existed": True}


def _handle_quipu_flush(
    store: Store, default_project_id: str | None, arguments: dict
) -> dict:
    project_id = _resolve_project_id(arguments, default_project_id)
    result = flush(project_id, store=store)
    out = {
        "enriched": result.get("enriched", 0),
        "skipped": result.get("skipped", False),
        "reason": result.get("reason"),
    }
    # After flush, trigger a push sync (best-effort; never raises).
    if project_id is not None:
        try:
            from quipu.sync.client import sync_now
            sr = sync_now(project_id, store=store, directions=("push",))
            out["sync_status"] = sr.status
            out["pushed"] = sr.pushed
        except Exception:
            logger.exception("quipu_flush: post-flush sync failed")
            out["sync_status"] = "offline"
            out["pushed"] = 0
    return out


def _handle_quipu_stats(
    store: Store, default_project_id: str | None, arguments: dict
) -> dict:
    project_id = _resolve_project_id(arguments, default_project_id)
    if project_id is None:
        raise _ToolError(
            "project_id is required for quipu_stats — set project_id arg or QUIPU_PROJECT_ID env"
        )
    atoms = store.list_by_project(project_id, include_invalidated=True)
    total = len(atoms)
    invalidated = sum(1 for a in atoms if a.invalidated)
    active = total - invalidated
    enriched_dates = [
        a.metadata["enriched_at"]
        for a in atoms
        if isinstance(a.metadata.get("enriched_at"), str)
    ]
    last_flush = max(enriched_dates) if enriched_dates else None

    # Sync status — never fails stats if sync info unavailable.
    sync_status = "never_configured"
    last_push: str | None = None
    last_pull: str | None = None
    try:
        from quipu.config import get_hub_config, get_client_id
        from quipu.sync.client import get_last_sync_status
        from quipu.sync.cursor import read_cursor_meta
        from quipu.sync._aad import aad_for

        cfg = get_hub_config()
        if cfg is None:
            sync_status = "never_configured"
        else:
            sync_status = get_last_sync_status()
            # Only attempt key derivation if a key source is present in the
            # environment — avoids a getpass.getpass() TTY hang when neither
            # QUIPU_KEY nor QUIPU_PASSPHRASE is set (mirrors sync_now guard).
            if os.environ.get("QUIPU_KEY") or os.environ.get("QUIPU_PASSPHRASE"):
                try:
                    from quipu.keystore._backend import get_or_derive_key
                    key = get_or_derive_key(project_id)
                    blinded = aad_for(project_id, key).decode()
                    client_id = get_client_id(store)
                    _, last_push = read_cursor_meta(store._conn, blinded, "push", client_id)
                    _, last_pull = read_cursor_meta(store._conn, blinded, "pull", client_id)
                except Exception:
                    pass  # key unavailable — leave last_push/last_pull as None
    except Exception:
        pass  # sync info always optional

    return {
        "total": total,
        "active": active,
        "invalidated": invalidated,
        "last_flush": last_flush,
        "sync_status": sync_status,
        "last_push": last_push,
        "last_pull": last_pull,
    }


def _handle_quipu_push(
    store: Store, default_project_id: str | None, arguments: dict
) -> dict:
    project_id = _resolve_project_id(arguments, default_project_id)
    if project_id is None:
        raise _ToolError(
            "project_id is required for quipu_push — set project_id arg or QUIPU_PROJECT_ID env"
        )
    from quipu.sync.client import sync_now
    sr = sync_now(project_id, store=store, directions=("push",))
    return {
        "sync_status": sr.status,
        "pushed": sr.pushed,
        "detail": sr.detail,
    }


def _handle_quipu_pull(
    store: Store, default_project_id: str | None, arguments: dict
) -> dict:
    project_id = _resolve_project_id(arguments, default_project_id)
    if project_id is None:
        raise _ToolError(
            "project_id is required for quipu_pull — set project_id arg or QUIPU_PROJECT_ID env"
        )
    from quipu.sync.client import sync_now
    sr = sync_now(project_id, store=store, directions=("pull",))
    return {
        "sync_status": sr.status,
        "pulled": sr.pulled,
        "detail": sr.detail,
    }


def _handle_quipu_prime(
    store: Store, default_project_id: str | None, arguments: dict
) -> dict:
    top_k = arguments.get("top_k", 8)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise _ToolError("top_k must be an integer >= 1")
    if top_k > 1000:
        raise _ToolError("top_k must be <= 1000")
    topic = arguments.get("topic")
    if topic is not None and not isinstance(topic, str):
        raise _ToolError("topic must be a string")
    if isinstance(topic, str) and len(topic) > 1000:
        raise _ToolError("topic exceeds 1000 characters")
    project_id = _resolve_project_id(arguments, default_project_id)  # may be None
    from quipu.behavior.prime import prime
    return prime(store, project_id, topic=topic, top_k=top_k)


def _handle_quipu_receipts(
    store: Store, default_project_id: str | None, arguments: dict
) -> dict:
    import hashlib

    project_id = _resolve_project_id(arguments, default_project_id)
    if project_id is None:
        raise _ToolError(
            "project_id is required for quipu_receipts — set project_id arg or QUIPU_PROJECT_ID env"
        )

    limit = arguments.get("limit")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise _ToolError("limit must be an integer >= 1")

    fmt = arguments.get("format", "json")
    if fmt not in ("json", "text"):
        raise _ToolError("format must be 'json' or 'text'")

    op_filter = arguments.get("op")
    if op_filter is not None:
        if not isinstance(op_filter, str) or op_filter not in ("write", "invalidate"):
            raise _ToolError("op must be 'write' or 'invalidate'")

    try:
        from quipu.keystore._backend import get_or_derive_key
        from quipu.sync._aad import aad_for

        key = get_or_derive_key(project_id)
        blinded = aad_for(project_id, key).decode()
    except Exception:
        raise _ToolError("project key unavailable — cannot scope receipts")

    from quipu.sync.oplog_store import OplogStore

    oplog = OplogStore(store._conn)
    db_op = None
    if op_filter == "write":
        db_op = "upsert"
    elif op_filter == "invalidate":
        db_op = "invalidate"

    entries = oplog.list_entries_by_project(blinded, op=db_op, limit=limit)

    receipts: list[dict] = []
    for e in entries:
        record_hash = hashlib.sha256(f"{e.record_id}:{project_id}".encode()).hexdigest()
        receipts.append({
            "op": "write" if e.op == "upsert" else e.op,
            "ts": e.ts,
            "record_hash": record_hash,
        })

    if fmt == "text":
        lines = [f"{r['op']}\t{r['ts']}\t{r['record_hash']}" for r in receipts]
        return {"receipts": "\n".join(lines)}

    return {"receipts": receipts}


def _handle_quipu_gc(
    store: Store, default_project_id: str | None, arguments: dict
) -> dict:
    project_id = _resolve_project_id(arguments, default_project_id)
    if project_id is None:
        raise _ToolError(
            "project_id is required for quipu_gc — set project_id arg or QUIPU_PROJECT_ID env"
        )

    dry_run = bool(arguments.get("dry_run", True))
    run_flag = bool(arguments.get("run", False))

    if run_flag:
        dry_run = False

    min_age_days = arguments.get("min_age_days", 90)
    if not isinstance(min_age_days, int) or isinstance(min_age_days, bool) or min_age_days < 0:
        raise _ToolError("min_age_days must be a non-negative integer")

    min_access_count = arguments.get("min_access_count", 3)
    if not isinstance(min_access_count, int) or isinstance(min_access_count, bool) or min_access_count < 0:
        raise _ToolError("min_access_count must be a non-negative integer")

    stale = store.list_stale(
        project_id,
        min_age_days=min_age_days,
        min_access_count=min_access_count,
    )

    invalidated = 0
    if not dry_run:
        for atom in stale:
            store.update_invalidated(atom.id, True)
            invalidated += 1

    return {
        "dry_run": dry_run,
        "stale_count": len(stale),
        "invalidated": invalidated,
        "stale": [
            {
                "id": a.id,
                "content": a.content[:160],
                "access_count": a.access_count,
                "last_accessed": a.last_accessed,
                "created_at": a.created_at,
            }
            for a in stale
        ],
    }


def _handle_quipu_graph(
    store: Store, default_project_id: str | None, arguments: dict
) -> dict:
    from quipu.retrieval import search as quipu_search

    entity = arguments.get("entity")
    if not isinstance(entity, str) or not entity:
        raise _ToolError("entity is required and must be a non-empty string")

    project_id = _resolve_project_id(arguments, default_project_id)
    if project_id is None:
        raise _ToolError(
            "project_id is required for quipu_graph — set project_id arg or QUIPU_PROJECT_ID env"
        )

    max_depth = arguments.get("max_depth", 2)
    if not isinstance(max_depth, int) or isinstance(max_depth, bool) or max_depth < 1:
        raise _ToolError("max_depth must be an integer >= 1")

    edge_types = arguments.get("edge_types")
    if edge_types is not None:
        if not isinstance(edge_types, list) or not all(isinstance(e, str) for e in edge_types):
            raise _ToolError("edge_types must be a list of strings")

    top_k = arguments.get("top_k", 3)
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise _ToolError("top_k must be an integer >= 1")

    as_of = arguments.get("as_of")
    if as_of is not None and not isinstance(as_of, str):
        raise _ToolError("as_of must be a string")

    # Determine roots: try exact atom_id first, fall back to search
    atom = store.get(entity)
    root_ids: list[str]
    if atom is not None and (atom.project_id == project_id):
        root_ids = [atom.id]
    else:
        # Treat entity as search_term — find top_k roots via R3 fusion
        results = quipu_search(
            entity, tier="R3", project_id=project_id, top_k=top_k, store=store,
        )
        root_ids = [r.atom.id for r in results]

    # Union-deduped traverse from all roots
    all_nodes: dict[str, dict] = {}
    all_edges: dict[int, dict] = {}

    for root_id in root_ids:
        subgraph = store.traverse(
            root_id,
            project_id=project_id,
            max_depth=max_depth,
            edge_types=edge_types,
            as_of=as_of,
        )
        for node in subgraph["nodes"]:
            all_nodes[node["id"]] = node
        for edge in subgraph["edges"]:
            eid = edge["id"]
            if eid not in all_edges:
                all_edges[eid] = edge

    return {
        "nodes": list(all_nodes.values()),
        "edges": list(all_edges.values()),
    }


# ---------------------------------------------------------------------------
# Handler registry
# ---------------------------------------------------------------------------

_HANDLERS: dict[str, Any] = {
    "quipu_write": _handle_quipu_write,
    "quipu_search": _handle_quipu_search,
    "quipu_get": _handle_quipu_get,
    "quipu_list": _handle_quipu_list,
    "quipu_invalidate": _handle_quipu_invalidate,
    "quipu_flush": _handle_quipu_flush,
    "quipu_stats": _handle_quipu_stats,
    "quipu_push": _handle_quipu_push,
    "quipu_pull": _handle_quipu_pull,
    "quipu_prime": _handle_quipu_prime,
    "quipu_receipts": _handle_quipu_receipts,
    "quipu_gc": _handle_quipu_gc,
    "quipu_graph": _handle_quipu_graph,
}


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="quipu_write",
        description=(
            "Write a memory record to the Quipu store. "
            "Returns {\"id\": \"<atom_id>\", \"conflicts\": [...]}. "
            "conflicts is a list of existing active same-project atoms whose cosine similarity "
            "to the new atom meets or exceeds the invalidation threshold "
            "(QUIPU_INVALIDATION_THRESHOLD, default 0.92). "
            "Each conflict entry has {\"id\", \"similarity\", \"snippet\"} (snippet = first 160 chars). "
            "The new atom is always persisted regardless of conflicts. "
            "To supersede a conflicting atom, call quipu_invalidate(id=<conflict_id>). "
            "To keep both atoms, take no further action."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Record content."},
                "metadata": {
                    "type": "object",
                    "description": "Optional metadata key/value pairs.",
                },
                "project_id": {
                    "type": "string",
                    "description": "Project identifier (defaults to QUIPU_PROJECT_ID env).",
                },
            },
            "required": ["content"],
        },
    ),
    Tool(
        name="quipu_search",
        description="Search Quipu records using multi-tier retrieval (BM25 + cosine fusion).",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query text."},
                "tier": {
                    "type": "string",
                    "description": "Retrieval tier: R0 (exact), R1 (cosine), R2 (BM25), R3 (fusion). Default R3.",
                    "default": "R3",
                },
                "project_id": {
                    "type": "string",
                    "description": "Project identifier (required; defaults to QUIPU_PROJECT_ID env).",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum results to return (default 10, minimum 1).",
                    "default": 10,
                    "minimum": 1,
                },
                "session_id": {
                    "type": "string",
                    "description": "Optional session filter. When set, restricts results to atoms from that session.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional tag filter. When set, only atoms with at least one matching tag are returned.",
                },
                "graph_expand": {
                    "type": "boolean",
                    "description": "When True, expand top results with connected atoms from the knowledge graph.",
                    "default": False,
                },
                "graph_depth": {
                    "type": "integer",
                    "description": "BFS depth for graph expansion (default 1, minimum 1).",
                    "default": 1,
                    "minimum": 1,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="quipu_get",
        description="Fetch a single Quipu record by its ID (scoped to project).",
        inputSchema={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Atom ID."},
                "project_id": {
                    "type": "string",
                    "description": "Project identifier (defaults to QUIPU_PROJECT_ID env).",
                },
            },
            "required": ["id"],
        },
    ),
    Tool(
        name="quipu_list",
        description="List Quipu records for a project, newest first.",
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "Project identifier (defaults to QUIPU_PROJECT_ID env).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum records to return (default: all, minimum 1).",
                    "minimum": 1,
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="quipu_invalidate",
        description="Mark a Quipu record as invalidated (soft-delete, scoped to project).",
        inputSchema={
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Atom ID to invalidate."},
                "project_id": {
                    "type": "string",
                    "description": "Project identifier (defaults to QUIPU_PROJECT_ID env).",
                },
            },
            "required": ["id"],
        },
    ),
    Tool(
        name="quipu_flush",
        description="Trigger Haiku enrichment pass on pending records.",
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "Project identifier (defaults to QUIPU_PROJECT_ID env).",
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="quipu_stats",
        description="Return record counts (total, active, invalidated), last-flush timestamp, and sync status.",
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "Project identifier (defaults to QUIPU_PROJECT_ID env).",
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="quipu_push",
        description="Manually trigger a push of local oplog entries to the hub.",
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "Project identifier (defaults to QUIPU_PROJECT_ID env).",
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="quipu_pull",
        description="Manually trigger a pull of oplog entries from the hub.",
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "Project identifier (defaults to QUIPU_PROJECT_ID env).",
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="quipu_prime",
        description=(
            "Session-start auto-recall: surface relevant memory at the beginning of a session. "
            "Call once per session before other tool calls. Always returns a well-formed payload; "
            "never raises. When project_id is unset, returns primed=false with an empty results list."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Seed query for recall (optional; defaults to a broad context seed).",
                },
                "project_id": {
                    "type": "string",
                    "description": "Project identifier (defaults to QUIPU_PROJECT_ID env).",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum memory records to surface (default 8, range 1..1000).",
                    "default": 8,
                    "minimum": 1,
                    "maximum": 1000,
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="quipu_receipts",
        description=(
            "Export a hashed/redacted operation log for audit without exposing plaintext content. "
            "Returns a list of {op, ts, record_hash} — never content. "
            "record_hash is SHA-256 of the record_id + project_id (deterministic but irreversible). "
            "Use the optional 'op' filter (write/invalidate) to narrow receipts."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "Project identifier (required; defaults to QUIPU_PROJECT_ID env).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum receipt entries to return (default: all, minimum 1).",
                    "minimum": 1,
                },
                "format": {
                    "type": "string",
                    "description": "Output format: 'json' (default) or 'text' (tab-separated).",
                    "enum": ["json", "text"],
                },
                "op": {
                    "type": "string",
                    "description": "Filter by operation: 'write' or 'invalidate' (default: all).",
                    "enum": ["write", "invalidate"],
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="quipu_gc",
        description=(
            "Garbage collect stale low-value atoms. "
            "By default (dry_run=true), lists candidates without taking action. "
            "Set run=true to soft-invalidate (reversible) matching atoms. "
            "Stale = non-invalidated atoms older than min_age_days with access_count < min_access_count."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "project_id": {
                    "type": "string",
                    "description": "Project identifier (defaults to QUIPU_PROJECT_ID env).",
                },
                "dry_run": {
                    "type": "boolean",
                    "description": "List stale atoms without invalidating (default true).",
                    "default": True,
                },
                "run": {
                    "type": "boolean",
                    "description": "Set to true to soft-invalidate stale atoms. Overrides dry_run.",
                    "default": False,
                },
                "min_age_days": {
                    "type": "integer",
                    "description": "Minimum age in days to consider stale (default 90).",
                    "default": 90,
                    "minimum": 1,
                },
                "min_access_count": {
                    "type": "integer",
                    "description": "Atoms with access_count below this are stale (default 3).",
                    "default": 3,
                    "minimum": 0,
                },
            },
            "required": [],
        },
    ),
    Tool(
        name="quipu_graph",
        description=(
            "Traverse the Quipu knowledge graph from a starting entity. "
            "The entity can be an atom_id (exact match) or a search term "
            "(finds top_k roots via R3 fusion). Returns the connected subgraph "
            "as {nodes: [atom dicts], edges: [edge dicts]}."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "entity": {
                    "type": "string",
                    "description": "Atom ID or search term to use as graph traversal root(s).",
                },
                "project_id": {
                    "type": "string",
                    "description": "Project identifier (defaults to QUIPU_PROJECT_ID env).",
                },
                "max_depth": {
                    "type": "integer",
                    "description": "BFS depth limit (default 2, minimum 1).",
                    "default": 2,
                    "minimum": 1,
                },
                "edge_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional filter: only follow edges of these types (supersedes, blocks, touches_file, decided_by, depends_on, causal).",
                },
                "top_k": {
                    "type": "integer",
                    "description": "When entity is a search term, number of top results to use as roots (default 3).",
                    "default": 3,
                    "minimum": 1,
                },
                "as_of": {
                    "type": "string",
                    "description": "Optional ISO-8601 timestamp for temporal filtering of triples.",
                },
            },
            "required": ["entity"],
        },
    ),
]


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _err(msg: str) -> list[TextContent]:
    return [TextContent(type="text", text=json.dumps({"error": msg}))]


def dispatch(
    name: str,
    *,
    store: Store,
    default_project_id: str | None,
    arguments: dict,
) -> list[TextContent]:
    """Dispatch a tool call by name. Always returns list[TextContent].

    Exceptions from handlers are caught and returned as structured error
    payloads so nothing propagates to the stdio transport.
    """
    try:
        handler = _HANDLERS[name]
    except KeyError:
        return _err(f"unknown tool: {name}")
    try:
        payload = handler(store, default_project_id, arguments or {})
        return [TextContent(type="text", text=json.dumps(payload))]
    except _ToolError as e:
        return _err(str(e))
    except Exception:
        logger.exception("tool %s failed", name)
        return _err("internal error")
