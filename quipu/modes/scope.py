"""Scope filtering helpers for multi-store (project/global/all) queries.

Self-contained: built only on Store's public API (list_by_project) and
quipu.retrieval.search (imported locally to avoid heavy load on import).

Injected stores are NEVER closed by this module — caller owns lifetime.

Scope semantics:
  project  -- atoms in project_store scoped to project_id
  global   -- atoms in global_store (or project_store) with scope=='global'
  all      -- union of project + global, dedup by id, sorted created_at DESC

NOTE (Phase 1 limitation, tracked E13): global-scope queries are keyed to the
current project_id — cross-project global atoms (different project_id) are NOT
returned.  HMAC-based cross-project access control arrives in E10/E13.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quipu.storage.store import Atom, Store
    from quipu.ranking.result import SearchResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_SCOPES: frozenset[str] = frozenset({"project", "global", "all"})

ScopeName = str  # type alias for clarity


# ---------------------------------------------------------------------------
# Scope resolution
# ---------------------------------------------------------------------------


def resolve_scope(scope: str | None) -> str:
    """Normalise scope string.

    None -> 'project'. Invalid value raises ValueError.
    """
    if scope is None:
        return "project"
    if scope not in VALID_SCOPES:
        raise ValueError(
            f"scope must be one of {sorted(VALID_SCOPES)}, got {scope!r}"
        )
    return scope


# ---------------------------------------------------------------------------
# DB path helpers
# ---------------------------------------------------------------------------


def global_db_path() -> Path:
    """Return the canonical global DB path (~/.quipu/global.db)."""
    return Path.home() / ".quipu" / "global.db"


def project_db_path() -> Path:
    """Return the project DB path based on current mode/env config."""
    from quipu.config import get_project_root
    return get_project_root() / ".quipu" / "quipu.db"


# ---------------------------------------------------------------------------
# Atom filtering
# ---------------------------------------------------------------------------


def filtered_atoms(
    scope: str | None,
    project_id: str,
    *,
    project_store: "Store",
    global_store: "Store | None" = None,
    include_invalidated: bool = False,
    limit: int | None = None,
) -> "list[Atom]":
    """Return atoms filtered by scope.

    Args:
        scope: 'project', 'global', 'all', or None (-> 'project').
        project_id: Project ID to scope project-side queries.
        project_store: Store for project records. Never closed here.
        global_store: Optional separate Store for global records.
                      If None, falls back to project_store.
        include_invalidated: Whether to include invalidated atoms.
        limit: Max results; applied AFTER merge (for 'all').

    Returns:
        List of Atom ordered by created_at DESC.
    """
    scope = resolve_scope(scope)

    if scope == "project":
        return project_store.list_by_project(
            project_id,
            include_invalidated=include_invalidated,
            limit=limit,
        )

    if scope == "global":
        source = global_store if global_store is not None else project_store
        # list_by_project scoped to project_id; filter on scope=='global'.
        # NOTE: For a dedicated global_store we use project_id as the key
        # because atoms written globally are still keyed to a project_id.
        # We filter atom.scope=='global' to be precise.
        atoms = source.list_by_project(
            project_id,
            include_invalidated=include_invalidated,
        )
        results = [a for a in atoms if a.scope == "global"]
        if limit is not None:
            results = results[:limit]
        return results

    # scope == 'all'
    project_atoms = project_store.list_by_project(
        project_id,
        include_invalidated=include_invalidated,
    )

    source = global_store if global_store is not None else project_store
    global_atoms_raw = source.list_by_project(
        project_id,
        include_invalidated=include_invalidated,
    )
    global_atoms = [a for a in global_atoms_raw if a.scope == "global"]

    # Union by id, project takes precedence on duplicate
    seen: set[str] = set()
    merged: list[Atom] = []
    for atom in project_atoms:
        if atom.id not in seen:
            seen.add(atom.id)
            merged.append(atom)
    for atom in global_atoms:
        if atom.id not in seen:
            seen.add(atom.id)
            merged.append(atom)

    # Re-sort by created_at DESC (ISO 8601 strings sort lexicographically)
    merged.sort(key=lambda a: a.created_at, reverse=True)

    if limit is not None:
        merged = merged[:limit]
    return merged


# ---------------------------------------------------------------------------
# Search result filtering
# ---------------------------------------------------------------------------


def filtered_search_results(
    scope: str | None,
    project_id: str,
    query: str,
    *,
    project_store: "Store",
    global_store: "Store | None" = None,
    tier: str = "R3",
    top_k: int = 10,
) -> "list[SearchResult]":
    """Return search results filtered by scope.

    Args:
        scope: 'project', 'global', 'all', or None (-> 'project').
        project_id: Project ID for scoping.
        query: Search query string.
        project_store: Store for project records. Never closed here.
        global_store: Optional separate Store for global records.
        tier: Retrieval tier ('R0'–'R3'). Default 'R3'.
        top_k: Max results per store; for 'all', applied after merge.

    Returns:
        List of SearchResult ordered by score DESC.
    """
    from quipu.retrieval import search

    scope = resolve_scope(scope)

    if scope == "project":
        return search(
            query,
            tier=tier,
            project_id=project_id,
            top_k=top_k,
            store=project_store,
        )

    if scope == "global":
        source = global_store if global_store is not None else project_store
        results = search(
            query,
            tier=tier,
            project_id=project_id,
            top_k=top_k,
            store=source,
        )
        return [r for r in results if r.atom.scope == "global"]

    # scope == 'all'
    project_results = search(
        query,
        tier=tier,
        project_id=project_id,
        top_k=top_k,
        store=project_store,
    )

    source = global_store if global_store is not None else project_store
    global_results_raw = search(
        query,
        tier=tier,
        project_id=project_id,
        top_k=top_k,
        store=source,
    )
    # Filter global leg: only atoms explicitly scoped 'global' (prevents
    # project-scoped atoms from a global store leaking into 'all' results).
    global_results = [r for r in global_results_raw if r.atom.scope == "global"]

    # Merge: dedup by atom.id, re-sort by score DESC, truncate to top_k
    seen: set[str] = set()
    merged: list[SearchResult] = []
    for r in project_results:
        if r.atom.id not in seen:
            seen.add(r.atom.id)
            merged.append(r)
    for r in global_results:
        if r.atom.id not in seen:
            seen.add(r.atom.id)
            merged.append(r)

    merged.sort(key=lambda r: r.score, reverse=True)
    return merged[:top_k]
