"""Public search() API for the Quipu retrieval pipeline.

Exposes a single `search()` function that dispatches to the appropriate
tier (R0–R3) after scoping atoms to a project via the Store.

Vec routing (R1 only):
  When quipu.vec.query_ready(store._conn) is True, R1 is served by
  vec.knn() (sqlite-vec KNN).  R3 always uses the pure-Python tier_r1
  cosine signal for exact score equivalence regardless of corpus size.
  When vec is unavailable or below-threshold (the default in tests and
  small deployments), R1 falls back to the existing pure-Python tier_r1
  path — results are byte-identical to pre-vec behaviour.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from quipu.ranking.result import SearchResult
from quipu.retrieval.tiers import tier_r0, tier_r1, tier_r2, tier_r3

if TYPE_CHECKING:
    from quipu.storage.store import Store

# Import embed at module level so tests can monkeypatch it.
from quipu.embeddings import embed

_VALID_TIERS = frozenset({"R0", "R1", "R2", "R3"})


def search(
    query: str,
    tier: str = "R3",
    project_id: str | None = None,
    top_k: int = 10,
    *,
    store: "Store | None" = None,
    w_cos: float = 0.55,
    w_bm25: float = 0.35,
    w_access: float = 0.1,
    session_id: str | None = None,
    tags: list[str] | None = None,
    graph_expand: bool = False,
    graph_depth: int = 1,
) -> "list[SearchResult]":
    """Retrieve atoms matching *query* from a single project.

    Args:
        query: The search query string.
        tier: One of "R0", "R1", "R2", "R3". Defaults to "R3" (best recall).
        project_id: Required. Scopes retrieval to a single project.
        top_k: Maximum number of results to return.
        store: Optional Store instance for dependency injection (tests).
               If None, opens the default store via quipu.storage.store().
        w_cos: Weight for cosine signal in R3 fusion (default 0.6).
        w_bm25: Weight for BM25 signal in R3 fusion (default 0.4).
        w_access: Weight for access-frequency boost in R3 fusion (default 0.1).
        session_id: Optional session filter. When set, restricts candidate atoms
                    to those with a matching session_id BEFORE tiering. When None,
                    all atoms (including NULL session_id) are considered.
        tags: Optional list of string tags. When set, restricts candidate atoms
              to those where atom.tags overlaps with requested tags
              (case-insensitive). Atoms with NULL tags are excluded.
        graph_expand: When True, expands top search results by adding atoms
                      connected via KG edges (BFS with graph_depth). Default
                      False (preserves byte-identical pre-graph behavior).
        graph_depth: BFS depth for graph_expand (default 1, min 1).

    Returns:
        List of SearchResult ordered by descending score, at most top_k items.

    Raises:
        ValueError: If project_id is None (cross-project scope not supported).
        ValueError: If tier is not one of R0, R1, R2, R3.
    """
    if project_id is None:
        raise ValueError(
            "project_id is required for Phase 1 retrieval; "
            "cross-project scope is not yet supported"
        )

    if not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"top_k must be a positive int, got {top_k!r}")

    if tier not in _VALID_TIERS:
        raise ValueError(f"tier must be one of {sorted(_VALID_TIERS)}, got {tier!r}")

    # Open store if not injected
    _owned = False
    if store is None:
        from quipu.storage import store as open_store
        store = open_store()
        _owned = True

    try:
        atoms = store.list_by_project(project_id, include_invalidated=False)

        # Session filter: restrict candidates BEFORE tiering when session_id is set.
        if session_id is not None:
            atoms = [a for a in atoms if a.session_id == session_id]

        # Tags filter: restrict candidates BEFORE tiering when tags is set.
        # Atoms with NULL tags are excluded. Matching is case-insensitive.
        if tags is not None:
            tags_lower = {t.lower() for t in tags}
            atoms = [
                a for a in atoms
                if a.tags is not None and tags_lower.intersection(
                    t.lower() for t in a.tags
                )
            ]

        results: list[SearchResult]

        if tier == "R0":
            results = tier_r0(atoms, query, top_k)

        elif tier == "R1":
            query_vec = embed(query)
            # Vec routing: use KNN index when available, else pure-Python.
            from quipu.vec import query_ready, knn as vec_knn
            if query_ready(store._conn):
                rowid_scores = vec_knn(store._conn, query_vec, project_id, top_k)
                results = _results_from_rowids(store, rowid_scores, "R1", project_id=project_id)
            else:
                results = tier_r1(atoms, query_vec, top_k)

        elif tier == "R2":
            results = tier_r2(atoms, query, top_k)

        else:
            # R3 — fusion of cosine + BM25 + access-frequency signals.
            query_vec = embed(query)
            results = tier_r3(atoms, query, query_vec, top_k,
                              w_cos=w_cos, w_bm25=w_bm25, w_access=w_access)

        # Graph expand: append connected atoms via BFS from top results
        if graph_expand and results:
            expand_n = min(3, len(results))
            connected_results: list[SearchResult] = []
            for r in results[:expand_n]:
                connected = store.get_connected_atoms(
                    r.atom.id,
                    project_id=project_id,
                    max_depth=graph_depth,
                )
                for atom in connected:
                    connected_results.append(SearchResult(
                        atom=atom,
                        score=r.score * 0.7,
                        tier=r.tier,
                    ))

            # Dedup by atom.id, keep highest score, re-sort, slice top_k
            seen: dict[str, SearchResult] = {}
            for r in results + connected_results:
                aid = r.atom.id
                if aid not in seen or r.score > seen[aid].score:
                    seen[aid] = r
            results = sorted(seen.values(), key=lambda r: r.score, reverse=True)[:top_k]

        # Increment access_count on every atom returned to the caller
        for r in results:
            store.increment_access(r.atom.id)

        return results

    finally:
        if _owned:
            store.close()


def _results_from_rowids(
    store: "Store",
    rowid_scores: "list[tuple[int, float]]",
    tier: str,
    project_id: str | None = None,
) -> "list[SearchResult]":
    """Reconstruct SearchResult list from (rowid, dot_score) pairs.

    Fetches each atom by rowid, preserving the score ordering.
    Rows whose rowid no longer exists in atoms are silently skipped.

    When *project_id* is supplied (required for the vec path) the re-fetch
    also filters by project_id AND invalidated=0 so a stale vec-index rowid
    that has been reassigned to a different project's atom cannot leak cross-
    project data.  Any rowid that fails this filter is silently skipped.
    """
    results: list[SearchResult] = []
    from quipu.storage.store import _row_to_atom
    for rowid, score in rowid_scores:
        if project_id is not None:
            row = store._conn.execute(
                "SELECT * FROM atoms WHERE rowid = ? AND project_id = ? AND invalidated = 0",
                (rowid, project_id),
            ).fetchone()
        else:
            row = store._conn.execute(
                "SELECT * FROM atoms WHERE rowid = ?", (rowid,)
            ).fetchone()
        if row is None:
            continue
        atom = _row_to_atom(row)
        results.append(SearchResult(atom=atom, score=score, tier=tier))
    return results
