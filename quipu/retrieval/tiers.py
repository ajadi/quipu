"""Tier implementations R0–R3 operating on pre-fetched Atom lists.

All tier functions accept a pre-fetched list[Atom] (already scoped to the
target project, non-invalidated) so they can be tested in isolation.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

from quipu.ranking.result import SearchResult
from quipu.ranking.cosine import dot
from quipu.ranking.fusion import fuse
from quipu.retrieval.bm25 import bm25_rank

if TYPE_CHECKING:
    from quipu.storage.store import Atom


def tier_r0(atoms: "list[Atom]", query: str, top_k: int) -> "list[SearchResult]":
    """R0 — exact string match. Score = 1.0 for each matching atom."""
    results = [
        SearchResult(atom=a, score=1.0, tier="R0")
        for a in atoms
        if a.content == query
    ]
    return results[:top_k]


def tier_r1(
    atoms: "list[Atom]",
    query_vec: "list[float]",
    top_k: int,
) -> "list[SearchResult]":
    """R1 — cosine similarity via dot product on L2-normalized embeddings.

    Atoms with embedding=None are skipped (not indexable by this tier).
    Results ordered by descending score.
    """
    from quipu.storage.store import unpack_embedding

    scored: list[tuple] = []
    for atom in atoms:
        if atom.embedding is None:
            continue
        try:
            vec = unpack_embedding(atom.embedding)
        except (ValueError, struct.error):
            continue
        score = dot(query_vec, vec)
        scored.append((atom, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [
        SearchResult(atom=a, score=s, tier="R1")
        for a, s in scored[:top_k]
    ]


def tier_r2(atoms: "list[Atom]", query: str, top_k: int) -> "list[SearchResult]":
    """R2 — BM25 via SQLite FTS5.

    Scores are negated FTS5 bm25() values (higher == more relevant).
    Results ordered by descending score.
    """
    scores = bm25_rank(atoms, query)
    if not scores:
        return []

    # Build atom lookup
    atom_by_id = {a.id: a for a in atoms}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = []
    for atom_id, score in ranked[:top_k]:
        if atom_id in atom_by_id:
            results.append(SearchResult(atom=atom_by_id[atom_id], score=score, tier="R2"))
    return results


def tier_r3(
    atoms: "list[Atom]",
    query: str,
    query_vec: "list[float]",
    top_k: int,
    w_cos: float = 0.55,
    w_bm25: float = 0.35,
    w_access: float = 0.1,
) -> "list[SearchResult]":
    """R3 — fusion of R1 cosine, R2 BM25, and access-frequency boost.

    Min-max normalizes each signal independently, linear-combines, deduplicates
    by atom.id, returns top_k by descending combined score.
    """
    # Run both signals over the full atom set (no top_k cap yet — need full
    # signal distribution for normalization)
    r1_full = tier_r1(atoms, query_vec, top_k=len(atoms))
    r2_full = tier_r2(atoms, query, top_k=len(atoms))

    access_counts: dict[str, int] = {a.id: a.access_count for a in atoms}

    fused = fuse(
        r1_full, r2_full,
        w_cos=w_cos, w_bm25=w_bm25,
        w_access=w_access, access_counts=access_counts,
    )
    return fused[:top_k]
