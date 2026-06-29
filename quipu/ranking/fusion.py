"""Score normalization and R3 fusion for multi-signal retrieval."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quipu.ranking.result import SearchResult


def normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize scores to [0, 1].

    - Empty list → empty list.
    - All-equal scores → all 1.0 (avoids division by zero).
    """
    if not scores:
        return []
    mn = min(scores)
    mx = max(scores)
    if mx == mn:
        return [1.0] * len(scores)
    rng = mx - mn
    return [(s - mn) / rng for s in scores]


def _access_boost(access_count: int) -> float:
    """Mild boost: log(access_count + 1)."""
    return math.log(access_count + 1)


def fuse(
    r1: "list[SearchResult]",
    r2: "list[SearchResult]",
    *,
    w_cos: float = 0.55,
    w_bm25: float = 0.35,
    w_access: float = 0.1,
    access_counts: dict[str, int] | None = None,
) -> "list[SearchResult]":
    """Merge R1 (cosine) and R2 (BM25) candidate sets into a fused ranking.

    Steps:
    1. Min-max normalize each signal's scores independently to [0, 1].
    2. Optionally compute and normalize an access-frequency boost term.
    3. Linear-combine per atom id (missing signal → 0).
    4. Deduplicate by atom.id (keep highest combined score — both signals
       map to same atom so scores are identical; dedup is structural).
    5. Return results tagged tier="R3", sorted by descending combined score.

    Combined score ∈ [0, 1] is query-relative; do not compare across queries.

    *access_counts* is an optional dict mapping atom_id → access_count.
    When provided, log(access_count+1) is min-max normalised across all
    candidates and added as a third fusion term weighted by *w_access*.
    """
    from quipu.ranking.result import SearchResult

    # Normalize R1 cosine scores
    r1_scores_norm = normalize_scores([r.score for r in r1])
    r1_normed: dict[str, float] = {
        r.atom.id: ns for r, ns in zip(r1, r1_scores_norm)
    }
    # Keep atom references
    atoms_by_id: dict[str, "object"] = {r.atom.id: r.atom for r in r1}

    # Normalize R2 BM25 scores
    r2_scores_norm = normalize_scores([r.score for r in r2])
    r2_normed: dict[str, float] = {
        r.atom.id: ns for r, ns in zip(r2, r2_scores_norm)
    }
    for r in r2:
        atoms_by_id.setdefault(r.atom.id, r.atom)

    # Normalize access-frequency boosts if provided
    access_normed: dict[str, float] = {}
    if access_counts is not None and w_access > 0:
        all_ids = set(r1_normed) | set(r2_normed)
        boosts = [_access_boost(access_counts.get(aid, 0)) for aid in all_ids]
        if boosts:
            normed = normalize_scores(boosts)
            access_normed = {aid: ns for aid, ns in zip(all_ids, normed)}

    # Combine over union of atom ids
    all_ids = set(r1_normed) | set(r2_normed)
    combined: list[tuple[str, float]] = []
    for aid in all_ids:
        score = (
            w_cos * r1_normed.get(aid, 0.0)
            + w_bm25 * r2_normed.get(aid, 0.0)
            + w_access * access_normed.get(aid, 0.0)
        )
        combined.append((aid, score))

    combined.sort(key=lambda x: x[1], reverse=True)

    return [
        SearchResult(atom=atoms_by_id[aid], score=score, tier="R3")
        for aid, score in combined
    ]
