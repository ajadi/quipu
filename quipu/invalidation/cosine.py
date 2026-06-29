"""Cosine-similarity-based supersession detection.

Embeddings are L2-normalized so cosine similarity == dot product.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quipu.storage.store import Atom, Store

logger = logging.getLogger(__name__)

_SNIPPET_LEN = 160

_DEFAULT_THRESHOLD = 0.92
_ENV_VAR = "QUIPU_INVALIDATION_THRESHOLD"


def cosine(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity as dot product (vectors must be L2-normalized).

    Args:
        a: L2-normalized vector.
        b: L2-normalized vector.

    Returns:
        Dot product (== cosine similarity for unit vectors).
    """
    return sum(x * y for x, y in zip(a, b))


def resolve_threshold() -> float:
    """Return the invalidation similarity threshold.

    Reads QUIPU_INVALIDATION_THRESHOLD from env. Falls back to 0.92 on missing
    or invalid value, logging a warning on bad values.

    Returns:
        Float threshold in range (0, 1].
    """
    raw = os.environ.get(_ENV_VAR)
    if raw is None:
        return _DEFAULT_THRESHOLD

    try:
        value = float(raw)
    except (ValueError, TypeError):
        logger.warning(
            "Invalid %s=%r — expected float, using default %.2f",
            _ENV_VAR,
            raw,
            _DEFAULT_THRESHOLD,
        )
        return _DEFAULT_THRESHOLD

    if not (0.0 < value <= 1.0):
        logger.warning(
            "Out-of-range %s=%r — expected (0, 1], using default %.2f",
            _ENV_VAR,
            raw,
            _DEFAULT_THRESHOLD,
        )
        return _DEFAULT_THRESHOLD

    return value


def find_superseded(
    new_vec: list[float],
    existing: list["Atom"],
    *,
    threshold: float | None = None,
    exclude_id: str | None = None,
) -> list[str]:
    """Return IDs of existing atoms whose similarity to new_vec meets threshold.

    Args:
        new_vec: The new atom's L2-normalized embedding vector.
        existing: List of existing Atom objects (from store.list_by_project).
        threshold: Similarity threshold; defaults to resolve_threshold().
        exclude_id: If set, skip the atom with this ID (prevents self-invalidation).

    Returns:
        List of atom IDs where cosine(new_vec, existing.embedding) >= threshold.
        Atoms with embedding=None are skipped. Atom with exclude_id is skipped.
    """
    from quipu.storage.store import unpack_embedding

    if threshold is None:
        threshold = resolve_threshold()

    superseded: list[str] = []
    for atom in existing:
        if exclude_id is not None and atom.id == exclude_id:
            continue
        if atom.embedding is None:
            continue
        existing_vec = unpack_embedding(atom.embedding)
        sim = cosine(new_vec, existing_vec)
        if sim >= threshold:
            superseded.append(atom.id)

    return superseded


def find_conflicts(
    new_vec: list[float],
    existing: list["Atom"],
    *,
    threshold: float | None = None,
    exclude_id: str | None = None,
) -> list[dict]:
    """Return structured conflict records for existing atoms above the similarity threshold.

    Args:
        new_vec: The new atom's L2-normalized embedding vector.
        existing: List of existing Atom objects (from store.list_by_project).
        threshold: Similarity threshold; defaults to resolve_threshold().
        exclude_id: If set, skip the atom with this ID (prevents self-conflict).

    Returns:
        List of dicts: [{"id": str, "similarity": float (4dp), "snippet": str}].
        Atoms with embedding=None are skipped. Atom with exclude_id is skipped.
        snippet = content[:160]; "…" appended only when content was longer than 160.
    """
    from quipu.storage.store import unpack_embedding

    if threshold is None:
        threshold = resolve_threshold()

    conflicts: list[dict] = []
    for atom in existing:
        if exclude_id is not None and atom.id == exclude_id:
            continue
        if atom.embedding is None:
            continue
        existing_vec = unpack_embedding(atom.embedding)
        sim = cosine(new_vec, existing_vec)
        if sim >= threshold:
            content = atom.content
            snippet = content[:_SNIPPET_LEN] + ("…" if len(content) > _SNIPPET_LEN else "")
            conflicts.append({
                "id": atom.id,
                "similarity": round(sim, 4),
                "snippet": snippet,
            })

    return conflicts


def invalidate_superseded(
    store: "Store",
    new_vec: list[float],
    project_id: str,
    *,
    threshold: float | None = None,
    exclude_id: str | None = None,
) -> list[str]:
    """Mark existing atoms superseded by new_vec as invalidated.

    Scans non-invalidated atoms for project_id, finds those with cosine
    similarity >= threshold to new_vec, and calls store.update_invalidated()
    on each.

    Args:
        store: Store instance to query and update.
        new_vec: The new atom's L2-normalized embedding.
        project_id: Project to scope the search.
        threshold: Similarity threshold; defaults to resolve_threshold().
        exclude_id: If set, skip the atom with this ID (prevents self-invalidation).

    Returns:
        List of atom IDs that were marked invalidated.
    """
    existing = store.list_by_project(project_id, include_invalidated=False)
    ids = find_superseded(new_vec, existing, threshold=threshold, exclude_id=exclude_id)
    for atom_id in ids:
        store.update_invalidated(atom_id, True)
        inv_atom = store.get(atom_id)
        if inv_atom is not None:
            from quipu.oplog.producer import emit
            emit(store, op="invalidate", atom=inv_atom, project_id=project_id)
    return ids
