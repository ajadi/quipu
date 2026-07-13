"""Cosine similarity via dot product for L2-normalized vectors."""

from __future__ import annotations


def dot(a: list[float], b: list[float]) -> float:
    """Compute dot product of two vectors.

    For L2-normalized vectors this equals cosine similarity.
    Both vectors must have the same length; a mismatch raises ValueError
    rather than silently truncating (which would yield wrong scores).
    """
    if len(a) != len(b):
        raise ValueError(f"dim mismatch: {len(a)} != {len(b)}")
    return sum(x * y for x, y in zip(a, b))
