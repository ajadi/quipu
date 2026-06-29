"""Cosine similarity via dot product for L2-normalized vectors."""

from __future__ import annotations


def dot(a: list[float], b: list[float]) -> float:
    """Compute dot product of two vectors.

    For L2-normalized vectors this equals cosine similarity.
    Both vectors must have the same length.
    """
    return sum(x * y for x, y in zip(a, b))
