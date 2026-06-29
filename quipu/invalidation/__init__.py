"""Quipu invalidation: cosine-similarity-based supersession detection."""

from .cosine import (
    cosine,
    resolve_threshold,
    find_superseded,
    find_conflicts,
    invalidate_superseded,
)

__all__ = [
    "cosine",
    "resolve_threshold",
    "find_superseded",
    "find_conflicts",
    "invalidate_superseded",
]
