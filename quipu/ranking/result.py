"""SearchResult dataclass for the retrieval/ranking pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quipu.storage.store import Atom


@dataclass
class SearchResult:
    """A single retrieval result with atom, score, and tier label."""

    atom: "Atom"
    score: float
    tier: str  # "R0" | "R1" | "R2" | "R3"
