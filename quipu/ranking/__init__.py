"""quipu.ranking — stateless scoring primitives and result type."""

from quipu.ranking.result import SearchResult
from quipu.ranking.cosine import dot
from quipu.ranking.fusion import normalize_scores, fuse

__all__ = ["SearchResult", "dot", "normalize_scores", "fuse"]
