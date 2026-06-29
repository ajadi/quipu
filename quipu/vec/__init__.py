"""quipu.vec — optional sqlite-vec acceleration for R1 cosine retrieval.

Public API:
    from quipu.vec import ensure_index, VecState, query_ready, knn

ensure_index(conn, *, threshold=None) -> VecState
    Idempotent lifecycle gate. Loads extension, checks threshold, builds.

query_ready(conn) -> bool
    True iff vec path is usable (extension loaded + build complete).

knn(conn, query_vec, project_id, top_k) -> list[tuple[int, float]]
    Returns [(rowid, dot_score)] pairs, dot_score in cosine scale.
"""

from quipu.vec.index import VecState, ensure_index
from quipu.vec._query import knn, query_ready

__all__ = ["VecState", "ensure_index", "query_ready", "knn"]
