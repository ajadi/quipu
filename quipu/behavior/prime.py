"""Session-start auto-recall helper.

``prime()`` is a pure helper that NEVER raises. It resolves project_id,
runs a single retrieval search at R3, and returns a well-formed dict.
Intended to be called from the quipu_prime MCP tool handler.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quipu.storage import Store

_DEFAULT_SEED = "decisions patterns context architecture important"


def prime(
    store: "Store",
    project_id: str | None,
    topic: str | None = None,
    top_k: int = 8,
) -> dict:
    """Session-start auto-recall. NEVER raises.

    Returns:
        {"primed": bool, "results": list, "note": str}

    Conditions:
        project_id is None -> {"primed": False, "results": [], "note": "no project_id"}
        search() raises   -> {"primed": True, "results": [], "note": "recall unavailable"}
        search() empty    -> {"primed": True, "results": [], "note": "no memory yet"}
        search() hits     -> {"primed": True, "results": [...], "note": "ok"}
    """
    if project_id is None:
        return {"primed": False, "results": [], "note": "no project_id"}

    query = topic if topic else _DEFAULT_SEED

    try:
        from quipu.retrieval import search

        results = search(query, tier="R3", project_id=project_id, top_k=top_k, store=store)
    except Exception:
        return {"primed": True, "results": [], "note": "recall unavailable"}

    if not results:
        return {"primed": True, "results": [], "note": "no memory yet"}

    return {
        "primed": True,
        "results": [
            {
                "id": r.atom.id,
                "content": r.atom.content,
                "score": r.score,
                "tier": r.tier,
                "type": r.atom.type,
                "scope": r.atom.scope,
                "invalidated": r.atom.invalidated,
                "metadata": r.atom.metadata,
            }
            for r in results
        ],
        "note": "ok",
    }
