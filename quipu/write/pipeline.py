"""Write pipeline: embed, extract, store, invalidate.

Network-free. Haiku enrichment is deferred to flush().
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quipu.storage.store import Store


def write(
    content: str,
    metadata: dict | None = None,
    project_id: str | None = None,
    *,
    store: "Store | None" = None,
    type: str = "diary",
    created_at: "str | None" = None,
    session_id: "str | None" = None,
) -> str:
    """Store a content record with embedding and local NLP extraction.

    Flow:
      1. embed(content) -> vec
      2. pack_embedding(vec) -> bytes
      3. extract_local(content) -> {entities, keywords}
      4. merge extraction + caller metadata + enriched:false
      5. Store.insert(...)
      6. emit oplog upsert for the new atom
      7. return atom.id

    DI seam: if store is None, opens and closes its own Store. If injected,
    uses caller's store and does NOT close it.

    Args:
        content: Text content to store.
        metadata: Optional caller-supplied metadata dict.
        project_id: Optional project scope.
        store: Optional injected Store; None opens a new one (and closes it).
        type: Atom type (default 'diary').
        created_at: Optional ISO-8601 UTC timestamp for event-time preservation.
                    Passed through to Store.insert(). None = SQL DEFAULT now.
        session_id: Optional session identifier for grouping atoms by capture
                    session. Passed through to Store.insert(). None = ungrouped.

    Returns:
        The new atom's ID string.
    """
    from quipu.embeddings import embed
    from quipu.storage import store as open_store
    from quipu.storage.store import pack_embedding
    from quipu.extraction import extract_local

    own_store = store is None
    if own_store:
        store = open_store()

    try:
        # 1. Embed
        vec = embed(content)

        # 2. Pack embedding
        embedding_bytes = pack_embedding(vec)

        # 3. Local extraction
        extracted = extract_local(content)

        # 3a. Derive tags from entities + keywords (top-5, deduped, order-stable)
        entities_lower = [e.lower() for e in extracted["entities"]]
        combined: list[str] = []
        seen: set[str] = set()
        for item in entities_lower + extracted["keywords"]:
            if item not in seen:
                seen.add(item)
                combined.append(item)
        derived_tags = combined[:5] if combined else None

        # 4. Build metadata: extraction results + caller metadata + enriched:false
        merged: dict = {}
        merged["entities"] = extracted["entities"]
        merged["keywords"] = extracted["keywords"]
        merged["enriched"] = False

        if metadata:
            # Caller metadata overlays extraction (but enriched flag always False at write)
            for k, v in metadata.items():
                if k != "enriched":
                    merged[k] = v

        # 5. Insert
        atom = store.insert(
            content=content,
            embedding=embedding_bytes,
            project_id=project_id,
            type=type,
            metadata=merged,
            created_at=created_at,
            session_id=session_id,
            tags=derived_tags,
        )

        # 6. Emit oplog entry (sync-active only; fail-isolated; after atom write)
        from quipu.oplog.producer import emit
        emit(store, op="upsert", atom=atom, project_id=project_id)

        # 7. Return atom id
        return atom.id

    finally:
        if own_store:
            store.close()
