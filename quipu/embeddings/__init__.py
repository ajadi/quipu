"""Embedding API for Quipu.

``EMBED_DIM`` is an ``int | None`` frozen at import time: it is ``None`` in
keyword-only mode and does not track later environment changes. Call
``embed_dim()`` when a live active-model dimension is required.
"""

from .engine import EMBED_DIM, embed, embed_batch, embed_dim

__all__ = ["embed", "embed_batch", "EMBED_DIM", "embed_dim"]
