"""Embedding API for Quipu.

Exposes ``embed`` and ``embed_batch`` backed by EmbeddingGemma-300m ONNX.
"""

from .engine import EMBED_DIM, embed, embed_batch

__all__ = ["embed", "embed_batch", "EMBED_DIM"]
