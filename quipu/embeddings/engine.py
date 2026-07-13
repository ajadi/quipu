"""Embedding engine: singleton _Engine wrapping ONNX session + tokenizer."""

from __future__ import annotations

from typing import List, Optional

from quipu.models.cache import active_dim, active_model

# Frozen at import time from the active model — environment changes after
# import are not reflected here. Live dimension checks must call embed_dim().
# In keyword-only mode EMBED_DIM is intentionally None; the server's test hook
# owns its separate numeric fallback and must not redefine this contract.
EMBED_DIM = active_dim() if active_model() is not None else None


def embed_dim() -> int:
    """Return the active model's embedding dimension (env-sensitive, live)."""
    return active_dim()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_input(inputs, keyword: str, all_names: list) -> str:
    """Return the name of the first input whose name contains *keyword*.

    Raises RuntimeError listing available names if not found.
    """
    for inp in inputs:
        if keyword in inp.name:
            return inp.name
    raise RuntimeError(
        f"Cannot find a '{keyword}' input among session inputs: {all_names}. "
        "Ensure the ONNX model exports standard HuggingFace input names."
    )


class _Engine:
    """Holds an ONNX InferenceSession and a tokenizers.Tokenizer.

    Build via ``_Engine.build()`` or supply stubs via ``set_engine()`` for tests.
    """

    def __init__(self, session, tokenizer) -> None:
        # Resolve tensor names by NAME, not position — guards against models with
        # extra inputs (token_type_ids, position_ids) or non-standard ordering.
        self._session = session
        self._tokenizer = tokenizer

        inputs = session.get_inputs()
        input_names = [inp.name for inp in inputs]

        self._input_ids_name: str = _find_input(inputs, "input_ids", input_names)
        self._attn_mask_name: str = _find_input(inputs, "attention_mask", input_names)

        # Collect optional auxiliary inputs (token_type_ids, position_ids, etc.)
        # that the model declares but we don't drive from the tokenizer.
        self._aux_inputs: list = [
            inp for inp in inputs
            if inp.name not in (self._input_ids_name, self._attn_mask_name)
        ]

        self._output_name: str = session.get_outputs()[0].name

    @classmethod
    def build(cls) -> "_Engine":
        """Instantiate from the on-disk model cache."""
        from quipu.models.loader import load_session
        from quipu.models.cache import model_dir
        from quipu.embeddings.tokenizer import load_tokenizer

        session = load_session()
        tokenizer = load_tokenizer(model_dir())
        return cls(session, tokenizer)

    # ------------------------------------------------------------------
    # Internal inference helpers
    # ------------------------------------------------------------------

    def _run(
        self,
        input_ids: "object",
        attention_mask: "object",
    ) -> "object":
        """Run ONNX session and return the raw output array."""
        import numpy as np

        feeds = {
            self._input_ids_name: input_ids,
            self._attn_mask_name: attention_mask,
        }

        # Feed zeros for any auxiliary inputs (e.g. token_type_ids, position_ids)
        # that the model declares but we don't populate from the tokenizer.
        for aux in self._aux_inputs:
            dtype = getattr(aux, "type", "tensor(int64)")
            np_dtype = np.int64 if "int" in str(dtype) else np.float32
            feeds[aux.name] = np.zeros_like(input_ids, dtype=np_dtype)

        outputs = self._session.run([self._output_name], feeds)
        return outputs[0]  # shape (N, T, D) or (N, D)

    def _pool_and_normalize(
        self,
        raw: "object",
        attention_mask: "object",
    ) -> "object":
        """Mean-pool if rank-3, then L2-normalize. Returns (N, EMBED_DIM) float32."""
        import numpy as np

        arr = np.array(raw, dtype=np.float32)

        if arr.ndim == 3:
            # (N, T, D) — mean-pool masked tokens
            mask = np.array(attention_mask, dtype=np.float32)  # (N, T)
            mask_exp = mask[:, :, np.newaxis]  # (N, T, 1)
            summed = (arr * mask_exp).sum(axis=1)  # (N, D)
            counts = mask.sum(axis=1, keepdims=True).clip(min=1.0)  # (N, 1)
            pooled = summed / counts
        elif arr.ndim == 2:
            # (N, D) — already pooled (e.g. sentence_embedding output)
            pooled = arr
        else:
            raise ValueError(
                f"Unexpected output rank {arr.ndim}; expected 2 or 3."
            )

        # L2 normalize so E3 can use plain dot product for cosine similarity.
        norms = np.linalg.norm(pooled, axis=1, keepdims=True).clip(min=1e-12)
        return pooled / norms

    # ------------------------------------------------------------------
    # Public encode
    # ------------------------------------------------------------------

    def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode a list of strings; returns list of EMBED_DIM float vectors."""
        import numpy as np
        from quipu.embeddings.tokenizer import encode_batch

        if not texts:
            return []

        input_ids, attention_mask = encode_batch(self._tokenizer, texts)
        raw = self._run(input_ids, attention_mask)
        vectors = self._pool_and_normalize(raw, attention_mask)  # (N, D)
        return vectors.tolist()


# ---------------------------------------------------------------------------
# Process-global singleton
# ---------------------------------------------------------------------------

_engine: Optional[_Engine] = None


def _get_engine() -> _Engine:
    """Return the singleton engine, building it on first call (lazy)."""
    global _engine
    if _engine is None:
        _engine = _Engine.build()
    return _engine


def set_engine(engine: Optional[_Engine]) -> None:
    """Inject a pre-built engine (for testing).

    Call ``_reset()`` in test teardown to restore lazy-build behaviour.
    """
    global _engine
    _engine = engine


def _reset() -> None:
    """Reset the singleton so the next call triggers a fresh lazy build.

    For use in test teardown fixtures.
    """
    global _engine
    _engine = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def embed(text: str) -> List[float]:
    """Embed a single string.

    Args:
        text: Input string.

    Returns:
        L2-normalized float vector of length ``embed_dim()`` (active model's dim).
    """
    return _get_engine().encode([text])[0]


def embed_batch(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings.

    Args:
        texts: Input strings. Empty list returns empty list.

    Returns:
        One L2-normalized ``embed_dim()``-dim vector per input, order preserved.
    """
    if not texts:
        return []
    return _get_engine().encode(texts)
