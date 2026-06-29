"""HuggingFace tokenizers-based tokenizer for EmbeddingGemma-300m."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

# numpy imported lazily inside functions so top-level import stays cheap.

MAX_SEQ_LEN_FALLBACK = 2048
# Hard upper bound: values above this clamp to the cap rather than allowing
# unbounded allocation from a tampered tokenizer_config.json (DoS mitigation).
MAX_SEQ_LEN_CAP = 32_768

_TOKENIZER_CONFIG_NAME = "tokenizer_config.json"
_MAX_LEN_KEYS = ("model_max_length", "max_position_embeddings", "max_seq_len")


def _resolve_max_seq_len(model_dir: Path) -> int:
    """Read max_length from tokenizer_config.json if present, else fallback.

    Values above MAX_SEQ_LEN_CAP (32768) are clamped to the cap to prevent
    runaway allocation from a tampered config file.
    Values absent or invalid fall back to MAX_SEQ_LEN_FALLBACK (2048).
    """
    cfg_path = model_dir / _TOKENIZER_CONFIG_NAME
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            for key in _MAX_LEN_KEYS:
                val = cfg.get(key)
                if isinstance(val, (int, float)) and 0 < int(val):
                    return min(int(val), MAX_SEQ_LEN_CAP)
        except Exception:  # noqa: BLE001
            pass
    return MAX_SEQ_LEN_FALLBACK


def load_tokenizer(model_dir: Path):
    """Load a HuggingFace Tokenizer from *model_dir*/tokenizer.json.

    Enables truncation at the model's configured max sequence length.

    Args:
        model_dir: Directory containing ``tokenizer.json`` (and optionally
            ``tokenizer_config.json``).

    Returns:
        A ``tokenizers.Tokenizer`` instance with truncation and padding enabled.
    """
    from tokenizers import Tokenizer  # lazy

    tok_path = model_dir / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(tok_path))

    max_len = _resolve_max_seq_len(model_dir)
    tokenizer.enable_truncation(max_length=max_len)
    tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", direction="right")

    return tokenizer


def encode_batch(
    tokenizer,
    texts: list[str],
) -> "Tuple[object, object]":
    """Encode a list of strings into padded int64 numpy arrays.

    Args:
        tokenizer: A ``tokenizers.Tokenizer`` with truncation/padding enabled.
        texts: Input strings. May be empty.

    Returns:
        Tuple ``(input_ids, attention_mask)`` each shaped ``(N, seq_len)``
        as int64 numpy arrays.
    """
    import numpy as np  # lazy

    if not texts:
        empty = np.empty((0, 0), dtype=np.int64)
        return empty, empty

    encodings = tokenizer.encode_batch(texts)

    input_ids = np.array([enc.ids for enc in encodings], dtype=np.int64)
    attention_mask = np.array([enc.attention_mask for enc in encodings], dtype=np.int64)

    return input_ids, attention_mask
