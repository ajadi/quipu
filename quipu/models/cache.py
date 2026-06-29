"""Model cache path resolution — multi-model registry."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS: dict[str, dict] = {
    "nomic-embed-v2": {
        "hf_repo": "nomic-ai/nomic-embed-v2",
        "local_dir": "nomic-embed-v2",
        "gated": False,
    },
    "nomic-embed-text-v1.5": {
        "hf_repo": "nomic-ai/nomic-embed-text-v1.5",
        "local_dir": "nomic-embed-text-v1.5",
        "gated": False,
    },
    "bge-small-en-v1.5": {
        "hf_repo": "BAAI/bge-small-en-v1.5",
        "local_dir": "bge-small-en-v1.5",
        "gated": False,
    },
    "bge-m3": {
        "hf_repo": "BAAI/bge-m3",
        "local_dir": "bge-m3",
        "gated": False,
    },
    "embeddinggemma-300m": {
        "hf_repo": "google/embeddinggemma-300m",
        "local_dir": "embeddinggemma-300m",
        "gated": True,
    },
}

DEFAULT_MODEL = "nomic-embed-v2"

_DEFAULT_BASE = Path.home() / ".quipu" / "models"


class ModelNotFoundError(ImportError):
    """Raised when the ONNX model file is absent from the cache directory.

    Includes exact download instructions in the message.
    """


def active_model() -> str:
    """Return the active model key.

    Reads QUIPU_EMBEDDING_MODEL env var fresh on every call.
    Falls back to DEFAULT_MODEL if unset or unknown key.
    """
    key = os.environ.get("QUIPU_EMBEDDING_MODEL", "")
    if key in MODELS:
        return key
    return DEFAULT_MODEL


def download_cmd(model_key: str) -> str:
    """Return the bare huggingface-cli download command for *model_key*.

    No --local-dir appended — caller supplies the resolved path.
    """
    hf_repo = MODELS[model_key]["hf_repo"]
    return f"huggingface-cli download {hf_repo}"


def is_gated(model_key: str) -> bool:
    """Return True if *model_key* requires HF authentication."""
    return bool(MODELS[model_key]["gated"])


# Backward-compat alias — FROZEN to DEFAULT_MODEL at import time.
# It is NOT env-sensitive and does NOT track active_model().
# Callers needing the active model must call download_cmd(active_model()) directly.
# Deprecated — will be removed in a future release.
DOWNLOAD_CMD = download_cmd(DEFAULT_MODEL)


def model_dir() -> Path:
    """Return the resolved model directory.

    Prefers $QUIPU_MODEL_DIR env var (with ~ expansion), falls back to
    ``~/.quipu/models/<active_model_local_dir>/``.
    """
    env = os.environ.get("QUIPU_MODEL_DIR")
    if env:
        return Path(env).expanduser().resolve()
    local_dir = MODELS[active_model()]["local_dir"]
    return _DEFAULT_BASE.expanduser().resolve() / local_dir


def onnx_path() -> Path:
    """Return expected path to the ONNX model file."""
    return model_dir() / "model.onnx"


def tokenizer_path() -> Path:
    """Return expected path to the tokenizer.json file."""
    return model_dir() / "tokenizer.json"
