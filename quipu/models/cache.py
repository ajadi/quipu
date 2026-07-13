"""Model cache path resolution — multi-model registry."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS: dict[str, dict] = {
    "nomic-embed-text-v1.5": {
        "hf_repo": "nomic-ai/nomic-embed-text-v1.5",
        "local_dir": "nomic-embed-text-v1.5",
        "gated": False,
        "dim": 768,
    },
    "bge-small-en-v1.5": {
        "hf_repo": "BAAI/bge-small-en-v1.5",
        "local_dir": "bge-small-en-v1.5",
        "gated": False,
        "dim": 384,
    },
    "bge-m3": {
        "hf_repo": "BAAI/bge-m3",
        "local_dir": "bge-m3",
        "gated": False,
        "dim": 1024,
    },
    "embeddinggemma-300m": {
        "hf_repo": "google/embeddinggemma-300m",
        "local_dir": "embeddinggemma-300m",
        "gated": True,
        "dim": 768,
    },
}

# RECOMMENDED_MODEL is a display/menu label only — the model installers
# suggest first and docs point at. It is NEVER an implicit runtime fallback:
# active_model() does not substitute it when the env var is unset/invalid.
RECOMMENDED_MODEL = "nomic-embed-text-v1.5"

_DEFAULT_BASE = Path.home() / ".quipu" / "models"


class ModelNotFoundError(ImportError):
    """Raised when the ONNX model file is absent from the cache directory.

    Includes exact download instructions in the message.
    """


class UnknownModelError(ValueError):
    """Raised when QUIPU_EMBEDDING_MODEL names an unrecognized, non-empty key.

    A typo must never silently resolve to a different real model — this
    error names the bad value and lists the valid keys (including ``none``).
    """


def active_model() -> str | None:
    """Return the active model key, or None for keyword-only mode.

    Reads QUIPU_EMBEDDING_MODEL env var fresh on every call.

    - Unset/empty, or ``none`` (case-insensitive) -> None. This is the
      keyword-only sentinel: a valid, explicit choice to run without any
      embedding model, not an error and not a silent substitution.
    - A registered MODELS key -> that key.
    - Any other non-empty value -> raises UnknownModelError (typo guard).
    """
    key = os.environ.get("QUIPU_EMBEDDING_MODEL", "")
    if not key or key.lower() == "none":
        return None
    if key in MODELS:
        return key
    valid = ", ".join(sorted(MODELS)) + ", none"
    raise UnknownModelError(
        f"Unknown QUIPU_EMBEDDING_MODEL value {key!r}. Valid values: {valid}"
    )


def active_dim() -> int:
    """Return the embedding dimension of the active model.

    Reads the active model fresh on every call (env-sensitive). Raises
    ValueError if the active model declares no ``dim`` — such a model can
    never embed anyway. Raises ValueError if the active model is None
    (keyword-only mode has no embedding dimension).
    """
    model = active_model()
    if model is None:
        raise ValueError(
            "active_dim() has no meaning in keyword-only mode "
            "(QUIPU_EMBEDDING_MODEL is unset/none). Set QUIPU_EMBEDDING_MODEL "
            "to a registered model key before calling this."
        )
    entry = MODELS[model]
    if "dim" not in entry:
        raise ValueError(f"model {model!r} has no declared embedding dim")
    return int(entry["dim"])


def download_cmd(model_key: str) -> str:
    """Return the bare huggingface-cli download command for *model_key*.

    No --local-dir appended — caller supplies the resolved path.
    """
    hf_repo = MODELS[model_key]["hf_repo"]
    return f"huggingface-cli download {hf_repo}"


def is_gated(model_key: str) -> bool:
    """Return True if *model_key* requires HF authentication."""
    return bool(MODELS[model_key]["gated"])


# Backward-compat alias — FROZEN to RECOMMENDED_MODEL at import time.
# It is NOT env-sensitive and does NOT track active_model().
# Callers needing the active model must call download_cmd(active_model()) directly.
# Deprecated — will be removed in a future release.
DOWNLOAD_CMD = download_cmd(RECOMMENDED_MODEL)


def model_dir() -> Path:
    """Return the resolved model directory.

    Prefers $QUIPU_MODEL_DIR env var (with ~ expansion), falls back to
    ``~/.quipu/models/<active_model_local_dir>/``.

    Raises ValueError (not a bare KeyError) if $QUIPU_MODEL_DIR is unset and
    the active model is None (keyword-only mode has no model directory).
    """
    env = os.environ.get("QUIPU_MODEL_DIR")
    if env:
        return Path(env).expanduser().resolve()
    model = active_model()
    if model is None:
        raise ValueError(
            "model_dir() has no meaning in keyword-only mode "
            "(QUIPU_EMBEDDING_MODEL is unset/none). Set QUIPU_EMBEDDING_MODEL "
            "to a registered model key, or set QUIPU_MODEL_DIR directly."
        )
    local_dir = MODELS[model]["local_dir"]
    return _DEFAULT_BASE.expanduser().resolve() / local_dir


def onnx_path() -> Path:
    """Return path to the ONNX model file.

    HF ONNX export repos place the weight either at the snapshot root
    (``model.onnx``) or under an ``onnx/`` subdirectory (``onnx/model.onnx``).
    Returns whichever exists; if neither exists yet (e.g. before download),
    returns the root path as the default.
    """
    root = model_dir() / "model.onnx"
    if root.exists():
        return root
    nested = model_dir() / "onnx" / "model.onnx"
    if nested.exists():
        return nested
    return root


def onnx_path_candidates() -> list[Path]:
    """Return both possible ONNX file locations, root first then onnx/ subdir."""
    d = model_dir()
    return [d / "model.onnx", d / "onnx" / "model.onnx"]


def tokenizer_path() -> Path:
    """Return expected path to the tokenizer.json file."""
    return model_dir() / "tokenizer.json"
