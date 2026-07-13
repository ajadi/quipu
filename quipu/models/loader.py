"""ONNX InferenceSession loader — model-agnostic with lazy auto-download."""

from __future__ import annotations

import hashlib
import os
import sys

from quipu.models.cache import (
    ModelNotFoundError,
    active_model,
    download_cmd,
    is_gated,
    model_dir,
    onnx_path,
    onnx_path_candidates,
)


def _try_auto_download(path, target, model_key) -> None:
    """Attempt to auto-download the model via huggingface_hub.

    Prints progress to stderr. Raises ModelNotFoundError if huggingface_hub
    is not installed (pip-install message must reach the caller verbatim).
    Raises RuntimeError on any download failure so load_session's generic
    except-Exception arm builds the richer "not found" / gated-login message.
    """
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        raise ModelNotFoundError(
            "huggingface_hub is not installed. Install it first:\n\n"
            "    pip install huggingface_hub\n\n"
            "Then re-run your command to auto-download the model."
        )

    hf_repo = _MODELS_DIRECT[model_key]
    print(f"[quipu] Downloading model {model_key} from {hf_repo} ...", file=sys.stderr)
    print(f"[quipu] This is a one-time download (~1 GB). Please wait.", file=sys.stderr)

    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=hf_repo,
            local_dir=str(target),
            allow_patterns=[
                "*.onnx",
                "*.onnx_data",
                "onnx/*",
                "*.json",
                "*.txt",
                "*.model",
            ],
        )
    except Exception as exc:
        raise RuntimeError(f"Auto-download of {model_key} failed: {exc}")

    print(f"[quipu] Model downloaded to {target}", file=sys.stderr)


# Direct mapping for auto-download (bypasses cache.py's MODELS table)
from quipu.models.cache import MODELS

_MODELS_DIRECT = {k: v["hf_repo"] for k, v in MODELS.items()}


def load_session():  # type: ignore[return]
    """Build and return an onnxruntime InferenceSession (CPUExecutionProvider).

    If the model file is not present, attempts to auto-download it via
    huggingface_hub before raising ModelNotFoundError.

    Raises:
        ModelNotFoundError: if the .onnx file cannot be found or downloaded,
            or if QUIPU_MODEL_SHA256 is set and the file digest does not match.
    """
    import onnxruntime as ort  # lazy — only needed at runtime

    path = onnx_path()
    if not path.exists():
        target = model_dir()
        model_key = active_model()
        try:
            _try_auto_download(path, target, model_key)
        except ModelNotFoundError:
            raise
        except Exception as exc:
            cmd = download_cmd(model_key)
            msg = (
                f"ONNX model not found at: {path}\n"
                f"Auto-download failed: {exc}\n"
                f"Download with:\n\n    {cmd} --local-dir {target}\n"
            )
            if is_gated(model_key):
                msg += (
                    "\nThis model requires Hugging Face authentication.\n"
                    "Log in first with:\n\n    huggingface-cli login\n"
                )
            raise ModelNotFoundError(msg)

        # Re-resolve after download — the snapshot may place the weight at
        # the root or in the onnx/ subdirectory depending on repo layout.
        path = onnx_path()
        if not path.exists():
            root, nested = onnx_path_candidates()
            cmd = download_cmd(model_key)
            msg = (
                f"ONNX model not found after download. Searched:\n"
                f"    {root}\n"
                f"    {nested}\n"
                f"Download manually with:\n\n    {cmd} --local-dir {target}\n"
            )
            if is_gated(model_key):
                msg += (
                    "\nThis model requires Hugging Face authentication.\n"
                    "Log in first with:\n\n    huggingface-cli login\n"
                )
            raise ModelNotFoundError(msg)

    # Optional integrity check
    expected_sha = os.environ.get("QUIPU_MODEL_SHA256")
    if expected_sha:
        actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            model_key = active_model()
            cmd = download_cmd(model_key)
            target = model_dir()
            msg = (
                f"model hash mismatch — re-download\n"
                f"expected: {expected_sha}\n"
                f"actual:   {actual_sha}\n"
                f"Download with:\n\n    {cmd} --local-dir {target}\n"
            )
            if is_gated(model_key):
                msg += (
                    "\nThis model requires Hugging Face authentication.\n"
                    "Log in first with:\n\n    huggingface-cli login\n"
                )
            raise ModelNotFoundError(msg)

    session_opts = ort.SessionOptions()
    session_opts.inter_op_num_threads = 1
    session_opts.intra_op_num_threads = 1

    return ort.InferenceSession(
        str(path),
        sess_options=session_opts,
        providers=["CPUExecutionProvider"],
    )
