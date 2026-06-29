"""ONNX InferenceSession loader — model-agnostic."""

from __future__ import annotations

import hashlib
import os

from quipu.models.cache import (
    ModelNotFoundError,
    active_model,
    download_cmd,
    is_gated,
    model_dir,
    onnx_path,
)

# SECURITY: Loading an untrusted .onnx executes arbitrary graph ops — treat the
# model file as code.  Users MUST download only from the official Hugging Face
# source.  Once the official file hash is published, set QUIPU_MODEL_SHA256 to
# the hex digest to enable integrity verification.
# TODO: pin the official SHA-256 hash here when it becomes known post-release.


def load_session():  # type: ignore[return]
    """Build and return an onnxruntime InferenceSession (CPUExecutionProvider).

    Raises:
        ModelNotFoundError: if the .onnx file is not present in the cache dir,
            or if QUIPU_MODEL_SHA256 is set and the file digest does not match.
    """
    import onnxruntime as ort  # lazy — only needed at runtime

    path = onnx_path()
    if not path.exists():
        target = model_dir()
        model_key = active_model()
        cmd = download_cmd(model_key)
        msg = (
            f"ONNX model not found at: {path}\n"
            f"Download with:\n\n    {cmd} --local-dir {target}\n"
        )
        if is_gated(model_key):
            msg += (
                "\nThis model requires Hugging Face authentication.\n"
                "Log in first with:\n\n    huggingface-cli login\n"
            )
        raise ModelNotFoundError(msg)

    # Optional integrity check — set QUIPU_MODEL_SHA256 to the expected hex digest.
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
