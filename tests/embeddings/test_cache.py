"""Unit tests for quipu.models.cache path resolution."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from quipu.models.cache import (
    DEFAULT_MODEL,
    DOWNLOAD_CMD,
    ModelNotFoundError,
    active_model,
    download_cmd,
    is_gated,
    model_dir,
    onnx_path,
    tokenizer_path,
)


class TestModelDir:
    def test_default_path(self, monkeypatch):
        monkeypatch.delenv("QUIPU_MODEL_DIR", raising=False)
        monkeypatch.delenv("QUIPU_EMBEDDING_MODEL", raising=False)
        result = model_dir()
        assert result == (Path.home() / ".quipu" / "models" / "nomic-embed-text-v1.5").resolve()

    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUIPU_MODEL_DIR", str(tmp_path))
        result = model_dir()
        assert result == tmp_path.resolve()

    def test_env_tilde_expands(self, monkeypatch):
        """QUIPU_MODEL_DIR with ~ must expand and produce an absolute path."""
        monkeypatch.setenv("QUIPU_MODEL_DIR", "~/.quipu/custom-model")
        result = model_dir()
        assert result.is_absolute(), "model_dir() must return an absolute path"
        assert "~" not in str(result), "tilde must be expanded"

    def test_default_path_is_absolute(self, monkeypatch):
        monkeypatch.delenv("QUIPU_MODEL_DIR", raising=False)
        monkeypatch.delenv("QUIPU_EMBEDDING_MODEL", raising=False)
        result = model_dir()
        assert result.is_absolute()

    def test_model_env_selects_correct_dir(self, monkeypatch):
        """QUIPU_EMBEDDING_MODEL=bge-m3 → model_dir() ends in bge-m3."""
        monkeypatch.delenv("QUIPU_MODEL_DIR", raising=False)
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-m3")
        result = model_dir()
        assert result.name == "bge-m3"

    def test_unknown_model_key_falls_back_to_default(self, monkeypatch):
        """Unknown QUIPU_EMBEDDING_MODEL key → active_model() returns DEFAULT_MODEL."""
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "not-a-real-model")
        assert active_model() == DEFAULT_MODEL

    def test_download_cmd_uses_active_model(self, monkeypatch):
        """download_cmd(active_model()) contains the correct HF repo."""
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-small-en-v1.5")
        result = download_cmd(active_model())
        assert "BAAI/bge-small-en-v1.5" in result


class TestPaths:
    def test_onnx_path_filename(self, monkeypatch):
        monkeypatch.delenv("QUIPU_MODEL_DIR", raising=False)
        assert onnx_path().name == "model.onnx"

    def test_tokenizer_path_filename(self, monkeypatch):
        monkeypatch.delenv("QUIPU_MODEL_DIR", raising=False)
        assert tokenizer_path().name == "tokenizer.json"


class TestModelNotFoundError:
    def test_is_import_error(self):
        err = ModelNotFoundError("test")
        assert isinstance(err, ImportError)

    def test_loader_raises_with_instructions(self, monkeypatch, tmp_path):
        """load_session() raises ModelNotFoundError with download cmd when missing."""
        monkeypatch.setenv("QUIPU_MODEL_DIR", str(tmp_path))
        monkeypatch.delenv("QUIPU_MODEL_SHA256", raising=False)
        monkeypatch.delenv("QUIPU_EMBEDDING_MODEL", raising=False)
        pytest.importorskip("onnxruntime")

        from quipu.models.loader import load_session

        with pytest.raises(ModelNotFoundError) as exc_info:
            load_session()

        msg = str(exc_info.value)
        assert "huggingface-cli" in msg
        assert str(tmp_path) in msg

    def test_gated_model_error_contains_hf_login_instructions(self, monkeypatch, tmp_path):
        """Gated model (embeddinggemma-300m) error must include HF-login lines."""
        monkeypatch.setenv("QUIPU_MODEL_DIR", str(tmp_path))
        monkeypatch.delenv("QUIPU_MODEL_SHA256", raising=False)
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "embeddinggemma-300m")
        pytest.importorskip("onnxruntime")

        from quipu.models.loader import load_session

        with pytest.raises(ModelNotFoundError) as exc_info:
            load_session()

        msg = str(exc_info.value)
        assert "huggingface-cli login" in msg
        assert "authentication" in msg.lower()

    def test_non_gated_model_error_omits_hf_login_instructions(self, monkeypatch, tmp_path):
        """Non-gated model error must NOT include HF-login lines."""
        monkeypatch.setenv("QUIPU_MODEL_DIR", str(tmp_path))
        monkeypatch.delenv("QUIPU_MODEL_SHA256", raising=False)
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-small-en-v1.5")
        pytest.importorskip("onnxruntime")

        from quipu.models.loader import load_session

        with pytest.raises(ModelNotFoundError) as exc_info:
            load_session()

        msg = str(exc_info.value)
        assert "huggingface-cli login" not in msg
        assert "authentication" not in msg.lower()

    def test_download_cmd_in_error(self):
        """ModelNotFoundError message must include download instructions."""
        err = ModelNotFoundError(f"missing\n{DOWNLOAD_CMD}")
        assert "huggingface-cli" in str(err)

    def test_error_message_has_exactly_one_local_dir(self, monkeypatch, tmp_path):
        """Loader error message must contain exactly one --local-dir flag."""
        monkeypatch.setenv("QUIPU_MODEL_DIR", str(tmp_path))
        monkeypatch.delenv("QUIPU_MODEL_SHA256", raising=False)
        monkeypatch.delenv("QUIPU_EMBEDDING_MODEL", raising=False)
        pytest.importorskip("onnxruntime")

        from quipu.models.loader import load_session

        with pytest.raises(ModelNotFoundError) as exc_info:
            load_session()

        msg = str(exc_info.value)
        assert msg.count("--local-dir") == 1, (
            f"Expected exactly 1 --local-dir in error, found {msg.count('--local-dir')}:\n{msg}"
        )

    def test_error_message_local_dir_points_to_resolved_dir(self, monkeypatch, tmp_path):
        """The --local-dir in the error must point at the resolved model_dir()."""
        monkeypatch.setenv("QUIPU_MODEL_DIR", str(tmp_path))
        monkeypatch.delenv("QUIPU_MODEL_SHA256", raising=False)
        monkeypatch.delenv("QUIPU_EMBEDDING_MODEL", raising=False)
        pytest.importorskip("onnxruntime")

        from quipu.models.loader import load_session

        with pytest.raises(ModelNotFoundError) as exc_info:
            load_session()

        msg = str(exc_info.value)
        expected_dir = str(tmp_path.resolve())
        assert expected_dir in msg, (
            f"Expected '{expected_dir}' in error message:\n{msg}"
        )

    def test_download_cmd_has_no_local_dir(self):
        """download_cmd must be a bare command — no --local-dir embedded."""
        assert "--local-dir" not in download_cmd(DEFAULT_MODEL)


class TestSHA256IntegrityCheck:
    def test_correct_hash_passes(self, monkeypatch, tmp_path):
        """If QUIPU_MODEL_SHA256 matches the file, load_session proceeds past hash check."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake-onnx-content")
        digest = hashlib.sha256(b"fake-onnx-content").hexdigest()

        monkeypatch.setenv("QUIPU_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("QUIPU_MODEL_SHA256", digest)
        monkeypatch.delenv("QUIPU_EMBEDDING_MODEL", raising=False)
        pytest.importorskip("onnxruntime")

        from quipu.models import loader as _loader
        import importlib
        importlib.reload(_loader)
        from quipu.models.loader import load_session

        # Hash matches — will fail at InferenceSession (invalid ONNX), not at hash check.
        with pytest.raises(Exception) as exc_info:
            load_session()

        assert "hash mismatch" not in str(exc_info.value).lower(), (
            "Should not raise hash mismatch when digest is correct"
        )

    def test_wrong_hash_raises(self, monkeypatch, tmp_path):
        """If QUIPU_MODEL_SHA256 does not match, ModelNotFoundError is raised."""
        onnx_file = tmp_path / "model.onnx"
        onnx_file.write_bytes(b"fake-onnx-content")

        monkeypatch.setenv("QUIPU_MODEL_DIR", str(tmp_path))
        monkeypatch.setenv("QUIPU_MODEL_SHA256", "deadbeef" * 8)  # 64 hex chars, wrong
        monkeypatch.delenv("QUIPU_EMBEDDING_MODEL", raising=False)
        pytest.importorskip("onnxruntime")

        from quipu.models.loader import load_session

        with pytest.raises(ModelNotFoundError) as exc_info:
            load_session()

        msg = str(exc_info.value)
        assert "hash mismatch" in msg

    def test_no_sha256_env_skips_check(self, monkeypatch, tmp_path):
        """Absent QUIPU_MODEL_SHA256 → no integrity check (current behavior preserved)."""
        monkeypatch.setenv("QUIPU_MODEL_DIR", str(tmp_path))
        monkeypatch.delenv("QUIPU_MODEL_SHA256", raising=False)
        monkeypatch.delenv("QUIPU_EMBEDDING_MODEL", raising=False)
        pytest.importorskip("onnxruntime")

        from quipu.models.loader import load_session

        # model.onnx absent → ModelNotFoundError for missing file, not hash.
        with pytest.raises(ModelNotFoundError) as exc_info:
            load_session()

        assert "not found" in str(exc_info.value).lower()
        assert "hash mismatch" not in str(exc_info.value).lower()
