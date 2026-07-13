"""Unit tests for quipu.models.cache path resolution."""

from __future__ import annotations

import hashlib
import os
import sys
import types
from pathlib import Path

import pytest

from quipu.models import cache
from quipu.models.cache import (
    DOWNLOAD_CMD,
    MODELS,
    ModelNotFoundError,
    RECOMMENDED_MODEL,
    UnknownModelError,
    active_dim,
    active_model,
    download_cmd,
    is_gated,
    model_dir,
    onnx_path,
    onnx_path_candidates,
    tokenizer_path,
)


# ---------------------------------------------------------------------------
# TASK-053 — active_dim() dimension resolution
# ---------------------------------------------------------------------------

class TestActiveDim:
    def test_recommended_model_dim_is_768(self, monkeypatch):
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", RECOMMENDED_MODEL)
        assert active_dim() == 768

    def test_bge_small_dim_is_384(self, monkeypatch):
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-small-en-v1.5")
        assert active_dim() == 384

    def test_bge_m3_dim_is_1024(self, monkeypatch):
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-m3")
        assert active_dim() == 1024

    def test_embeddinggemma_dim_is_768(self, monkeypatch):
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "embeddinggemma-300m")
        assert active_dim() == 768

    def test_synthetic_model_without_dim_raises_value_error(self, monkeypatch):
        """A registered model lacking a 'dim' key — active_dim() must raise, not guess.

        Drives the dim-less branch with a synthetic monkeypatched entry
        (TASK-054: the previously-used nomic-embed-v2 phantom was removed
        from MODELS entirely — it can no longer be used to exercise this path).
        """
        monkeypatch.setitem(
            cache.MODELS,
            "synthetic-no-dim",
            {
                "hf_repo": "example/synthetic-no-dim",
                "local_dir": "synthetic-no-dim",
                "gated": False,
            },
        )
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "synthetic-no-dim")
        with pytest.raises(ValueError, match="synthetic-no-dim"):
            active_dim()

    def test_unknown_model_key_raises(self, monkeypatch):
        """Unknown model key raises via active_model(), no silent fallback."""
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "not-a-real-model")
        with pytest.raises(UnknownModelError, match="not-a-real-model"):
            active_dim()

    def test_raises_when_model_is_none(self, monkeypatch):
        """active_dim() raises a clear error (not KeyError) in keyword-only mode."""
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "none")
        with pytest.raises(ValueError, match="keyword-only"):
            active_dim()

    def test_reads_env_fresh_each_call(self, monkeypatch):
        """active_dim() is not cached — switching env mid-process changes the result."""
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-small-en-v1.5")
        assert active_dim() == 384
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-m3")
        assert active_dim() == 1024


# ---------------------------------------------------------------------------
# TASK-062 — active_model() None sentinel (keyword-only mode) + typo guard
# ---------------------------------------------------------------------------

class TestActiveModel:
    def test_unset_returns_none(self, monkeypatch):
        """No QUIPU_EMBEDDING_MODEL set -> keyword-only sentinel, not a default model."""
        monkeypatch.delenv("QUIPU_EMBEDDING_MODEL", raising=False)
        assert active_model() is None

    def test_explicit_empty_string_returns_none(self, monkeypatch):
        """QUIPU_EMBEDDING_MODEL explicitly set to "" (present, empty) -> None,
        same sentinel as unset — must not be treated as an unknown key."""
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "")
        assert active_model() is None

    @pytest.mark.parametrize("value", ["none", "None", "NONE", "NoNe"])
    def test_none_keyword_is_case_insensitive(self, monkeypatch, value):
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", value)
        assert active_model() is None

    def test_unknown_key_raises(self, monkeypatch):
        """A typo/unknown key must never silently resolve to a real model."""
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "nomic-embed-v1")
        with pytest.raises(UnknownModelError, match="nomic-embed-v1"):
            active_model()

    def test_unknown_key_error_lists_valid_keys_and_none(self, monkeypatch):
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "not-a-real-model")
        with pytest.raises(UnknownModelError) as exc_info:
            active_model()
        msg = str(exc_info.value)
        for key in MODELS:
            assert key in msg
        assert "none" in msg

    def test_valid_key_returns_it(self, monkeypatch):
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-m3")
        assert active_model() == "bge-m3"


class TestModelRegistry:
    """TASK-054 — nomic-embed-v2 (nonexistent model) removed from MODELS."""

    def test_models_has_exactly_four_real_models(self):
        assert set(MODELS.keys()) == {
            "nomic-embed-text-v1.5",
            "bge-small-en-v1.5",
            "bge-m3",
            "embeddinggemma-300m",
        }

    def test_nomic_embed_v2_not_in_models(self):
        assert "nomic-embed-v2" not in MODELS

    def test_recommended_model_is_nomic_embed_text_v1_5(self):
        assert RECOMMENDED_MODEL == "nomic-embed-text-v1.5"

    def test_recommended_model_is_a_registered_key(self):
        assert RECOMMENDED_MODEL in MODELS

    def test_every_model_entry_declares_a_dim(self):
        """Guards active_dim(): a registered model must never hit the dim-less branch."""
        missing = [key for key, entry in MODELS.items() if "dim" not in entry]
        assert missing == [], f"models missing 'dim': {missing}"

    @pytest.mark.parametrize(
        "model_key,expected_dim",
        [
            ("nomic-embed-text-v1.5", 768),
            ("bge-small-en-v1.5", 384),
            ("bge-m3", 1024),
            ("embeddinggemma-300m", 768),
        ],
    )
    def test_model_dim_matches_expected(self, model_key, expected_dim):
        assert MODELS[model_key]["dim"] == expected_dim


class TestModelDir:
    def test_default_path(self, monkeypatch):
        monkeypatch.delenv("QUIPU_MODEL_DIR", raising=False)
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", RECOMMENDED_MODEL)
        result = model_dir()
        assert result == (Path.home() / ".quipu" / "models" / "nomic-embed-text-v1.5").resolve()

    def test_raises_when_model_is_none(self, monkeypatch):
        """model_dir() raises a clear error (not KeyError) in keyword-only mode."""
        monkeypatch.delenv("QUIPU_MODEL_DIR", raising=False)
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "none")
        with pytest.raises(ValueError, match="keyword-only"):
            model_dir()

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
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", RECOMMENDED_MODEL)
        result = model_dir()
        assert result.is_absolute()

    def test_model_env_selects_correct_dir(self, monkeypatch):
        """QUIPU_EMBEDDING_MODEL=bge-m3 → model_dir() ends in bge-m3."""
        monkeypatch.delenv("QUIPU_MODEL_DIR", raising=False)
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-m3")
        result = model_dir()
        assert result.name == "bge-m3"

    def test_unknown_model_key_raises(self, monkeypatch):
        """Unknown QUIPU_EMBEDDING_MODEL key → active_model() raises, no fallback."""
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "not-a-real-model")
        with pytest.raises(UnknownModelError):
            active_model()

    def test_download_cmd_uses_active_model(self, monkeypatch):
        """download_cmd(active_model()) contains the correct HF repo."""
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-small-en-v1.5")
        result = download_cmd(active_model())
        assert "BAAI/bge-small-en-v1.5" in result


class TestPaths:
    def test_onnx_path_filename(self, monkeypatch, semantic_model):
        monkeypatch.delenv("QUIPU_MODEL_DIR", raising=False)
        assert onnx_path().name == "model.onnx"

    def test_tokenizer_path_filename(self, monkeypatch, semantic_model):
        monkeypatch.delenv("QUIPU_MODEL_DIR", raising=False)
        assert tokenizer_path().name == "tokenizer.json"

    def test_onnx_path_resolves_to_root_when_present(self, monkeypatch, tmp_path):
        """onnx_path() must return the root model.onnx when it exists there."""
        monkeypatch.setenv("QUIPU_MODEL_DIR", str(tmp_path))
        monkeypatch.delenv("QUIPU_EMBEDDING_MODEL", raising=False)
        (tmp_path / "model.onnx").write_bytes(b"fake-onnx")

        assert onnx_path() == tmp_path.resolve() / "model.onnx"

    def test_onnx_path_resolves_to_nested_onnx_subdir(self, monkeypatch, tmp_path):
        """onnx_path() must fall back to onnx/model.onnx when only that exists."""
        monkeypatch.setenv("QUIPU_MODEL_DIR", str(tmp_path))
        monkeypatch.delenv("QUIPU_EMBEDDING_MODEL", raising=False)
        onnx_dir = tmp_path / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"fake-onnx")

        assert onnx_path() == onnx_dir.resolve() / "model.onnx"

    def test_onnx_path_prefers_root_when_both_exist(self, monkeypatch, tmp_path):
        """onnx_path() must prefer the root path when both root and nested exist."""
        monkeypatch.setenv("QUIPU_MODEL_DIR", str(tmp_path))
        monkeypatch.delenv("QUIPU_EMBEDDING_MODEL", raising=False)
        (tmp_path / "model.onnx").write_bytes(b"root")
        onnx_dir = tmp_path / "onnx"
        onnx_dir.mkdir()
        (onnx_dir / "model.onnx").write_bytes(b"nested")

        assert onnx_path() == tmp_path.resolve() / "model.onnx"

    def test_onnx_path_candidates_returns_both_locations(self, monkeypatch, tmp_path):
        """onnx_path_candidates() must return [root, onnx/] regardless of existence."""
        monkeypatch.setenv("QUIPU_MODEL_DIR", str(tmp_path))
        monkeypatch.delenv("QUIPU_EMBEDDING_MODEL", raising=False)

        candidates = onnx_path_candidates()

        assert candidates == [
            tmp_path.resolve() / "model.onnx",
            tmp_path.resolve() / "onnx" / "model.onnx",
        ]


class TestModelNotFoundError:
    def test_is_import_error(self):
        err = ModelNotFoundError("test")
        assert isinstance(err, ImportError)

    def test_loader_raises_with_instructions(self, monkeypatch, tmp_path):
        """load_session() raises ModelNotFoundError with download cmd when missing."""
        monkeypatch.setenv("QUIPU_MODEL_DIR", str(tmp_path))
        monkeypatch.delenv("QUIPU_MODEL_SHA256", raising=False)
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", RECOMMENDED_MODEL)
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
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", RECOMMENDED_MODEL)
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
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", RECOMMENDED_MODEL)
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
        assert "--local-dir" not in download_cmd(RECOMMENDED_MODEL)

    def test_missing_onnx_after_download_names_both_candidate_paths(self, monkeypatch, tmp_path):
        """Post-download miss: error must name BOTH root and onnx/ candidate paths."""
        monkeypatch.setenv("QUIPU_MODEL_DIR", str(tmp_path))
        monkeypatch.delenv("QUIPU_MODEL_SHA256", raising=False)
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", RECOMMENDED_MODEL)
        pytest.importorskip("onnxruntime")

        # Simulate a "successful" auto-download that places no .onnx file
        # anywhere (e.g. only tokenizer/config files) — no real network.
        fake_hf = types.ModuleType("huggingface_hub")

        def _fake_snapshot_download(**kwargs):
            local_dir = Path(kwargs["local_dir"])
            local_dir.mkdir(parents=True, exist_ok=True)
            (local_dir / "tokenizer.json").write_text("{}")

        fake_hf.snapshot_download = _fake_snapshot_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        from quipu.models.loader import load_session

        with pytest.raises(ModelNotFoundError) as exc_info:
            load_session()

        msg = str(exc_info.value)
        root, nested = onnx_path_candidates()
        assert str(root) in msg, f"root candidate {root} missing from error:\n{msg}"
        assert str(nested) in msg, f"nested candidate {nested} missing from error:\n{msg}"

    def test_load_session_reresolves_path_after_download_to_nested_subdir(
        self, monkeypatch, tmp_path
    ):
        """load_session must re-check onnx_path() after download, not reuse the
        stale pre-download path — regression test for TASK-052."""
        monkeypatch.setenv("QUIPU_MODEL_DIR", str(tmp_path))
        monkeypatch.delenv("QUIPU_MODEL_SHA256", raising=False)
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", RECOMMENDED_MODEL)
        pytest.importorskip("onnxruntime")

        # Simulate a "successful" auto-download that places the weight under
        # onnx/ only (root model.onnx never exists) — no real network.
        fake_hf = types.ModuleType("huggingface_hub")

        def _fake_snapshot_download(**kwargs):
            local_dir = Path(kwargs["local_dir"])
            onnx_dir = local_dir / "onnx"
            onnx_dir.mkdir(parents=True, exist_ok=True)
            (onnx_dir / "model.onnx").write_bytes(b"not-a-real-onnx-protobuf")

        fake_hf.snapshot_download = _fake_snapshot_download
        monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hf)

        from quipu.models.loader import load_session

        # A stale (unresolved) pre-download path never exists post-download
        # (only onnx/model.onnx does) → load_session would raise
        # ModelNotFoundError without ever reaching InferenceSession. If the
        # path was correctly re-resolved to the nested location, it reaches
        # InferenceSession and fails there instead (invalid protobuf content),
        # and onnxruntime's own error message echoes the exact path it opened.
        _, nested = onnx_path_candidates()
        with pytest.raises(Exception) as exc_info:
            load_session()

        assert not isinstance(exc_info.value, ModelNotFoundError), (
            f"load_session raised ModelNotFoundError instead of re-resolving "
            f"to the nested path and attempting InferenceSession: {exc_info.value}"
        )
        msg = str(exc_info.value)
        assert str(nested) in msg, (
            f"Expected re-resolved nested path {nested} in onnxruntime error, got:\n{msg}"
        )


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
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", RECOMMENDED_MODEL)
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
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", RECOMMENDED_MODEL)
        pytest.importorskip("onnxruntime")

        from quipu.models.loader import load_session

        # model.onnx absent → ModelNotFoundError for missing file, not hash.
        with pytest.raises(ModelNotFoundError) as exc_info:
            load_session()

        assert "not found" in str(exc_info.value).lower()
        assert "hash mismatch" not in str(exc_info.value).lower()
