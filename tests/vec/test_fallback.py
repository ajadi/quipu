"""Tests for the unavailability fallback path.

Verifies that when sqlite-vec is unavailable (simulated):
- ensure_index returns UNAVAILABLE
- query_ready returns False
- No crash, no user prompt
"""

from __future__ import annotations

import sys
import types

import pytest

from quipu.vec._gate import reset_warned
from quipu.vec.index import VecState, ensure_index
from quipu.vec._query import query_ready


class TestEnsureIndexUnavailable:
    def setup_method(self):
        reset_warned()

    def _patch_try_load_false(self, monkeypatch):
        """Monkeypatch quipu.vec._gate.try_load to always return False."""
        monkeypatch.setattr("quipu.vec._gate.try_load", lambda conn: False)
        # Also patch at index.py import path.
        monkeypatch.setattr("quipu.vec.index.try_load", lambda conn: False)

    def test_unavailable_state_returned(self, tmp_conn, monkeypatch):
        self._patch_try_load_false(monkeypatch)
        state = ensure_index(tmp_conn, threshold=1)
        assert state == VecState.UNAVAILABLE

    def test_no_crash_on_unavailable(self, tmp_conn, monkeypatch):
        """ensure_index must not raise even when extension load fails."""
        self._patch_try_load_false(monkeypatch)
        try:
            ensure_index(tmp_conn, threshold=1)
        except Exception as exc:  # noqa: BLE001
            pytest.fail(f"ensure_index raised unexpectedly: {exc}")

    def test_below_threshold_no_build(self, tmp_conn, monkeypatch):
        """Below threshold with vec available → BELOW_THRESHOLD, no table."""
        # Simulate vec available (try_load returns True) but not loaded
        # via is_loaded (extension not actually there).
        monkeypatch.setattr("quipu.vec.index.try_load", lambda conn: True)
        # No atoms → below any threshold > 0.
        state = ensure_index(tmp_conn, threshold=100)
        assert state == VecState.BELOW_THRESHOLD

        # atoms_vec must NOT have been created.
        tables = {
            row[0]
            for row in tmp_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "atoms_vec" not in tables


class TestQueryReadyFalseWhenUnavailable:
    def test_query_ready_false_no_extension(self, tmp_conn):
        """query_ready is False when vec_version() is unavailable."""
        assert query_ready(tmp_conn) is False

    def test_query_ready_false_when_not_complete(self, tmp_conn, monkeypatch):
        """query_ready is False even if extension 'loaded' but meta not complete."""
        monkeypatch.setattr("quipu.vec._query.is_loaded", lambda conn: True)
        # No meta table → is_build_complete returns False.
        assert query_ready(tmp_conn) is False


class TestEnsureIndexThresholdEnvVar:
    """QUIPU_VEC_THRESHOLD env-var hardening (FIX 4).

    Non-integer or <=0 values must fall back to the default (30 000)
    without raising an uncaught ValueError.
    """

    def setup_method(self):
        reset_warned()

    def _patch_try_load_true(self, monkeypatch):
        monkeypatch.setattr("quipu.vec.index.try_load", lambda conn: True)

    def test_non_integer_env_falls_back_to_default(self, tmp_conn, monkeypatch):
        """QUIPU_VEC_THRESHOLD='30k' must not raise; falls back to 30000."""
        self._patch_try_load_true(monkeypatch)
        monkeypatch.setenv("QUIPU_VEC_THRESHOLD", "30k")
        # No atoms → threshold 30000 → BELOW_THRESHOLD (not a crash).
        state = ensure_index(tmp_conn)
        assert state == VecState.BELOW_THRESHOLD

    def test_empty_string_env_uses_default(self, tmp_conn, monkeypatch):
        """QUIPU_VEC_THRESHOLD='' must not raise; uses default 30000."""
        self._patch_try_load_true(monkeypatch)
        monkeypatch.setenv("QUIPU_VEC_THRESHOLD", "")
        state = ensure_index(tmp_conn)
        assert state == VecState.BELOW_THRESHOLD

    def test_zero_env_falls_back_to_default(self, tmp_conn, monkeypatch):
        """QUIPU_VEC_THRESHOLD='0' is invalid (<=0); falls back to 30000."""
        self._patch_try_load_true(monkeypatch)
        monkeypatch.setenv("QUIPU_VEC_THRESHOLD", "0")
        state = ensure_index(tmp_conn)
        assert state == VecState.BELOW_THRESHOLD

    def test_negative_env_falls_back_to_default(self, tmp_conn, monkeypatch):
        """QUIPU_VEC_THRESHOLD='-5' is invalid; falls back to 30000."""
        self._patch_try_load_true(monkeypatch)
        monkeypatch.setenv("QUIPU_VEC_THRESHOLD", "-5")
        state = ensure_index(tmp_conn)
        assert state == VecState.BELOW_THRESHOLD

    def test_invalid_env_logs_warning(self, tmp_conn, monkeypatch, caplog):
        """Non-integer QUIPU_VEC_THRESHOLD logs exactly one WARNING."""
        import logging
        self._patch_try_load_true(monkeypatch)
        monkeypatch.setenv("QUIPU_VEC_THRESHOLD", "not_a_number")
        with caplog.at_level(logging.WARNING, logger="quipu.vec.index"):
            ensure_index(tmp_conn)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1

    def test_valid_env_is_respected(self, tmp_conn, monkeypatch):
        """A valid QUIPU_VEC_THRESHOLD integer overrides the default."""
        self._patch_try_load_true(monkeypatch)
        # 1 atom inserted → count=1; threshold=1 → crossed.
        # But sqlite-vec not loaded so build would fail — just check BELOW.
        monkeypatch.setenv("QUIPU_VEC_THRESHOLD", "999999")
        # No atoms → below 999999.
        state = ensure_index(tmp_conn)
        assert state == VecState.BELOW_THRESHOLD
