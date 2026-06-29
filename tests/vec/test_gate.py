"""Tests for quipu.vec._gate — availability gate for sqlite-vec.

All tests run without sqlite-vec installed. Load failures are simulated
via monkeypatch.
"""

from __future__ import annotations

import logging
import sqlite3

import pytest

from quipu.vec._gate import is_loaded, reset_warned, try_load


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fresh_conn():
    """Return an in-memory sqlite3 connection (no quipu schema needed)."""
    return sqlite3.connect(":memory:")


# ---------------------------------------------------------------------------
# try_load — failure paths (always-run, no sqlite-vec required)
# ---------------------------------------------------------------------------

class TestTryLoadFailure:
    def setup_method(self):
        reset_warned()

    def test_import_error_returns_false(self, monkeypatch):
        """ImportError (sqlite-vec not installed) → False, no crash."""
        monkeypatch.setattr("builtins.__import__", _raise_import_on_sqlite_vec)
        conn = _fresh_conn()
        result = try_load(conn)
        assert result is False

    def test_import_error_logs_warning_once(self, monkeypatch, caplog):
        """First failure logs exactly one WARNING."""
        monkeypatch.setattr("builtins.__import__", _raise_import_on_sqlite_vec)
        conn = _fresh_conn()
        with caplog.at_level(logging.WARNING, logger="quipu.vec._gate"):
            try_load(conn)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1

    def test_import_error_warns_only_once(self, monkeypatch, caplog):
        """Second failure does NOT log another warning (dedup)."""
        monkeypatch.setattr("builtins.__import__", _raise_import_on_sqlite_vec)
        conn = _fresh_conn()
        with caplog.at_level(logging.WARNING, logger="quipu.vec._gate"):
            try_load(conn)  # first call: warns
            try_load(conn)  # second call: silent
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1

    def test_enable_load_extension_raises_not_supported(self, monkeypatch):
        """NotSupportedError from enable_load_extension → False, no crash."""
        # Cannot monkeypatch a C-level read-only attr on sqlite3.Connection;
        # use a plain stub that exposes the same surface try_load touches.
        class _StubConn:
            def enable_load_extension(self, flag: bool) -> None:  # noqa: FBT001
                raise sqlite3.NotSupportedError("disabled by hardened build")

        result = try_load(_StubConn())  # type: ignore[arg-type]
        assert result is False

    def test_operational_error_returns_false(self, monkeypatch):
        """OperationalError during load → False."""
        import sys
        import types

        # Inject a fake sqlite_vec module whose load() raises OperationalError.
        fake_mod = types.ModuleType("sqlite_vec")
        fake_mod.load = lambda conn: (_ for _ in ()).throw(
            sqlite3.OperationalError("cannot open shared object")
        )
        monkeypatch.setitem(sys.modules, "sqlite_vec", fake_mod)

        conn = _fresh_conn()
        result = try_load(conn)
        assert result is False

    def test_attribute_error_returns_false(self, monkeypatch):
        """AttributeError (module has no .load) → False."""
        import sys
        import types

        fake_mod = types.ModuleType("sqlite_vec")
        # No .load attribute at all.
        monkeypatch.setitem(sys.modules, "sqlite_vec", fake_mod)

        conn = _fresh_conn()
        result = try_load(conn)
        assert result is False


# ---------------------------------------------------------------------------
# is_loaded — without extension
# ---------------------------------------------------------------------------

class TestIsLoaded:
    def test_false_without_extension(self):
        """is_loaded returns False when vec_version() is not available."""
        conn = _fresh_conn()
        assert is_loaded(conn) is False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__


def _raise_import_on_sqlite_vec(name, *args, **kwargs):
    if name == "sqlite_vec":
        raise ImportError("sqlite_vec not installed (simulated)")
    return _real_import(name, *args, **kwargs)
