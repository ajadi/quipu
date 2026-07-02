"""Tests for quipu.cli.get_version and __version__ import."""

from __future__ import annotations

import quipu
from quipu.cli import get_version


class TestGetVersion:
    def test_returns_string(self):
        v = get_version()
        assert isinstance(v, str)
        assert len(v) > 0

    def test_starts_with_digit(self):
        v = get_version()
        assert v[0].isdigit(), f"Expected version to start with a digit, got {v!r}"

    def test_matches_init_version(self):
        assert get_version() == quipu.__version__

    def test_init_version_is_semver_like(self):
        parts = quipu.__version__.split(".")
        assert len(parts) == 3, f"Expected 3 semver parts, got {quipu.__version__!r}"
        for p in parts:
            assert p.isdigit(), f"Expected numeric part, got {p!r}"

    def test_baked_version_used_when_present(self, monkeypatch):
        monkeypatch.setattr(quipu, "__version__", "1.2.3")
        assert quipu.__version__ == "1.2.3"
        assert get_version() == "1.2.3"
