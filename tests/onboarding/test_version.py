"""Tests for quipu.cli.get_version and __version__ fallback."""

from __future__ import annotations

import importlib
import sys
import types
from unittest.mock import patch

import pytest

from quipu.cli import __version__, get_version


class TestGetVersion:
    def test_returns_string(self):
        v = get_version()
        assert isinstance(v, str)
        assert len(v) > 0

    def test_starts_with_digit(self):
        """Version string should look like semver (major.minor.patch)."""
        v = get_version()
        assert v[0].isdigit(), f"Expected version to start with a digit, got {v!r}"

    def test_fallback_when_metadata_absent(self):
        """When importlib.metadata raises, get_version() falls back to __version__."""
        def _raise(_name):
            raise importlib.metadata.PackageNotFoundError("quipu-mcp")

        with patch("importlib.metadata.version", side_effect=_raise):
            v = get_version()
        assert v == __version__

    def test_module_level_version_constant(self):
        assert __version__ == "0.2.0"

    def test_get_version_matches_metadata_when_installed(self, monkeypatch):
        """When metadata is available, get_version() returns its value."""
        monkeypatch.setattr("importlib.metadata.version", lambda _name: "9.8.7")
        v = get_version()
        assert v == "9.8.7"
