"""Unit tests for the quipu_prime handler via dispatch(...)."""

from __future__ import annotations

import json

import pytest

from quipu.mcp.tools import dispatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(result) -> dict:
    assert len(result) == 1
    return json.loads(result[0].text)


# ---------------------------------------------------------------------------
# quipu_prime via dispatch
# ---------------------------------------------------------------------------


class TestQuipuPrimeDispatch:
    def test_no_project_id_returns_primed_false_no_error(self, tmp_store):
        """No project_id (default None) → primed=false, no 'error' key."""
        result = dispatch(
            "quipu_prime",
            store=tmp_store,
            default_project_id=None,
            arguments={},
        )
        data = _parse(result)
        assert "error" not in data
        assert data["primed"] is False
        assert data["results"] == []
        assert "note" in data

    def test_happy_path_empty_store(self, tmp_store, project_id, fake_engine):
        """Bound project, empty store → primed=true, results=[], no error."""
        result = dispatch(
            "quipu_prime",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert "error" not in data
        assert data["primed"] is True
        assert data["results"] == []
        assert data["note"] == "no memory yet"

    def test_write_then_prime_finds_record(self, tmp_store, project_id, fake_engine):
        """Write a record, then prime → written id appears in results."""
        write_result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "quipu prime test marker beta"},
        )
        written_id = json.loads(write_result[0].text)["id"]

        prime_result = dispatch(
            "quipu_prime",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"topic": "prime test marker"},
        )
        data = _parse(prime_result)
        assert "error" not in data
        assert data["primed"] is True
        result_ids = [r["id"] for r in data["results"]]
        assert written_id in result_ids

    def test_with_topic_arg(self, tmp_store, project_id, fake_engine):
        """topic argument is accepted and used."""
        result = dispatch(
            "quipu_prime",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"topic": "architecture decisions"},
        )
        data = _parse(result)
        assert "error" not in data
        assert "primed" in data

    def test_invalid_top_k_zero(self, tmp_store, project_id):
        """top_k=0 → structured error (not a crash)."""
        result = dispatch(
            "quipu_prime",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"top_k": 0},
        )
        data = _parse(result)
        assert "error" in data
        assert "top_k" in data["error"]

    def test_invalid_top_k_bool(self, tmp_store, project_id):
        """top_k=True (bool subclass of int) → error."""
        result = dispatch(
            "quipu_prime",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"top_k": True},
        )
        data = _parse(result)
        assert "error" in data

    def test_invalid_top_k_over_limit(self, tmp_store, project_id):
        """top_k=5000 → structured error."""
        result = dispatch(
            "quipu_prime",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"top_k": 5000},
        )
        data = _parse(result)
        assert "error" in data
        assert "1000" in data["error"]

    def test_extra_args_tolerated(self, tmp_store, project_id, fake_engine):
        """Extra unknown arguments are silently ignored."""
        result = dispatch(
            "quipu_prime",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"unknown_arg": "value", "another": 123},
        )
        data = _parse(result)
        assert "error" not in data
        assert "primed" in data

    def test_result_fields_present(self, tmp_store, project_id, fake_engine):
        """Prime results have the same fields as quipu_search results."""
        dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "field check memory record"},
        )
        result = dispatch(
            "quipu_prime",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        if data["results"]:
            r = data["results"][0]
            for field in ("id", "content", "score", "tier", "type", "scope",
                          "invalidated", "metadata"):
                assert field in r, f"missing field: {field}"
            assert "embedding" not in r

    def test_scope_lock_respected_on_bound_server(self, tmp_store):
        """On a bound server, supplying a different project_id is rejected."""
        result = dispatch(
            "quipu_prime",
            store=tmp_store,
            default_project_id="project_a",
            arguments={"project_id": "project_b"},
        )
        data = _parse(result)
        assert "error" in data
        assert "not permitted" in data["error"] or "bound" in data["error"]

    def test_top_k_1000_accepted(self, tmp_store, project_id, fake_engine):
        """top_k=1000 (max boundary) is accepted."""
        result = dispatch(
            "quipu_prime",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"top_k": 1000},
        )
        data = _parse(result)
        assert "error" not in data

    def test_topic_non_string_returns_error(self, tmp_store, project_id):
        """topic=42 (non-string) → structured error."""
        result = dispatch(
            "quipu_prime",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"topic": 42},
        )
        data = _parse(result)
        assert "error" in data
        assert "string" in data["error"]

    def test_topic_over_1000_chars_returns_error(self, tmp_store, project_id):
        """topic longer than 1000 characters → structured error."""
        result = dispatch(
            "quipu_prime",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"topic": "x" * 1001},
        )
        data = _parse(result)
        assert "error" in data
        assert "1000" in data["error"]
