"""Tests for quipu.write.flush.flush()."""

from __future__ import annotations

import json
import sys

import pytest
from tests._semantic import TEST_EMBED_DIM

from quipu.storage.store import pack_embedding
from quipu.write.flush import flush, ENRICHED_FLAG
from tests.write.conftest import get_flush_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(dim: int, index: int) -> list[float]:
    v = [0.0] * dim
    v[index] = 1.0
    return v


def _insert_atom(store, content: str, project_id: str = "proj", enriched: bool = False):
    """Insert an atom with optional enriched flag in metadata."""
    emb = pack_embedding(_unit_vec(TEST_EMBED_DIM, 0))
    metadata = {"enriched": enriched} if enriched else {}
    return store.insert(
        content=content,
        embedding=emb,
        project_id=project_id,
        metadata=metadata,
    )


def _canned_response(summary="Test summary", entities=None, keywords=None) -> dict:
    """Build a minimal Haiku messages-API JSON response."""
    if entities is None:
        entities = ["Entity1"]
    if keywords is None:
        keywords = ["keyword1"]
    body = json.dumps({
        "summary": summary,
        "entities": entities,
        "keywords": keywords,
    })
    return {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": body}],
        "model": "claude-haiku-4-5",
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 20},
    }


# ---------------------------------------------------------------------------
# Tests: ANTHROPIC_API_KEY absent
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("semantic_model")
class TestFlushNoApiKey:
    def test_returns_skipped_true_when_key_absent(self, tmp_store, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = flush(store=tmp_store)
        assert result["skipped"] is True

    def test_returns_zero_enriched_when_key_absent(self, tmp_store, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = flush(store=tmp_store)
        assert result["enriched"] == 0

    def test_reason_is_no_api_key(self, tmp_store, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = flush(store=tmp_store)
        assert result["reason"] == "no_api_key"

    def test_zero_http_calls_when_key_absent(self, tmp_store, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        _insert_atom(tmp_store, "some content")

        flush_mod = get_flush_module()
        calls = []

        def _fake_http(url, headers, payload):
            calls.append((url, headers, payload))
            return {}

        monkeypatch.setattr(flush_mod, "_http_post_json", _fake_http)
        flush(store=tmp_store)
        assert calls == [], f"Expected zero calls, got {len(calls)}"

    def test_does_not_raise_when_key_absent(self, tmp_store, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # Should not raise
        result = flush(store=tmp_store)
        assert result is not None


# ---------------------------------------------------------------------------
# Tests: ANTHROPIC_API_KEY present — metadata writeback
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("semantic_model")
class TestFlushWithApiKey:
    def test_enriches_unenriched_atom(self, tmp_store, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        atom = _insert_atom(tmp_store, "content to enrich", project_id="proj")

        flush_mod = get_flush_module()
        monkeypatch.setattr(
            flush_mod, "_http_post_json",
            lambda url, h, p: _canned_response("Summary text", ["Alice"], ["learning"])
        )

        result = flush(project_id="proj", store=tmp_store)
        assert result["enriched"] == 1
        assert result["skipped"] is False

    def test_writeback_sets_enriched_true(self, tmp_store, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        atom = _insert_atom(tmp_store, "content to enrich", project_id="proj")

        flush_mod = get_flush_module()
        monkeypatch.setattr(
            flush_mod, "_http_post_json",
            lambda url, h, p: _canned_response()
        )

        flush(project_id="proj", store=tmp_store)

        updated = tmp_store.get(atom.id)
        assert updated.metadata.get(ENRICHED_FLAG) is True

    def test_writeback_stores_summary(self, tmp_store, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        atom = _insert_atom(tmp_store, "content to enrich", project_id="proj")

        flush_mod = get_flush_module()
        monkeypatch.setattr(
            flush_mod, "_http_post_json",
            lambda url, h, p: _canned_response(summary="My summary")
        )

        flush(project_id="proj", store=tmp_store)
        updated = tmp_store.get(atom.id)
        assert updated.metadata.get("summary") == "My summary"

    def test_writeback_stores_entities(self, tmp_store, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        atom = _insert_atom(tmp_store, "content to enrich", project_id="proj")

        flush_mod = get_flush_module()
        monkeypatch.setattr(
            flush_mod, "_http_post_json",
            lambda url, h, p: _canned_response(entities=["Alice", "Bob"])
        )

        flush(project_id="proj", store=tmp_store)
        updated = tmp_store.get(atom.id)
        assert updated.metadata.get("entities") == ["Alice", "Bob"]

    def test_writeback_stores_keywords(self, tmp_store, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        atom = _insert_atom(tmp_store, "content to enrich", project_id="proj")

        flush_mod = get_flush_module()
        monkeypatch.setattr(
            flush_mod, "_http_post_json",
            lambda url, h, p: _canned_response(keywords=["kw1", "kw2"])
        )

        flush(project_id="proj", store=tmp_store)
        updated = tmp_store.get(atom.id)
        assert updated.metadata.get("keywords") == ["kw1", "kw2"]

    def test_writeback_sets_enriched_at_iso8601(self, tmp_store, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        atom = _insert_atom(tmp_store, "content", project_id="proj")

        flush_mod = get_flush_module()
        monkeypatch.setattr(
            flush_mod, "_http_post_json",
            lambda url, h, p: _canned_response()
        )

        flush(project_id="proj", store=tmp_store)
        updated = tmp_store.get(atom.id)
        enriched_at = updated.metadata.get("enriched_at")
        assert enriched_at is not None
        # Basic ISO-8601 check: should contain 'T' separator
        assert "T" in enriched_at

    def test_already_enriched_atom_skipped(self, tmp_store, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        atom = _insert_atom(tmp_store, "already enriched", enriched=True)

        flush_mod = get_flush_module()
        calls = []
        monkeypatch.setattr(
            flush_mod, "_http_post_json",
            lambda url, h, p: calls.append(p) or _canned_response()
        )

        result = flush(store=tmp_store)
        assert result["enriched"] == 0
        assert calls == []

    def test_skipped_false_when_key_present(self, tmp_store, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

        flush_mod = get_flush_module()
        monkeypatch.setattr(
            flush_mod, "_http_post_json",
            lambda url, h, p: _canned_response()
        )

        result = flush(store=tmp_store)
        assert result["skipped"] is False
        assert result["reason"] is None

    def test_bad_json_response_leaves_atom_unenriched(self, tmp_store, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        atom = _insert_atom(tmp_store, "bad json content", project_id="proj")

        flush_mod = get_flush_module()
        monkeypatch.setattr(
            flush_mod, "_http_post_json",
            lambda url, h, p: {
                "content": [{"type": "text", "text": "NOT JSON {{{"}]
            }
        )

        # Should not raise
        result = flush(project_id="proj", store=tmp_store)
        assert result["enriched"] == 0

        updated = tmp_store.get(atom.id)
        assert not updated.metadata.get(ENRICHED_FLAG, False)

    def test_empty_content_blocks_leaves_atom_unenriched(self, tmp_store, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        atom = _insert_atom(tmp_store, "some content", project_id="proj")

        flush_mod = get_flush_module()
        monkeypatch.setattr(
            flush_mod, "_http_post_json",
            lambda url, h, p: {"content": []}
        )

        result = flush(project_id="proj", store=tmp_store)
        assert result["enriched"] == 0
        assert not tmp_store.get(atom.id).metadata.get(ENRICHED_FLAG, False)

    def test_non_string_summary_dropped_silently(self, tmp_store, monkeypatch):
        """Non-string summary is dropped; enriched:true is still set."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        atom = _insert_atom(tmp_store, "content", project_id="proj")

        flush_mod = get_flush_module()
        bad_response = {
            "content": [{"type": "text", "text": json.dumps({
                "summary": 12345,       # not a string
                "entities": ["Alice"],
                "keywords": ["kw"],
            })}]
        }
        monkeypatch.setattr(flush_mod, "_http_post_json", lambda u, h, p: bad_response)

        result = flush(project_id="proj", store=tmp_store)
        assert result["enriched"] == 1  # still counts as enriched
        updated = tmp_store.get(atom.id)
        assert updated.metadata.get(ENRICHED_FLAG) is True
        assert "summary" not in updated.metadata  # dropped

    def test_non_list_entities_dropped_silently(self, tmp_store, monkeypatch):
        """Non-list entities and keywords are dropped; enriched:true is still set."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")
        atom = _insert_atom(tmp_store, "content", project_id="proj")

        flush_mod = get_flush_module()
        bad_response = {
            "content": [{"type": "text", "text": json.dumps({
                "summary": "good summary",
                "entities": "not a list",   # not a list
                "keywords": {"k": "v"},     # not a list
            })}]
        }
        monkeypatch.setattr(flush_mod, "_http_post_json", lambda u, h, p: bad_response)

        result = flush(project_id="proj", store=tmp_store)
        assert result["enriched"] == 1
        updated = tmp_store.get(atom.id)
        assert updated.metadata.get(ENRICHED_FLAG) is True
        assert updated.metadata.get("summary") == "good summary"
        assert "entities" not in updated.metadata  # dropped
        assert "keywords" not in updated.metadata  # dropped


# ---------------------------------------------------------------------------
# Tests: flush is NOT called by write
# ---------------------------------------------------------------------------

class TestFlushNotCalledByWrite:
    def test_write_does_not_trigger_flush(self, fake_engine, tmp_store, monkeypatch):
        """Confirm flush is separate from write — no enrichment at write time."""
        flush_mod = get_flush_module()

        http_calls = []
        monkeypatch.setattr(
            flush_mod, "_http_post_json",
            lambda url, h, p: http_calls.append(p) or {}
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

        from quipu.write.pipeline import write
        write("content to write", project_id="proj", store=tmp_store)

        assert http_calls == [], "write() must not make Haiku calls"

    def test_enriched_false_immediately_after_write(self, fake_engine, tmp_store):
        """After write, atom.metadata.enriched is False (not enriched yet)."""
        from quipu.write.pipeline import write
        atom_id = write("content", store=tmp_store)
        atom = tmp_store.get(atom_id)
        assert atom.metadata.get("enriched") is False


# ---------------------------------------------------------------------------
# Tests: DI seam — own store management
# ---------------------------------------------------------------------------

class TestFlushDISeam:
    def test_flush_without_injected_store_does_not_raise(self, monkeypatch, tmp_path):
        """flush() with no store= arg opens its own store (no crash)."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # Use a temp path to avoid opening the real DB
        monkeypatch.setenv("QUIPU_DB_PATH", str(tmp_path / "t.db"))
        result = flush()
        assert result["skipped"] is True
