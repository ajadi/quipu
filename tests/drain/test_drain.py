"""Tests for quipu.capture.drain.drain().

All tests use fake_engine (injected via conftest) so the real ONNX model is
never loaded. tmp_store provides a fresh SQLite DB per test. Queue files are
written by the test into tmp_path.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from quipu.capture.drain import drain


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_record(
    content: str = "hello world",
    project_id: str = "P",
    ts: str = "2026-01-02T03:04:05Z",
    metadata: dict | None = None,
    **extra,
) -> dict:
    record: dict = {
        "v": 1,
        "source": "test",
        "agent": "test-agent",
        "task_id": "T1",
        "project_id": project_id,
        "ts": ts,
        "content": content,
    }
    if metadata is not None:
        record["metadata"] = metadata
    record.update(extra)
    return record


def _write_queue(path: Path, records: list) -> None:
    """Write records as JSONL to path."""
    lines = [json.dumps(r) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Test 1 — Happy path: 3 valid records written, queue gone
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_three_records_written(self, fake_engine, tmp_store, tmp_path):
        queue = tmp_path / "capture-queue.jsonl"
        records = [
            _make_record(content="record one", project_id="P"),
            _make_record(content="record two", project_id="P"),
            _make_record(content="record three", project_id="P"),
        ]
        _write_queue(queue, records)

        counts = drain(queue_path=str(queue), project_id="P", store=tmp_store)

        assert counts["written"] == 3
        assert counts["skipped_malformed"] == 0
        assert counts["skipped_secret"] == 0
        assert counts["skipped_foreign"] == 0

    def test_queue_file_gone_after_drain(self, fake_engine, tmp_store, tmp_path):
        queue = tmp_path / "capture-queue.jsonl"
        _write_queue(queue, [_make_record(content="hello", project_id="P")])

        drain(queue_path=str(queue), project_id="P", store=tmp_store)

        assert not queue.exists(), "queue file should be removed after drain"

    def test_atoms_in_store(self, fake_engine, tmp_store, tmp_path):
        queue = tmp_path / "capture-queue.jsonl"
        records = [
            _make_record(content="record one", project_id="P"),
            _make_record(content="record two", project_id="P"),
            _make_record(content="record three", project_id="P"),
        ]
        _write_queue(queue, records)

        drain(queue_path=str(queue), project_id="P", store=tmp_store)

        atoms = tmp_store.list_by_project("P")
        assert len(atoms) == 3


# ---------------------------------------------------------------------------
# Test 2 — Malformed skip: bad JSON + missing content + valid records
# ---------------------------------------------------------------------------

class TestMalformedSkip:
    def test_skips_bad_json_and_missing_content(self, fake_engine, tmp_store, tmp_path):
        queue = tmp_path / "capture-queue.jsonl"
        lines = [
            "this is not json at all",
            json.dumps({"project_id": "P", "ts": "2026-01-01T00:00:00Z"}),  # no content
            json.dumps(_make_record(content="valid one", project_id="P")),
            json.dumps(_make_record(content="valid two", project_id="P")),
        ]
        queue.write_text("\n".join(lines) + "\n", encoding="utf-8")

        counts = drain(queue_path=str(queue), project_id="P", store=tmp_store)

        assert counts["written"] == 2
        assert counts["skipped_malformed"] == 2
        assert counts["skipped_secret"] == 0
        assert counts["skipped_foreign"] == 0

    def test_does_not_abort_on_bad_lines(self, fake_engine, tmp_store, tmp_path):
        """Drain continues processing after skipping malformed records."""
        queue = tmp_path / "capture-queue.jsonl"
        lines = [
            "{bad json",
            json.dumps(_make_record(content="survived", project_id="P")),
        ]
        queue.write_text("\n".join(lines) + "\n", encoding="utf-8")

        counts = drain(queue_path=str(queue), project_id="P", store=tmp_store)

        assert counts["written"] == 1
        assert counts["skipped_malformed"] == 1


# ---------------------------------------------------------------------------
# Test 3 — Secret refusal
# ---------------------------------------------------------------------------

class TestSecretRefusal:
    def test_skips_openai_key(self, fake_engine, tmp_store, tmp_path):
        queue = tmp_path / "capture-queue.jsonl"
        secret_content = "sk-" + "a" * 40
        records = [
            _make_record(content=secret_content, project_id="P"),
            _make_record(content="totally normal content", project_id="P"),
        ]
        _write_queue(queue, records)

        counts = drain(queue_path=str(queue), project_id="P", store=tmp_store)

        assert counts["written"] == 1
        assert counts["skipped_secret"] == 1

    def test_secret_not_written_to_store(self, fake_engine, tmp_store, tmp_path):
        queue = tmp_path / "capture-queue.jsonl"
        secret_content = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpA==\n-----END RSA PRIVATE KEY-----"
        records = [
            _make_record(content=secret_content, project_id="P"),
        ]
        _write_queue(queue, records)

        drain(queue_path=str(queue), project_id="P", store=tmp_store)

        atoms = tmp_store.list_by_project("P")
        assert len(atoms) == 0

    def test_pem_key_skipped(self, fake_engine, tmp_store, tmp_path):
        queue = tmp_path / "capture-queue.jsonl"
        records = [
            _make_record(content="-----BEGIN PRIVATE KEY-----\nMIIEvQ==\n-----END PRIVATE KEY-----", project_id="P"),
            _make_record(content="clean record here", project_id="P"),
        ]
        _write_queue(queue, records)

        counts = drain(queue_path=str(queue), project_id="P", store=tmp_store)

        assert counts["skipped_secret"] == 1
        assert counts["written"] == 1


# ---------------------------------------------------------------------------
# Test 4 — Rotation/truncation: .processing temp is cleaned up
# ---------------------------------------------------------------------------

class TestRotationTruncation:
    def test_processing_file_not_present_after_drain(self, fake_engine, tmp_store, tmp_path):
        queue = tmp_path / "capture-queue.jsonl"
        _write_queue(queue, [_make_record(content="hello", project_id="P")])

        drain(queue_path=str(queue), project_id="P", store=tmp_store)

        processing = Path(str(queue) + ".processing")
        assert not processing.exists(), ".processing temp file should be removed"

    def test_original_queue_removed_after_drain(self, fake_engine, tmp_store, tmp_path):
        queue = tmp_path / "capture-queue.jsonl"
        _write_queue(queue, [_make_record(content="hello", project_id="P")])

        drain(queue_path=str(queue), project_id="P", store=tmp_store)

        assert not queue.exists(), "original queue file should be removed"


# ---------------------------------------------------------------------------
# Test 5 — Idempotency: re-drain on absent queue returns zeros, no exception
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_drain_absent_queue_returns_zeros(self, fake_engine, tmp_store, tmp_path):
        queue = tmp_path / "capture-queue.jsonl"
        # queue does not exist
        counts = drain(queue_path=str(queue), project_id="P", store=tmp_store)

        assert counts == {
            "written": 0,
            "skipped_malformed": 0,
            "skipped_secret": 0,
            "skipped_foreign": 0,
        }

    def test_second_drain_after_first_is_noop(self, fake_engine, tmp_store, tmp_path):
        queue = tmp_path / "capture-queue.jsonl"
        _write_queue(queue, [_make_record(content="one", project_id="P")])

        drain(queue_path=str(queue), project_id="P", store=tmp_store)

        # Second drain on the now-absent queue
        counts2 = drain(queue_path=str(queue), project_id="P", store=tmp_store)

        assert counts2 == {
            "written": 0,
            "skipped_malformed": 0,
            "skipped_secret": 0,
            "skipped_foreign": 0,
        }


# ---------------------------------------------------------------------------
# Test 6 — Bound-scope: foreign project_id skipped
# ---------------------------------------------------------------------------

class TestBoundScope:
    def test_foreign_project_id_skipped(self, fake_engine, tmp_store, tmp_path):
        queue = tmp_path / "capture-queue.jsonl"
        records = [
            _make_record(content="should be skipped", project_id="OTHER"),
            _make_record(content="should be written", project_id="P"),
        ]
        _write_queue(queue, records)

        counts = drain(queue_path=str(queue), project_id="P", store=tmp_store)

        assert counts["written"] == 1
        assert counts["skipped_foreign"] == 1

    def test_foreign_record_not_in_store(self, fake_engine, tmp_store, tmp_path):
        queue = tmp_path / "capture-queue.jsonl"
        records = [
            _make_record(content="foreign content", project_id="OTHER"),
        ]
        _write_queue(queue, records)

        drain(queue_path=str(queue), project_id="P", store=tmp_store)

        atoms_p = tmp_store.list_by_project("P")
        atoms_other = tmp_store.list_by_project("OTHER")
        assert len(atoms_p) == 0
        assert len(atoms_other) == 0


# ---------------------------------------------------------------------------
# Test 7 — EVENT-TIME happy: drained atom.created_at == queue ts
# ---------------------------------------------------------------------------

class TestEventTimeHappy:
    def test_created_at_equals_ts(self, fake_engine, tmp_store, tmp_path):
        queue = tmp_path / "capture-queue.jsonl"
        ts = "2026-01-02T03:04:05Z"
        _write_queue(queue, [_make_record(content="timed content", project_id="P", ts=ts)])

        drain(queue_path=str(queue), project_id="P", store=tmp_store)

        atoms = tmp_store.list_by_project("P")
        assert len(atoms) == 1
        assert atoms[0].created_at == ts, (
            f"Expected created_at={ts!r}, got {atoms[0].created_at!r}"
        )

    def test_millisecond_precision_ts_accepted(self, fake_engine, tmp_store, tmp_path):
        """Millisecond-precision ts is also valid and preserved."""
        queue = tmp_path / "capture-queue.jsonl"
        ts = "2026-03-15T12:30:45.123Z"
        _write_queue(queue, [_make_record(content="ms precision", project_id="P", ts=ts)])

        drain(queue_path=str(queue), project_id="P", store=tmp_store)

        atoms = tmp_store.list_by_project("P")
        assert len(atoms) == 1
        assert atoms[0].created_at == ts


# ---------------------------------------------------------------------------
# Test 8 — EVENT-TIME malformed: bad ts → atom still created, created_at is valid
# ---------------------------------------------------------------------------

class TestEventTimeMalformed:
    def test_bad_ts_drain_continues(self, fake_engine, tmp_store, tmp_path):
        queue = tmp_path / "capture-queue.jsonl"
        _write_queue(queue, [
            _make_record(content="has bad ts", project_id="P", ts="not-a-date"),
        ])

        counts = drain(queue_path=str(queue), project_id="P", store=tmp_store)

        # Drain must continue — record still written
        assert counts["written"] == 1

    def test_bad_ts_atom_created_at_non_empty(self, fake_engine, tmp_store, tmp_path):
        queue = tmp_path / "capture-queue.jsonl"
        _write_queue(queue, [
            _make_record(content="malformed ts record", project_id="P", ts="not-a-date"),
        ])

        drain(queue_path=str(queue), project_id="P", store=tmp_store)

        atoms = tmp_store.list_by_project("P")
        assert len(atoms) == 1
        created_at = atoms[0].created_at
        assert created_at, "created_at should not be empty"
        # Should start with a 4-digit year (fallback to now)
        assert len(created_at) >= 4
        assert created_at[:4].isdigit(), f"Expected 4-digit year prefix, got {created_at!r}"

    def test_missing_ts_field_also_defaults(self, fake_engine, tmp_store, tmp_path):
        """Record with no ts field at all → falls back to now, not an error."""
        queue = tmp_path / "capture-queue.jsonl"
        record = {
            "v": 1,
            "source": "test",
            "project_id": "P",
            "content": "no ts field",
        }
        _write_queue(queue, [record])

        counts = drain(queue_path=str(queue), project_id="P", store=tmp_store)
        assert counts["written"] == 1

        atoms = tmp_store.list_by_project("P")
        assert atoms[0].created_at


# ---------------------------------------------------------------------------
# TASK-023 — drain threads session_id from queue record to atom
# ---------------------------------------------------------------------------

class TestDrainSessionId:
    def test_session_id_passed_to_atom(self, fake_engine, tmp_store, tmp_path):
        queue = tmp_path / "capture-queue.jsonl"
        _write_queue(queue, [
            _make_record(
                content="session record", project_id="P",
                session_id="my-session",
            ),
        ])

        drain(queue_path=str(queue), project_id="P", store=tmp_store)

        atoms = tmp_store.list_by_project("P")
        assert len(atoms) == 1
        assert atoms[0].session_id == "my-session"

    def test_session_id_null_when_absent_from_record(self, fake_engine, tmp_store, tmp_path):
        queue = tmp_path / "capture-queue.jsonl"
        _write_queue(queue, [
            _make_record(
                content="no session field", project_id="P",
                # session_id key absent entirely
            ),
        ])

        drain(queue_path=str(queue), project_id="P", store=tmp_store)

        atoms = tmp_store.list_by_project("P")
        assert len(atoms) == 1
        assert atoms[0].session_id is None

    def test_session_id_null_when_empty_string_in_record(self, fake_engine, tmp_store, tmp_path):
        queue = tmp_path / "capture-queue.jsonl"
        _write_queue(queue, [
            _make_record(
                content="empty session", project_id="P",
                session_id="",
            ),
        ])

        drain(queue_path=str(queue), project_id="P", store=tmp_store)

        atoms = tmp_store.list_by_project("P")
        assert len(atoms) == 1
        assert atoms[0].session_id is None

    def test_session_id_null_when_not_a_string(self, fake_engine, tmp_store, tmp_path):
        queue = tmp_path / "capture-queue.jsonl"
        _write_queue(queue, [
            _make_record(
                content="numeric session", project_id="P",
                session_id=12345,
            ),
        ])

        drain(queue_path=str(queue), project_id="P", store=tmp_store)

        atoms = tmp_store.list_by_project("P")
        assert len(atoms) == 1
        assert atoms[0].session_id is None


# ---------------------------------------------------------------------------
# Test 9 — Regression: write() and Store.insert() without created_at → non-empty
# ---------------------------------------------------------------------------

class TestRegressionDefaultCreatedAt:
    def test_write_without_created_at_has_non_empty_created_at(self, fake_engine, tmp_store):
        from quipu.write import write

        import datetime
        current_year = str(datetime.datetime.now().year)

        atom_id = write(
            "regression test content",
            project_id="P",
            store=tmp_store,
        )

        atom = tmp_store.get(atom_id)
        assert atom is not None
        assert atom.created_at, "created_at should not be empty or None"
        assert atom.created_at.startswith(current_year), (
            f"Expected created_at to start with {current_year}, got {atom.created_at!r}"
        )

    def test_store_insert_without_created_at_has_non_empty_created_at(self, fake_engine, tmp_store):
        import datetime
        current_year = str(datetime.datetime.now().year)

        atom = tmp_store.insert(
            content="direct insert regression",
            project_id="P",
        )

        assert atom.created_at, "created_at should not be empty or None"
        assert atom.created_at.startswith(current_year), (
            f"Expected created_at to start with {current_year}, got {atom.created_at!r}"
        )
