"""Tests for quipu.oplog.backfill — one-shot re-emit of pre-existing atoms.

Covers:
  (a) re-run emits 0 (idempotency via entries_for_record, NOT the UNIQUE backstop)
  (b) backfill + live producer do NOT double-emit the same atom
  (c) live atom -> "upsert", invalidated atom -> "invalidate"
  (d) ts == atom.updated_at on emitted entries
  (e) no key / no hub -> BackfillResult(status="inactive"), no crash, no rows
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from quipu.oplog.backfill import BackfillResult, backfill_project
from quipu.oplog.producer import emit, reset_cache
from quipu.storage import store as open_store
from quipu.sync._aad import aad_for

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_KEY = bytes(range(32))
TEST_KEY_B64 = base64.b64encode(TEST_KEY).decode()
PROJECT_ID = "proj-backfill-test"
CLIENT_ID = "client-backfill-test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blinded() -> str:
    return aad_for(PROJECT_ID, TEST_KEY).decode()


def _oplog_rows(store) -> list:
    return store._conn.execute(
        "SELECT * FROM oplog_entries ORDER BY sequence_no"
    ).fetchall()


def _count_rows(store) -> int:
    return len(_oplog_rows(store))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_producer_cache():
    reset_cache()
    yield
    reset_cache()


@pytest.fixture()
def tmp_store(tmp_path):
    s = open_store(str(tmp_path / "test.db"))
    yield s
    s.close()


@pytest.fixture()
def active_env(monkeypatch, tmp_path):
    """Set env vars so the producer/backfill gate considers sync active."""
    monkeypatch.setenv("QUIPU_KEY", TEST_KEY_B64)
    monkeypatch.setenv("QUIPU_HUB_URL", "http://hub-fake")
    monkeypatch.setenv("QUIPU_HUB_TOKEN", "fake-token")
    monkeypatch.setenv("QUIPU_CLIENT_ID", CLIENT_ID)
    monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))


# ---------------------------------------------------------------------------
# (a) Idempotency: re-run emits 0
# ---------------------------------------------------------------------------

class TestReRunIsNoOp:
    def test_second_run_emits_zero(self, tmp_store, active_env):
        # Insert atoms directly (no producer involvement -> no oplog rows yet).
        tmp_store.insert(content="atom one", project_id=PROJECT_ID)
        tmp_store.insert(content="atom two", project_id=PROJECT_ID)
        tmp_store.insert(content="atom three", project_id=PROJECT_ID)
        assert _count_rows(tmp_store) == 0

        r1 = backfill_project(tmp_store, PROJECT_ID)
        assert r1.status == "ok"
        assert r1.emitted == 3
        rows_after_first = _count_rows(tmp_store)
        assert rows_after_first == 3

        # Re-run: dedup pre-check must skip all; emitted=0; row count unchanged.
        r2 = backfill_project(tmp_store, PROJECT_ID)
        assert r2.status == "ok"
        assert r2.emitted == 0, "re-run must not emit any new entries"
        assert r2.skipped == 3
        assert _count_rows(tmp_store) == rows_after_first


# ---------------------------------------------------------------------------
# (b) Backfill + live producer: no double-emit
# ---------------------------------------------------------------------------

class TestNoDoubleEmitWithLive:
    def test_live_emitted_atom_is_skipped_by_backfill(self, tmp_store, active_env):
        # An atom the live producer already emitted.
        atom = tmp_store.insert(content="live atom", project_id=PROJECT_ID)
        emit(tmp_store, op="upsert", atom=atom, project_id=PROJECT_ID)
        assert _count_rows(tmp_store) == 1

        # A second atom that the producer never saw.
        tmp_store.insert(content="cold atom", project_id=PROJECT_ID)

        r = backfill_project(tmp_store, PROJECT_ID)
        assert r.status == "ok"
        assert r.emitted == 1, "only the cold atom should be emitted"
        assert r.skipped == 1, "the live-emitted atom should be skipped"
        # Total rows: 1 (live) + 1 (backfilled cold) = 2, no duplicate for live atom.
        assert _count_rows(tmp_store) == 2

        blinded = _blinded()
        from quipu.sync.oplog_store import OplogStore
        oplog = OplogStore(tmp_store._conn)
        assert len(oplog.entries_for_record(blinded, atom.id)) == 1


# ---------------------------------------------------------------------------
# (c) op mapping: live -> upsert, invalidated -> invalidate
# ---------------------------------------------------------------------------

class TestOpMapping:
    def test_live_upsert_invalidated_invalidate(self, tmp_store, active_env):
        live = tmp_store.insert(content="live", project_id=PROJECT_ID)
        dead = tmp_store.insert(content="dead", project_id=PROJECT_ID)
        tmp_store.update_invalidated(dead.id, True)

        r = backfill_project(tmp_store, PROJECT_ID)
        assert r.status == "ok"
        assert r.emitted == 2

        rows = {row["record_id"]: row["op"] for row in _oplog_rows(tmp_store)}
        assert rows[live.id] == "upsert"
        assert rows[dead.id] == "invalidate"


# ---------------------------------------------------------------------------
# (d) ts == atom.updated_at on emitted entries
# ---------------------------------------------------------------------------

class TestTsEqualsUpdatedAt:
    def test_ts_is_event_time(self, tmp_store, active_env):
        a1 = tmp_store.insert(
            content="fixed ts",
            project_id=PROJECT_ID,
            created_at="2020-01-02T03:04:05Z",
        )
        a2 = tmp_store.insert(content="default ts", project_id=PROJECT_ID)

        r = backfill_project(tmp_store, PROJECT_ID)
        assert r.emitted == 2

        ts_by_record = {
            row["record_id"]: row["ts"] for row in _oplog_rows(tmp_store)
        }
        assert ts_by_record[a1.id] == tmp_store.get(a1.id).updated_at
        assert ts_by_record[a1.id] == "2020-01-02T03:04:05Z"
        assert ts_by_record[a2.id] == tmp_store.get(a2.id).updated_at


# ---------------------------------------------------------------------------
# (e) no key / no hub -> inactive, no crash, no rows
# ---------------------------------------------------------------------------

class TestInactiveGate:
    def test_no_key_no_hub_is_inactive(self, tmp_store, monkeypatch, tmp_path):
        monkeypatch.delenv("QUIPU_KEY", raising=False)
        monkeypatch.delenv("QUIPU_PASSPHRASE", raising=False)
        monkeypatch.delenv("QUIPU_HUB_URL", raising=False)
        monkeypatch.delenv("QUIPU_HUB_TOKEN", raising=False)
        monkeypatch.setenv("QUIPU_CLIENT_ID", CLIENT_ID)
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))

        tmp_store.insert(content="local only", project_id=PROJECT_ID)

        r = backfill_project(tmp_store, PROJECT_ID)
        assert r == BackfillResult(status="inactive")
        assert _count_rows(tmp_store) == 0

    def test_no_hub_key_present_is_inactive(self, tmp_store, monkeypatch, tmp_path):
        monkeypatch.setenv("QUIPU_KEY", TEST_KEY_B64)
        monkeypatch.delenv("QUIPU_HUB_URL", raising=False)
        monkeypatch.delenv("QUIPU_HUB_TOKEN", raising=False)
        monkeypatch.setenv("QUIPU_CLIENT_ID", CLIENT_ID)
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))

        tmp_store.insert(content="no hub", project_id=PROJECT_ID)

        r = backfill_project(tmp_store, PROJECT_ID)
        assert r.status == "inactive"
        assert _count_rows(tmp_store) == 0
