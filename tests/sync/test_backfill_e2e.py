"""HEADLINE E2E test for the oplog backfill.

Models the core TASK-018 problem: atoms created BEFORE sync was enabled never
reach oplog_entries, so a fresh client cannot reconstruct them. Backfill closes
that gap.

Flow (mirrors tests/sync/test_producer_e2e.py harness):
  - Insert atoms on client A while sync is INACTIVE (producer emits 0 — asserted).
  - Activate sync + run backfill_project (emits one entry per atom).
  - Push A -> a REAL hub via hub.main.create_app wrapped in FastAPI TestClient
    (HttpTransport._request routed through the TestClient).
  - Pull on a FRESH client B (same key/bpid).
  - ASSERT: pre-existing atoms (content + invalidation state) reconstructed on B;
    one invalidated atom propagates as a tombstone on B; backfill re-run is a no-op.
"""

from __future__ import annotations

import base64
import hashlib
import os
import sys
import urllib.error
from pathlib import Path
from typing import Any

import pytest

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Set HUB_TOKENS before importing hub.main.
os.environ.setdefault("HUB_TOKENS", "test-sentinel-token-for-import")

from fastapi.testclient import TestClient

from hub.config import Config
from hub.main import create_app

from quipu.oplog.backfill import backfill_project
from quipu.oplog.producer import emit, reset_cache
from quipu.storage import store as open_store
from quipu.sync._aad import aad_for
from quipu.sync.client import HttpTransport
from quipu.sync.oplog_store import OplogStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_KEY = bytes(range(32))
TEST_KEY_B64 = base64.b64encode(TEST_KEY).decode()
TOKEN = "e2e-backfill-token"
TOKEN_HASH = hashlib.sha256(TOKEN.encode()).hexdigest()
PROJECT_ID = "proj-e2e-backfill"


# ---------------------------------------------------------------------------
# Hub TestClient fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def hub_client(tmp_path):
    cfg = Config.__new__(Config)
    cfg.allowed_token_hashes = frozenset([TOKEN_HASH])
    cfg.db_path = str(tmp_path / "hub.db")
    cfg.audit_path = str(tmp_path / "audit.log")
    cfg.rate_limit = 10000
    cfg.rate_window = 3600
    cfg.max_body_bytes = 10 * 1024 * 1024
    cfg.max_entries = 1000
    cfg.max_pull = 500
    cfg.tls_cert = None
    cfg.tls_key = None
    app = create_app(config=cfg)
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def _make_transport(hub_client: TestClient, token: str) -> HttpTransport:
    t = HttpTransport("http://hub-testclient", token)

    def _request_via_client(
        method: str,
        path: str,
        body: dict | None,
        params: dict | None = None,
    ) -> Any:
        headers = {"Authorization": f"Bearer {token}"}
        if method == "POST":
            resp = hub_client.post(path, json=body, headers=headers)
        elif method == "GET":
            resp = hub_client.get(path, params=params or {}, headers=headers)
        else:
            raise NotImplementedError(f"Unsupported method: {method}")
        if resp.status_code == 200:
            return resp.json()
        raise urllib.error.HTTPError(
            url=path,
            code=resp.status_code,
            msg=str(resp.status_code),
            hdrs=None,  # type: ignore[arg-type]
            fp=None,
        )

    t._request = _request_via_client  # type: ignore[method-assign]
    return t


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_producer_cache():
    reset_cache()
    yield
    reset_cache()


@pytest.fixture()
def active_env(monkeypatch, tmp_path):
    """Set env so sync is active on client A."""
    monkeypatch.setenv("QUIPU_KEY", TEST_KEY_B64)
    monkeypatch.setenv("QUIPU_HUB_URL", "http://hub-e2e-fake")
    monkeypatch.setenv("QUIPU_HUB_TOKEN", TOKEN)
    monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("QUIPU_CLIENT_ID", "client-A-backfill")


def _count_rows(store) -> int:
    return store._conn.execute(
        "SELECT COUNT(*) AS n FROM oplog_entries"
    ).fetchone()["n"]


# ---------------------------------------------------------------------------
# E2E: pre-sync atoms on A -> backfill -> push -> pull on B
# ---------------------------------------------------------------------------

class TestBackfillE2E:
    def test_pre_existing_atoms_reconstructed_on_b(
        self, hub_client, tmp_path, active_env
    ):
        store_a = open_store(str(tmp_path / "a.db"))
        store_b = open_store(str(tmp_path / "b.db"))
        try:
            # --- Phase 1: atoms exist BEFORE sync produced anything ---
            # Insert directly via the Store (NOT the write pipeline) so the
            # producer never runs — models atoms created before sync was enabled.
            live = store_a.insert(
                content="pre-sync live atom",
                project_id=PROJECT_ID,
                created_at="2025-01-01T00:00:01Z",
            )
            dead = store_a.insert(
                content="pre-sync dead atom",
                project_id=PROJECT_ID,
                created_at="2025-01-01T00:00:02Z",
            )
            store_a.update_invalidated(dead.id, True)

            # Producer emitted 0 for these (no oplog rows yet).
            assert _count_rows(store_a) == 0, "producer must not have emitted"

            # --- Phase 2: activate sync + backfill ---
            result = backfill_project(store_a, PROJECT_ID)
            assert result.status == "ok"
            assert result.emitted == 2, f"expected 2 emitted, got {result.emitted}"

            blinded = aad_for(PROJECT_ID, TEST_KEY).decode()
            oplog_a = OplogStore(store_a._conn)
            unpushed = oplog_a.unpushed(blinded, "client-A-backfill")
            ops = sorted(e.op for e in unpushed)
            assert ops == ["invalidate", "upsert"], f"unexpected ops: {ops}"

            # --- Phase 3: push A -> hub ---
            from quipu.sync.push import push
            transport_a = _make_transport(hub_client, TOKEN)
            pushed = push(
                PROJECT_ID,
                store=store_a,
                transport=transport_a,
                key=TEST_KEY,
                client_id="client-A-backfill",
            )
            assert pushed == 2, f"push shipped {pushed} (expected 2)"

            # --- Phase 4: pull on fresh client B ---
            from quipu.sync.pull import pull
            transport_b = _make_transport(hub_client, TOKEN)
            pulled = pull(
                PROJECT_ID,
                store=store_b,
                transport=transport_b,
                key=TEST_KEY,
                client_id="client-B-backfill",
            )
            assert pulled > 0, f"pull got {pulled} records"

            # --- Phase 5: assert reconstruction on B ---
            live_b = store_b.get(live.id)
            dead_b = store_b.get(dead.id)
            assert live_b is not None, "live atom missing on B"
            assert live_b.content == "pre-sync live atom"
            assert not live_b.invalidated, "live atom must not be invalidated on B"

            assert dead_b is not None, "dead atom missing on B"
            assert dead_b.invalidated, "invalidated atom must be a tombstone on B"

            # --- Phase 6: backfill re-run is a no-op ---
            rows_before = _count_rows(store_a)
            rerun = backfill_project(store_a, PROJECT_ID)
            assert rerun.status == "ok"
            assert rerun.emitted == 0, "re-run must emit nothing"
            assert _count_rows(store_a) == rows_before

        finally:
            store_a.close()
            store_b.close()

    def test_backfill_skips_live_emitted_atom(
        self, hub_client, tmp_path, active_env
    ):
        """An atom the live producer already emitted is not re-emitted by backfill."""
        store_a = open_store(str(tmp_path / "a.db"))
        try:
            # Live-emitted atom.
            hot = store_a.insert(content="hot", project_id=PROJECT_ID)
            emit(store_a, op="upsert", atom=hot, project_id=PROJECT_ID)
            # Pre-existing cold atom (no producer).
            store_a.insert(content="cold", project_id=PROJECT_ID)

            assert _count_rows(store_a) == 1

            result = backfill_project(store_a, PROJECT_ID)
            assert result.status == "ok"
            assert result.emitted == 1, "only the cold atom should be emitted"
            assert result.skipped == 1
            assert _count_rows(store_a) == 2

        finally:
            store_a.close()
