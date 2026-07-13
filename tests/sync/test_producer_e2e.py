"""HEADLINE E2E test for the oplog producer.

Writes atoms + triggers invalidation on client A -> pushes to a REAL hub via
hub.main.create_app(cfg) wrapped in FastAPI TestClient with HttpTransport._request
overridden to route through the TestClient (reuses the seam from
tests/sync/test_client_hub_integration.py, known-issues L18) -> quipu pull on
client B (fresh store, SAME key/bpid) -> asserts:
  - atom content is reconstructed on B
  - invalidation is latched on B
  - the real push ships >0 entries
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
from tests._semantic import TEST_EMBED_DIM

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Set HUB_TOKENS before importing hub.main.
os.environ.setdefault("HUB_TOKENS", "test-sentinel-token-for-import")

from fastapi.testclient import TestClient

from hub.config import Config
from hub.main import create_app

from quipu.oplog.producer import reset_cache
from quipu.storage import store as open_store
from quipu.sync._aad import aad_for
from quipu.sync.client import HttpTransport, sync_now
from quipu.sync.oplog_store import OplogStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEST_KEY = bytes(range(32))
TEST_KEY_B64 = base64.b64encode(TEST_KEY).decode()
TOKEN = "e2e-producer-token"
TOKEN_HASH = hashlib.sha256(TOKEN.encode()).hexdigest()
PROJECT_ID = "proj-e2e-producer"
# ---------------------------------------------------------------------------
# Fake embedding engine (no ONNX needed)
# ---------------------------------------------------------------------------

class _N:
    def __init__(self, name):
        self.name = name
        self.type = "tensor(int64)"


class _FakeSession:
    def get_inputs(self):
        return [_N("input_ids"), _N("attention_mask")]
    def get_outputs(self):
        return [_N("sentence_embedding")]
    def run(self, output_names, feeds):
        import numpy as np
        n = feeds["input_ids"].shape[0]
        return [np.ones((n, TEST_EMBED_DIM), dtype=np.float32)]


class _FakeTokenizer:
    def encode_batch(self, texts):
        class _Enc:
            ids = [1] * 8
            attention_mask = [1] * 8
        return [_Enc() for _ in texts]


class _UnitVecSession:
    """Returns a unit vector along dim 0 (so all writes are near-duplicates)."""
    def get_inputs(self):
        return [_N("input_ids"), _N("attention_mask")]
    def get_outputs(self):
        return [_N("sentence_embedding")]
    def run(self, output_names, feeds):
        import numpy as np
        n = feeds["input_ids"].shape[0]
        arr = np.zeros((n, TEST_EMBED_DIM), dtype=np.float32)
        arr[:, 0] = 1.0
        return [arr]


def _install_fake_engine():
    from quipu.embeddings.engine import _Engine, set_engine, _reset
    _reset()
    set_engine(_Engine(session=_FakeSession(), tokenizer=_FakeTokenizer()))


def _install_unit_engine():
    from quipu.embeddings.engine import _Engine, set_engine, _reset
    _reset()
    set_engine(_Engine(session=_UnitVecSession(), tokenizer=_FakeTokenizer()))


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


# ---------------------------------------------------------------------------
# HttpTransport routed through TestClient
# ---------------------------------------------------------------------------

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


@pytest.fixture(autouse=True)
def _reset_embed():
    from quipu.embeddings.engine import _reset
    yield
    _reset()


@pytest.fixture()
def active_env(semantic_model, monkeypatch, tmp_path):
    """Set env so the producer considers sync active on client A."""
    monkeypatch.setenv("QUIPU_KEY", TEST_KEY_B64)
    monkeypatch.setenv("QUIPU_HUB_URL", "http://hub-e2e-fake")
    monkeypatch.setenv("QUIPU_HUB_TOKEN", TOKEN)
    monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("QUIPU_CLIENT_ID", "client-A-e2e")


# ---------------------------------------------------------------------------
# E2E: write on A -> push to hub -> pull on B -> content + invalidation on B
# ---------------------------------------------------------------------------

class TestProducerE2E:
    def test_write_upsert_propagates_to_b(self, hub_client, tmp_path, active_env):
        """Producer emits upsert -> push ships >0 entries -> pull on B reconstructs atom."""
        store_a = open_store(str(tmp_path / "a.db"))
        store_b = open_store(str(tmp_path / "b.db"))
        try:
            # Write atom on A (producer emits upsert into oplog_entries)
            _install_fake_engine()
            from quipu.write.pipeline import write
            atom_id = write("hello from A", project_id=PROJECT_ID, store=store_a)

            # Verify at least one oplog entry was produced locally
            oplog_a = OplogStore(store_a._conn)
            blinded = aad_for(PROJECT_ID, TEST_KEY).decode()
            unpushed = oplog_a.unpushed(blinded, "client-A-e2e")
            assert len(unpushed) > 0, "Producer emitted no oplog entries for the write"

            # Push A -> hub via real TestClient
            from quipu.sync.push import push
            transport_a = _make_transport(hub_client, TOKEN)
            pushed = push(
                PROJECT_ID,
                store=store_a,
                transport=transport_a,
                key=TEST_KEY,
                client_id="client-A-e2e",
            )
            assert pushed > 0, f"push shipped {pushed} entries (expected >0)"

            # Pull on B (fresh store, no producer setup needed — merge writes atom)
            from quipu.sync.pull import pull
            transport_b = _make_transport(hub_client, TOKEN)
            pulled = pull(
                PROJECT_ID,
                store=store_b,
                transport=transport_b,
                key=TEST_KEY,
                client_id="client-B-e2e",
            )
            assert pulled > 0, f"pull got {pulled} records (expected >0)"

            # Atom content reconstructed on B
            atom_b = store_b.get(atom_id)
            assert atom_b is not None, f"Atom {atom_id!r} not found on client B"
            assert atom_b.content == "hello from A"

        finally:
            store_a.close()
            store_b.close()

    def test_invalidation_latched_on_b(self, hub_client, tmp_path, active_env):
        """Write + explicit quipu_invalidate on A -> push -> pull on B -> invalidation latched on B.

        write() no longer auto-invalidates; caller calls quipu_invalidate to supersede.
        Verifies: oplog invalidate entry emitted with ts==atom.updated_at, AC3 path.
        """
        store_a = open_store(str(tmp_path / "a.db"))
        store_b = open_store(str(tmp_path / "b.db"))
        try:
            from quipu.write.pipeline import write

            # First atom (unit vec dim 0)
            _install_unit_engine()
            old_id = write("first memory", project_id=PROJECT_ID, store=store_a)

            # Second atom with same direction — write() no longer auto-invalidates
            _install_unit_engine()
            new_id = write("second memory", project_id=PROJECT_ID, store=store_a)

            # Verify old_id is NOT auto-invalidated by write()
            assert not store_a.get(old_id).invalidated, "old atom must be ACTIVE after write() (no auto-invalidation)"

            # Caller supersedes explicitly via quipu_invalidate (AC3 path)
            from quipu.mcp.tools import dispatch
            inv_result = dispatch(
                "quipu_invalidate",
                store=store_a,
                default_project_id=PROJECT_ID,
                arguments={"id": old_id},
            )
            import json as _json
            inv_data = _json.loads(inv_result[0].text)
            assert inv_data["invalidated"] is True, f"quipu_invalidate failed: {inv_data}"

            # Now old_id is invalidated
            assert store_a.get(old_id).invalidated, "old atom must be invalidated after explicit call"

            # Verify oplog has both upserts + at least 1 invalidate (emitted by quipu_invalidate)
            blinded = aad_for(PROJECT_ID, TEST_KEY).decode()
            oplog_a = OplogStore(store_a._conn)
            unpushed = oplog_a.unpushed(blinded, "client-A-e2e")
            ops = [e.op for e in unpushed]
            assert "upsert" in ops, f"No upsert in unpushed: {ops}"
            assert "invalidate" in ops, f"No invalidate in unpushed: {ops}"

            # ts on invalidate entry must equal atom.updated_at (event-time, AC3)
            inv_entries = [e for e in unpushed if e.op == "invalidate"]
            assert inv_entries, "No invalidate oplog entry found"
            old_atom = store_a.get(old_id)
            assert inv_entries[0].ts == old_atom.updated_at, (
                f"invalidate oplog ts {inv_entries[0].ts!r} != atom.updated_at {old_atom.updated_at!r}"
            )

            # Push all entries to hub
            from quipu.sync.push import push
            transport_a = _make_transport(hub_client, TOKEN)
            pushed = push(
                PROJECT_ID,
                store=store_a,
                transport=transport_a,
                key=TEST_KEY,
                client_id="client-A-e2e",
            )
            assert pushed > 0

            # Pull on B
            from quipu.sync.pull import pull
            transport_b = _make_transport(hub_client, TOKEN)
            pulled = pull(
                PROJECT_ID,
                store=store_b,
                transport=transport_b,
                key=TEST_KEY,
                client_id="client-B-e2e",
            )
            assert pulled > 0

            # Both atoms should be on B
            atom_old_b = store_b.get(old_id)
            atom_new_b = store_b.get(new_id)
            assert atom_old_b is not None, "old atom missing on B"
            assert atom_new_b is not None, "new atom missing on B"

            # Invalidation must be latched on B
            assert atom_old_b.invalidated, "old atom not invalidated on B"
            assert not atom_new_b.invalidated, "new atom should NOT be invalidated on B"

        finally:
            store_a.close()
            store_b.close()

    def test_real_push_ships_positive_entries(self, hub_client, tmp_path, active_env):
        """The main regression: a real quipu push now ships >0 entries (was 0 before producer)."""
        store_a = open_store(str(tmp_path / "a.db"))
        try:
            _install_fake_engine()
            from quipu.write.pipeline import write

            write("entry one", project_id=PROJECT_ID, store=store_a)
            write("entry two", project_id=PROJECT_ID, store=store_a)

            blinded = aad_for(PROJECT_ID, TEST_KEY).decode()
            oplog_a = OplogStore(store_a._conn)
            unpushed = oplog_a.unpushed(blinded, "client-A-e2e")
            assert len(unpushed) >= 2, (
                f"Expected >=2 unpushed entries after 2 writes, got {len(unpushed)}"
            )

            from quipu.sync.push import push
            transport_a = _make_transport(hub_client, TOKEN)
            pushed = push(
                PROJECT_ID,
                store=store_a,
                transport=transport_a,
                key=TEST_KEY,
                client_id="client-A-e2e",
            )
            assert pushed >= 2, (
                f"push shipped {pushed} entries; expected >=2 (producer regression)"
            )

        finally:
            store_a.close()
