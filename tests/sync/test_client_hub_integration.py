"""Real client<->hub round-trip integration tests.

Drives HttpTransport against a real hub FastAPI app via fastapi.testclient.TestClient.
_request is overridden on each transport instance to route calls through the TestClient
instead of real HTTP, so push/pull's error-mapping and defensive-parse run against
real hub responses.

Covers:
  (a) convergence: store A push -> store B pull -> B sees A's records
  (b) bad token -> SyncAuthError (real 401)
  (c) tampered payload -> DecryptError surfaced (real GCM integrity failure)
  (d) hub unreachable -> offline degrade, local ops unaffected
"""

from __future__ import annotations

import base64
import hashlib
import os
import urllib.error
import urllib.request
from typing import Any

import pytest

# Set HUB_TOKENS before importing hub.main (mirrors hub/tests/conftest.py).
os.environ.setdefault("HUB_TOKENS", "test-sentinel-token-for-import")

from fastapi.testclient import TestClient

from hub.config import Config
from hub.main import create_app

from quipu.crypto.errors import DecryptError
from quipu.storage import store as open_store
from quipu.sync._aad import aad_for
from quipu.sync.client import HttpTransport, sync_now
from quipu.sync.errors import SyncAuthError
from quipu.sync.oplog_store import OplogStore
from quipu.sync.push import push
from quipu.sync.pull import pull
from quipu.oplog.codec import encode_entry
from quipu.oplog.entry import OplogEntry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TOKEN = "integration-test-secret-token"
TOKEN_HASH = hashlib.sha256(TOKEN.encode()).hexdigest()
BAD_TOKEN = "wrong-token-xyz"

# Reuse the same fixed 32-byte test key from conftest (independent, no fixture dep)
KEY = bytes(range(32))
PROJECT_ID = "proj-hub-integration"


# ---------------------------------------------------------------------------
# Hub TestClient fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def hub_client(tmp_path):
    """TestClient wrapping a real hub app with tmp SQLite DB."""
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
# HttpTransport subclass routed through TestClient
# ---------------------------------------------------------------------------

def _make_hub_transport(hub_client: TestClient, token: str) -> HttpTransport:
    """Build an HttpTransport whose _request routes calls through a TestClient."""

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

        # Re-raise as HTTPError so _translate_exc (called in push/pull) maps it
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
# Local write helper (mirrors tests/sync/conftest.py::write_local)
# ---------------------------------------------------------------------------

def _write_local(store, key, project_id, client_id, *, op, record_id, ts, content=""):
    blinded = aad_for(project_id, key).decode()
    oplog = OplogStore(store._conn)
    seq = oplog.next_sequence_no(client_id)
    entry_id = OplogEntry.compute_entry_id(client_id, seq)
    op_fields = {
        "op": op,
        "record_id": record_id,
        "ts": ts,
        "content": content,
        "type": "diary",
        "scope": "project",
        "metadata": {},
        "refs": [],
        "project_id": project_id,
    }
    payload = encode_entry(op_fields, key, blinded)
    entry = OplogEntry(
        entry_id=entry_id,
        client_id=client_id,
        sequence_no=seq,
        op=op,
        record_id=record_id,
        blinded_project_id=blinded,
        ts=ts,
        payload=payload,
    )
    oplog.append_local(entry)
    return entry


# ---------------------------------------------------------------------------
# (a) Convergence: A pushes, B pulls, B sees A's records
# ---------------------------------------------------------------------------

class TestConvergence:
    def test_push_then_pull_converges(self, hub_client, tmp_path):
        store_a = open_store(str(tmp_path / "a.db"))
        store_b = open_store(str(tmp_path / "b.db"))
        try:
            _write_local(
                store_a, KEY, PROJECT_ID, "client-A",
                op="upsert", record_id="rec-1",
                ts="2026-06-21T00:00:01Z", content="hello-from-A",
            )

            transport_a = _make_hub_transport(hub_client, TOKEN)
            transport_b = _make_hub_transport(hub_client, TOKEN)

            pushed = push(PROJECT_ID, store=store_a, transport=transport_a, key=KEY, client_id="client-A")
            assert pushed == 1

            pulled = pull(PROJECT_ID, store=store_b, transport=transport_b, key=KEY, client_id="client-B")
            assert pulled == 1

            # B now has the record A wrote
            atom = store_b.get("rec-1")
            assert atom is not None
            assert atom.content == "hello-from-A"
        finally:
            store_a.close()
            store_b.close()

    def test_push_multiple_records_all_converge(self, hub_client, tmp_path):
        store_a = open_store(str(tmp_path / "a.db"))
        store_b = open_store(str(tmp_path / "b.db"))
        try:
            for i in range(3):
                _write_local(
                    store_a, KEY, PROJECT_ID, "client-A",
                    op="upsert", record_id=f"rec-{i}",
                    ts=f"2026-06-21T00:00:0{i+1}Z", content=f"content-{i}",
                )

            transport_a = _make_hub_transport(hub_client, TOKEN)
            transport_b = _make_hub_transport(hub_client, TOKEN)

            pushed = push(PROJECT_ID, store=store_a, transport=transport_a, key=KEY, client_id="client-A")
            assert pushed == 3

            pulled = pull(PROJECT_ID, store=store_b, transport=transport_b, key=KEY, client_id="client-B")
            assert pulled == 3

            for i in range(3):
                atom = store_b.get(f"rec-{i}")
                assert atom is not None
                assert atom.content == f"content-{i}"
        finally:
            store_a.close()
            store_b.close()


# ---------------------------------------------------------------------------
# (b) Bad token -> SyncAuthError (real 401 from hub)
# ---------------------------------------------------------------------------

class TestBadToken:
    def test_push_bad_token_raises_auth_error(self, hub_client, tmp_path):
        store = open_store(str(tmp_path / "bad.db"))
        try:
            _write_local(
                store, KEY, PROJECT_ID, "client-X",
                op="upsert", record_id="rec-bad",
                ts="2026-06-21T00:00:01Z", content="test",
            )
            transport_bad = _make_hub_transport(hub_client, BAD_TOKEN)
            with pytest.raises(SyncAuthError):
                push(PROJECT_ID, store=store, transport=transport_bad, key=KEY, client_id="client-X")
        finally:
            store.close()

    def test_pull_bad_token_raises_auth_error(self, hub_client, tmp_path):
        store = open_store(str(tmp_path / "bad.db"))
        try:
            transport_bad = _make_hub_transport(hub_client, BAD_TOKEN)
            with pytest.raises(SyncAuthError):
                pull(PROJECT_ID, store=store, transport=transport_bad, key=KEY, client_id="client-X")
        finally:
            store.close()


# ---------------------------------------------------------------------------
# (c) Tampered payload -> DecryptError surfaced
# ---------------------------------------------------------------------------

class TestTamperedPayload:
    def test_tampered_payload_raises_decrypt_error(self, hub_client, tmp_path):
        """Push a valid entry; receive it back; corrupt the ciphertext; assert DecryptError."""
        store_a = open_store(str(tmp_path / "a.db"))
        store_b = open_store(str(tmp_path / "b.db"))
        try:
            blinded = aad_for(PROJECT_ID, KEY).decode()
            _write_local(
                store_a, KEY, PROJECT_ID, "client-T",
                op="upsert", record_id="rec-tamper",
                ts="2026-06-21T00:00:01Z", content="tamper-me",
            )

            transport_a = _make_hub_transport(hub_client, TOKEN)
            push(PROJECT_ID, store=store_a, transport=transport_a, key=KEY, client_id="client-T")

            # Pull from hub with a transport that corrupts every payload byte 0
            real_transport = _make_hub_transport(hub_client, TOKEN)
            real_pull_fn = real_transport.pull

            def _tampered_pull(bpid, cursor):
                entries, next_cursor = real_pull_fn(bpid, cursor)
                tampered = []
                for e in entries:
                    raw = base64.b64decode(e["payload"])
                    # Corrupt the last 16 bytes (GCM auth tag) to guarantee
                    # AESGCM.decrypt raises InvalidTag -> DecryptError, not a
                    # version/struct error from corrupting the header bytes.
                    b = bytearray(raw)
                    for i in range(1, 17):
                        b[-i] ^= 0xFF
                    tampered.append({**e, "payload": base64.b64encode(bytes(b)).decode("ascii")})
                return tampered, next_cursor

            real_transport.pull = _tampered_pull  # type: ignore[method-assign]

            with pytest.raises(DecryptError):
                pull(PROJECT_ID, store=store_b, transport=real_transport, key=KEY, client_id="client-B")
        finally:
            store_a.close()
            store_b.close()


# ---------------------------------------------------------------------------
# (d) Hub unreachable -> offline degrade, local ops unaffected
# ---------------------------------------------------------------------------

class TestOfflineDegrade:
    def _patch_sync_now_deps(self, monkeypatch):
        """Patch all sync_now dependencies: hub config, key derivation, client_id, transport."""
        from quipu.config import HubConfig

        fake_cfg = HubConfig(
            url="http://127.0.0.1:19999",  # nothing listening
            token=TOKEN,
            verify=None,
        )

        # sync_now does `from quipu.config import get_hub_config, get_client_id`
        monkeypatch.setattr("quipu.config.get_hub_config", lambda: fake_cfg)
        monkeypatch.setattr("quipu.config.get_client_id", lambda store: "client-offline")

        # sync_now does `from quipu.keystore._backend import get_or_derive_key`
        monkeypatch.setattr(
            "quipu.keystore._backend.get_or_derive_key",
            lambda project_id: KEY,
        )

        # Patch HttpTransport so _request always raises URLError
        original_init = HttpTransport.__init__

        def _patched_init(self, base_url, token, **kwargs):
            original_init(self, base_url, token, **kwargs)

            def _raise_url_error(method, path, body, params=None):
                raise urllib.error.URLError("Connection refused")

            self._request = _raise_url_error

        monkeypatch.setattr(HttpTransport, "__init__", _patched_init)

    def test_sync_now_returns_offline_when_hub_unreachable(self, tmp_path, monkeypatch):
        """sync_now must return status='offline', never raise, when hub is down."""
        self._patch_sync_now_deps(monkeypatch)
        store = open_store(str(tmp_path / "offline.db"))
        try:
            result = sync_now(PROJECT_ID, store=store, directions=("pull", "push"))
            assert result.status == "offline"
        finally:
            store.close()

    def test_local_ops_succeed_after_offline_sync(self, tmp_path, monkeypatch):
        """Local store read/write must succeed even after sync returns offline."""
        self._patch_sync_now_deps(monkeypatch)
        store = open_store(str(tmp_path / "local.db"))
        try:
            result = sync_now(PROJECT_ID, store=store)
            assert result.status == "offline"

            # Local write + read must work fine after the failed sync
            atom = store.insert(content="local-only", project_id=PROJECT_ID)
            fetched = store.get(atom.id)
            assert fetched is not None
            assert fetched.content == "local-only"
        finally:
            store.close()
