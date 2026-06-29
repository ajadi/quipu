"""test_ratelimit — per-(token_hash, blinded_project_id) fixed-window."""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from hub.config import Config
from hub.main import create_app
from hub.tests.conftest import BPID, BPID2, VALID_TOKEN, make_entry

TOKEN2 = "second-token-xyz789"
TOKEN2_HASH = hashlib.sha256(TOKEN2.encode()).hexdigest()


@pytest.fixture()
def tight_client(tmp_path):
    """Client with limit=2 to make testing feasible."""
    cfg = Config.__new__(Config)
    cfg.allowed_token_hashes = frozenset([
        hashlib.sha256(VALID_TOKEN.encode()).hexdigest(),
        TOKEN2_HASH,
    ])
    cfg.db_path = str(tmp_path / "hub.db")
    cfg.audit_path = str(tmp_path / "audit.log")
    cfg.rate_limit = 2
    cfg.rate_window = 3600
    cfg.max_body_bytes = 10 * 1024 * 1024
    cfg.max_entries = 1000
    cfg.tls_cert = None
    cfg.tls_key = None

    app = create_app(config=cfg)
    with TestClient(app) as c:
        yield c


def _auth(token=VALID_TOKEN):
    return {"Authorization": f"Bearer {token}"}


def test_exceed_threshold_returns_429(tight_client):
    """After limit requests, the next returns 429."""
    # Use limit=2; 3rd request should 429
    for _ in range(2):
        r = tight_client.get(f"/oplog/{BPID}", headers=_auth())
        assert r.status_code == 200

    r = tight_client.get(f"/oplog/{BPID}", headers=_auth())
    assert r.status_code == 429


def test_429_has_retry_after_header(tight_client):
    """429 response includes Retry-After header."""
    for _ in range(2):
        tight_client.get(f"/oplog/{BPID}", headers=_auth())

    r = tight_client.get(f"/oplog/{BPID}", headers=_auth())
    assert r.status_code == 429
    assert "retry-after" in {k.lower() for k in r.headers}
    assert int(r.headers.get("Retry-After", "0")) > 0


def test_different_bpid_same_token_is_separate_bucket(tight_client):
    """Different blinded_project_id under same token = separate bucket."""
    # Exhaust bucket for BPID
    for _ in range(2):
        tight_client.get(f"/oplog/{BPID}", headers=_auth())
    r = tight_client.get(f"/oplog/{BPID}", headers=_auth())
    assert r.status_code == 429

    # BPID2 bucket is independent — still has capacity
    r2 = tight_client.get(f"/oplog/{BPID2}", headers=_auth())
    assert r2.status_code == 200


def test_different_token_same_bpid_is_separate_bucket(tight_client):
    """Different token for same blinded_project_id = separate bucket."""
    # Exhaust TOKEN1 bucket
    for _ in range(2):
        tight_client.get(f"/oplog/{BPID}", headers=_auth(VALID_TOKEN))
    r = tight_client.get(f"/oplog/{BPID}", headers=_auth(VALID_TOKEN))
    assert r.status_code == 429

    # TOKEN2 + same BPID is independent
    r2 = tight_client.get(f"/oplog/{BPID}", headers=_auth(TOKEN2))
    assert r2.status_code == 200
