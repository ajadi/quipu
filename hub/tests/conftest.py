"""hub.tests.conftest — shared fixtures for hub tests."""

from __future__ import annotations

import base64
import hashlib
import os
import tempfile

import pytest
from fastapi.testclient import TestClient

# Set HUB_TOKENS before importing hub.main so the module-level create_app()
# call (the Uvicorn entry-point) does not raise ValueError during test collection.
# The token here is a test-only sentinel; individual fixtures construct Config
# via Config.__new__ with their own token sets.
os.environ.setdefault("HUB_TOKENS", "test-sentinel-token-for-import")

from hub.config import Config
from hub.main import create_app

# ---------------------------------------------------------------------------
# Constants shared across test modules
# ---------------------------------------------------------------------------

VALID_TOKEN = "test-secret-token-abc123"
VALID_TOKEN_HASH = hashlib.sha256(VALID_TOKEN.encode()).hexdigest()

# A valid 64-hex blinded_project_id
BPID = "a" * 64
BPID2 = "b" * 64


def make_entry(
    *,
    entry_id: str = "e1" + "0" * 62,
    client_id: str = "client-001",
    sequence_no: int = 1,
    op: str = "upsert",
    record_id: str = "rec-001",
    blinded_project_id: str = BPID,
    ts: str = "2026-01-01T00:00:00Z",
    payload: bytes | None = None,
) -> dict:
    """Build a hub-visible entry dict (no quipu imports)."""
    if payload is None:
        payload = b"\x00\x01\x02\x03opaque-ciphertext"
    return {
        "entry_id": entry_id,
        "client_id": client_id,
        "sequence_no": sequence_no,
        "op": op,
        "record_id": record_id,
        "blinded_project_id": blinded_project_id,
        "ts": ts,
        "payload": base64.b64encode(payload).decode("ascii"),
    }


@pytest.fixture()
def tmp_db(tmp_path):
    """Temporary DB path."""
    return str(tmp_path / "hub.db")


@pytest.fixture()
def tmp_audit(tmp_path):
    """Temporary audit log path."""
    return str(tmp_path / "audit.log")


@pytest.fixture()
def client(tmp_db, tmp_audit):
    """TestClient with valid token config and temporary DB/audit."""
    cfg = Config.__new__(Config)
    cfg.allowed_token_hashes = frozenset([VALID_TOKEN_HASH])
    cfg.db_path = tmp_db
    cfg.audit_path = tmp_audit
    cfg.rate_limit = 1000
    cfg.rate_window = 3600
    cfg.max_body_bytes = 10 * 1024 * 1024
    cfg.max_entries = 1000
    cfg.tls_cert = None
    cfg.tls_key = None

    app = create_app(config=cfg)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def auth_headers():
    """Valid auth headers."""
    return {"Authorization": f"Bearer {VALID_TOKEN}"}
