"""test_security_caps — lock in the two MEDIUM security fixes.

TEST 1: Oversized raw body with NO Content-Length header -> 413.
    Exercises the second-phase (raw-body) cap in SizeLimitMiddleware.dispatch().
    The fast-path skips when Content-Length is absent; the raw-body read must
    still reject.  A generator body forces httpx to use Transfer-Encoding:
    chunked, omitting Content-Length entirely — this is the exact bypass path
    the fix closes.

TEST 2: Per-entry payload over _MAX_PAYLOAD_BYTES -> 422; exactly at cap -> 200.
    Exercises the payload_must_be_base64 validator in hub.models.PushEntry.
    Uses the real 1 MB default (_MAX_PAYLOAD_BYTES = 1048576) so no module
    reload is required.  Over-cap must be rejected; at-cap must be accepted
    and round-trippable.
"""

from __future__ import annotations

import base64
import hashlib

import pytest
from fastapi.testclient import TestClient

from hub.config import Config
from hub.main import create_app
from hub.models import _MAX_PAYLOAD_BYTES
from hub.tests.conftest import BPID, VALID_TOKEN, VALID_TOKEN_HASH, make_entry


# ---------------------------------------------------------------------------
# Fixture: small-cap client (1 KB body limit) for TEST 1.
#
# We cannot use the shared `client` fixture because its max_body_bytes is 10 MB,
# which would require sending >10 MB over the wire to trigger 413.  Instead we
# build a dedicated app with a 1 KB cap so the test stays fast.
# ---------------------------------------------------------------------------

SMALL_BODY_CAP = 1024  # 1 KB — small enough for a fast test


@pytest.fixture()
def small_cap_client(tmp_path):
    """TestClient whose SizeLimitMiddleware cap is 1 KB (SMALL_BODY_CAP)."""
    cfg = Config.__new__(Config)
    cfg.allowed_token_hashes = frozenset([VALID_TOKEN_HASH])
    cfg.db_path = str(tmp_path / "hub.db")
    cfg.audit_path = str(tmp_path / "audit.log")
    cfg.rate_limit = 1000
    cfg.rate_window = 3600
    cfg.max_body_bytes = SMALL_BODY_CAP
    cfg.max_entries = 1000
    cfg.max_pull = 500
    cfg.tls_cert = None
    cfg.tls_key = None

    app = create_app(config=cfg)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def auth_headers_local():
    """Valid auth headers (local copy avoids importing conftest auth_headers fixture)."""
    return {"Authorization": f"Bearer {VALID_TOKEN}"}


# ---------------------------------------------------------------------------
# TEST 1 — oversized body WITHOUT Content-Length header -> 413
#
# A generator passed as `content=` makes httpx use Transfer-Encoding: chunked
# and omit Content-Length entirely.  This forces the middleware to reach the
# second-phase raw-body read.  If the raw-body cap were absent, the request
# would proceed to FastAPI body parsing; with it present it must return 413.
# ---------------------------------------------------------------------------


def test_oversized_body_no_content_length_returns_413(small_cap_client, auth_headers_local):
    """Raw-body cap fires when Content-Length header is absent (chunked body).

    The fast-path in SizeLimitMiddleware skips when Content-Length is not sent.
    The second-phase unconditional body read must still catch the oversize body
    and return 413 — proving the no-Content-Length bypass is closed.

    FORCES the no-Content-Length path: content=generator -> httpx uses
    Transfer-Encoding: chunked, Content-Length header is NOT sent.
    """
    oversized_body = b"X" * (SMALL_BODY_CAP + 1)  # 1 byte over the 1 KB cap

    # Passing a generator as `content` forces Transfer-Encoding: chunked.
    # httpx does NOT add a Content-Length header in this case.
    def chunked_body():
        yield oversized_body

    r = small_cap_client.post(
        f"/oplog/{BPID}",
        content=chunked_body(),
        headers={**auth_headers_local, "Content-Type": "application/json"},
    )

    # The raw-body cap in SizeLimitMiddleware must reject this.
    assert r.status_code == 413, (
        f"Expected 413 from raw-body cap (no Content-Length path), got {r.status_code}. "
        f"Response body: {r.text[:200]}"
    )


def test_oversized_body_with_content_length_fast_path_returns_413(
    small_cap_client, auth_headers_local
):
    """Content-Length fast-path also rejects oversized bodies.

    This is the sibling path: when Content-Length IS present and oversized,
    the middleware rejects before reading the body at all.  Confirms neither
    path regresses independently.
    """
    oversized_body = b"X" * (SMALL_BODY_CAP + 1)

    # Passing `content=bytes` makes httpx set Content-Length automatically.
    r = small_cap_client.post(
        f"/oplog/{BPID}",
        content=oversized_body,
        headers={**auth_headers_local, "Content-Type": "application/json"},
    )

    assert r.status_code == 413, (
        f"Expected 413 from Content-Length fast-path, got {r.status_code}."
    )


# ---------------------------------------------------------------------------
# TEST 2 — per-entry payload cap in models.PushEntry
#
# _MAX_PAYLOAD_BYTES is read at import time from HUB_MAX_PAYLOAD_BYTES
# (default 1 MB = 1048576 bytes).  We use the real default rather than
# reloading the module with a patched env var.
#
# Over-cap: decoded payload is _MAX_PAYLOAD_BYTES + 1 bytes -> 422.
# At-cap:   decoded payload is exactly _MAX_PAYLOAD_BYTES bytes -> 200.
#
# The over-cap entry is otherwise a fully valid entry (8 correct fields,
# valid base64, op='upsert', blinded_project_id matches path) so the ONLY
# reason for rejection is the size cap.
# ---------------------------------------------------------------------------


def test_per_entry_payload_over_cap_returns_422(client, auth_headers):
    """Entry whose decoded payload exceeds _MAX_PAYLOAD_BYTES is rejected with 422.

    The entry is structurally valid in every other respect — correct field count,
    valid base64, valid op, correct blinded_project_id.  The sole violation is
    decoded payload size of _MAX_PAYLOAD_BYTES + 1 bytes.
    """
    raw_over_cap = b"P" * (_MAX_PAYLOAD_BYTES + 1)
    entry = make_entry(payload=raw_over_cap)
    # Sanity: confirm the decoded size is indeed over the cap.
    assert len(base64.b64decode(entry["payload"])) == _MAX_PAYLOAD_BYTES + 1

    r = client.post(
        f"/oplog/{BPID}",
        json={"entries": [entry]},
        headers=auth_headers,
    )

    # Pydantic field_validator raises ValueError -> FastAPI wraps as 422.
    assert r.status_code == 422, (
        f"Expected 422 for payload {_MAX_PAYLOAD_BYTES + 1} bytes decoded "
        f"(over cap of {_MAX_PAYLOAD_BYTES}), got {r.status_code}. "
        f"Response: {r.text[:300]}"
    )


def test_per_entry_payload_at_cap_is_accepted(client, auth_headers):
    """Entry whose decoded payload is exactly _MAX_PAYLOAD_BYTES bytes is accepted (200).

    Boundary value: the cap is inclusive (<=), so exactly-at-cap must succeed.
    We then confirm the entry is retrievable with the payload byte-identical,
    proving the 200 is not a vacuous no-op.
    """
    raw_at_cap = b"Q" * _MAX_PAYLOAD_BYTES
    entry = make_entry(
        payload=raw_at_cap,
        entry_id="c1" + "0" * 62,  # distinct entry_id from other tests
        sequence_no=99,
    )
    # Sanity: confirm the decoded size is exactly the cap.
    assert len(base64.b64decode(entry["payload"])) == _MAX_PAYLOAD_BYTES

    push_r = client.post(
        f"/oplog/{BPID}",
        json={"entries": [entry]},
        headers=auth_headers,
    )

    assert push_r.status_code == 200, (
        f"Expected 200 for payload exactly at cap ({_MAX_PAYLOAD_BYTES} bytes decoded), "
        f"got {push_r.status_code}. Response: {push_r.text[:300]}"
    )

    # Confirm the entry was persisted and the payload round-trips correctly.
    pull_r = client.get(f"/oplog/{BPID}", headers=auth_headers)
    assert pull_r.status_code == 200
    entries = pull_r.json()["entries"]
    pushed = next(
        (e for e in entries if e["entry_id"] == entry["entry_id"]),
        None,
    )
    assert pushed is not None, "At-cap entry was not found in pull response"
    returned_bytes = base64.b64decode(pushed["payload"])
    assert returned_bytes == raw_at_cap, (
        "At-cap payload round-trip mismatch: "
        f"expected {len(raw_at_cap)} bytes, got {len(returned_bytes)} bytes"
    )
