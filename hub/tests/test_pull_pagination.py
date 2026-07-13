"""test_pull_pagination — GET /oplog LIMIT + has_more pagination (TASK-058).

Uses a dedicated small_pull_client fixture (max_pull=5) so backlogs large
enough to exercise pagination stay fast.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from hub.config import Config
from hub.main import create_app
from hub.tests.conftest import BPID, VALID_TOKEN, VALID_TOKEN_HASH, make_entry

PAGE_SIZE = 5


@pytest.fixture()
def small_pull_client(tmp_path):
    """TestClient whose pull page size (max_pull) is small (5) for fast pagination tests."""
    cfg = Config.__new__(Config)
    cfg.allowed_token_hashes = frozenset([VALID_TOKEN_HASH])
    cfg.db_path = str(tmp_path / "hub.db")
    cfg.audit_path = str(tmp_path / "audit.log")
    cfg.rate_limit = 10000
    cfg.rate_window = 3600
    cfg.max_body_bytes = 10 * 1024 * 1024
    cfg.max_entries = 1000
    cfg.max_pull = PAGE_SIZE
    cfg.tls_cert = None
    cfg.tls_key = None

    app = create_app(config=cfg)
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def pull_client_limit_1(tmp_path):
    """TestClient with max_pull=1 to exercise degenerate one-row-per-page pagination."""
    cfg = Config.__new__(Config)
    cfg.allowed_token_hashes = frozenset([VALID_TOKEN_HASH])
    cfg.db_path = str(tmp_path / "hub.db")
    cfg.audit_path = str(tmp_path / "audit.log")
    cfg.rate_limit = 10000
    cfg.rate_window = 3600
    cfg.max_body_bytes = 10 * 1024 * 1024
    cfg.max_entries = 1000
    cfg.max_pull = 1
    cfg.tls_cert = None
    cfg.tls_key = None

    app = create_app(config=cfg)
    with TestClient(app) as c:
        yield c


def _auth():
    return {"Authorization": f"Bearer {VALID_TOKEN}"}


def _push_n(client, n, start=1):
    entries = [
        make_entry(
            entry_id="e" + str(i).zfill(63),
            sequence_no=i,
            record_id=f"rec-{i}",
        )
        for i in range(start, start + n)
    ]
    r = client.post(f"/oplog/{BPID}", json={"entries": entries}, headers=_auth())
    assert r.status_code == 200
    return entries


def test_backlog_larger_than_limit_returns_exactly_limit_and_has_more(small_pull_client):
    """(a) Backlog > limit: first pull returns exactly `limit` entries, has_more=True, usable cursor."""
    _push_n(small_pull_client, PAGE_SIZE + 3)  # 8 entries, page size 5

    r = small_pull_client.get(f"/oplog/{BPID}", headers=_auth())
    assert r.status_code == 200
    data = r.json()
    assert len(data["entries"]) == PAGE_SIZE
    assert data["has_more"] is True
    assert data["cursor"] is not None and data["cursor"] != "0"


def test_paging_through_since_drains_backlog_final_page_has_more_false(small_pull_client):
    """(b) Paging with since=next_cursor walks the whole backlog; final page has_more=False."""
    total = PAGE_SIZE * 2 + 2  # 12 entries -> pages of 5, 5, 2
    pushed = _push_n(small_pull_client, total)

    seen_ids = []
    cursor = None
    pages = 0
    while True:
        params = {"since": cursor} if cursor is not None else None
        r = small_pull_client.get(f"/oplog/{BPID}", params=params, headers=_auth())
        assert r.status_code == 200
        data = r.json()
        seen_ids.extend(e["entry_id"] for e in data["entries"])
        cursor = data["cursor"]
        pages += 1
        if not data["has_more"]:
            break
        assert pages < 20, "pagination did not terminate"

    assert seen_ids == [e["entry_id"] for e in pushed]
    assert pages == 3

    # One more pull from the final cursor: empty + has_more False
    r = small_pull_client.get(f"/oplog/{BPID}", params={"since": cursor}, headers=_auth())
    data = r.json()
    assert data["entries"] == []
    assert data["has_more"] is False
    assert data["cursor"] == cursor


def test_exact_boundary_count_equals_limit_has_more_false(small_pull_client):
    """(c) Backlog count exactly equals the limit: has_more=False, all entries returned."""
    _push_n(small_pull_client, PAGE_SIZE)  # exactly 5, page size 5

    r = small_pull_client.get(f"/oplog/{BPID}", headers=_auth())
    assert r.status_code == 200
    data = r.json()
    assert len(data["entries"]) == PAGE_SIZE
    assert data["has_more"] is False


def test_empty_backlog_returns_empty_has_more_false(small_pull_client):
    """(d) Empty backlog: entries empty, has_more False, cursor '0'."""
    r = small_pull_client.get(f"/oplog/{BPID}", headers=_auth())
    assert r.status_code == 200
    data = r.json()
    assert data["entries"] == []
    assert data["has_more"] is False
    assert data["cursor"] == "0"


def test_limit_one_degenerate_paging_walks_backlog_one_row_at_a_time(pull_client_limit_1):
    """(e) limit=1: each page returns exactly 1 entry, in order, no skips/dupes, final has_more=False."""
    total = 4
    pushed = _push_n(pull_client_limit_1, total)

    seen_ids = []
    cursor = None
    pages = 0
    while True:
        params = {"since": cursor} if cursor is not None else None
        r = pull_client_limit_1.get(f"/oplog/{BPID}", params=params, headers=_auth())
        assert r.status_code == 200
        data = r.json()
        assert len(data["entries"]) <= 1
        seen_ids.extend(e["entry_id"] for e in data["entries"])
        cursor = data["cursor"]
        pages += 1
        if not data["has_more"]:
            break
        assert pages < 20, "pagination did not terminate"

    assert seen_ids == [e["entry_id"] for e in pushed]
    assert pages == total


def test_out_of_range_cursor_returns_empty_has_more_false(small_pull_client):
    """(d) Out-of-range (but valid) cursor: empty result + has_more=False, not a crash."""
    _push_n(small_pull_client, 2)

    r = small_pull_client.get(
        f"/oplog/{BPID}", params={"since": "999999999"}, headers=_auth()
    )
    assert r.status_code == 200
    data = r.json()
    assert data["entries"] == []
    assert data["has_more"] is False
    assert data["cursor"] == "999999999"
