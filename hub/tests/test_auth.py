"""test_auth — bearer token auth enforcement."""

from __future__ import annotations

from hub.tests.conftest import BPID, VALID_TOKEN, make_entry


def test_no_auth_header_returns_401(client):
    resp = client.get(f"/oplog/{BPID}")
    assert resp.status_code == 401


def test_malformed_auth_header_returns_401(client):
    """Not a Bearer scheme."""
    resp = client.get(f"/oplog/{BPID}", headers={"Authorization": "Basic abc"})
    assert resp.status_code == 401


def test_wrong_token_returns_401(client):
    resp = client.get(
        f"/oplog/{BPID}",
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert resp.status_code == 401


def test_empty_bearer_returns_401(client):
    resp = client.get(f"/oplog/{BPID}", headers={"Authorization": "Bearer "})
    assert resp.status_code == 401


def test_valid_token_pull_returns_200(client):
    resp = client.get(
        f"/oplog/{BPID}",
        headers={"Authorization": f"Bearer {VALID_TOKEN}"},
    )
    assert resp.status_code == 200


def test_valid_token_push_returns_200(client):
    entry = make_entry()
    resp = client.post(
        f"/oplog/{BPID}",
        json={"entries": [entry]},
        headers={"Authorization": f"Bearer {VALID_TOKEN}"},
    )
    assert resp.status_code == 200


def test_health_no_auth(client):
    """Confirm /health is unauthenticated."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
