"""test_health — GET /health returns 200 {"status":"ok"}, no auth required."""

from __future__ import annotations


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_health_no_auth_required(client):
    """Health must work without any Authorization header."""
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_with_invalid_token_still_ok(client):
    """Health is exempt from auth even if a wrong token is provided."""
    resp = client.get("/health", headers={"Authorization": "Bearer wrong"})
    assert resp.status_code == 200
