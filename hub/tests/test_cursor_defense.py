"""test_cursor_defense — parse_cursor rejects bad cursors with 400, never 500."""

from __future__ import annotations

import pytest

from hub.tests.conftest import BPID


def _pull(client, auth_headers, since):
    return client.get(f"/oplog/{BPID}", params={"since": since}, headers=auth_headers)


def test_alpha_cursor_returns_400(client, auth_headers):
    r = _pull(client, auth_headers, "abc")
    assert r.status_code == 400


def test_negative_cursor_returns_400(client, auth_headers):
    r = _pull(client, auth_headers, "-1")
    assert r.status_code == 400


def test_float_cursor_returns_400(client, auth_headers):
    """Strings like '1e9' or '1.5' must be rejected."""
    r = _pull(client, auth_headers, "1e9")
    assert r.status_code == 400


def test_float_decimal_cursor_returns_400(client, auth_headers):
    r = _pull(client, auth_headers, "1.5")
    assert r.status_code == 400


def test_oversized_cursor_returns_400(client, auth_headers):
    """40-digit number exceeds max cursor length -> 400."""
    r = _pull(client, auth_headers, "1" * 40)
    assert r.status_code == 400


def test_empty_string_cursor_is_none(client, auth_headers):
    """Empty string cursor treated as None (first pull from start)."""
    r = _pull(client, auth_headers, "")
    assert r.status_code == 200


def test_valid_large_cursor_returns_200_empty(client, auth_headers):
    """In-range valid cursor with no rows -> 200 empty (not a crash)."""
    r = _pull(client, auth_headers, "999999999")
    assert r.status_code == 200
    data = r.json()
    assert data["entries"] == []


def test_zero_cursor_returns_200(client, auth_headers):
    r = _pull(client, auth_headers, "0")
    assert r.status_code == 200


def test_bad_cursor_never_500(client, auth_headers):
    """Bad cursors must never produce 500."""
    bad_values = ["abc", "-1", "1e9", "1.5", "1" * 40, "--1", "null", "None"]
    for val in bad_values:
        r = _pull(client, auth_headers, val)
        assert r.status_code != 500, f"Got 500 for cursor={val!r}"
