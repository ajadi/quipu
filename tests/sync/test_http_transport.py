"""Tests for HttpTransport: push/pull happy path, error mapping, defensive parsing,
TLS verify default, and token-not-in-logs.
"""

from __future__ import annotations

import json
import logging
import socket
import ssl
import urllib.error
import urllib.request
from io import BytesIO
from typing import Any

import pytest

from quipu.sync.client import HttpTransport, _build_ssl_context
from quipu.sync.errors import SyncAuthError, SyncProtocolError, SyncUnavailableError


# ---------------------------------------------------------------------------
# Fake _request helper
# ---------------------------------------------------------------------------


def _make_transport_with_fake(responses: list) -> tuple[HttpTransport, list]:
    """Return transport + call-log. responses is a list of dicts or exceptions."""
    calls: list[dict] = []
    t = HttpTransport("http://hub", "tok")

    idx = 0

    def _fake_request(method, path, body, params=None):
        nonlocal idx
        calls.append({"method": method, "path": path, "body": body, "params": params})
        resp = responses[idx]
        idx += 1
        if isinstance(resp, Exception):
            raise resp
        return resp

    t._request = _fake_request  # type: ignore[method-assign]
    return t, calls


# ---------------------------------------------------------------------------
# Push happy path
# ---------------------------------------------------------------------------


class TestPushHappyPath:
    def test_push_calls_correct_endpoint(self):
        t, calls = _make_transport_with_fake([{"cursor": "3"}])
        t.push("a" * 64, [{"entry_id": "e1"}])
        assert len(calls) == 1
        assert calls[0]["method"] == "POST"
        assert calls[0]["path"] == f"/oplog/{'a'*64}"

    def test_push_sends_entries_in_body(self):
        t, calls = _make_transport_with_fake([{"cursor": "3"}])
        entries = [{"entry_id": "e1", "payload": "abc"}]
        t.push("a" * 64, entries)
        assert calls[0]["body"] == {"entries": entries}

    def test_push_returns_none(self):
        t, calls = _make_transport_with_fake([{"cursor": "3"}])
        result = t.push("a" * 64, [])
        assert result is None

    def test_push_ignores_returned_cursor(self):
        # If hub returns {"cursor": "999"}, push() silently ignores it.
        t, _ = _make_transport_with_fake([{"cursor": "999"}])
        t.push("a" * 64, [])  # No assertion needed — must not raise


# ---------------------------------------------------------------------------
# Pull happy path
# ---------------------------------------------------------------------------


class TestPullHappyPath:
    def _bpid(self):
        return "b" * 64

    def test_pull_returns_entries_and_cursor(self):
        bpid = self._bpid()
        entries = [{"entry_id": "e1"}]
        t, calls = _make_transport_with_fake([{"entries": entries, "cursor": "5"}])
        got_entries, got_cursor = t.pull(bpid, None)
        assert got_entries == entries
        assert got_cursor == "5"

    def test_pull_sends_since_param_when_cursor(self):
        bpid = self._bpid()
        t, calls = _make_transport_with_fake([{"entries": [], "cursor": "5"}])
        t.pull(bpid, "3")
        assert calls[0]["params"] == {"since": "3"}

    def test_pull_sends_no_since_when_no_cursor(self):
        bpid = self._bpid()
        t, calls = _make_transport_with_fake([{"entries": [], "cursor": "0"}])
        t.pull(bpid, None)
        assert calls[0]["params"] is None

    def test_pull_null_cursor_allowed(self):
        bpid = self._bpid()
        t, _ = _make_transport_with_fake([{"entries": [], "cursor": None}])
        _, cursor = t.pull(bpid, None)
        assert cursor is None

    def test_pull_calls_correct_endpoint(self):
        bpid = self._bpid()
        t, calls = _make_transport_with_fake([{"entries": [], "cursor": "0"}])
        t.pull(bpid, None)
        assert calls[0]["path"] == f"/oplog/{bpid}"
        assert calls[0]["method"] == "GET"


# ---------------------------------------------------------------------------
# Error mapping
# ---------------------------------------------------------------------------


def _http_error(status: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="http://x", code=status, msg=str(status), hdrs=None, fp=None
    )


class TestErrorMapping:
    def test_401_raises_auth_error(self):
        t, _ = _make_transport_with_fake([_http_error(401)])
        with pytest.raises(SyncAuthError):
            t.push("a" * 64, [])

    def test_400_raises_protocol_error(self):
        t, _ = _make_transport_with_fake([_http_error(400)])
        with pytest.raises(SyncProtocolError):
            t.push("a" * 64, [])

    def test_413_raises_protocol_error(self):
        t, _ = _make_transport_with_fake([_http_error(413)])
        with pytest.raises(SyncProtocolError):
            t.push("a" * 64, [])

    def test_422_raises_protocol_error(self):
        t, _ = _make_transport_with_fake([_http_error(422)])
        with pytest.raises(SyncProtocolError):
            t.push("a" * 64, [])

    def test_500_raises_unavailable(self):
        t, _ = _make_transport_with_fake([_http_error(500)])
        with pytest.raises(SyncUnavailableError):
            t.push("a" * 64, [])

    def test_503_raises_unavailable(self):
        t, _ = _make_transport_with_fake([_http_error(503)])
        with pytest.raises(SyncUnavailableError):
            t.push("a" * 64, [])

    def test_429_raises_unavailable(self):
        t, _ = _make_transport_with_fake([_http_error(429)])
        with pytest.raises(SyncUnavailableError):
            t.push("a" * 64, [])

    def test_url_error_raises_unavailable(self):
        t, _ = _make_transport_with_fake([urllib.error.URLError(reason="unreachable")])
        with pytest.raises(SyncUnavailableError):
            t.push("a" * 64, [])

    def test_socket_timeout_raises_unavailable(self):
        t, _ = _make_transport_with_fake([socket.timeout("timed out")])
        with pytest.raises(SyncUnavailableError):
            t.push("a" * 64, [])

    def test_timeout_error_raises_unavailable(self):
        t, _ = _make_transport_with_fake([TimeoutError("timed out")])
        with pytest.raises(SyncUnavailableError):
            t.push("a" * 64, [])


# ---------------------------------------------------------------------------
# Defensive parse of pull response
# ---------------------------------------------------------------------------


class TestPullDefensiveParse:
    def _bpid(self):
        return "c" * 64

    def test_non_dict_body_raises_protocol_error(self):
        t, _ = _make_transport_with_fake(["not a dict"])
        with pytest.raises(SyncProtocolError, match="not a JSON object"):
            t.pull(self._bpid(), None)

    def test_missing_entries_raises_protocol_error(self):
        t, _ = _make_transport_with_fake([{"cursor": "0"}])
        with pytest.raises(SyncProtocolError, match="entries"):
            t.pull(self._bpid(), None)

    def test_entries_not_list_raises_protocol_error(self):
        t, _ = _make_transport_with_fake([{"entries": "nope", "cursor": "0"}])
        with pytest.raises(SyncProtocolError, match="entries"):
            t.pull(self._bpid(), None)

    def test_entry_not_dict_raises_protocol_error(self):
        t, _ = _make_transport_with_fake([{"entries": ["bad"], "cursor": "0"}])
        with pytest.raises(SyncProtocolError, match="dict"):
            t.pull(self._bpid(), None)

    def test_cursor_non_string_raises_protocol_error(self):
        t, _ = _make_transport_with_fake([{"entries": [], "cursor": 123}])
        with pytest.raises(SyncProtocolError, match="cursor"):
            t.pull(self._bpid(), None)

    def test_cursor_non_digit_raises_protocol_error(self):
        t, _ = _make_transport_with_fake([{"entries": [], "cursor": "abc"}])
        with pytest.raises(SyncProtocolError, match="cursor"):
            t.pull(self._bpid(), None)

    def test_cursor_too_long_raises_protocol_error(self):
        t, _ = _make_transport_with_fake([{"entries": [], "cursor": "1" * 20}])
        with pytest.raises(SyncProtocolError, match="cursor"):
            t.pull(self._bpid(), None)

    def test_cursor_exactly_19_digits_ok(self):
        t, _ = _make_transport_with_fake([{"entries": [], "cursor": "1" * 19}])
        _, cursor = t.pull(self._bpid(), None)
        assert cursor == "1" * 19

    def test_cursor_zero_ok(self):
        t, _ = _make_transport_with_fake([{"entries": [], "cursor": "0"}])
        _, cursor = t.pull(self._bpid(), None)
        assert cursor == "0"


# ---------------------------------------------------------------------------
# Token never appears in logs
# ---------------------------------------------------------------------------


class TestTokenNotLogged:
    def test_token_never_in_log_records(self, caplog):
        secret_token = "SUPER_SECRET_TOKEN_XYZ"
        t = HttpTransport("http://hub", secret_token)

        # Override _request to simulate an auth failure (which would log)
        def _fake(method, path, body, params=None):
            raise SyncAuthError("401 Unauthorized from hub")

        t._request = _fake  # type: ignore[method-assign]

        with caplog.at_level(logging.DEBUG):
            try:
                t.push("a" * 64, [])
            except SyncAuthError:
                pass

        for record in caplog.records:
            assert secret_token not in record.getMessage(), (
                f"Token found in log: {record.getMessage()}"
            )


# ---------------------------------------------------------------------------
# M3 — blinded_project_id validation (path-traversal defense-in-depth)
# ---------------------------------------------------------------------------


class TestBpidValidation:
    """HttpTransport.push/pull must reject non-64-hex bpid before building URL."""

    def _good_bpid(self):
        return "a" * 64

    def _transport(self):
        """Transport with a _request that should NEVER be called."""
        t = HttpTransport("http://hub", "tok")

        def _should_not_be_called(*args, **kwargs):
            raise AssertionError("_request called despite invalid bpid")

        t._request = _should_not_be_called  # type: ignore[method-assign]
        return t

    def test_push_path_traversal_raises_protocol_error(self):
        t = self._transport()
        with pytest.raises(SyncProtocolError):
            t.push("../etc/passwd", [])

    def test_push_short_hex_raises_protocol_error(self):
        t = self._transport()
        with pytest.raises(SyncProtocolError):
            t.push("abc123", [])

    def test_push_uppercase_hex_raises_protocol_error(self):
        t = self._transport()
        with pytest.raises(SyncProtocolError):
            t.push("A" * 64, [])

    def test_push_64_char_non_hex_raises_protocol_error(self):
        t = self._transport()
        with pytest.raises(SyncProtocolError):
            t.push("g" * 64, [])

    def test_push_empty_raises_protocol_error(self):
        t = self._transport()
        with pytest.raises(SyncProtocolError):
            t.push("", [])

    def test_pull_path_traversal_raises_protocol_error(self):
        t = self._transport()
        with pytest.raises(SyncProtocolError):
            t.pull("../etc/passwd", None)

    def test_pull_short_bpid_raises_protocol_error(self):
        t = self._transport()
        with pytest.raises(SyncProtocolError):
            t.pull("abc", None)

    def test_pull_uppercase_raises_protocol_error(self):
        t = self._transport()
        with pytest.raises(SyncProtocolError):
            t.pull("B" * 64, None)

    def test_push_valid_bpid_does_not_raise(self):
        t, _ = _make_transport_with_fake([{"cursor": "0"}])
        t.push(self._good_bpid(), [])  # must not raise

    def test_pull_valid_bpid_does_not_raise(self):
        t, _ = _make_transport_with_fake([{"entries": [], "cursor": "0"}])
        t.pull(self._good_bpid(), None)  # must not raise


# ---------------------------------------------------------------------------
# TLS verify default is ON
# ---------------------------------------------------------------------------


class TestTLSVerify:
    def test_verify_none_creates_validating_context(self):
        ctx = _build_ssl_context(None)
        assert ctx is not None
        assert isinstance(ctx, ssl.SSLContext)
        # CERT_NONE == 0; validation ON means not CERT_NONE
        assert ctx.verify_mode != ssl.CERT_NONE

    def test_verify_true_creates_validating_context(self):
        ctx = _build_ssl_context(True)
        assert ctx is not None
        assert ctx.verify_mode != ssl.CERT_NONE

    def test_verify_false_disables_verification(self):
        ctx = _build_ssl_context(False)
        assert ctx is not None
        assert ctx.verify_mode == ssl.CERT_NONE
        assert ctx.check_hostname is False

    def test_verify_str_uses_cafile_path(self, tmp_path, monkeypatch):
        # Monkeypatch ssl.create_default_context to avoid loading an actual PEM.
        import quipu.sync.client as client_mod

        captured: list[dict] = []
        fake_ctx = ssl.create_default_context()  # real validating context

        def _fake_create(cafile=None):
            captured.append({"cafile": cafile})
            return fake_ctx

        monkeypatch.setattr(client_mod.ssl, "create_default_context", _fake_create)

        ca_path = str(tmp_path / "ca.crt")
        ctx = _build_ssl_context(ca_path)

        # _build_ssl_context must have called create_default_context(cafile=ca_path)
        assert len(captured) == 1
        assert captured[0]["cafile"] == ca_path
        # Returned context is the real validating context (validation ON)
        assert ctx is not None
        assert ctx.verify_mode != ssl.CERT_NONE
