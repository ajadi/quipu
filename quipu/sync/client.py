"""quipu.sync.client — HttpTransport + SyncResult + sync_now.

HttpTransport speaks the real hub wire contract (stdlib urllib.request only,
matching the outbound-HTTP precedent in quipu/write/flush.py).

Single _request seam: tests override self._request(method, path, body)
to drive a hub TestClient or inject fakes without subprocess overhead.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import socket
import ssl
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from quipu.sync.errors import (
    SyncAuthError,
    SyncProtocolError,
    SyncUnavailableError,
)

logger = logging.getLogger(__name__)

_CURSOR_RE = re.compile(r"^\d+$")
_CURSOR_MAX_LEN = 19
_BPID_RE = re.compile(r"^[0-9a-f]{64}$")


# ---------------------------------------------------------------------------
# HttpTransport
# ---------------------------------------------------------------------------


class HttpTransport:
    """Drop-in Transport implementation backed by stdlib urllib.request.

    Constructor:
        base_url: Hub root URL, e.g. "https://hub.example.com".
        token: Bearer token (from QUIPU_HUB_TOKEN env; NEVER logged).
        timeout: Request timeout in seconds (default 30).
        verify: TLS verification mode.
            None / True  -> ssl.create_default_context() (validation ON, default).
            str          -> path to CA bundle (cafile=).
            False        -> disabled (self-hosted dev only; documented, never default).
    """

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        timeout: float = 30.0,
        verify: bool | str | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._token = token
        self._timeout = timeout
        self._ssl_ctx = _build_ssl_context(verify)

    # ------------------------------------------------------------------
    # Single network seam — override in tests to avoid real HTTP
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: dict | None,
        params: dict | None = None,
    ) -> Any:
        """Execute an HTTP request; return parsed JSON body.

        Raises typed SyncError subclasses on all failure modes.
        """
        url = self._base_url + path
        if params:
            url = url + "?" + urllib.parse.urlencode(params)

        headers = {"Authorization": f"Bearer {self._token}"}
        data: bytes | None = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self._timeout, context=self._ssl_ctx) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw)
        except urllib.error.HTTPError as exc:
            status = exc.code
            if status == 401:
                raise SyncAuthError(f"401 Unauthorized from hub") from exc
            if status in (400, 413, 422):
                raise SyncProtocolError(f"HTTP {status} from hub") from exc
            if status == 429 or (500 <= status < 600):
                raise SyncUnavailableError(f"HTTP {status} from hub") from exc
            # Other HTTP errors treated as protocol errors
            raise SyncProtocolError(f"Unexpected HTTP {status} from hub") from exc
        except urllib.error.URLError as exc:
            raise SyncUnavailableError(f"Network error: {exc.reason}") from exc
        except socket.timeout as exc:
            raise SyncUnavailableError(f"Request timed out") from exc
        except TimeoutError as exc:
            raise SyncUnavailableError(f"Request timed out") from exc
        except json.JSONDecodeError as exc:
            raise SyncProtocolError(f"Malformed JSON from hub: {exc}") from exc

    # ------------------------------------------------------------------
    # Exception translation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _translate_exc(exc: BaseException) -> None:
        """Re-raise *exc* as a typed SyncError.

        Called when self._request() (which may be overridden in tests) raises a
        raw exception that the real _request would have translated internally.
        This ensures the same typed errors appear on both the live path and the
        test-override path.
        """
        if isinstance(exc, (SyncAuthError, SyncProtocolError, SyncUnavailableError)):
            raise exc
        if isinstance(exc, urllib.error.HTTPError):
            status = exc.code
            if status == 401:
                raise SyncAuthError(f"401 Unauthorized from hub") from exc
            if status in (400, 413, 422):
                raise SyncProtocolError(f"HTTP {status} from hub") from exc
            if status == 429 or (500 <= status < 600):
                raise SyncUnavailableError(f"HTTP {status} from hub") from exc
            raise SyncProtocolError(f"Unexpected HTTP {status} from hub") from exc
        if isinstance(exc, urllib.error.URLError):
            raise SyncUnavailableError(f"Network error: {exc.reason}") from exc
        if isinstance(exc, (socket.timeout, TimeoutError)):
            raise SyncUnavailableError("Request timed out") from exc
        if isinstance(exc, ConnectionError):
            raise SyncUnavailableError(f"Connection error: {exc}") from exc
        raise exc

    # ------------------------------------------------------------------
    # Transport Protocol implementation
    # ------------------------------------------------------------------

    def push(self, blinded_project_id: str, entries: list[dict]) -> None:
        """POST /oplog/{bpid} with entries.

        Ignores the hub's returned {"cursor": ...} — push.py tracks its own cursor.
        """
        if not _BPID_RE.fullmatch(blinded_project_id):
            raise SyncProtocolError(
                f"blinded_project_id failed validation (expected 64 lowercase hex chars)"
            )
        try:
            self._request(
                "POST",
                f"/oplog/{blinded_project_id}",
                {"entries": entries},
            )
        except BaseException as exc:
            self._translate_exc(exc)

    def pull(
        self, blinded_project_id: str, cursor: str | None
    ) -> tuple[list[dict], str | None]:
        """GET /oplog/{bpid}[?since=cursor]. Defensive-parses the envelope."""
        if not _BPID_RE.fullmatch(blinded_project_id):
            raise SyncProtocolError(
                f"blinded_project_id failed validation (expected 64 lowercase hex chars)"
            )
        params = {"since": cursor} if cursor is not None else None
        try:
            raw = self._request("GET", f"/oplog/{blinded_project_id}", None, params)
        except BaseException as exc:
            self._translate_exc(exc)
            raise  # unreachable; satisfies type-checker

        # Defensive-parse envelope
        if not isinstance(raw, dict):
            raise SyncProtocolError("Hub pull response is not a JSON object")
        entries = raw.get("entries")
        next_cursor = raw.get("cursor")
        if not isinstance(entries, list):
            raise SyncProtocolError("Hub pull response missing 'entries' list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise SyncProtocolError("Hub pull entry is not a dict")
        # Validate cursor: must be None or a digit-string of max 19 chars
        if next_cursor is not None:
            if not isinstance(next_cursor, str):
                raise SyncProtocolError("Hub pull cursor is not a string")
            if len(next_cursor) > _CURSOR_MAX_LEN or not _CURSOR_RE.match(next_cursor):
                raise SyncProtocolError(
                    f"Hub pull cursor failed validation: {next_cursor!r}"
                )

        return entries, next_cursor


# ---------------------------------------------------------------------------
# TLS context builder
# ---------------------------------------------------------------------------


def _build_ssl_context(verify: bool | str | None) -> ssl.SSLContext | None:
    """Build an SSLContext based on the verify parameter.

    None/True -> create_default_context() (validation ON, the safe default).
    str       -> create_default_context(cafile=verify) (custom CA bundle).
    False     -> disabled context (self-hosted dev only, documented).
    """
    if verify is False:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        return ctx
    if isinstance(verify, str):
        return ssl.create_default_context(cafile=verify)
    # None or True -> default validating context
    return ssl.create_default_context()


# ---------------------------------------------------------------------------
# SyncResult dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SyncResult:
    """Result returned by sync_now."""

    direction: str  # "pull" | "push" | "pull+push"
    status: str     # "ok" | "offline" | "never_configured"
    pushed: int = 0
    pulled: int = 0
    detail: str | None = None


# ---------------------------------------------------------------------------
# Process-lifetime sync status marker
# ---------------------------------------------------------------------------

# "ok" | "offline" | "never_configured" — updated by sync_now each run
_last_sync_status: str = "never_configured"


def get_last_sync_status() -> str:
    return _last_sync_status


def reset_last_sync_status() -> None:
    """Reset the process-lifetime sync status to 'never_configured'.

    Call this in test fixtures/teardown to prevent status leaking between tests.
    """
    global _last_sync_status
    _last_sync_status = "never_configured"


def _set_last_sync_status(status: str) -> None:
    global _last_sync_status
    _last_sync_status = status


# ---------------------------------------------------------------------------
# sync_now
# ---------------------------------------------------------------------------


def sync_now(
    project_id: str,
    *,
    store: Any,
    directions: tuple[str, ...] = ("pull", "push"),
) -> SyncResult:
    """Sync the local store with the hub.

    Steps:
    1. get_hub_config() -> None => never_configured, no network/key.
    2. Derive key once via get_or_derive_key(project_id) (NO store arg).
       Failure -> WARNING + offline, return.
    3. Build HttpTransport; get client_id.
    4. Run requested legs (pull-then-push by default).
    5. Catch SyncUnavailableError per leg -> offline, stop.
       Other SyncError/DecryptError -> WARNING + offline.
    6. All success -> ok.

    NEVER raises. NEVER prompts. NEVER retries.
    """
    from quipu.config import get_hub_config, get_client_id

    cfg = get_hub_config()
    if cfg is None:
        _set_last_sync_status("never_configured")
        return SyncResult(
            direction="+".join(directions),
            status="never_configured",
        )

    # Derive key (headless: QUIPU_KEY or cached; failure => offline)
    try:
        from quipu.keystore._backend import get_or_derive_key
        key = get_or_derive_key(project_id)
    except Exception as exc:
        logger.warning("sync_now: key derivation failed for %s: %s", project_id, exc)
        _set_last_sync_status("offline")
        return SyncResult(
            direction="+".join(directions),
            status="offline",
            detail="internal_error",
        )

    transport = HttpTransport(cfg.url, cfg.token, verify=cfg.verify)
    client_id = get_client_id(store)

    from quipu.sync.push import push
    from quipu.sync.pull import pull

    pushed = 0
    pulled = 0

    for direction in directions:
        try:
            if direction == "push":
                pushed = push(project_id, store=store, transport=transport, key=key, client_id=client_id)
            elif direction == "pull":
                pulled = pull(project_id, store=store, transport=transport, key=key, client_id=client_id)
        except SyncUnavailableError as exc:
            logger.warning("sync_now: hub unavailable during %s: %s", direction, exc)
            _set_last_sync_status("offline")
            return SyncResult(
                direction=direction,
                status="offline",
                pushed=pushed,
                pulled=pulled,
                detail="offline",
            )
        except SyncAuthError as exc:
            logger.warning("sync_now: auth failed during %s: %s", direction, exc)
            _set_last_sync_status("offline")
            return SyncResult(
                direction=direction,
                status="offline",
                pushed=pushed,
                pulled=pulled,
                detail="auth_failed",
            )
        except SyncProtocolError as exc:
            logger.warning("sync_now: protocol error during %s: %s", direction, exc)
            _set_last_sync_status("offline")
            return SyncResult(
                direction=direction,
                status="offline",
                pushed=pushed,
                pulled=pulled,
                detail="protocol_error",
            )
        except Exception as exc:
            logger.warning("sync_now: %s failed: %s", direction, exc, exc_info=True)
            _set_last_sync_status("offline")
            return SyncResult(
                direction=direction,
                status="offline",
                pushed=pushed,
                pulled=pulled,
                detail="internal_error",
            )

    _set_last_sync_status("ok")
    return SyncResult(
        direction="+".join(directions),
        status="ok",
        pushed=pushed,
        pulled=pulled,
    )
