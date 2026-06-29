"""Quipu configuration: mode detection, project-id derivation, DB path routing,
hub configuration.

This module is import-acyclic: it NEVER imports quipu.storage.paths.
quipu.storage.paths imports this module (local import in the else-branch).

Environment variables:
  QUIPU_MODE            'project' | 'global'; default 'project' (unrecognized -> 'project')
  QUIPU_PROJECT_ROOT    Root dir for project-mode DB; default cwd.
  QUIPU_HUB_URL         Hub URL (also readable from config.json hub_url).
  QUIPU_HUB_TOKEN       Bearer token for hub auth (env ONLY — never written to disk).
  QUIPU_HUB_CA          Path to CA bundle for TLS verification (optional).
  QUIPU_CLIENT_ID       Override client_id (env -> config.json -> generate uuid4().hex).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Any


def get_mode() -> str:
    """Return the current mode: 'project' or 'global'.

    Reads QUIPU_MODE; defaults to 'project'; unrecognized values -> 'project'.
    """
    raw = os.environ.get("QUIPU_MODE", "").strip().lower()
    if raw == "global":
        return "global"
    return "project"


def get_project_root() -> Path:
    """Return the resolved project root path.

    Uses QUIPU_PROJECT_ROOT env var if set, otherwise cwd.
    """
    env = os.environ.get("QUIPU_PROJECT_ROOT")
    if env:
        return Path(env).resolve()
    return Path.cwd().resolve()


def get_project_id(path: str | None = None) -> str:
    """Derive a deterministic 16-hex-char project id from a directory path.

    Canonicalization:
      - Resolve to absolute path (strict=False — need not exist).
      - Convert to POSIX string.
      - On Windows (os.name == 'nt'): lowercase (case-insensitive FS).
      - Strip trailing slash (unless the root '/').
      - SHA-256 of UTF-8 encoded canonical string, first 16 hex chars.

    Args:
        path: Directory path to identify. Defaults to os.getcwd().

    Returns:
        16-character lowercase hex string (stable across runs for the same path).

    Note:
        This ID is an internal partition key, NOT an auth token; do not rely on
        it for access control (HMAC project-id arrives in E10/E13).
    """
    p = Path(path or os.getcwd()).resolve(strict=False)
    s = p.as_posix()
    if os.name == "nt":
        s = s.lower()
    s = s.rstrip("/") or "/"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def resolve_mode_db_path() -> Path:
    """Return the DB path based on mode/env. Does NOT mkdir (paths.py owns mkdir).

    Precedence within this function:
      QUIPU_MODE unset  -> ~/.quipu/quipu.db      (COMPAT: keeps existing tests green)
      QUIPU_MODE=global -> ~/.quipu/global.db
      QUIPU_MODE=project -> <project-root>/.quipu/quipu.db
    """
    if "QUIPU_MODE" not in os.environ:
        return Path.home() / ".quipu" / "quipu.db"
    if get_mode() == "global":
        return Path.home() / ".quipu" / "global.db"
    return get_project_root() / ".quipu" / "quipu.db"


# ---------------------------------------------------------------------------
# Hub configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class HubConfig:
    """Resolved hub connection parameters.

    token is sourced from QUIPU_HUB_TOKEN env ONLY — never written to disk.
    verify is the TLS verification mode passed to HttpTransport:
        None -> ssl.create_default_context() (validation ON, default)
        str  -> CA bundle path
    """

    url: str
    token: str = dataclasses.field(repr=False)
    verify: str | None


def _read_config_json(config_path: Path) -> dict:
    """Read config.json at path; return empty dict on missing/invalid."""
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _config_json_path() -> Path:
    """Return the config.json path consistent with cli._config_path logic."""
    mode = get_mode()
    if mode == "global":
        return Path.home() / ".quipu" / "config.json"
    return get_project_root() / ".quipu" / "config.json"


def get_hub_config() -> HubConfig | None:
    """Return HubConfig if hub is configured, else None.

    Resolution order:
      url:    QUIPU_HUB_URL env > config.json hub_url
      token:  QUIPU_HUB_TOKEN env ONLY (never from disk)
      verify: QUIPU_HUB_CA env (path) or None

    Returns None if url or token is missing (-> offline mode, no error).
    """
    url = os.environ.get("QUIPU_HUB_URL")
    if not url:
        cfg = _read_config_json(_config_json_path())
        url = cfg.get("hub_url") or None
    if not url:
        return None

    token = os.environ.get("QUIPU_HUB_TOKEN")
    if not token:
        return None

    verify: str | None = os.environ.get("QUIPU_HUB_CA") or None
    return HubConfig(url=url, token=token, verify=verify)


def get_client_id(store: Any = None) -> str:
    """Return a stable client_id for this Quipu installation.

    Resolution order:
    1. QUIPU_CLIENT_ID env
    2. config.json client_id
    3. Generate uuid4().hex, persist to config.json once

    The store parameter is accepted but unused (kept for call-site compatibility);
    client_id lives in config.json, not the SQLite store.
    """
    env_id = os.environ.get("QUIPU_CLIENT_ID")
    if env_id:
        return env_id

    config_path = _config_json_path()
    cfg = _read_config_json(config_path)
    if "client_id" in cfg and cfg["client_id"]:
        return cfg["client_id"]

    # Generate and persist
    import uuid
    new_id = uuid.uuid4().hex
    cfg["client_id"] = new_id
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    return new_id
