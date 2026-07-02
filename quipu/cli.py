"""Quipu CLI helpers: version, init command.

This module is imported by quipu/__main__.py and keeps new logic separate from
the existing entry-point wiring so mirror/serve remain byte-for-byte stable.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

def get_version() -> str:
    """Return the running package version.

    Resolves from quipu/__init__.py fallback chain:
    _build_version -> importlib.metadata -> 0.0.0.
    """
    from quipu import __version__

    return __version__


def _config_path(mode: str, project_root: Path) -> Path:
    """Return the Path where config.json should be written.

    project mode: <project_root>/.quipu/config.json
    global mode:  ~/.quipu/config.json
    """
    if mode == "global":
        return Path.home() / ".quipu" / "config.json"
    return project_root / ".quipu" / "config.json"


def _write_config(
    path: Path,
    mode: str,
    project_id: str,
    project_root: Optional[Path],
    *,
    hub_url: Optional[str] = None,
    client_id: Optional[str] = None,
) -> dict:
    """Write (or idempotently update) config.json at *path*.

    Idempotency rules:
    - If the file already exists, READ it first.
    - PRESERVE existing ``project_id`` and ``created`` fields.
    - REFRESH ``quipu_version`` and ``last_init``.
    - Write atomically (json.dump + newline).

    hub_url: written only if provided (QUIPU_HUB_URL at init time).
    client_id: written only if provided; NEVER overwrites an existing value.
    NEVER writes the token.

    Returns the final dict that was written.
    """
    existing: dict = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {}

    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    config: dict = {
        "mode": mode,
        "project_id": existing.get("project_id", project_id),
        "quipu_version": get_version(),
        "created": existing.get("created", now_iso),
        "last_init": now_iso,
    }
    if project_root is not None:
        config["project_root"] = str(project_root)
    if hub_url is not None:
        config["hub_url"] = hub_url
    # Preserve existing client_id; only write new one if not present
    existing_cid = existing.get("client_id")
    if existing_cid:
        config["client_id"] = existing_cid
    elif client_id:
        config["client_id"] = client_id

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def cmd_drain(
    queue_path: Optional[str],
    db_path: Optional[str],
    project_id: Optional[str],
) -> int:
    """Implement ``quipu drain [--queue-path ...] [--db-path ...] [--project-id ...]``.

    Reads the capture queue, writes valid records to the store, prints a
    one-line summary, and returns 0.
    """
    from quipu.storage import store as open_store
    import quipu.capture as capture_mod

    s = open_store(db_path or None)
    try:
        counts = capture_mod.drain(
            queue_path=queue_path or None,
            project_id=project_id or None,
            store=s,
        )
    except Exception as exc:
        print(f"quipu drain: error: {exc}", file=sys.stderr)
        return 1
    finally:
        s.close()

    print(
        f"quipu drain: written={counts['written']} "
        f"skipped_malformed={counts['skipped_malformed']} "
        f"skipped_secret={counts['skipped_secret']} "
        f"skipped_foreign={counts['skipped_foreign']}"
    )
    return 0


def cmd_backfill(
    db_path: Optional[str],
    project_id: Optional[str],
) -> int:
    """Implement ``quipu backfill [--db-path ...] [--project-id ...]``.

    One-shot re-emit of pre-existing atoms into oplog_entries so a subsequent
    ``quipu push`` ships them. Mirrors cmd_drain: open store, resolve project_id
    (default get_project_id(); --project-id override), run backfill_project,
    print a one-line summary, return 0.

    When sync is not configured / no key is available the run is a clean
    local-only no-op (status="inactive") — clear message, return 0, no crash.
    """
    from quipu.config import get_project_id
    from quipu.oplog.backfill import backfill_project
    from quipu.storage import store as open_store

    pid = project_id or get_project_id()

    s = None
    try:
        s = open_store(db_path or None)
        result = backfill_project(s, pid)
    except Exception as exc:
        print(f"quipu backfill: error: {type(exc).__name__}", file=sys.stderr)
        return 1
    finally:
        if s is not None:
            s.close()

    if result.status == "inactive":
        print(
            "quipu backfill: sync not configured or key unavailable — "
            "nothing to backfill (local-only)"
        )
        return 0

    print(
        f"quipu backfill: emitted={result.emitted} skipped={result.skipped}"
    )
    return 0


def cmd_gc(
    db_path: Optional[str],
    project_id: Optional[str],
    dry_run: bool,
    run_flag: bool,
    min_age_days: int,
    min_access_count: int,
) -> int:
    """Implement ``quipu gc [--project-id ...] [--db-path ...] [--dry-run] [--run]``.

    Soft-invalidates stale low-value atoms. ``--run`` is required to execute;
    without it (default: dry-run) only lists candidates.
    """
    from quipu.config import get_project_id
    from quipu.storage import store as open_store

    if run_flag:
        dry_run = False

    s = None
    try:
        s = open_store(db_path or None)
        pid = project_id or get_project_id()
        if pid is None:
            print("quipu gc: error: project_id required (pass --project-id or set QUIPU_PROJECT_ID)", file=sys.stderr)
            return 1
        stale = s.list_stale(
            project_id=pid,
            min_age_days=min_age_days,
            min_access_count=min_access_count,
        )
        invalidated = 0
        if not dry_run:
            for atom in stale:
                s.update_invalidated(atom.id, True)
                invalidated += 1

        if dry_run:
            print(f"quipu gc: dry-run — {len(stale)} stale candidates")
        else:
            print(f"quipu gc: invalidated {invalidated} of {len(stale)} stale atoms")
        for a in stale:
            print(
                f"  {a.id[:12]}... access={a.access_count} "
                f"last_accessed={a.last_accessed or 'never'} "
                f"created={a.created_at} "
                f"content={a.content[:80].replace(chr(10), ' ')}"
            )
    except Exception as exc:
        print(f"quipu gc: error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if s is not None:
            s.close()
    return 0


def cmd_receipts(
    db_path: Optional[str],
    project_id: Optional[str],
    limit: Optional[int],
    fmt: str,
    op_filter: Optional[str],
) -> int:
    """Implement ``quipu receipts [--db-path ...] [--project-id ...] [--limit N] [--format json|text] [--op write|invalidate]``.

    Exports a hashed/redacted operation log for audit without plaintext content.
    """
    import hashlib

    from quipu.config import get_project_id
    from quipu.keystore._backend import get_or_derive_key
    from quipu.storage import store as open_store
    from quipu.sync._aad import aad_for
    from quipu.sync.oplog_store import OplogStore

    pid = project_id or get_project_id()

    s = None
    try:
        s = open_store(db_path or None)
        key = get_or_derive_key(pid)
        blinded = aad_for(pid, key).decode()

        oplog = OplogStore(s._conn)
        db_op = None
        if op_filter == "write":
            db_op = "upsert"
        elif op_filter == "invalidate":
            db_op = "invalidate"

        entries = oplog.list_entries_by_project(blinded, op=db_op, limit=limit)

        if fmt == "text":
            for e in entries:
                op_label = "write" if e.op == "upsert" else e.op
                rh = hashlib.sha256(f"{e.record_id}:{pid}".encode()).hexdigest()
                print(f"{op_label}\t{e.ts}\t{rh}")
        else:
            import json
            receipts = []
            for e in entries:
                op_label = "write" if e.op == "upsert" else e.op
                rh = hashlib.sha256(f"{e.record_id}:{pid}".encode()).hexdigest()
                receipts.append({"op": op_label, "ts": e.ts, "record_hash": rh})
            print(json.dumps(receipts, indent=2))
        return 0
    except Exception as exc:
        print(f"quipu receipts: error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        if s is not None:
            s.close()


def _install_venv_python() -> Optional[str]:
    """Return the install-script venv Python path if it exists, else None."""
    venv = Path.home() / ".quipu" / "venv"
    if sys.platform == "win32":
        candidate = venv / "Scripts" / "python.exe"
    else:
        candidate = venv / "bin" / "python"
    return str(candidate) if candidate.is_file() else None


def cmd_init(mode: Optional[str]) -> int:
    """Implement ``quipu init [--mode project|global|server]``.

    Args:
        mode: One of "project", "global", "server", or None (→ "project").

    Returns:
        0 on success, 2 for server-staged (not yet implemented).
    """
    if mode is None:
        mode = "project"

    from quipu.config import get_project_id, get_project_root
    from quipu.storage import store as _store_factory

    if mode == "server":
        # Server mode overlays the project DB (same location as project mode).
        os.environ["QUIPU_MODE"] = "project"
        project_root = get_project_root()
        project_id = get_project_id(str(project_root))
        db_path = project_root / ".quipu" / "quipu.db"

        # Open (and init-if-absent) the store.
        s = _store_factory(str(db_path))
        s.close()

        # Generate or preserve client_id (never generate token here).
        import uuid
        config_file = _config_path("server", project_root)
        # Read existing to check for client_id
        existing: dict = {}
        if config_file.exists():
            try:
                existing = json.loads(config_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {}
        new_client_id = existing.get("client_id") or uuid.uuid4().hex

        hub_url = os.environ.get("QUIPU_HUB_URL") or None
        config = _write_config(
            config_file,
            "server",
            project_id,
            project_root,
            hub_url=hub_url,
            client_id=new_client_id,
        )
        final_client_id = config.get("client_id", new_client_id)

        print(f"quipu init: mode=server, project_id={config['project_id']}")
        print(f"  DB:        {db_path}")
        print(f"  Config:    {config_file}")
        print(f"  hub_url:   {hub_url or 'unset — set QUIPU_HUB_URL'}")
        print(f"  client_id: {final_client_id}")
        print()
        print("QUIPU_HUB_TOKEN must be set in the environment (never written to disk).")
        print()
        print("Example .mcp.json snippet:")
        if _install_venv_python() is not None:
            python_cmd = str(_install_venv_python())
        else:
            python_cmd = sys.executable
        snippet = {
            "command": python_cmd,
            "args": ["-m", "quipu", "serve"],
            "env": {
                "QUIPU_MODE": "project",
                "QUIPU_PROJECT_ROOT": str(project_root),
                "QUIPU_HUB_URL": hub_url or "<set-your-hub-url>",
                "QUIPU_HUB_TOKEN": "<set-your-hub-token>",
            },
        }
        print(json.dumps(snippet, indent=2))
        return 0

    if mode == "global":
        # Force QUIPU_MODE=global so resolve_mode_db_path() returns global.db
        os.environ["QUIPU_MODE"] = "global"
        project_root = Path.home() / ".quipu"
        project_id = "global"
        db_path = Path.home() / ".quipu" / "global.db"
    else:
        # mode == "project"
        os.environ["QUIPU_MODE"] = "project"
        project_root = get_project_root()
        project_id = get_project_id(str(project_root))
        db_path = project_root / ".quipu" / "quipu.db"

    # Open (and init-if-absent) the store — idempotent; schema uses CREATE IF NOT EXISTS.
    s = _store_factory(str(db_path))
    s.close()

    # Write config.json — idempotent: preserves project_id/created on re-run.
    config_file = _config_path(mode, project_root)
    config = _write_config(
        config_file,
        mode,
        project_id,
        project_root if mode == "project" else None,
    )

    final_project_id = config["project_id"]

    print(f"quipu init: mode={mode}, project_id={final_project_id}")
    print(f"  DB:     {db_path}")
    print(f"  Config: {config_file}")
    print()
    if mode == "project":
        print("Set these env vars in your .mcp.json (or shell):")
        print(f'  QUIPU_MODE=project')
        print(f'  QUIPU_PROJECT_ROOT={project_root}')
    else:
        print("Set this env var in your .mcp.json (or shell):")
        print(f'  QUIPU_MODE=global')
    print()
    print("Example .mcp.json snippet:")
    # Use the install-script venv if it exists, otherwise the current Python
    # (handles both install.sh and pip install quipu-mcp)
    if _install_venv_python() is not None:
        python_cmd = _install_venv_python()
    else:
        python_cmd = sys.executable
    snippet = {
        "command": str(python_cmd),
        "args": ["-m", "quipu", "serve"],
        "env": {
            "QUIPU_MODE": mode,
            **({"QUIPU_PROJECT_ROOT": str(project_root)} if mode == "project" else {}),
        },
    }
    print(json.dumps(snippet, indent=2))

    return 0
