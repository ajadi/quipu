"""Drain the quipu capture queue into the write pipeline.

Reads .quipu/capture-queue.jsonl, validates each record, filters secrets and
foreign project_ids, then calls quipu.write.write() for each valid record.

Durability model (atomic rename):
  1. queue absent/empty → zero-counts no-op.
  2. os.replace(queue, queue+".processing") — atomic rename; producer keeps
     appending to the original path without data loss.
  3. Read .processing line-by-line, write() each valid record.
  4. Clean completion → os.remove(.processing) (== rotation/truncation).
  5. Stale .processing at entry → drain it first.
  If write() raises mid-drain: leave .processing in place and re-raise after
  logging; next run reprocesses the whole batch (may double-write succeeded
  records; accepted — cosine invalidate_superseded mitigates duplicates).
  Skips (malformed/foreign/secret) never leave .processing.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quipu.storage.store import Store

# ---------------------------------------------------------------------------
# Timestamp validation
# ---------------------------------------------------------------------------

# Accepts second-precision (quipu-capture.sh) and millisecond-precision (DB DEFAULT).
# Pattern: YYYY-MM-DDTHH:MM:SS[.fraction]Z
_TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")


def _validate_ts(ts: object) -> "str | None":
    """Validate an ISO-8601 UTC timestamp string.

    Returns the original string if valid, else None.
    Accepts second-precision (YYYY-MM-DDTHH:MM:SSZ) and millisecond-precision
    (YYYY-MM-DDTHH:MM:SS.fffZ).
    """
    if not isinstance(ts, str):
        return None
    if not _TS_RE.match(ts):
        return None
    # Round-trip validation (catches out-of-range dates like month=13)
    try:
        # Normalise to a form datetime.strptime can parse.
        ts_clean = ts.rstrip("Z")
        if "." in ts_clean:
            # Truncate/pad fractional to 6 digits for %f
            parts = ts_clean.split(".")
            frac = parts[1][:6].ljust(6, "0")
            ts_clean = f"{parts[0]}.{frac}"
            datetime.strptime(ts_clean, "%Y-%m-%dT%H:%M:%S.%f").replace(
                tzinfo=timezone.utc
            )
        else:
            datetime.strptime(ts_clean, "%Y-%m-%dT%H:%M:%S").replace(
                tzinfo=timezone.utc
            )
    except ValueError:
        return None
    return ts


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_LINE_BYTES = 1 * 1024 * 1024  # 1 MB


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def drain(
    queue_path: "str | Path | None" = None,
    project_id: "str | None" = None,
    *,
    store: "Store | None" = None,
) -> dict:
    """Drain the capture queue and write valid records to the store.

    Args:
        queue_path: Path to capture-queue.jsonl. If None, resolves via
                    QUIPU_PROJECT_ROOT (or cwd) + /.quipu/capture-queue.jsonl.
        project_id: Bound project scope. If None, reads QUIPU_PROJECT_ID from
                    env; if still None, accepts any record project_id (unbound).
        store: Optional injected Store. If None, opens one for the drain and
               closes it when done.

    Returns:
        dict with keys: written, skipped_malformed, skipped_secret,
        skipped_foreign.
    """
    from quipu.capture.secrets import looks_like_secret
    from quipu.write import write
    from quipu.storage import store as open_store

    counts: dict[str, int] = {
        "written": 0,
        "skipped_malformed": 0,
        "skipped_secret": 0,
        "skipped_foreign": 0,
    }

    # Resolve queue path
    if queue_path is None:
        from quipu.config import get_project_root
        root = get_project_root()
        resolved_queue = Path(root) / ".quipu" / "capture-queue.jsonl"
    else:
        resolved_queue = Path(queue_path)

    processing_path = Path(str(resolved_queue) + ".processing")

    # Resolve bound project_id
    if project_id is None:
        project_id = os.environ.get("QUIPU_PROJECT_ID") or None

    # Open store if not injected
    own_store = store is None
    if own_store:
        store = open_store()

    try:
        _drain_inner(
            resolved_queue,
            processing_path,
            project_id,
            store,
            counts,
            looks_like_secret,
            write,
        )
    finally:
        if own_store:
            store.close()

    return counts


def _drain_inner(
    queue: Path,
    processing: Path,
    project_id: "str | None",
    store: "Store",
    counts: dict,
    looks_like_secret_fn,
    write_fn,
) -> None:
    """Core drain logic (separated for testability)."""

    # Handle stale .processing from a prior crash — drain it first.
    if processing.exists():
        _drain_file(
            processing,
            project_id,
            store,
            counts,
            looks_like_secret_fn,
            write_fn,
        )
        # If drain succeeds, remove stale processing file.  If removal fails,
        # return early to avoid os.replace(queue, processing) overwriting the
        # not-yet-removed file with a fresh batch.
        try:
            os.remove(processing)
        except OSError as exc:
            print(
                f"quipu drain: warning: could not remove stale .processing: {exc}",
                file=sys.stderr,
            )
            return

    # Check if queue file exists and is non-empty.
    if not queue.exists():
        return
    if queue.stat().st_size == 0:
        return

    # Atomic rename: producer can keep writing to the original path.
    try:
        os.replace(queue, processing)
    except OSError as exc:
        print(
            f"quipu drain: error: could not rename queue to processing: {exc}",
            file=sys.stderr,
        )
        return

    # Drain the .processing file.
    _drain_file(
        processing,
        project_id,
        store,
        counts,
        looks_like_secret_fn,
        write_fn,
    )

    # Clean completion: remove .processing.
    try:
        os.remove(processing)
    except OSError as exc:
        print(
            f"quipu drain: warning: could not remove .processing after drain: {exc}",
            file=sys.stderr,
        )


def _drain_file(
    path: Path,
    project_id: "str | None",
    store: "Store",
    counts: dict,
    looks_like_secret_fn,
    write_fn,
) -> None:
    """Read path line-by-line and process each record.

    Raises if write_fn raises (caller handles cleanup). Skips do NOT raise.
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        print(
            f"quipu drain: error: could not read {path}: {exc}",
            file=sys.stderr,
        )
        return

    for raw_line in data.split(b"\n"):
        line = raw_line.strip()
        if not line:
            continue

        # 1. Size guard
        if len(line) > _MAX_LINE_BYTES:
            print(
                "quipu drain: skipped malformed record: line exceeds 1MB",
                file=sys.stderr,
            )
            counts["skipped_malformed"] += 1
            continue

        # 2. JSON parse
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            print(
                f"quipu drain: skipped malformed record: JSONDecodeError: {exc}",
                file=sys.stderr,
            )
            counts["skipped_malformed"] += 1
            continue

        if not isinstance(record, dict):
            print(
                "quipu drain: skipped malformed record: not a JSON object",
                file=sys.stderr,
            )
            counts["skipped_malformed"] += 1
            continue

        # 3. Required fields: content (non-empty str), project_id (str)
        content = record.get("content")
        rec_project_id = record.get("project_id")

        if not isinstance(content, str) or not content.strip():
            print(
                "quipu drain: skipped malformed record: missing or empty content",
                file=sys.stderr,
            )
            counts["skipped_malformed"] += 1
            continue

        if not isinstance(rec_project_id, str) or not rec_project_id:
            print(
                "quipu drain: skipped malformed record: missing or invalid project_id",
                file=sys.stderr,
            )
            counts["skipped_malformed"] += 1
            continue

        # 4. Bound-scope check
        if project_id is not None and rec_project_id != project_id:
            print(
                f"quipu drain: skipped foreign record: project_id={rec_project_id!r} "
                f"(bound to {project_id!r})",
                file=sys.stderr,
            )
            counts["skipped_foreign"] += 1
            continue

        # 5. Secret scanner — log WITHOUT echoing content
        if looks_like_secret_fn(content):
            print(
                "quipu drain: skipped secret record: content looks like a credential",
                file=sys.stderr,
            )
            counts["skipped_secret"] += 1
            continue

        # 6. Validate metadata
        raw_metadata = record.get("metadata")
        metadata = raw_metadata if isinstance(raw_metadata, dict) else None

        # 7. Validate type
        raw_type = record.get("type")
        rec_type = raw_type if isinstance(raw_type, str) and raw_type else "diary"

        # 8. Validate event timestamp
        valid_ts = _validate_ts(record.get("ts"))
        created_at = valid_ts  # None if invalid → write() uses SQL DEFAULT now

        if record.get("ts") and valid_ts is None:
            print(
                "quipu drain: malformed ts for record — using drain time as created_at",
                file=sys.stderr,
            )

        # 9. Validate session_id
        raw_session_id = record.get("session_id")
        session_id = raw_session_id if isinstance(raw_session_id, str) and raw_session_id else None

        # 10. Write (may raise — caller handles cleanup)
        write_fn(
            content,
            metadata=metadata,
            project_id=rec_project_id,
            store=store,
            type=rec_type,
            created_at=created_at,
            session_id=session_id,
        )
        counts["written"] += 1
