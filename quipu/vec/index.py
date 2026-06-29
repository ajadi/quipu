"""Public entry point for the vec index lifecycle.

ensure_index(conn, *, threshold=None) -> VecState

Idempotent orchestrator:
1. Gate: try_load sqlite-vec. If unavailable → UNAVAILABLE.
2. Threshold: if atom count < threshold → BELOW_THRESHOLD.
3. Build: call build(conn) (idempotent). Returns BUILDING mid-build,
   READY when status == 'complete'.
"""

from __future__ import annotations

import enum
import logging
import os
import sqlite3

from quipu.vec._gate import try_load
from quipu.vec._build import atom_count, build, is_build_complete
from quipu.vec._meta import ensure_meta_table, get_build_status

_logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 30_000


class VecState(enum.Enum):
    UNAVAILABLE = "unavailable"
    BELOW_THRESHOLD = "below"
    BUILDING = "building"
    READY = "ready"


def ensure_index(
    conn: sqlite3.Connection,
    *,
    threshold: int | None = None,
) -> VecState:
    """Idempotent vec-index lifecycle gate.

    Args:
        conn: An open sqlite3.Connection (post get_connection).
        threshold: Atom-count threshold to trigger build. Defaults to
                   int(QUIPU_VEC_THRESHOLD env) or 30 000.

    Returns:
        VecState indicating current state. Never raises on missing extension.
    """
    if threshold is None:
        raw = os.environ.get("QUIPU_VEC_THRESHOLD", "")
        if raw:
            try:
                parsed = int(raw)
                if parsed < 1:
                    raise ValueError("threshold must be >= 1")
                threshold = parsed
            except ValueError:
                _logger.warning(
                    "QUIPU_VEC_THRESHOLD=%r is not a valid positive integer; "
                    "falling back to default %d",
                    raw,
                    _DEFAULT_THRESHOLD,
                )
                threshold = _DEFAULT_THRESHOLD
        else:
            threshold = _DEFAULT_THRESHOLD

    # --- 1. Availability gate ---
    if not try_load(conn):
        return VecState.UNAVAILABLE

    # --- 2. Threshold check (skip if already built) ---
    ensure_meta_table(conn)
    status = get_build_status(conn)

    if status != "complete":
        count = atom_count(conn)
        if count < threshold:
            return VecState.BELOW_THRESHOLD

    # --- 3. Already complete? ---
    if status == "complete":
        return VecState.READY

    # --- 4. Build (idempotent) ---
    if status == "building":
        # Previous run crashed mid-build; resume.
        _logger.info("Resuming interrupted atoms_vec build…")

    build(conn)

    # After build(), check actual status.
    if is_build_complete(conn):
        return VecState.READY

    # Still in progress (shouldn't normally happen in synchronous path).
    return VecState.BUILDING
