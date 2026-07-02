"""Tests for core/hooks/quipu-capture.sh and core/hooks/pre-compact.sh.

Runs the shell scripts via subprocess.  Each test uses an isolated temp
directory as CLAUDE_PROJECT_DIR so no state bleeds between tests.

Coverage:
  (a) QUIPU_PROJECT_ID unset  → exit 0, no spool file created.
  (b) Valid event (JSON stdin with agent + TASK-XXX) → exactly one valid-JSON
      line; correct source, task_id, project_id, non-empty ts.
  (c) Content with double-quotes AND newline → single line, valid JSON, newline
      flattened to space.
  (d) jq-absent path (QUIPU_CAPTURE_NO_JQ=1) → still valid JSON line.
  (e) Unwritable .quipu dir (pre-created as a regular file) → stderr warning,
      exit 0.
  (f) pre-compact.sh: produces handoffs/precompact-*.md snapshot AND (when
      QUIPU_PROJECT_ID set) appends a capture line with source=pre_compact.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from glob import glob
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="shell hooks require Unix paths")

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CAPTURE_HOOK = _REPO_ROOT / "core" / "hooks" / "quipu-capture.sh"
_PRECOMPACT_HOOK = _REPO_ROOT / "core" / "hooks" / "pre-compact.sh"

# Resolve the shell to use. On Windows with Git Bash we want bash (sh is
# a thin wrapper that may not support all constructs).
_SH = "bash"


def _run_capture(
    *,
    tmpdir: Path,
    env_extra: dict[str, str] | None = None,
    stdin_text: str = "",
    capture_source: str = "test_source",
) -> subprocess.CompletedProcess:
    """Helper: run quipu-capture.sh with a clean isolated CLAUDE_PROJECT_DIR."""
    env = os.environ.copy()
    env["CLAUDE_PROJECT_DIR"] = str(tmpdir)
    env["QUIPU_CAPTURE_SOURCE"] = capture_source
    # Remove any inherited project id by default; callers add it explicitly.
    env.pop("QUIPU_PROJECT_ID", None)
    env.pop("QUIPU_TASK_ID", None)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [_SH, str(_CAPTURE_HOOK)],
        input=stdin_text,
        capture_output=True,
        text=True,
        env=env,
    )


def _spool_file(tmpdir: Path) -> Path:
    return tmpdir / ".quipu" / "capture-queue.jsonl"


# ---------------------------------------------------------------------------
# (a) QUIPU_PROJECT_ID unset → exit 0, no spool file created
# ---------------------------------------------------------------------------

def test_no_project_id_exits_clean(tmp_path):
    result = _run_capture(tmpdir=tmp_path, stdin_text='{"agent_name":"pm"}')
    assert result.returncode == 0, f"stderr: {result.stderr}"
    spool = _spool_file(tmp_path)
    assert not spool.exists(), "spool file must NOT be created when QUIPU_PROJECT_ID unset"


# ---------------------------------------------------------------------------
# (b) Valid event → exactly one valid-JSON line; correct fields
# ---------------------------------------------------------------------------

def test_valid_event_appends_one_line(tmp_path):
    stdin_json = '{"agent_name":"developer","task":"TASK-007 — do stuff"}'
    result = _run_capture(
        tmpdir=tmp_path,
        env_extra={
            "QUIPU_PROJECT_ID": "proj-abc",
            "QUIPU_TASK_ID": "TASK-007",
        },
        stdin_text=stdin_json,
        capture_source="pm_close",
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    spool = _spool_file(tmp_path)
    assert spool.exists(), "spool file must be created"
    lines = [l for l in spool.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1, f"expected exactly 1 line, got {len(lines)}: {lines}"

    record = json.loads(lines[0])  # raises if invalid JSON
    assert record["source"] == "pm_close"
    assert record["task_id"] == "TASK-007"
    assert record["project_id"] == "proj-abc"
    assert record["ts"], "ts must be non-empty"
    assert record["v"] == 1
    assert record["metadata"]["captured_by"] == "quipu-capture.sh"


# ---------------------------------------------------------------------------
# (c) Content with double-quotes + newline → single line, valid JSON, flattened
# ---------------------------------------------------------------------------

def test_content_quotes_and_newlines_are_flattened(tmp_path):
    # Raw text stdin (not JSON) so content = raw stdin; contains quote + newline.
    stdin_text = 'She said "hello"\nSecond line here'
    result = _run_capture(
        tmpdir=tmp_path,
        env_extra={"QUIPU_PROJECT_ID": "proj-q"},
        stdin_text=stdin_text,
        capture_source="retro",
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    spool = _spool_file(tmp_path)
    assert spool.exists()
    lines = [l for l in spool.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1, f"expected exactly 1 line, got {len(lines)}: {lines}"

    record = json.loads(lines[0])  # must be valid JSON
    # The newline must be flattened to a space (not embedded as actual LF).
    assert "\n" not in record["content"], "newline must be flattened in content"
    # Double-quote must survive AND "hello" must be present.
    assert '"' in record["content"], "double-quote must be preserved in content"
    assert "hello" in record["content"], "'hello' must be present in content"


# ---------------------------------------------------------------------------
# (d) jq-absent path (QUIPU_CAPTURE_NO_JQ=1) → still valid JSON
# ---------------------------------------------------------------------------

def test_jq_absent_still_valid_json(tmp_path):
    """Force the sed fallback path via QUIPU_CAPTURE_NO_JQ=1.

    The hook checks this env var before `command -v jq` and bypasses jq entirely
    when set to 1.  This is the documented approach for environments where PATH
    manipulation is unreliable (e.g. Windows MSYS).
    """
    stdin_json = '{"agent_name":"tester","summary":"test summary TASK-007"}'
    result = _run_capture(
        tmpdir=tmp_path,
        env_extra={
            "QUIPU_PROJECT_ID": "proj-nojq",
            "QUIPU_TASK_ID": "TASK-007",
            "QUIPU_CAPTURE_NO_JQ": "1",
        },
        stdin_text=stdin_json,
        capture_source="reality_check",
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    spool = _spool_file(tmp_path)
    assert spool.exists(), "spool file must be created even without jq"
    lines = [l for l in spool.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1, f"expected exactly 1 line, got {len(lines)}: {lines}"

    record = json.loads(lines[0])  # must be valid JSON
    assert record["source"] == "reality_check"
    assert record["project_id"] == "proj-nojq"
    assert record["v"] == 1


# ---------------------------------------------------------------------------
# (e) Unwritable .quipu dir → stderr warning, exit 0
# ---------------------------------------------------------------------------

def test_unwritable_quipu_dir_exits_clean(tmp_path):
    """Pre-create .quipu as a regular FILE so mkdir -p .quipu fails."""
    quipu_path = tmp_path / ".quipu"
    quipu_path.write_text("i am a file not a dir")  # blocks mkdir -p

    result = _run_capture(
        tmpdir=tmp_path,
        env_extra={"QUIPU_PROJECT_ID": "proj-nw"},
        stdin_text="some content",
        capture_source="oq_resolution",
    )
    assert result.returncode == 0, f"process must exit 0, stderr: {result.stderr}"
    assert "warning" in result.stderr.lower(), (
        f"expected 'warning' in stderr, got: {result.stderr!r}"
    )


# ---------------------------------------------------------------------------
# (f) pre-compact.sh: snapshot file created + capture line with source=pre_compact
# ---------------------------------------------------------------------------

def test_precompact_produces_snapshot_and_capture(tmp_path):
    """Run pre-compact.sh from the project root (it references tasks/, tz.md etc.)."""
    env = os.environ.copy()
    env["CLAUDE_PROJECT_DIR"] = str(tmp_path)
    env["QUIPU_PROJECT_ID"] = "proj-pc"
    env.pop("QUIPU_TASK_ID", None)

    # Run pre-compact.sh with cwd = tmp_path (so tasks/, handoffs/ etc. resolve).
    result = subprocess.run(
        [_SH, str(_PRECOMPACT_HOOK)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, f"pre-compact.sh exited non-zero: {result.stderr}"

    # Check snapshot created.
    handoffs = tmp_path / "handoffs"
    snapshots = list(handoffs.glob("precompact-*.md")) if handoffs.exists() else []
    assert snapshots, f"no precompact-*.md snapshot found under {handoffs}"

    snap_content = snapshots[0].read_text(encoding="utf-8")
    assert "SESSION STATE BEFORE COMPACTION" in snap_content

    # Check spool written with source=pre_compact.
    spool = _spool_file(tmp_path)
    assert spool.exists(), "capture-queue.jsonl must be written by pre-compact.sh"
    lines = [l for l in spool.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert lines, "spool must contain at least one line"

    record = json.loads(lines[0])
    assert record["source"] == "pre_compact", f"expected source=pre_compact, got {record['source']}"
    assert record["project_id"] == "proj-pc"


# ---------------------------------------------------------------------------
# (g) sed-fallback path: literal TAB in content → valid JSON, TAB escaped
# ---------------------------------------------------------------------------

def test_jq_absent_tab_in_content_produces_valid_json(tmp_path):
    """FIX 1: _esc() must escape TAB to \\t so the output is valid JSON.

    Forces the sed fallback via QUIPU_CAPTURE_NO_JQ=1 and feeds content
    containing a literal TAB character.  Asserts json.loads succeeds and
    no raw TAB survives in content.
    """
    stdin_text = "column1\tcolumn2\tcolumn3"
    result = _run_capture(
        tmpdir=tmp_path,
        env_extra={
            "QUIPU_PROJECT_ID": "proj-tab",
            "QUIPU_CAPTURE_NO_JQ": "1",
        },
        stdin_text=stdin_text,
        capture_source="test_tab",
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    spool = _spool_file(tmp_path)
    assert spool.exists(), "spool file must be created"
    lines = [l for l in spool.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1, f"expected exactly 1 line, got {len(lines)}: {lines}"

    # (b) Raw file bytes must NOT contain a 0x09 byte — the TAB was escaped as \t.
    raw_bytes = spool.read_bytes()
    assert b"\x09" not in raw_bytes, "raw 0x09 TAB byte must not appear in spool file"

    # (a) Must parse as valid JSON — raises JSONDecodeError if TAB is unescaped.
    record = json.loads(lines[0])

    # (c) After JSON decoding, \t round-trips back to a real tab character.
    assert "\t" in record["content"], "JSON \\t must decode to real tab after round-trip"
    assert "column1" in record["content"] and "column2" in record["content"]
