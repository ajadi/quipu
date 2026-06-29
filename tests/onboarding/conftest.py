"""conftest.py for tests/onboarding — ensures quipu package is importable and
isolates direct os.environ mutations made by quipu.cli.cmd_init().

cmd_init() does direct os.environ["QUIPU_MODE"] = "project"/"global" (not via
monkeypatch), so monkeypatch.delenv/setenv alone cannot revert those writes.
The autouse fixture below snapshots the relevant keys before each test and
restores exact prior state in teardown, preventing leaks into other suites
(e.g. tests/storage/test_paths.py::TestResolveDbPath::test_default_path_is_under_home_when_no_override_no_env).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Insert repo root so `import quipu` works without editable install.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Keys that cmd_init() mutates directly.
_WATCHED_KEYS = (
    "QUIPU_MODE",
    "QUIPU_PROJECT_ROOT",
    "QUIPU_DB_PATH",
    "QUIPU_PROJECT_ID",
)


@pytest.fixture(autouse=True)
def _isolate_quipu_env():
    """Snapshot QUIPU_* env keys before each test; restore exact prior state after."""
    snapshot = {k: os.environ.get(k) for k in _WATCHED_KEYS}
    yield
    for key, prior in snapshot.items():
        if prior is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = prior
