"""conftest.py for tests/capture — shared fixtures for quipu-capture hook tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Insert repo root so `import quipu` works without editable install.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
