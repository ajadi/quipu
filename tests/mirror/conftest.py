"""conftest.py for tests/mirror — ensures quipu package is importable."""

import sys
from pathlib import Path

# Insert repo root so `import quipu` works when pytest is run without
# the package installed in editable mode.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
