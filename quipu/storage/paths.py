"""DB path resolution for Quipu storage.

Precedence (highest to lowest):
  1. Explicit override argument
  2. QUIPU_DB_PATH environment variable
  3. Mode-based routing via quipu.config.resolve_mode_db_path():
       QUIPU_MODE unset  -> ~/.quipu/quipu.db   (compat default)
       QUIPU_MODE=global -> ~/.quipu/global.db
       QUIPU_MODE=project -> <QUIPU_PROJECT_ROOT or cwd>/.quipu/quipu.db
"""

import os
from pathlib import Path


def resolve_db_path(override: str | Path | None = None) -> Path:
    """Return the resolved Path for the Quipu SQLite DB.

    Creates the parent directory if it does not exist.

    Precedence:
        explicit arg > QUIPU_DB_PATH env > mode-based routing (quipu.config)
    """
    if override is not None:
        path = Path(override)
    elif (env := os.environ.get("QUIPU_DB_PATH")):
        path = Path(env)
    else:
        from quipu.config import resolve_mode_db_path
        path = resolve_mode_db_path()

    path.parent.mkdir(parents=True, exist_ok=True)
    return path
