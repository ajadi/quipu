"""Migration 0003 — add session_id column and index to atoms table."""

VERSION: int = 3

UP: str = """
ALTER TABLE atoms ADD COLUMN session_id TEXT;

CREATE INDEX IF NOT EXISTS idx_atoms_session ON atoms (project_id, session_id, created_at DESC);
"""

DOWN: str = """
DROP INDEX IF EXISTS idx_atoms_session;
ALTER TABLE atoms DROP COLUMN session_id;
"""
