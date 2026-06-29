"""Migration 0004 — add access_count and last_accessed columns to atoms table."""

VERSION: int = 4

UP: str = """
ALTER TABLE atoms ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0;
ALTER TABLE atoms ADD COLUMN last_accessed TEXT;

CREATE INDEX IF NOT EXISTS idx_atoms_access ON atoms (project_id, access_count, created_at DESC);
"""

DOWN: str = """
DROP INDEX IF EXISTS idx_atoms_access;
ALTER TABLE atoms DROP COLUMN last_accessed;
ALTER TABLE atoms DROP COLUMN access_count;
"""
