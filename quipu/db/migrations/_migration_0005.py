"""Migration 0005 — add tags TEXT column to atoms table."""

VERSION: int = 5

UP: str = """
ALTER TABLE atoms ADD COLUMN tags TEXT;
"""

DOWN: str = """
ALTER TABLE atoms DROP COLUMN tags;
"""
