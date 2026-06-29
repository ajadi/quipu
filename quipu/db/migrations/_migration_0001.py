"""Migration 0001 — atoms table, indexes, updated_at trigger."""

VERSION: int = 1

UP: str = """
CREATE TABLE IF NOT EXISTS atoms (
    id          TEXT     NOT NULL,
    type        TEXT     NOT NULL,
    scope       TEXT     NOT NULL DEFAULT 'project',
    content     TEXT     NOT NULL,
    embedding   BLOB,
    metadata    TEXT,
    project_id  TEXT,
    refs        TEXT,
    invalidated INTEGER  NOT NULL DEFAULT 0,
    created_at  TEXT     NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    updated_at  TEXT     NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    CONSTRAINT atoms_pk PRIMARY KEY (id),
    CONSTRAINT atoms_type_ck CHECK (type IN (
        'decision', 'pattern', 'diary', 'entity',
        'oq-resolution', 'infra-fact', 'server', 'deploy-target'
    )),
    CONSTRAINT atoms_scope_ck CHECK (scope IN ('project', 'global', 'all')),
    CONSTRAINT atoms_invalidated_ck CHECK (invalidated IN (0, 1))
);

CREATE INDEX IF NOT EXISTS idx_atoms_project_id   ON atoms (project_id);
CREATE INDEX IF NOT EXISTS idx_atoms_invalidated  ON atoms (invalidated);
CREATE INDEX IF NOT EXISTS idx_atoms_type         ON atoms (type);
CREATE INDEX IF NOT EXISTS idx_atoms_created_at   ON atoms (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_atoms_project_valid ON atoms (project_id, invalidated, created_at DESC);

CREATE TRIGGER IF NOT EXISTS atoms_updated_at
AFTER UPDATE ON atoms
FOR EACH ROW
BEGIN
    UPDATE atoms
    SET updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
    WHERE id = NEW.id;
END;
"""

DOWN: str = """
DROP TRIGGER IF EXISTS atoms_updated_at;
DROP INDEX IF EXISTS idx_atoms_project_valid;
DROP INDEX IF EXISTS idx_atoms_created_at;
DROP INDEX IF EXISTS idx_atoms_type;
DROP INDEX IF EXISTS idx_atoms_invalidated;
DROP INDEX IF EXISTS idx_atoms_project_id;
DROP TABLE IF EXISTS atoms;
"""
