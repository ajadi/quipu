"""Migration 0006 — add kg_triples and kg_edges tables for knowledge graph."""

VERSION: int = 6

UP: str = """
CREATE TABLE IF NOT EXISTS kg_triples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    subject     TEXT    NOT NULL,
    predicate   TEXT    NOT NULL,
    object      TEXT    NOT NULL,
    valid_from  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    valid_to    TEXT,
    confidence  REAL    NOT NULL DEFAULT 1.0,
    source_ref  TEXT,
    project_id  TEXT,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    CONSTRAINT kg_triples_ck_conf CHECK (confidence >= 0.0 AND confidence <= 1.0)
);
CREATE INDEX IF NOT EXISTS idx_kg_triples_subj   ON kg_triples (subject, project_id);
CREATE INDEX IF NOT EXISTS idx_kg_triples_obj    ON kg_triples (object, project_id);
CREATE INDEX IF NOT EXISTS idx_kg_triples_pred   ON kg_triples (predicate);
CREATE INDEX IF NOT EXISTS idx_kg_triples_proj   ON kg_triples (project_id, valid_from);

CREATE TABLE IF NOT EXISTS kg_edges (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    from_atom_id    TEXT    NOT NULL,
    to_atom_id      TEXT    NOT NULL,
    edge_type       TEXT    NOT NULL,
    project_id      TEXT,
    metadata        TEXT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    CONSTRAINT kg_edges_type_ck CHECK (edge_type IN (
        'supersedes', 'blocks', 'touches_file', 'decided_by', 'depends_on', 'causal'
    ))
);
CREATE INDEX IF NOT EXISTS idx_kg_edges_from     ON kg_edges (from_atom_id, project_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_to       ON kg_edges (to_atom_id, project_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_type     ON kg_edges (edge_type);
CREATE INDEX IF NOT EXISTS idx_kg_edges_proj     ON kg_edges (project_id, edge_type);
"""

DOWN: str = """
DROP INDEX IF EXISTS idx_kg_edges_proj;
DROP INDEX IF EXISTS idx_kg_edges_type;
DROP INDEX IF EXISTS idx_kg_edges_to;
DROP INDEX IF EXISTS idx_kg_edges_from;
DROP TABLE IF EXISTS kg_edges;
DROP INDEX IF EXISTS idx_kg_triples_proj;
DROP INDEX IF EXISTS idx_kg_triples_pred;
DROP INDEX IF EXISTS idx_kg_triples_obj;
DROP INDEX IF EXISTS idx_kg_triples_subj;
DROP TABLE IF EXISTS kg_triples;
"""
