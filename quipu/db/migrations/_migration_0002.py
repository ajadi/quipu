"""Migration 0002 — oplog_entries table, sync_cursors table, indexes."""

VERSION: int = 2

UP: str = """
-- ---------------------------------------------------------------------------
-- oplog_entries: append-only log of upsert/invalidate operations.
--
-- SINGLE TABLE for both locally-produced and remotely-pulled entries.
-- Rationale: idempotent apply requires UNIQUE(client_id, sequence_no) to span
-- the whole log; two tables would need a cross-table JOIN on every pull.  A
-- `source` column ('local'|'remote') covers the push-path filter
-- ("entries I haven't pushed yet") without a separate table.
--
-- COLUMN CLASSIFICATION
--   hub-visible  — columns whose values are included in the JSON array sent to
--                  the hub on push (zero-knowledge hub sees plaintext routing
--                  metadata but never plaintext content or real project_id).
--   local-only   — never transmitted; exist only on the client.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS oplog_entries (
    -- Synthetic surrogate for local ordering; NEVER sent to the hub.
    -- local-only.
    rowid           INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Opaque content-addressed identifier for this entry.
    -- SHA-256(client_id || ':' || sequence_no) hex-64, computed client-side.
    -- hub-visible: used by hub as a dedup / idempotency key on ingest.
    entry_id        TEXT    NOT NULL,

    -- Originating client's stable identifier (UUID or SHA-256 of device key).
    -- hub-visible: routing metadata; hub uses this to scatter/gather per peer.
    client_id       TEXT    NOT NULL,

    -- Monotonically increasing counter within this client's namespace.
    -- Starts at 1, gaps not allowed (enforced by application layer).
    -- hub-visible: ordering metadata used by pull cursor arithmetic.
    sequence_no     INTEGER NOT NULL,

    -- Operation kind.  Plaintext because the hub needs to apply merge policy
    -- (last-write-wins / invalidate-wins) without decryption.
    -- Values: 'upsert' | 'invalidate'
    -- hub-visible.
    op              TEXT    NOT NULL,

    -- The atom id this operation targets.  Plaintext so the hub can dedup
    -- same-record operations in merge without decryption.
    -- hub-visible.
    record_id       TEXT    NOT NULL,

    -- Blinded project id: HMAC-SHA256(project_id, sub-key) → 64-hex.
    -- This is the hub's partition key.  Real project_id NEVER stored in any
    -- column of this table.
    -- hub-visible: hub routes push/pull by this value.
    blinded_project_id  TEXT    NOT NULL,

    -- ISO-8601 UTC timestamp of the write event on the originating client.
    -- Plaintext: the merge layer (last-write-wins) compares timestamps across
    -- entries without decryption.
    -- hub-visible.
    ts              TEXT    NOT NULL,

    -- Encrypted payload BLOB produced by quipu/crypto encrypt_record().
    -- Layout: |1B version|1B key_version|12B nonce|ciphertext+16B GCM tag|
    -- AAD = blinded_project_id.encode() (cross-project replay protection per
    -- TASK-011 handoff requirement — MUST be passed on encrypt AND decrypt).
    -- local-only: hub stores it opaque; only the originating client and peers
    -- with the project key can decrypt.
    payload         BLOB    NOT NULL,

    -- 'local'  — entry was produced by this client (push candidate).
    -- 'remote' — entry was pulled from the hub and applied to this replica.
    -- local-only: never sent to hub.
    source          TEXT    NOT NULL DEFAULT 'local',

    -- 0 = not yet pushed to hub; 1 = confirmed pushed.
    -- local-only: tracks push progress for 'local' entries; always 0 for
    -- 'remote' entries (they came FROM the hub).
    pushed          INTEGER NOT NULL DEFAULT 0,

    -- Constraints
    CONSTRAINT oplog_entries_op_ck CHECK (op IN ('upsert', 'invalidate')),
    CONSTRAINT oplog_entries_source_ck CHECK (source IN ('local', 'remote')),
    CONSTRAINT oplog_entries_pushed_ck CHECK (pushed IN (0, 1)),

    -- THE IDEMPOTENCY ANCHOR.
    -- (client_id, sequence_no) is globally unique: each client issues a
    -- monotonic sequence and never reuses a number.  Applying an entry twice
    -- (replay, re-pull, re-push) hits this constraint and is silently ignored
    -- (INSERT OR IGNORE).  This is the primary dedup mechanism for both push
    -- idempotency and pull idempotency.
    CONSTRAINT oplog_entries_dedup UNIQUE (client_id, sequence_no),

    -- entry_id uniqueness: content-addressed; duplicate content = same entry.
    CONSTRAINT oplog_entries_entry_id_uq UNIQUE (entry_id)
);

-- Index: push query — "local entries not yet pushed, ordered for a project"
--   SELECT * FROM oplog_entries
--   WHERE blinded_project_id = ? AND source = 'local' AND pushed = 0
--   ORDER BY sequence_no
--   AND client_id = ?    (this client only)
CREATE INDEX IF NOT EXISTS idx_oplog_push
    ON oplog_entries (blinded_project_id, client_id, pushed, sequence_no);

-- Index: pull/merge apply — "all entries for a project, ordered by (client, seq)"
--   SELECT * FROM oplog_entries
--   WHERE blinded_project_id = ?
--   ORDER BY client_id, sequence_no
CREATE INDEX IF NOT EXISTS idx_oplog_pull
    ON oplog_entries (blinded_project_id, client_id, sequence_no);

-- Index: dedup lookup — fast EXISTS check on (client_id, sequence_no) before
-- INSERT OR IGNORE; also used by merge to find latest entry per record_id.
CREATE INDEX IF NOT EXISTS idx_oplog_record
    ON oplog_entries (blinded_project_id, record_id, ts DESC);

-- Index: ts ordering — merge needs last-write-wins by timestamp per record_id
CREATE INDEX IF NOT EXISTS idx_oplog_ts
    ON oplog_entries (blinded_project_id, record_id, ts DESC, op);

-- ---------------------------------------------------------------------------
-- sync_cursors: persists push/pull progress per (blinded_project_id, peer_id).
--
-- Push cursor: last sequence_no this client has confirmed pushed for this
--   blinded_project_id.  Push sends entries WHERE sequence_no > last_pushed_seq.
--
-- Pull cursor: opaque cursor token returned by the hub after each pull
--   (e.g. a server-side offset or the entry_id of the last received entry).
--   Stored as TEXT to be hub-implementation-agnostic.
--
-- peer_id: identifies the counterpart.  For the push cursor this is the
--   local client_id (one row per local client per project).  For pull cursors
--   this is the remote client_id whose entries were fetched (one row per
--   remote peer per project so incremental pull per peer is possible).
--   A single `direction` column disambiguates ('push'|'pull').
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sync_cursors (
    -- local-only: surrogate pk
    rowid               INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Partition key matching oplog_entries.blinded_project_id.
    -- local-only (the blinded value is hub-visible in oplog_entries;
    -- the cursor table itself is never transmitted).
    blinded_project_id  TEXT    NOT NULL,

    -- 'push': last_seq is the highest sequence_no confirmed pushed for
    --         this client's own outbound log.
    -- 'pull': last_seq / last_cursor is the progress marker for entries
    --         received from peer_id.
    direction           TEXT    NOT NULL,

    -- For direction='push': this client's own client_id.
    -- For direction='pull': the remote peer's client_id.
    peer_id             TEXT    NOT NULL,

    -- Last sequence_no successfully pushed (direction='push') or
    -- last sequence_no received from peer_id (direction='pull').
    -- NULL = never pushed / never pulled from this peer.
    last_seq            INTEGER,

    -- Opaque hub cursor string for incremental pull (direction='pull' only).
    -- May be a server-side offset, entry_id, or timestamp — hub-defined.
    -- NULL before first pull.
    last_cursor         TEXT,

    -- ISO-8601 UTC timestamp of when this cursor was last updated.
    -- local-only.
    updated_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),

    CONSTRAINT sync_cursors_direction_ck CHECK (direction IN ('push', 'pull')),

    -- One cursor row per (project, direction, peer).
    CONSTRAINT sync_cursors_pk UNIQUE (blinded_project_id, direction, peer_id)
);

-- Index: cursor lookup — "get push cursor for this project+client"
--   SELECT last_seq FROM sync_cursors
--   WHERE blinded_project_id = ? AND direction = 'push' AND peer_id = ?
CREATE INDEX IF NOT EXISTS idx_cursors_lookup
    ON sync_cursors (blinded_project_id, direction, peer_id);
"""

DOWN: str = """
DROP INDEX IF EXISTS idx_cursors_lookup;
DROP TABLE IF EXISTS sync_cursors;
DROP INDEX IF EXISTS idx_oplog_ts;
DROP INDEX IF EXISTS idx_oplog_record;
DROP INDEX IF EXISTS idx_oplog_pull;
DROP INDEX IF EXISTS idx_oplog_push;
DROP TABLE IF EXISTS oplog_entries;
"""
