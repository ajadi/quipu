"""Unit tests for DB auto-init, schema versioning, and migration idempotency.

Covers AC1: DB initializes automatically if absent; schema versioned.
"""

import sqlite3

import pytest

from quipu.storage import store
from quipu.db import get_connection
from quipu.db.migrate import init_db, MIGRATIONS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db_path(tmp_path):
    """Return a path to a DB file that does NOT yet exist."""
    return tmp_path / "init_test.db"


# ---------------------------------------------------------------------------
# AC1 — auto-init: file created, migration runs, user_version == latest migration
# ---------------------------------------------------------------------------

class TestAutoInit:
    def test_store_creates_file_on_absent_path(self, db_path):
        assert not db_path.exists()
        with store(str(db_path)) as s:
            pass
        assert db_path.exists()

    def test_user_version_matches_latest_migration_after_first_connect(self, db_path):
        with store(str(db_path)):
            pass
        conn = sqlite3.connect(str(db_path))
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        conn.close()
        assert version == MIGRATIONS[-1].VERSION

    def test_atoms_table_exists_after_init(self, db_path):
        with store(str(db_path)):
            pass
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='atoms'"
        ).fetchall()
        conn.close()
        assert len(rows) == 1

    def test_reopening_same_db_is_idempotent_no_error(self, db_path):
        with store(str(db_path)) as s1:
            s1.insert(content="before reopen")
        # Second open must not raise
        with store(str(db_path)) as s2:
            result = s2.insert(content="after reopen")
        assert result is not None

    def test_reopening_keeps_user_version_at_latest(self, db_path):
        with store(str(db_path)):
            pass
        # Open again
        with store(str(db_path)):
            pass
        conn = sqlite3.connect(str(db_path))
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        conn.close()
        assert version == MIGRATIONS[-1].VERSION

    def test_reopening_preserves_existing_data(self, db_path):
        with store(str(db_path)) as s:
            atom = s.insert(content="persistent")
            saved_id = atom.id
        with store(str(db_path)) as s2:
            fetched = s2.get(saved_id)
        assert fetched is not None
        assert fetched.content == "persistent"


# ---------------------------------------------------------------------------
# AC1 — migration mechanism: init_db is idempotent when version already set
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# TASK-023 — migration 0003 back-compat: pre-migration DB → session_id column + NULL
# ---------------------------------------------------------------------------

class TestMigration0003BackCompat:
    def test_db_at_v2_upgrades_to_v3_and_existing_atom_has_null_session_id(self, db_path):
        """Create DB at user_version=2, insert an atom, run init_db, assert
        user_version==3, session_id column exists, and the pre-existing atom
        has session_id IS NULL."""
        import sqlite3
        from quipu.db.migrate import init_db, MIGRATIONS

        conn = sqlite3.connect(str(db_path))
        # Run migrations 1 and 2 only
        m1, m2 = MIGRATIONS[0], MIGRATIONS[1]
        assert m1.VERSION == 1
        assert m2.VERSION == 2
        conn.executescript(m1.UP)
        conn.executescript(m2.UP)
        conn.execute("PRAGMA user_version = 2")
        conn.commit()

        # Insert a pre-migration atom
        conn.execute(
            "INSERT INTO atoms (id, type, scope, content, project_id) "
            "VALUES (?, ?, ?, ?, ?)",
            ("pre-001", "diary", "project", "pre-migration atom", "proj-P"),
        )
        conn.commit()
        conn.close()

        # Now run init_db — should apply migration 3
        conn2 = sqlite3.connect(str(db_path))
        init_db(conn2)
        version = conn2.execute("PRAGMA user_version").fetchone()[0]
        assert version >= 3  # at least v3 (session_id); later migrations also applied

        # session_id column exists
        cols = {
            row[1]
            for row in conn2.execute("PRAGMA table_info(atoms)").fetchall()
        }
        assert "session_id" in cols

        # Pre-existing atom has NULL session_id
        sid = conn2.execute(
            "SELECT session_id FROM atoms WHERE id = ?", ("pre-001",)
        ).fetchone()[0]
        assert sid is None

        # idx_atoms_session index exists
        indexes = {
            row[1]
            for row in conn2.execute(
                "SELECT type, name FROM sqlite_master WHERE type='index'"
            )
        }
        assert "idx_atoms_session" in indexes

        conn2.close()


# ---------------------------------------------------------------------------
# TASK-024 — migration 0005 back-compat: pre-migration DB → tags column + NULL
# ---------------------------------------------------------------------------

class TestMigration0005BackCompat:
    def test_db_at_v4_upgrades_to_v5_and_existing_atom_has_null_tags(self, db_path):
        """Create DB at user_version=4, insert an atom, run init_db, assert
        user_version==5, tags column exists, and the pre-existing atom
        has tags IS NULL."""
        import sqlite3
        from quipu.db.migrate import init_db, MIGRATIONS

        conn = sqlite3.connect(str(db_path))
        # Run migrations 1 through 4 (0001, 0002, 0003). No 0004 exists.
        m1, m2, m3 = MIGRATIONS[0], MIGRATIONS[1], MIGRATIONS[2]
        assert m1.VERSION == 1
        assert m2.VERSION == 2
        assert m3.VERSION == 3
        conn.executescript(m1.UP)
        conn.executescript(m2.UP)
        conn.executescript(m3.UP)
        conn.execute("PRAGMA user_version = 3")
        conn.commit()

        # Insert a pre-migration atom (no tags column yet)
        conn.execute(
            "INSERT INTO atoms (id, type, scope, content, project_id) "
            "VALUES (?, ?, ?, ?, ?)",
            ("pre-005", "diary", "project", "pre-0005 atom", "proj-P"),
        )
        conn.commit()
        conn.close()

        # Now run init_db — should apply migration 5
        conn2 = sqlite3.connect(str(db_path))
        init_db(conn2)
        version = conn2.execute("PRAGMA user_version").fetchone()[0]
        assert version >= 5

        # tags column exists
        cols = {
            row[1]
            for row in conn2.execute("PRAGMA table_info(atoms)").fetchall()
        }
        assert "tags" in cols

        # Pre-existing atom has NULL tags
        tags_val = conn2.execute(
            "SELECT tags FROM atoms WHERE id = ?", ("pre-005",)
        ).fetchone()[0]
        assert tags_val is None

        conn2.close()

    def test_db_at_v3_with_already_applied_v4_also_upgrades_to_v5(self, db_path):
        """If user_version was 4, init_db upgrades to 5 by applying m5."""
        import sqlite3
        from quipu.db.migrate import init_db, MIGRATIONS

        conn = sqlite3.connect(str(db_path))
        # MIGRATIONS has at least 5 entries (0001, 0002, 0003, 0004, 0005, ...)
        m1, m2, m3, m4, m5, *_ = MIGRATIONS
        conn.executescript(m1.UP)
        conn.executescript(m2.UP)
        conn.executescript(m3.UP)
        conn.executescript(m4.UP)
        # Do NOT run m5 — let init_db apply it during upgrade
        conn.execute("PRAGMA user_version = 4")
        conn.commit()

        conn.execute(
            "INSERT INTO atoms (id, type, scope, content, project_id) "
            "VALUES (?, ?, ?, ?, ?)",
            ("pre-v4", "diary", "project", "pre-v4 atom", "proj-P"),
        )
        conn.commit()
        conn.close()

        # init_db should see user_version=4 and apply migration 5 (and any later)
        conn2 = sqlite3.connect(str(db_path))
        init_db(conn2)
        version = conn2.execute("PRAGMA user_version").fetchone()[0]
        assert version >= 5

        tags_val = conn2.execute(
            "SELECT tags FROM atoms WHERE id = ?", ("pre-v4",)
        ).fetchone()[0]
        assert tags_val is None

        conn2.close()


# ---------------------------------------------------------------------------
# TASK-022 — migration 0006 back-compat: pre-migration DB → kg tables
# ---------------------------------------------------------------------------

class TestMigration0006BackCompat:
    def test_db_at_v5_upgrades_to_v6_and_kg_tables_exist(self, db_path):
        """Create DB at user_version=5, run init_db, assert v6 and kg tables exist."""
        import sqlite3
        from quipu.db.migrate import init_db, MIGRATIONS

        conn = sqlite3.connect(str(db_path))
        m1, m2, m3, m4, m5 = MIGRATIONS[:5]
        conn.executescript(m1.UP)
        conn.executescript(m2.UP)
        conn.executescript(m3.UP)
        conn.executescript(m4.UP)
        conn.executescript(m5.UP)
        conn.execute("PRAGMA user_version = 5")
        conn.commit()
        conn.close()

        conn2 = sqlite3.connect(str(db_path))
        init_db(conn2)
        version = conn2.execute("PRAGMA user_version").fetchone()[0]
        assert version == 6

        tables = {
            row[0] for row in conn2.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "kg_triples" in tables
        assert "kg_edges" in tables

        conn2.close()

    def test_db_at_v5_upgrade_preserves_existing_atoms(self, db_path):
        """Upgrade from v5 to v6 does not affect existing atoms."""
        import sqlite3
        from quipu.db.migrate import init_db, MIGRATIONS

        conn = sqlite3.connect(str(db_path))
        m1, m2, m3, m4, m5 = MIGRATIONS[:5]
        conn.executescript(m1.UP)
        conn.executescript(m2.UP)
        conn.executescript(m3.UP)
        conn.executescript(m4.UP)
        conn.executescript(m5.UP)
        conn.execute("PRAGMA user_version = 5")
        conn.execute(
            "INSERT INTO atoms (id, type, scope, content, project_id) "
            "VALUES (?, ?, ?, ?, ?)",
            ("pre-v5", "diary", "project", "pre-migration atom", "proj-P"),
        )
        conn.commit()
        conn.close()

        conn2 = sqlite3.connect(str(db_path))
        init_db(conn2)
        content = conn2.execute(
            "SELECT content FROM atoms WHERE id = ?", ("pre-v5",)
        ).fetchone()[0]
        assert content == "pre-migration atom"
        conn2.close()

    def test_down_migration_0006_drops_kg_tables(self, db_path):
        import sqlite3
        from quipu.db.migrate import init_db, MIGRATIONS
        from quipu.db.migrations import _migration_0006

        conn = sqlite3.connect(str(db_path))
        # Run all migrations
        for m in MIGRATIONS:
            conn.executescript(m.UP)
        conn.execute("PRAGMA user_version = 6")
        conn.commit()

        tables_before = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "kg_triples" in tables_before
        assert "kg_edges" in tables_before

        # Run down migration
        conn.executescript(_migration_0006.DOWN)
        tables_after = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "kg_triples" not in tables_after
        assert "kg_edges" not in tables_after

        conn.close()


class TestInitDbIdempotency:
    def test_init_db_idempotent_on_already_migrated_db(self, db_path):
        conn = get_connection(str(db_path))
        v_before = conn.execute("PRAGMA user_version").fetchone()[0]
        # Run init again on already-initialized connection
        init_db(conn)
        v_after = conn.execute("PRAGMA user_version").fetchone()[0]
        conn.close()
        assert v_before == v_after == MIGRATIONS[-1].VERSION

    def test_migration_list_is_non_empty(self):
        assert len(MIGRATIONS) >= 1

    def test_migration_0001_has_required_attributes(self):
        m = MIGRATIONS[0]
        assert hasattr(m, "VERSION")
        assert hasattr(m, "UP")
        assert hasattr(m, "DOWN")
        assert m.VERSION == 1

    def test_migration_0001_up_creates_atoms_table(self, db_path):
        conn = sqlite3.connect(str(db_path))
        conn.executescript(MIGRATIONS[0].UP)
        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "atoms" in tables

    def test_migration_0001_down_drops_atoms_table(self, db_path):
        conn = sqlite3.connect(str(db_path))
        conn.executescript(MIGRATIONS[0].UP)
        conn.executescript(MIGRATIONS[0].DOWN)
        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "atoms" not in tables

    def test_parent_directory_created_automatically(self, tmp_path):
        nested = tmp_path / "deep" / "nested" / "quipu.db"
        assert not nested.parent.exists()
        with store(str(nested)) as s:
            s.insert(content="nested dir test")
        assert nested.exists()


# ---------------------------------------------------------------------------
# AC1 — expected indexes created
# ---------------------------------------------------------------------------

class TestIndexesCreated:
    def test_composite_covering_index_exists(self, db_path):
        with store(str(db_path)):
            pass
        conn = sqlite3.connect(str(db_path))
        indexes = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        conn.close()
        assert "idx_atoms_project_valid" in indexes

    def test_project_id_index_exists(self, db_path):
        with store(str(db_path)):
            pass
        conn = sqlite3.connect(str(db_path))
        indexes = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        conn.close()
        assert "idx_atoms_project_id" in indexes
