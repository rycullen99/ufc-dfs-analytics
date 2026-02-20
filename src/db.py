"""SQLite connection helpers for the UFC DFS analytics pipeline."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .config import DB_PATH


def get_connection(db_path: Path | None = None, readonly: bool = False) -> sqlite3.Connection:
    """Get a SQLite connection with WAL mode and foreign keys enabled."""
    path = db_path or DB_PATH
    if readonly:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    else:
        conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def db_connection(db_path: Path | None = None, readonly: bool = False):
    """Context manager that auto-commits on success and rolls back on error."""
    conn = get_connection(db_path, readonly=readonly)
    try:
        yield conn
        if not readonly:
            conn.commit()
    except Exception:
        if not readonly:
            conn.rollback()
        raise
    finally:
        conn.close()


def execute_migration(conn: sqlite3.Connection, sql_path: Path) -> None:
    """Execute a SQL migration file."""
    sql = sql_path.read_text()
    conn.executescript(sql)


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """Check if a table exists in the database."""
    row = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?",
        (table_name,),
    ).fetchone()
    return row[0] > 0


def row_count(conn: sqlite3.Connection, table_name: str) -> int:
    """Get the row count for a table."""
    return conn.execute(f"SELECT COUNT(*) FROM [{table_name}]").fetchone()[0]
