"""Export curated analytical tables to Supabase."""

import pandas as pd
import sqlite3
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def build_fighter_dim(conn: sqlite3.Connection) -> pd.DataFrame:
    """Build the fighter dimension table from canonical fighters."""
    query = """
    SELECT
        canonical_id,
        canonical_name,
        weight_class
    FROM canonical_fighter
    """
    try:
        return pd.read_sql_query(query, conn)
    except Exception:
        logger.warning("canonical_fighter table not found; building from players")
        query = """
        SELECT DISTINCT
            player_id AS canonical_id,
            full_name AS canonical_name,
            NULL AS weight_class
        FROM players
        """
        return pd.read_sql_query(query, conn)


def build_event_dim(conn: sqlite3.Connection) -> pd.DataFrame:
    """Build the event dimension table."""
    query = """
    SELECT
        date_id,
        MIN(contest_date) AS event_date,
        COUNT(DISTINCT contest_id) AS n_contests
    FROM contests
    GROUP BY date_id
    ORDER BY date_id
    """
    return pd.read_sql_query(query, conn)


def build_contest_dim(conn: sqlite3.Connection) -> pd.DataFrame:
    """Build the contest dimension table."""
    query = """
    SELECT
        contest_id,
        date_id,
        contest_name,
        entry_cost,
        contest_size,
        multi_entry_max,
        cash_line
    FROM contests
    """
    return pd.read_sql_query(query, conn)


def build_fighter_event_snapshot(conn: sqlite3.Connection) -> pd.DataFrame:
    """Build the fighter-event snapshot fact table."""
    query = """
    SELECT
        p.player_id,
        c.date_id,
        p.salary,
        p.ownership AS ownership_actual,
        p.actual_points,
        fo.implied_prob
    FROM players p
    JOIN contests c ON p.contest_id = c.contest_id
    LEFT JOIN fighter_odds fo ON fo.player_id = p.player_id AND fo.date_id = c.date_id
    GROUP BY p.player_id, c.date_id
    """
    return pd.read_sql_query(query, conn)


def export_to_supabase(
    conn: sqlite3.Connection,
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
) -> dict:
    """
    Export curated tables to Supabase.

    Parameters
    ----------
    conn : sqlite3.Connection
        Source SQLite database connection.
    supabase_url : str, optional
        Supabase project URL. If None, reads from environment.
    supabase_key : str, optional
        Supabase service role key. If None, reads from environment.

    Returns
    -------
    dict
        Row counts for each exported table.
    """
    import os

    url = supabase_url or os.environ.get("SUPABASE_URL")
    key = supabase_key or os.environ.get("SUPABASE_KEY")

    if not url or not key:
        logger.warning("Supabase credentials not configured. Skipping export.")
        return {}

    try:
        from supabase import create_client
    except ImportError:
        logger.error("supabase package not installed. Run: uv pip install supabase")
        return {}

    client = create_client(url, key)

    tables = {
        "fighter_dim": build_fighter_dim(conn),
        "event_dim": build_event_dim(conn),
        "contest_dim": build_contest_dim(conn),
        "fighter_event_snapshot": build_fighter_event_snapshot(conn),
    }

    # Add predictions and signals if available
    for table_name in ["ownership_predictions", "sharp_signals"]:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
            if len(df) > 0:
                tables[table_name] = df
        except Exception:
            logger.info("Table %s not found in SQLite, skipping", table_name)

    counts = {}
    for name, df in tables.items():
        logger.info("Exporting %s: %d rows", name, len(df))
        records = df.to_dict(orient="records")

        # Batch upsert in chunks of 500
        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            client.table(name).upsert(batch).execute()

        counts[name] = len(df)

    logger.info("Export complete: %s", counts)
    return counts
