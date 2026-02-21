"""
Export UFC DFS analytics data to Supabase via REST API.

Step 1: Run the DDL SQL in Supabase SQL Editor (printed by --schema-only)
Step 2: Run this script to insert data via REST API

Usage:
    python scripts/export_to_supabase.py --schema-only    # print DDL to run in SQL editor
    python scripts/export_to_supabase.py                   # insert data via REST API
    python scripts/export_to_supabase.py --dry-run         # show row counts only
"""

import os
import sys
import sqlite3
import argparse
import logging
import json

import pandas as pd
import requests

DB_PATH = os.path.expanduser("~/Desktop/resultsdb_ufc.db")

SUPABASE_URL = "https://wzuhwewlousjryuktbkw.supabase.co"
SUPABASE_SERVICE_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Ind6dWh3ZXdsb3VzanJ5dWt0Ymt3Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3MTYzMDA5MCwiZXhwIjoyMDg3MjA2MDkwfQ."
    "GaO5ytyk2X8vyT-plZpsS7N_uVWefSABS2LBELO7peA"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema DDL (run in Supabase SQL Editor)
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
-- UFC DFS Analytics Schema
-- Run this in the Supabase SQL Editor before inserting data.

DROP TABLE IF EXISTS fighter_odds_enriched CASCADE;
DROP TABLE IF EXISTS sharp_signals CASCADE;
DROP TABLE IF EXISTS ownership_predictions CASCADE;
DROP TABLE IF EXISTS fighter_event_snapshot CASCADE;
DROP TABLE IF EXISTS contest_dim CASCADE;
DROP TABLE IF EXISTS event_dim CASCADE;
DROP TABLE IF EXISTS fighter_dim CASCADE;

CREATE TABLE fighter_dim (
    canonical_id INTEGER PRIMARY KEY,
    canonical_name TEXT NOT NULL
);

CREATE TABLE event_dim (
    date_id INTEGER PRIMARY KEY,
    event_date DATE NOT NULL,
    n_contests INTEGER,
    n_fighters INTEGER
);

CREATE TABLE contest_dim (
    contest_id INTEGER PRIMARY KEY,
    date_id INTEGER REFERENCES event_dim(date_id),
    entry_cost REAL,
    contest_size INTEGER,
    multi_entry_max INTEGER,
    contest_type TEXT,
    cash_line INTEGER
);

CREATE TABLE fighter_event_snapshot (
    id SERIAL PRIMARY KEY,
    date_id INTEGER REFERENCES event_dim(date_id),
    canonical_id INTEGER REFERENCES fighter_dim(canonical_id),
    player_id INTEGER,
    full_name TEXT,
    salary INTEGER,
    ownership REAL,
    actual_points REAL,
    card_section TEXT,
    salary_tier TEXT
);

CREATE TABLE sharp_signals (
    id SERIAL PRIMARY KEY,
    date_id INTEGER,
    canonical_id INTEGER REFERENCES fighter_dim(canonical_id),
    player_id INTEGER,
    full_name TEXT,
    field_exposure_pct REAL,
    sharp_exposure_pct REAL,
    sharp_vs_field_delta REAL,
    sharp_confidence REAL,
    n_sharp_users INTEGER
);

CREATE TABLE fighter_odds_enriched (
    id SERIAL PRIMARY KEY,
    date_id INTEGER,
    canonical_id INTEGER REFERENCES fighter_dim(canonical_id),
    odds_name TEXT,
    player_id INTEGER,
    dfs_name TEXT,
    salary INTEGER,
    actual_ownership REAL,
    actual_points REAL,
    open_prob REAL,
    close_prob REAL,
    line_move REAL,
    is_favorite INTEGER,
    open_n_books INTEGER,
    close_n_books INTEGER,
    opponent_odds_name TEXT,
    opponent_canonical_id INTEGER
);

-- Indexes
CREATE INDEX idx_fes_date ON fighter_event_snapshot(date_id);
CREATE INDEX idx_fes_canonical ON fighter_event_snapshot(canonical_id);
CREATE INDEX idx_ss_date ON sharp_signals(date_id);
CREATE INDEX idx_ss_canonical ON sharp_signals(canonical_id);
CREATE INDEX idx_foe_date ON fighter_odds_enriched(date_id);
CREATE INDEX idx_foe_canonical ON fighter_odds_enriched(canonical_id);
CREATE INDEX idx_foe_favorite ON fighter_odds_enriched(is_favorite);
CREATE INDEX idx_contest_date ON contest_dim(date_id);

-- Disable RLS for service_role access
ALTER TABLE fighter_dim ENABLE ROW LEVEL SECURITY;
ALTER TABLE event_dim ENABLE ROW LEVEL SECURITY;
ALTER TABLE contest_dim ENABLE ROW LEVEL SECURITY;
ALTER TABLE fighter_event_snapshot ENABLE ROW LEVEL SECURITY;
ALTER TABLE sharp_signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE fighter_odds_enriched ENABLE ROW LEVEL SECURITY;

-- Allow service_role full access
CREATE POLICY "service_role_all" ON fighter_dim FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "service_role_all" ON event_dim FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "service_role_all" ON contest_dim FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "service_role_all" ON fighter_event_snapshot FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "service_role_all" ON sharp_signals FOR ALL USING (true) WITH CHECK (true);
CREATE POLICY "service_role_all" ON fighter_odds_enriched FOR ALL USING (true) WITH CHECK (true);
"""


# ---------------------------------------------------------------------------
# Data extraction from SQLite
# ---------------------------------------------------------------------------


def extract_fighter_dim(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT canonical_id, canonical_name FROM canonical_fighter ORDER BY canonical_id",
        conn,
    )


def extract_event_dim(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("""
        SELECT
            date_id,
            DATE(
                SUBSTR(CAST(date_id AS TEXT), 1, 4) || '-' ||
                SUBSTR(CAST(date_id AS TEXT), 5, 2) || '-' ||
                SUBSTR(CAST(date_id AS TEXT), 7, 2)
            ) AS event_date,
            COUNT(DISTINCT contest_id) AS n_contests,
            (SELECT COUNT(DISTINCT p.player_id)
             FROM players p
             JOIN contests c2 ON p.contest_id = c2.contest_id
             WHERE c2.date_id = c.date_id) AS n_fighters
        FROM contests c
        GROUP BY date_id
        ORDER BY date_id
    """, conn)


def extract_contest_dim(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("""
        SELECT
            contest_id, date_id, entry_cost, contest_size,
            multi_entry_max,
            CASE
                WHEN multi_entry_max = 1 THEN 'Single Entry'
                WHEN multi_entry_max = 3 THEN '3-Max'
                WHEN multi_entry_max = 20 THEN '20-Max'
                WHEN multi_entry_max = 150 THEN '150-Max'
                WHEN multi_entry_max BETWEEN 2 AND 19 THEN 'Limited'
                ELSE 'MME'
            END AS contest_type,
            cash_line
        FROM contests
        WHERE contest_id != 142690805
        ORDER BY date_id, contest_id
    """, conn)


def extract_fighter_event_snapshot(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("""
        SELECT DISTINCT
            c.date_id,
            cf.canonical_id,
            p.player_id,
            p.full_name,
            p.salary,
            p.ownership,
            p.actual_points,
            fcp.card_section,
            CASE
                WHEN p.salary >= 10000 THEN '$10K+'
                WHEN p.salary >= 9000 THEN '$9K-$10K'
                WHEN p.salary >= 8000 THEN '$8K-$9K'
                WHEN p.salary >= 7000 THEN '$7K-$8K'
                WHEN p.salary >= 6000 THEN '$6K-$7K'
                ELSE 'Under $6K'
            END AS salary_tier
        FROM players p
        JOIN contests c ON p.contest_id = c.contest_id
        LEFT JOIN canonical_fighter cf ON cf.canonical_name = p.full_name
        LEFT JOIN fighter_card_position fcp ON fcp.date_id = c.date_id AND fcp.player_id = p.player_id
        WHERE c.contest_id != 142690805
        GROUP BY c.date_id, p.full_name
        ORDER BY c.date_id, p.salary DESC
    """, conn)


def extract_sharp_signals(conn: sqlite3.Connection) -> pd.DataFrame:
    try:
        return pd.read_sql_query("""
            SELECT
                ss.date_id,
                cf.canonical_id,
                ss.player_id,
                ss.full_name,
                ss.field_exposure_pct,
                ss.sharp_exposure_pct,
                ss.sharp_vs_field_delta,
                ss.sharp_confidence,
                ss.n_sharp_users
            FROM sharp_signals ss
            LEFT JOIN canonical_fighter cf ON cf.canonical_name = ss.full_name
            ORDER BY ss.date_id, ss.sharp_vs_field_delta DESC
        """, conn)
    except Exception as e:
        log.warning("sharp_signals not found: %s", e)
        return pd.DataFrame()


def extract_fighter_odds(conn: sqlite3.Connection) -> pd.DataFrame:
    try:
        return pd.read_sql_query("""
            SELECT
                date_id, canonical_id, odds_name, player_id, dfs_name,
                salary, actual_ownership, actual_points,
                open_prob, close_prob, line_move, is_favorite,
                open_n_books, close_n_books,
                opponent_odds_name, opponent_canonical_id
            FROM fighter_odds_enriched
            ORDER BY date_id, salary DESC
        """, conn)
    except Exception as e:
        log.warning("fighter_odds_enriched not found: %s", e)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# REST API upload
# ---------------------------------------------------------------------------


def upload_via_rest(table_name: str, df: pd.DataFrame, batch_size: int = 200):
    """Insert rows via PostgREST API using service_role key."""
    if df.empty:
        log.info("  %s: empty, skipping", table_name)
        return

    # Clean NaN → None, then to JSON-safe dicts
    df = df.where(pd.notnull(df), None)

    # Drop the 'id' column if it exists (SERIAL, auto-generated)
    if "id" in df.columns:
        df = df.drop(columns=["id"])

    records = df.to_dict("records")

    # Integer columns that Postgres expects as int, not float
    int_columns = {
        "canonical_id", "date_id", "contest_id", "player_id", "salary",
        "contest_size", "multi_entry_max", "cash_line", "n_contests",
        "n_fighters", "n_sharp_users", "is_favorite", "open_n_books",
        "close_n_books", "opponent_canonical_id",
    }

    # Clean up values for JSON serialization
    for record in records:
        for k, v in record.items():
            if v is None:
                continue
            elif isinstance(v, float) and (v != v):  # NaN check
                record[k] = None
            elif hasattr(v, 'item'):  # numpy types
                val = v.item()
                record[k] = int(val) if k in int_columns and val == int(val) else val
            elif isinstance(v, float) and k in int_columns and v == int(v):
                record[k] = int(v)

    url = f"{SUPABASE_URL}/rest/v1/{table_name}"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=ignore-duplicates",
    }

    total_inserted = 0
    errors = 0

    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        resp = requests.post(url, headers=headers, json=batch, timeout=30)

        if resp.status_code in (200, 201):
            total_inserted += len(batch)
        else:
            errors += 1
            if errors <= 3:
                log.warning("  %s batch %d failed (%d): %s",
                            table_name, i // batch_size, resp.status_code,
                            resp.text[:200])

    log.info("  %s: inserted %d rows (%d batch errors)", table_name, total_inserted, errors)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Export to Supabase via REST API")
    parser.add_argument("--schema-only", action="store_true",
                        help="Print DDL SQL to run in SQL Editor")
    parser.add_argument("--dry-run", action="store_true",
                        help="Extract data but don't upload")
    args = parser.parse_args()

    if args.schema_only:
        print(SCHEMA_SQL)
        print("\n-- Copy and paste the above into the Supabase SQL Editor and click Run.")
        return

    # Extract from SQLite
    sqlite_conn = sqlite3.connect(DB_PATH)
    log.info("Connected to SQLite: %s", DB_PATH)

    log.info("=" * 60)
    log.info("Extracting data from SQLite")
    log.info("=" * 60)

    tables = {}
    tables["fighter_dim"] = extract_fighter_dim(sqlite_conn)
    log.info("  fighter_dim: %d rows", len(tables["fighter_dim"]))

    tables["event_dim"] = extract_event_dim(sqlite_conn)
    log.info("  event_dim: %d rows", len(tables["event_dim"]))

    tables["contest_dim"] = extract_contest_dim(sqlite_conn)
    log.info("  contest_dim: %d rows", len(tables["contest_dim"]))

    tables["fighter_event_snapshot"] = extract_fighter_event_snapshot(sqlite_conn)
    log.info("  fighter_event_snapshot: %d rows", len(tables["fighter_event_snapshot"]))

    tables["sharp_signals"] = extract_sharp_signals(sqlite_conn)
    log.info("  sharp_signals: %d rows", len(tables["sharp_signals"]))

    tables["fighter_odds_enriched"] = extract_fighter_odds(sqlite_conn)
    log.info("  fighter_odds_enriched: %d rows", len(tables["fighter_odds_enriched"]))

    sqlite_conn.close()

    if args.dry_run:
        total = sum(len(df) for df in tables.values())
        log.info("DRY RUN — would upload %d total rows across %d tables", total, len(tables))
        return

    # Verify REST API is reachable
    test_resp = requests.get(
        f"{SUPABASE_URL}/rest/v1/",
        headers={"apikey": SUPABASE_SERVICE_KEY},
        timeout=10,
    )
    if test_resp.status_code != 200:
        log.error("Supabase REST API not reachable (status %d). Run --schema-only first?",
                  test_resp.status_code)
        return

    log.info("=" * 60)
    log.info("Uploading data via REST API")
    log.info("=" * 60)

    # Order matters for foreign keys
    upload_order = [
        "fighter_dim",
        "event_dim",
        "contest_dim",
        "fighter_event_snapshot",
        "sharp_signals",
        "fighter_odds_enriched",
    ]

    for table_name in upload_order:
        if table_name in tables and not tables[table_name].empty:
            upload_via_rest(table_name, tables[table_name])

    log.info("=" * 60)
    log.info("DONE — Data exported to Supabase")
    log.info("Dashboard: https://supabase.com/dashboard/project/wzuhwewlousjryuktbkw")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
