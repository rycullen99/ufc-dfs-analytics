"""
Per-event and per-fighter odds quality flags.

Extends date_odds_quality with fighter-level quality tiers so downstream
models can gate odds features based on data availability.
"""

import sqlite3

import pandas as pd

from ..config import FREEROLL_CONTEST_ID


def compute_fighter_odds_quality(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Compute per-fighter odds quality tier for every (date_id, player_id).

    Tiers:
      - 'high': fighter has moneyline + at least one prop (ITD or rounds)
      - 'medium': fighter has moneyline only
      - 'none': no odds data available
    """
    df = pd.read_sql(
        """
        SELECT DISTINCT
            c.date_id,
            p.player_id,
            fo.implied_prob,
            fo.itd_prob
        FROM players p
        JOIN contests c ON p.contest_id = c.contest_id
        LEFT JOIN fighter_odds fo ON fo.player_id = p.player_id AND fo.date_id = c.date_id
        WHERE c.contest_id != ?
        """,
        conn,
        params=(FREEROLL_CONTEST_ID,),
    )

    if df.empty:
        return pd.DataFrame()

    # Deduplicate
    df = df.drop_duplicates(subset=["date_id", "player_id"])

    def quality_tier(row):
        if pd.isna(row["implied_prob"]):
            return "none"
        if pd.notna(row["itd_prob"]):
            return "high"
        return "medium"

    df["odds_quality_tier"] = df.apply(quality_tier, axis=1)

    return df[["date_id", "player_id", "odds_quality_tier"]]


def write_fighter_odds_quality(conn: sqlite3.Connection) -> int:
    """
    Create fighter_odds_quality table with per-fighter quality tiers.

    Returns the number of rows written.
    """
    quality_df = compute_fighter_odds_quality(conn)
    if quality_df.empty:
        return 0

    conn.execute("DROP TABLE IF EXISTS fighter_odds_quality")
    conn.execute("""
        CREATE TABLE fighter_odds_quality (
            date_id INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            odds_quality_tier TEXT NOT NULL,
            PRIMARY KEY (date_id, player_id)
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_foq_tier ON fighter_odds_quality(odds_quality_tier)"
    )

    rows = quality_df.to_dict("records")
    conn.executemany(
        """
        INSERT OR REPLACE INTO fighter_odds_quality
        (date_id, player_id, odds_quality_tier)
        VALUES (:date_id, :player_id, :odds_quality_tier)
        """,
        rows,
    )
    conn.commit()

    return len(rows)
