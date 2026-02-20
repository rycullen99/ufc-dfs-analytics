"""
Per-slate sharp exposure signals.

Computes how differently sharp users roster fighters compared to the
overall field, producing actionable signals for lineup construction.
"""

import numpy as np
import pandas as pd
import sqlite3
import logging
from typing import Optional

from ..config import FREEROLL_CONTEST_ID
from .user_skill import compute_user_skill

logger = logging.getLogger(__name__)


def compute_sharp_signals(
    conn: sqlite3.Connection,
    user_skill_df: Optional[pd.DataFrame] = None,
    min_sharp_users: int = 5,
) -> pd.DataFrame:
    """
    Compute sharp exposure signals for each fighter on each slate.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    user_skill_df : pd.DataFrame, optional
        Pre-computed user skill table. If None, computes from scratch.
    min_sharp_users : int
        Minimum sharp users in a contest to compute reliable signals.

    Returns
    -------
    pd.DataFrame with columns:
        date_id, player_id, full_name,
        field_exposure_pct, sharp_exposure_pct,
        sharp_vs_field_delta, sharp_confidence, n_sharp_users
    """
    if user_skill_df is None:
        user_skill_df = compute_user_skill(conn)

    sharp_usernames = set(
        user_skill_df.loc[user_skill_df["is_sharp"], "username"]
    )
    logger.info("Using %d sharp users for signal computation", len(sharp_usernames))

    # Get all lineups with fighter rosters per event
    query = f"""
    SELECT
        c.date_id,
        lp.player_id,
        p.full_name,
        lu.username,
        l.lineup_id,
        l.contest_id
    FROM lineup_players lp
    JOIN lineups l ON lp.lineup_id = l.lineup_id
    JOIN contests c ON l.contest_id = c.contest_id
    JOIN players p ON lp.player_id = p.player_id AND p.contest_id = c.contest_id
    JOIN lineup_usernames lu ON l.lineup_id = lu.lineup_id
    WHERE c.contest_id != {FREEROLL_CONTEST_ID}
    """

    logger.info("Loading lineup-player-user data (this may take a while) ...")
    df = pd.read_sql_query(query, conn)
    logger.info("Loaded %d rows", len(df))

    # Tag sharp users
    df["is_sharp"] = df["username"].isin(sharp_usernames)

    # Compute per (date_id, player_id) signals
    results = []

    for (date_id, player_id), group in df.groupby(["date_id", "player_id"]):
        full_name = group["full_name"].iloc[0]
        total_lineups = group["lineup_id"].nunique()
        sharp_lineups = group.loc[group["is_sharp"], "lineup_id"].nunique()
        n_sharp = group.loc[group["is_sharp"], "username"].nunique()

        field_exposure = total_lineups  # raw count
        sharp_exposure = sharp_lineups

        # Normalize to percentages within this event
        results.append({
            "date_id": date_id,
            "player_id": player_id,
            "full_name": full_name,
            "field_lineups": total_lineups,
            "sharp_lineups": sharp_lineups,
            "n_sharp_users": n_sharp,
        })

    signals = pd.DataFrame(results)

    if len(signals) == 0:
        return pd.DataFrame()

    # Normalize within each date_id
    date_totals = signals.groupby("date_id").agg(
        total_field=("field_lineups", "sum"),
        total_sharp=("sharp_lineups", "sum"),
    )

    signals = signals.merge(date_totals, on="date_id")

    signals["field_exposure_pct"] = (
        signals["field_lineups"] / signals["total_field"].clip(lower=1) * 100
    )
    signals["sharp_exposure_pct"] = (
        signals["sharp_lineups"] / signals["total_sharp"].clip(lower=1) * 100
    )
    signals["sharp_vs_field_delta"] = (
        signals["sharp_exposure_pct"] - signals["field_exposure_pct"]
    )

    # Confidence based on number of sharp users
    signals["sharp_confidence"] = np.clip(
        signals["n_sharp_users"] / min_sharp_users, 0, 1
    )

    # Clean up intermediate columns
    keep_cols = [
        "date_id", "player_id", "full_name",
        "field_exposure_pct", "sharp_exposure_pct",
        "sharp_vs_field_delta", "sharp_confidence", "n_sharp_users",
    ]

    result = signals[keep_cols].sort_values(
        ["date_id", "sharp_vs_field_delta"], ascending=[True, False]
    ).reset_index(drop=True)

    logger.info("Computed sharp signals for %d fighter-events", len(result))
    return result


def save_sharp_signals(conn: sqlite3.Connection, signals_df: pd.DataFrame) -> None:
    """Write sharp signals to the database."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sharp_signals (
            date_id INTEGER NOT NULL,
            player_id INTEGER NOT NULL,
            full_name TEXT,
            field_exposure_pct REAL,
            sharp_exposure_pct REAL,
            sharp_vs_field_delta REAL,
            sharp_confidence REAL,
            n_sharp_users INTEGER,
            PRIMARY KEY (date_id, player_id)
        )
    """)

    signals_df.to_sql("sharp_signals", conn, if_exists="replace", index=False)
    conn.commit()
    logger.info("Saved %d sharp signal rows to DB.", len(signals_df))
