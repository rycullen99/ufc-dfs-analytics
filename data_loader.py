"""
Data Loader — UFC DFS Backtesting
Translates resultsdb_ufc.db SQL queries into pandas DataFrames with derived columns.
"""

import sqlite3
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

from .config import (
    DB_PATH,
    SLATE_SIZE_BINS, SLATE_SIZE_LABELS,
    FIELD_SIZE_BINS, FIELD_SIZE_LABELS,
    ENTRY_FEE_BINS, ENTRY_FEE_LABELS,
    TOTAL_OWNERSHIP_BINS, TOTAL_OWNERSHIP_LABELS,
    SALARY_REMAINING_BINS, SALARY_REMAINING_LABELS,
    SALARY_SPREAD_BINS, SALARY_SPREAD_LABELS,
    FAVORITE_COUNT_BINS, FAVORITE_COUNT_LABELS,
    PROB_BAND_BINS, PROB_BAND_LABELS,
    PAYOUT_TOLERANCE,
    CONTEST_150MAX_PATTERNS,
)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Open a SQLite connection to the UFC results database."""
    return sqlite3.connect(db_path or DB_PATH)


# ---------------------------------------------------------------------------
# Contest loading & classification
# ---------------------------------------------------------------------------

def load_contests(db_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load all contests with derived classification columns.

    Key columns added:
        slate_size_label   : SHORT / STANDARD / FULL
        field_size_label   : small / medium / large / mega
        fee_label          : MICRO / LOW / MID / HIGH
        is_150max          : True for all mass-market GPPs (proven by dupe data)
        is_well_scraped    : True when payout totals match prizes within 5%
        fight_count        : parsed from fight_count column
    """
    query = """
        SELECT
            c.contest_id,
            c.contest_name,
            c.contest_date,
            c.entry_cost,
            c.contest_size,
            c.total_prizes,
            c.multi_entry_max,
            c.lineups_complete,
            c.fight_count,
            -- Payout completeness check: sum(payout * entries) vs advertised prize pool
            COALESCE(ps.scraped_prizes, 0) AS scraped_prizes
        FROM contests c
        LEFT JOIN (
            SELECT contest_id,
                   SUM(payout * user_count) AS scraped_prizes
            FROM lineups
            GROUP BY contest_id
        ) ps ON ps.contest_id = c.contest_id
        WHERE c.lineups_complete = 1
    """
    with get_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn)

    # --- Derived columns ---

    # Slate size from fight count
    df["slate_size_label"] = pd.cut(
        df["fight_count"],
        bins=SLATE_SIZE_BINS,
        labels=SLATE_SIZE_LABELS,
        right=True,
    )

    # Field size from contest entries
    df["field_size_label"] = pd.cut(
        df["contest_size"],
        bins=FIELD_SIZE_BINS,
        labels=FIELD_SIZE_LABELS,
        right=True,
    )

    # Entry fee tier
    df["fee_label"] = pd.cut(
        df["entry_cost"],
        bins=ENTRY_FEE_BINS,
        labels=ENTRY_FEE_LABELS,
        right=True,
    )

    # 150-max flag: all mass-market contests (NULL multi_entry_max)
    # The $555 Knockout series has explicit multi_entry_max values
    df["is_150max"] = df["multi_entry_max"].isna()

    # Well-scraped: payout totals within 5% of advertised prize pool
    df["payout_ratio"] = np.where(
        df["total_prizes"] > 0,
        df["scraped_prizes"] / df["total_prizes"],
        np.nan,
    )
    df["is_well_scraped"] = df["payout_ratio"].between(
        1 - PAYOUT_TOLERANCE, 1 + PAYOUT_TOLERANCE
    )

    return df


# ---------------------------------------------------------------------------
# Lineup loading
# ---------------------------------------------------------------------------

def load_lineups(
    db_path: Optional[Path] = None,
    well_scraped_only: bool = True,
    mass_market_only: bool = True,
) -> pd.DataFrame:
    """
    Load lineups joined to contest metadata with derived analytical columns.

    Args:
        well_scraped_only : Filter to contests where payouts balance within 5%.
                            Required for unbiased ROI. Default True.
        mass_market_only  : Filter to 150-max GPPs (exclude $555 Knockout series).
                            Default True.

    Returns:
        DataFrame with one row per unique lineup (use user_count for weighting).

    Key columns:
        roi                : payout / entry_cost (per-lineup)
        salary_remaining   : 50000 - total_salary
        salary_spread      : max_salary - min_salary (Stars & Scrubs signal)
        total_own_label    : binned total_ownership
        fav_count_label    : binned favorite_count
    """
    contests = load_contests(db_path)

    if well_scraped_only:
        contests = contests[contests["is_well_scraped"]]
    if mass_market_only:
        contests = contests[contests["is_150max"]]

    contest_ids = tuple(contests["contest_id"].tolist())
    if not contest_ids:
        return pd.DataFrame()

    # Placeholders for SQL IN clause
    placeholders = ",".join("?" * len(contest_ids))

    query = f"""
        SELECT
            l.id              AS lineup_id,
            l.contest_id,
            l.lineup_hash,
            l.points,
            l.total_salary,
            l.total_ownership,
            l.min_ownership,
            l.max_ownership,
            l.payout,
            l.lineup_percentile,
            l.favorite_count,
            l.user_count,
            l.lineup_count,
            l.odds_coverage,
            l.implied_prob_sum,
            l.own_prob_ratio,
            l.tossup_count,
            c.entry_cost,
            c.contest_size,
            c.fight_count,
            c.total_prizes
        FROM lineups l
        JOIN contests c ON c.contest_id = l.contest_id
        WHERE l.contest_id IN ({placeholders})
    """
    with get_connection(db_path) as conn:
        df = pd.read_sql_query(query, conn, params=list(contest_ids))

    # --- Derived columns ---

    df["roi"] = df["payout"] / df["entry_cost"].replace(0, np.nan)

    df["salary_remaining"] = 50_000 - df["total_salary"]

    # Binned dimensions for group-by analysis
    df["slate_size_label"] = pd.cut(
        df["fight_count"], bins=SLATE_SIZE_BINS, labels=SLATE_SIZE_LABELS, right=True
    )
    df["field_size_label"] = pd.cut(
        df["contest_size"], bins=FIELD_SIZE_BINS, labels=FIELD_SIZE_LABELS, right=True
    )
    df["fee_label"] = pd.cut(
        df["entry_cost"], bins=ENTRY_FEE_BINS, labels=ENTRY_FEE_LABELS, right=True
    )
    df["total_own_label"] = pd.cut(
        df["total_ownership"],
        bins=TOTAL_OWNERSHIP_BINS,
        labels=TOTAL_OWNERSHIP_LABELS,
        right=True,
    )
    df["salary_remaining_label"] = pd.cut(
        df["salary_remaining"],
        bins=SALARY_REMAINING_BINS,
        labels=SALARY_REMAINING_LABELS,
        right=True,
    )
    df["fav_count_label"] = pd.cut(
        df["favorite_count"],
        bins=FAVORITE_COUNT_BINS,
        labels=FAVORITE_COUNT_LABELS,
        right=False,
    )

    return df


# ---------------------------------------------------------------------------
# Fighter-level data
# ---------------------------------------------------------------------------

def load_players(
    db_path: Optional[Path] = None,
    contest_ids: Optional[list] = None,
) -> pd.DataFrame:
    """
    Load fighter-level data (salary, ownership, actual points per contest).

    Optionally filtered to a list of contest_ids.
    """
    where_clause = ""
    params: list = []
    if contest_ids:
        placeholders = ",".join("?" * len(contest_ids))
        where_clause = f"WHERE contest_id IN ({placeholders})"
        params = list(contest_ids)

    query = f"""
        SELECT player_id, contest_id, full_name, salary, ownership, actual_points
        FROM players
        {where_clause}
    """
    with get_connection(db_path) as conn:
        return pd.read_sql_query(query, conn, params=params)


def load_lineup_players(
    db_path: Optional[Path] = None,
    lineup_ids: Optional[list] = None,
) -> pd.DataFrame:
    """
    Load the lineup → player mapping table (12M rows total).
    Always filter by lineup_ids to avoid loading the full table.
    """
    if not lineup_ids:
        raise ValueError("lineup_ids required — loading all 12M rows would be too slow")

    placeholders = ",".join("?" * len(lineup_ids))
    query = f"""
        SELECT lineup_id, player_id
        FROM lineup_players
        WHERE lineup_id IN ({placeholders})
    """
    with get_connection(db_path) as conn:
        return pd.read_sql_query(query, conn, params=list(lineup_ids))


# ---------------------------------------------------------------------------
# ROI helper
# ---------------------------------------------------------------------------

def weighted_roi(df: pd.DataFrame) -> float:
    """
    Compute true ROI weighted by user_count (entries per unique lineup).

    IMPORTANT: Simple mean(roi) overstates performance because it treats
    a lineup entered once the same as one entered 150 times. We weight by
    user_count so mass-entered lineups have proportional influence.

    Formula: SUM(payout * user_count) / SUM(entry_cost * user_count)
    """
    total_payout = (df["payout"] * df["user_count"]).sum()
    total_cost = (df["entry_cost"] * df["user_count"]).sum()
    return total_payout / total_cost if total_cost > 0 else np.nan
