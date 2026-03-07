"""
Contest and lineup loader for ROI backtesting (Phase 6).
Builds on src/db.py — uses same connection, adds backtesting-specific queries.
"""

import numpy as np
import pandas as pd

from ..db import db_connection
from ..config import DB_PATH, FREEROLL_CONTEST_ID
from .regime import PAYOUT_TOLERANCE, add_regime_columns


def load_contests(well_scraped_only: bool = True) -> pd.DataFrame:
    """
    Load contests with payout completeness check and regime labels.

    well_scraped_only=True filters to contests where scraped payouts
    match the advertised prize pool within 5% — required for valid ROI.
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
            -- fight_count not stored; derive from players (2 fighters per fight)
            fc.fight_count,
            COALESCE(ps.scraped_prizes, 0) AS scraped_prizes
        FROM contests c
        LEFT JOIN (
            SELECT contest_id, SUM(payout * user_count) AS scraped_prizes
            FROM lineups
            GROUP BY contest_id
        ) ps ON ps.contest_id = c.contest_id
        LEFT JOIN (
            SELECT contest_id, COUNT(DISTINCT player_id) / 2 AS fight_count
            FROM players
            GROUP BY contest_id
        ) fc ON fc.contest_id = c.contest_id
        WHERE c.lineups_complete = 1
          AND c.contest_id != ?
    """
    with db_connection(readonly=True) as conn:
        df = pd.read_sql_query(query, conn, params=[FREEROLL_CONTEST_ID])

    # All complete contests are 150-max (NULL multi_entry_max)
    df["is_150max"] = df["multi_entry_max"].isna()

    # Payout completeness ratio
    df["payout_ratio"] = np.where(
        df["total_prizes"] > 0,
        df["scraped_prizes"] / df["total_prizes"],
        np.nan,
    )
    df["is_well_scraped"] = df["payout_ratio"].between(
        1 - PAYOUT_TOLERANCE, 1 + PAYOUT_TOLERANCE
    )

    if well_scraped_only:
        df = df[df["is_well_scraped"]]

    return add_regime_columns(df)


def load_lineups(well_scraped_only: bool = True) -> pd.DataFrame:
    """
    Load all lineups joined to contest metadata, with regime columns.

    ROI Note: payout / entry_cost is per-lineup. For aggregate ROI, always
    use weighted_roi() which weights by user_count (entries per unique lineup).
    """
    contests = load_contests(well_scraped_only=well_scraped_only)
    contest_ids = contests["contest_id"].tolist()

    if not contest_ids:
        return pd.DataFrame()

    placeholders = ",".join("?" * len(contest_ids))
    query = f"""
        SELECT
            l.id              AS lineup_id,
            l.contest_id,
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
            fc.fight_count,
            c.total_prizes,
            c.contest_date
        FROM lineups l
        JOIN contests c ON c.contest_id = l.contest_id
        LEFT JOIN (
            SELECT contest_id, COUNT(DISTINCT player_id) / 2 AS fight_count
            FROM players
            GROUP BY contest_id
        ) fc ON fc.contest_id = c.contest_id
        WHERE l.contest_id IN ({placeholders})
    """
    with db_connection(readonly=True) as conn:
        df = pd.read_sql_query(query, conn, params=contest_ids)

    return add_regime_columns(df)


def weighted_roi(df: pd.DataFrame) -> float:
    """
    Weighted ROI: SUM(payout * user_count) / SUM(entry_cost * user_count).
    Simple mean(payout/entry_cost) overstates ROI — a lineup entered 150 times
    should count 150x more than one entered once.
    """
    total_payout = (df["payout"] * df["user_count"]).sum()
    total_cost = (df["entry_cost"] * df["user_count"]).sum()
    return total_payout / total_cost if total_cost > 0 else np.nan
