"""
Ownership prediction feature engineering.

All features are available pre-contest (no leakage). Card position and odds
features are joined where available; missing values are left as NaN for
downstream imputation.
"""

import numpy as np
import pandas as pd
import sqlite3
import logging

from ..config import (
    FREEROLL_CONTEST_ID,
    SALARY_CAP,
    SALARY_TIER_BINS,
    SALARY_TIER_LABELS,
    CARD_SECTIONS,
)
from ..phase0_data.leakage_guard import assert_no_leakage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature column lists (for downstream model / leakage checks)
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = [
    "salary",
    "salary_rank_overall",
    "salary_pct",
    "card_position",
    "ownership_lag1",
    "ownership_lag2",
    "ownership_rolling_3_mean",
    "historical_avg_ownership",
    "is_favorite",
    "consensus_ml_prob",
    "days_since_last_fight",
    "dfs_sample_size",
    "log_field_size",
    "contest_size",
    "entry_cost",
    "num_fighters_on_slate",
]

CATEGORICAL_FEATURES = [
    "card_section",
    "salary_tier",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ---------------------------------------------------------------------------
# SQL queries
# ---------------------------------------------------------------------------
_BASE_QUERY = f"""
SELECT
    p.player_id,
    p.contest_id,
    p.full_name,
    p.salary,
    p.ownership,
    c.date_id,
    c.contest_date,
    c.entry_cost,
    c.contest_size
FROM players p
JOIN contests c ON p.contest_id = c.contest_id
WHERE c.contest_id != {FREEROLL_CONTEST_ID}
  AND p.ownership IS NOT NULL
ORDER BY p.player_id, c.date_id
"""

_ODDS_QUERY = """
SELECT
    fo.date_id,
    fo.player_id,
    fo.implied_prob  AS consensus_ml_prob,
    fo.is_favorite,
    doq.is_high_quality AS odds_high_quality
FROM fighter_odds fo
LEFT JOIN date_odds_quality doq ON fo.date_id = doq.date_id
"""


# ---------------------------------------------------------------------------
# Helper: compute card position from salary rank within a date_id
# ---------------------------------------------------------------------------
def _derive_card_position(df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximate card position from salary rank within each event.

    Higher-salaried fighters are typically on the main card / main event.
    We rank fighters by salary (descending) within each date_id and assign
    a positional rank (1 = highest salary = likely main event).
    """
    df = df.copy()
    df["card_position"] = (
        df.groupby("date_id")["salary"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    # Map positional rank to card section labels
    df["card_section"] = df["card_position"].map(CARD_SECTIONS).fillna("early_prelim")
    return df


# ---------------------------------------------------------------------------
# Helper: salary features
# ---------------------------------------------------------------------------
def _add_salary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add salary-derived features."""
    df = df.copy()
    df["salary_pct"] = df["salary"] / SALARY_CAP
    df["salary_rank_overall"] = (
        df.groupby(["date_id", "contest_id"])["salary"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    df["salary_tier"] = pd.cut(
        df["salary"],
        bins=SALARY_TIER_BINS,
        labels=SALARY_TIER_LABELS,
        right=True,
        include_lowest=True,
    )
    return df


# ---------------------------------------------------------------------------
# Helper: lag / rolling ownership features (grouped by player_id)
# ---------------------------------------------------------------------------
def _add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ownership lag, rolling, and expanding-mean features.

    IMPORTANT: all calculations use shift(1) so we never peek at the
    current row's ownership value.
    """
    df = df.sort_values(["player_id", "date_id"]).copy()

    grp = df.groupby("player_id")["ownership"]

    df["ownership_lag1"] = grp.shift(1)
    df["ownership_lag2"] = grp.shift(2)

    # Rolling 3-event mean of *shifted* ownership (i.e., last 3 prior events)
    shifted_own = grp.shift(1)
    df["ownership_rolling_3_mean"] = (
        shifted_own.groupby(df["player_id"])
        .rolling(window=3, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Expanding mean of *shifted* ownership
    df["historical_avg_ownership"] = (
        shifted_own.groupby(df["player_id"])
        .expanding(min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df


# ---------------------------------------------------------------------------
# Helper: days since last fight
# ---------------------------------------------------------------------------
def _add_recency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add days_since_last_fight and dfs_sample_size."""
    df = df.sort_values(["player_id", "date_id"]).copy()
    df["contest_date_dt"] = pd.to_datetime(df["contest_date"])

    grp = df.groupby("player_id")

    # Days since previous DFS appearance (shift to avoid leakage)
    df["prev_date"] = grp["contest_date_dt"].shift(1)
    df["days_since_last_fight"] = (
        (df["contest_date_dt"] - df["prev_date"]).dt.days
    )

    # Cumulative count of prior appearances (excluding current row)
    df["dfs_sample_size"] = grp.cumcount()  # 0-indexed = count of prior rows

    df.drop(columns=["prev_date", "contest_date_dt"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# Helper: contest / slate-level features
# ---------------------------------------------------------------------------
def _add_contest_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add contest-level and slate-level features."""
    df = df.copy()
    df["log_field_size"] = np.log1p(df["contest_size"])

    # Number of unique fighters on the same slate (same date_id, same contest)
    fighters_per_slate = (
        df.groupby(["date_id", "contest_id"])["player_id"]
        .transform("nunique")
    )
    df["num_fighters_on_slate"] = fighters_per_slate
    return df


# ---------------------------------------------------------------------------
# Helper: merge odds features
# ---------------------------------------------------------------------------
def _merge_odds(df: pd.DataFrame, conn: sqlite3.Connection) -> pd.DataFrame:
    """
    Left-join fighter odds onto the main DataFrame.

    Odds features are gated by the quality tier: if the event has low-quality
    odds coverage (< 80%), we null out the odds columns so the model learns
    to ignore them when coverage is poor.
    """
    odds_df = pd.read_sql_query(_ODDS_QUERY, conn)

    df = df.merge(
        odds_df,
        on=["date_id", "player_id"],
        how="left",
    )

    # Gate by quality: null-out odds where coverage is low
    low_quality_mask = df["odds_high_quality"].fillna(0) == 0
    df.loc[low_quality_mask, "consensus_ml_prob"] = np.nan
    df.loc[low_quality_mask, "is_favorite"] = np.nan

    df.drop(columns=["odds_high_quality"], inplace=True)
    return df


def build_ownership_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """Build the full ownership-prediction feature matrix."""
    logger.info("Loading base player/contest data ...")
    df = pd.read_sql_query(_BASE_QUERY, conn)
    logger.info("Base rows: %d", len(df))
    df = df.drop_duplicates(subset=["player_id", "contest_id"])
    logger.info("Adding salary features ...")
    df = _add_salary_features(df)
    logger.info("Deriving card position from salary rank ...")
    df = _derive_card_position(df)
    logger.info("Computing lag / rolling ownership features ...")
    df = _add_lag_features(df)
    logger.info("Computing recency features ...")
    df = _add_recency_features(df)
    logger.info("Adding contest-level features ...")
    df = _add_contest_features(df)
    logger.info("Merging fighter odds ...")
    df = _merge_odds(df, conn)
    df = df.sort_values(["date_id", "player_id", "contest_id"]).reset_index(drop=True)
    assert_no_leakage(ALL_FEATURES)
    logger.info(
        "Feature matrix ready: %d rows x %d feature cols + target",
        len(df),
        len(ALL_FEATURES),
    )
    return df
