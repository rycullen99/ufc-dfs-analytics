"""
Discover — UFC DFS Backtesting
ROI analysis across ownership, salary, odds, and fighter dimensions.
Each function takes a lineups DataFrame and returns a summary table.
"""

import pandas as pd
import numpy as np

from .data_loader import weighted_roi


# ---------------------------------------------------------------------------
# Core aggregation helper
# ---------------------------------------------------------------------------

def _roi_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Group by a dimension column and compute weighted ROI per bucket.

    Returns columns: [group_col, n_lineups, n_entries, weighted_roi, top1pct_rate, cash_rate]
    """
    results = []
    for label, group in df.groupby(group_col, observed=True):
        n_lineups = len(group)
        n_entries = group["user_count"].sum()
        roi = weighted_roi(group)
        top1 = (group["lineup_percentile"] <= 1).mean()
        cash = (group["payout"] > 0).mean()
        results.append({
            group_col: label,
            "n_lineups": n_lineups,
            "n_entries": int(n_entries),
            "weighted_roi": round(roi, 4),
            "top1pct_rate": round(top1, 4),
            "cash_rate": round(cash, 4),
        })
    return pd.DataFrame(results).sort_values(group_col)


# ---------------------------------------------------------------------------
# Ownership analysis
# ---------------------------------------------------------------------------

def analyze_ownership(df: pd.DataFrame) -> pd.DataFrame:
    """ROI by total lineup ownership band."""
    return _roi_table(df, "total_own_label")


def analyze_ownership_by_regime(
    df: pd.DataFrame,
    regime_col: str = "slate_size_label",
) -> pd.DataFrame:
    """
    ROI by ownership band, broken out by a regime dimension.
    Reveals reversals — e.g. chalk works on SHORT slates, fails on FULL slates.
    """
    results = []
    for regime, regime_df in df.groupby(regime_col, observed=True):
        tbl = _roi_table(regime_df, "total_own_label")
        tbl.insert(0, regime_col, regime)
        results.append(tbl)
    return pd.concat(results, ignore_index=True)


def analyze_ownership_composition(df: pd.DataFrame) -> dict:
    """
    ROI by min_ownership (most contrarian fighter) and max_ownership (chalk anchor).
    Returns dict with 'min_own' and 'max_own' DataFrames.
    """
    min_bins = [0, 3, 8, 15, 100]
    min_labels = ["<3%", "3-8%", "8-15%", "15%+"]
    max_bins = [0, 25, 35, 50, 100]
    max_labels = ["<25%", "25-35%", "35-50%", "50%+"]

    df = df.copy()
    df["min_own_label"] = pd.cut(df["min_ownership"], bins=min_bins, labels=min_labels, right=True)
    df["max_own_label"] = pd.cut(df["max_ownership"], bins=max_bins, labels=max_labels, right=True)

    return {
        "min_own": _roi_table(df, "min_own_label"),
        "max_own": _roi_table(df, "max_own_label"),
    }


# ---------------------------------------------------------------------------
# Salary analysis
# ---------------------------------------------------------------------------

def analyze_salary_remaining(df: pd.DataFrame) -> pd.DataFrame:
    """ROI by salary remaining (distance from $50K cap)."""
    return _roi_table(df, "salary_remaining_label")


def analyze_salary_spread(df: pd.DataFrame) -> pd.DataFrame:
    """
    ROI by salary spread (max fighter salary - min fighter salary).
    High spread = Stars & Scrubs construction. Low spread = balanced.
    """
    # salary_spread requires player-level data; use a proxy if not loaded
    if "salary_spread" not in df.columns:
        raise ValueError("salary_spread column not present — join player salary data first")

    spread_bins = [0, 500, 1000, 1500, 2000, float("inf")]
    spread_labels = ["<$500", "$500-1K", "$1-1.5K", "$1.5-2K", ">$2K"]
    df = df.copy()
    df["spread_label"] = pd.cut(df["salary_spread"], bins=spread_bins, labels=spread_labels, right=True)
    return _roi_table(df, "spread_label")


# ---------------------------------------------------------------------------
# Odds / favorites analysis
# ---------------------------------------------------------------------------

def analyze_favorite_count(df: pd.DataFrame) -> pd.DataFrame:
    """ROI by number of favorites (>50% implied probability) in the lineup."""
    odds_df = df.dropna(subset=["favorite_count"])
    if odds_df.empty:
        return pd.DataFrame()
    return _roi_table(odds_df, "fav_count_label")


def analyze_favorite_count_by_slate(df: pd.DataFrame) -> pd.DataFrame:
    """ROI by favorite count × slate size — validates slate-adjusted loading rules."""
    odds_df = df.dropna(subset=["favorite_count"])
    results = []
    for slate, slate_df in odds_df.groupby("slate_size_label", observed=True):
        tbl = _roi_table(slate_df, "fav_count_label")
        tbl.insert(0, "slate_size_label", slate)
        results.append(tbl)
    return pd.concat(results, ignore_index=True)


def analyze_implied_prob_sum(df: pd.DataFrame) -> pd.DataFrame:
    """
    ROI by total implied probability sum across all 6 fighters.
    Sweet spot: 3.0-3.75 (roughly 4-5 favorites at 55% each + 1-2 dogs at 40%).
    """
    odds_df = df.dropna(subset=["implied_prob_sum"])
    if odds_df.empty:
        return pd.DataFrame()

    bins = [0, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, float("inf")]
    labels = ["<2.5", "2.5-2.75", "2.75-3.0", "3.0-3.25", "3.25-3.5", "3.5-3.75", "3.75+"]
    odds_df = odds_df.copy()
    odds_df["prob_sum_label"] = pd.cut(odds_df["implied_prob_sum"], bins=bins, labels=labels, right=True)
    return _roi_table(odds_df, "prob_sum_label")


def analyze_own_prob_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """
    ROI by own_prob_ratio (avg ownership / implied win probability).
    Low ratio = fighters owned below their win probability = value.
    """
    odds_df = df.dropna(subset=["own_prob_ratio"])
    if odds_df.empty:
        return pd.DataFrame()

    bins = [0, 0.50, 0.75, 1.0, 1.25, float("inf")]
    labels = ["<0.50", "0.50-0.75", "0.75-1.0", "1.0-1.25", "1.25+"]
    odds_df = odds_df.copy()
    odds_df["ratio_label"] = pd.cut(odds_df["own_prob_ratio"], bins=bins, labels=labels, right=True)
    return _roi_table(odds_df, "ratio_label")


def analyze_tossup_count(df: pd.DataFrame) -> pd.DataFrame:
    """
    ROI by number of toss-up fighters (40-60% implied prob) in lineup.
    Finding: 0 toss-ups = 3.53x ROI. Avoid murky pick-em spots.
    """
    odds_df = df.dropna(subset=["tossup_count"])
    if odds_df.empty:
        return pd.DataFrame()
    odds_df = odds_df.copy()
    odds_df["tossup_label"] = odds_df["tossup_count"].astype(int).astype(str)
    return _roi_table(odds_df, "tossup_label")


# ---------------------------------------------------------------------------
# Duplication analysis
# ---------------------------------------------------------------------------

def analyze_duplication(df: pd.DataFrame) -> pd.DataFrame:
    """
    ROI by lineup_count (how many contests the same lineup was entered in).
    Unique lineups (count=1) consistently outperform duplicated ones.
    """
    bins = [0, 1, 5, 20, 100, float("inf")]
    labels = ["1 (unique)", "2-5", "6-20", "21-100", "101+"]
    df = df.copy()
    df["dupe_label"] = pd.cut(df["lineup_count"], bins=bins, labels=labels, right=True)
    return _roi_table(df, "dupe_label")


# ---------------------------------------------------------------------------
# Full dimension sweep
# ---------------------------------------------------------------------------

def run_full_discovery(df: pd.DataFrame) -> dict:
    """
    Run all analyses and return results as a dict of DataFrames.
    Use this for a complete refresh of all validated rules.
    """
    return {
        "ownership": analyze_ownership(df),
        "ownership_by_slate": analyze_ownership_by_regime(df, "slate_size_label"),
        "ownership_by_fee": analyze_ownership_by_regime(df, "fee_label"),
        "ownership_composition": analyze_ownership_composition(df),
        "salary_remaining": analyze_salary_remaining(df),
        "favorite_count": analyze_favorite_count(df),
        "favorite_count_by_slate": analyze_favorite_count_by_slate(df),
        "implied_prob_sum": analyze_implied_prob_sum(df),
        "own_prob_ratio": analyze_own_prob_ratio(df),
        "tossup_count": analyze_tossup_count(df),
        "duplication": analyze_duplication(df),
    }
