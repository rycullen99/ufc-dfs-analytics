"""
ROI discovery analysis (Phase 6).
Each function groups by a dimension and returns weighted ROI per bucket.
"""

import pandas as pd
import numpy as np

from .loader import weighted_roi


def _roi_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Group by dimension, compute weighted ROI + supporting metrics per bucket."""
    rows = []
    for label, g in df.groupby(group_col, observed=True):
        rows.append({
            group_col:      label,
            "n_lineups":    len(g),
            "n_entries":    int(g["user_count"].sum()),
            "weighted_roi": round(weighted_roi(g), 4),
            "top1pct_rate": round((g["lineup_percentile"] <= 1).mean(), 4),
            "cash_rate":    round((g["payout"] > 0).mean(), 4),
        })
    return pd.DataFrame(rows)


def ownership(df: pd.DataFrame) -> pd.DataFrame:
    """ROI by total lineup ownership band."""
    return _roi_table(df, "total_own_band")


def ownership_by(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    """ROI by ownership band within each level of a regime column."""
    results = []
    for val, sub in df.groupby(regime, observed=True):
        tbl = _roi_table(sub, "total_own_band")
        tbl.insert(0, regime, val)
        results.append(tbl)
    return pd.concat(results, ignore_index=True)


def ownership_composition(df: pd.DataFrame) -> dict:
    """ROI by min_ownership (contrarian floor) and max_ownership (chalk anchor)."""
    df = df.copy()
    df["min_own_band"] = pd.cut(df["min_ownership"],
                                bins=[0, 3, 8, 15, 100],
                                labels=["<3%", "3-8%", "8-15%", "15%+"], right=True)
    df["max_own_band"] = pd.cut(df["max_ownership"],
                                bins=[0, 25, 35, 50, 100],
                                labels=["<25%", "25-35%", "35-50%", "50%+"], right=True)
    return {
        "min_own": _roi_table(df, "min_own_band"),
        "max_own": _roi_table(df, "max_own_band"),
    }


def salary_remaining(df: pd.DataFrame) -> pd.DataFrame:
    """ROI by distance from $50K salary cap."""
    return _roi_table(df, "salary_remaining_band")


def favorite_count(df: pd.DataFrame) -> pd.DataFrame:
    """ROI by number of favorites (>50% implied prob) per lineup."""
    return _roi_table(df.dropna(subset=["favorite_count"]), "fav_count_band")


def favorite_count_by(df: pd.DataFrame, regime: str) -> pd.DataFrame:
    """ROI by favorite count within each regime level."""
    sub = df.dropna(subset=["favorite_count"])
    results = []
    for val, g in sub.groupby(regime, observed=True):
        tbl = _roi_table(g, "fav_count_band")
        tbl.insert(0, regime, val)
        results.append(tbl)
    return pd.concat(results, ignore_index=True)


def implied_prob_sum(df: pd.DataFrame) -> pd.DataFrame:
    """ROI by sum of all 6 fighters' implied win probabilities."""
    return _roi_table(df.dropna(subset=["implied_prob_sum"]), "prob_sum_band")


def own_prob_ratio(df: pd.DataFrame) -> pd.DataFrame:
    """ROI by ownership/probability ratio — low = fighters underowned vs market."""
    return _roi_table(df.dropna(subset=["own_prob_ratio"]), "own_prob_band")


def tossup_count(df: pd.DataFrame) -> pd.DataFrame:
    """ROI by number of toss-up fighters (40-60% implied prob) in lineup."""
    sub = df.dropna(subset=["tossup_count"]).copy()
    sub["tossup_label"] = sub["tossup_count"].astype(int).astype(str)
    return _roi_table(sub, "tossup_label")


def duplication(df: pd.DataFrame) -> pd.DataFrame:
    """ROI by lineup duplication count across contests."""
    return _roi_table(df, "dupe_band")


def run_all(df: pd.DataFrame) -> dict:
    """Run every analysis and return results as {name: DataFrame}."""
    return {
        "ownership":              ownership(df),
        "ownership_by_slate":     ownership_by(df, "slate_size"),
        "ownership_by_fee":       ownership_by(df, "fee_tier"),
        "ownership_composition":  ownership_composition(df),
        "salary_remaining":       salary_remaining(df),
        "favorite_count":         favorite_count(df),
        "fav_count_by_slate":     favorite_count_by(df, "slate_size"),
        "fav_count_by_fee":       favorite_count_by(df, "fee_tier"),
        "implied_prob_sum":       implied_prob_sum(df),
        "own_prob_ratio":         own_prob_ratio(df),
        "tossup_count":           tossup_count(df),
        "duplication":            duplication(df),
    }
