"""
ROI discovery analysis (Phase 6).
Each function groups by a dimension and returns weighted ROI per bucket.

v2: Bootstrap CIs + p-values on every bucket, FDR correction across all tests,
    contest family dimension, time-split validation runner.
"""

import pandas as pd
import numpy as np

from .loader import weighted_roi
from .bootstrap_roi import grouped_bootstrap_ci
from .fdr import apply_fdr_to_results


# ─── Core table builders ─────────────────────────────────────────────────────

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


def _roi_table_with_ci(df: pd.DataFrame, group_col: str) -> dict:
    """
    Group by dimension, compute weighted ROI with bootstrap CIs + p-values.

    Returns dict of {group_value: {point_estimate, ci_lower, ci_upper, p_value, ...}}.
    """
    ci_results = grouped_bootstrap_ci(df, group_col)

    # Enrich with supporting metrics
    for label, g in df.groupby(group_col, observed=True):
        if label in ci_results:
            ci_results[label]["n_lineups"] = len(g)
            ci_results[label]["n_entries"] = int(g["user_count"].sum())
            ci_results[label]["top1pct_rate"] = round(
                (g["lineup_percentile"] <= 1).mean(), 4
            )
            ci_results[label]["cash_rate"] = round((g["payout"] > 0).mean(), 4)

    return ci_results


# ─── Individual analyses ──────────────────────────────────────────────────────

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


def contest_family(df: pd.DataFrame) -> pd.DataFrame:
    """ROI by contest family (mini-MAX, Throwdown, Special, etc.)."""
    return _roi_table(df, "contest_family")


def ownership_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """ROI by ownership band within each contest family."""
    return ownership_by(df, "contest_family")


def fav_count_by_family(df: pd.DataFrame) -> pd.DataFrame:
    """ROI by favorite count within each contest family."""
    return favorite_count_by(df, "contest_family")


# ─── Fight-count regime analyses ──────────────────────────────────────────────

def _cross_tab(df: pd.DataFrame, regime_col: str, metric_col: str,
               dropna_col: str = None) -> pd.DataFrame:
    """Generic cross-tab: ROI by metric_col within each regime_col level."""
    sub = df.dropna(subset=[dropna_col]) if dropna_col else df
    results = []
    for val, g in sub.groupby(regime_col, observed=True):
        tbl = _roi_table(g, metric_col)
        tbl.insert(0, regime_col, val)
        results.append(tbl)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def fav_count_by_fights(df: pd.DataFrame) -> pd.DataFrame:
    """ROI by favorite count within each fight count band."""
    return _cross_tab(df, "fight_count_band", "fav_count_band", "favorite_count")


def ownership_by_fights(df: pd.DataFrame) -> pd.DataFrame:
    """ROI by ownership band within each fight count band."""
    return _cross_tab(df, "fight_count_band", "total_own_band")


def salary_by_fights(df: pd.DataFrame) -> pd.DataFrame:
    """ROI by salary remaining within each fight count band."""
    return _cross_tab(df, "fight_count_band", "salary_remaining_band")


def duplication_by_fights(df: pd.DataFrame) -> pd.DataFrame:
    """ROI by duplication within each fight count band."""
    return _cross_tab(df, "fight_count_band", "dupe_band")


def fav_count_by_fights_arch(df: pd.DataFrame) -> pd.DataFrame:
    """ROI by favorite count within each fight count × archetype combination."""
    sub = df.dropna(subset=["favorite_count"])
    if "contest_archetype" not in sub.columns:
        return pd.DataFrame()
    results = []
    for (fc, arch), g in sub.groupby(["fight_count_band", "contest_archetype"], observed=True):
        tbl = _roi_table(g, "fav_count_band")
        tbl.insert(0, "fight_count_band", fc)
        tbl.insert(1, "contest_archetype", arch)
        results.append(tbl)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def ownership_by_fights_arch(df: pd.DataFrame) -> pd.DataFrame:
    """ROI by ownership within each fight count × archetype combination."""
    if "contest_archetype" not in df.columns:
        return pd.DataFrame()
    results = []
    for (fc, arch), g in df.groupby(["fight_count_band", "contest_archetype"], observed=True):
        tbl = _roi_table(g, "total_own_band")
        tbl.insert(0, "fight_count_band", fc)
        tbl.insert(1, "contest_archetype", arch)
        results.append(tbl)
    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


# ─── Runners ──────────────────────────────────────────────────────────────────

def run_all(df: pd.DataFrame) -> dict:
    """Run every analysis and return results as {name: DataFrame}."""
    return {
        "ownership":              ownership(df),
        "ownership_by_slate":     ownership_by(df, "slate_size"),
        "ownership_by_fee":       ownership_by(df, "fee_tier"),
        "ownership_by_family":    ownership_by_family(df),
        "ownership_composition":  ownership_composition(df),
        "salary_remaining":       salary_remaining(df),
        "favorite_count":         favorite_count(df),
        "fav_count_by_slate":     favorite_count_by(df, "slate_size"),
        "fav_count_by_fee":       favorite_count_by(df, "fee_tier"),
        "fav_count_by_family":    fav_count_by_family(df),
        "implied_prob_sum":       implied_prob_sum(df),
        "own_prob_ratio":         own_prob_ratio(df),
        "tossup_count":           tossup_count(df),
        "duplication":            duplication(df),
        "contest_family":         contest_family(df),
        # Fight-count regime cross-tabs
        "fav_count_by_fights":      fav_count_by_fights(df),
        "ownership_by_fights":      ownership_by_fights(df),
        "salary_by_fights":         salary_by_fights(df),
        "duplication_by_fights":    duplication_by_fights(df),
        "fav_count_by_fights_arch": fav_count_by_fights_arch(df),
        "ownership_by_fights_arch": ownership_by_fights_arch(df),
    }


def run_all_with_ci(df: pd.DataFrame, fdr_q: float = 0.10) -> dict:
    """
    Run every analysis with bootstrap CIs, p-values, and FDR correction.

    Returns dict of {analysis_name: {group_value: result_dict}}.
    Each result_dict has: point_estimate, ci_lower, ci_upper, p_value,
    n_lineups, n_entries, top1pct_rate, cash_rate, survives_fdr.
    """
    # Simple analyses: one group column
    simple = {
        "ownership":         ("total_own_band", df),
        "salary_remaining":  ("salary_remaining_band", df),
        "favorite_count":    ("fav_count_band", df.dropna(subset=["favorite_count"])),
        "implied_prob_sum":  ("prob_sum_band", df.dropna(subset=["implied_prob_sum"])),
        "own_prob_ratio":    ("own_prob_band", df.dropna(subset=["own_prob_ratio"])),
        "duplication":       ("dupe_band", df),
        "contest_family":    ("contest_family", df),
    }

    # Tossup needs label column
    tossup_df = df.dropna(subset=["tossup_count"]).copy()
    tossup_df["tossup_label"] = tossup_df["tossup_count"].astype(int).astype(str)
    simple["tossup_count"] = ("tossup_label", tossup_df)

    # Ownership composition
    comp_df = df.copy()
    comp_df["min_own_band"] = pd.cut(
        comp_df["min_ownership"], bins=[0, 3, 8, 15, 100],
        labels=["<3%", "3-8%", "8-15%", "15%+"], right=True,
    )
    comp_df["max_own_band"] = pd.cut(
        comp_df["max_ownership"], bins=[0, 25, 35, 50, 100],
        labels=["<25%", "25-35%", "35-50%", "50%+"], right=True,
    )
    simple["min_ownership"] = ("min_own_band", comp_df)
    simple["max_ownership"] = ("max_own_band", comp_df)

    results = {}
    for name, (col, sub_df) in simple.items():
        results[name] = _roi_table_with_ci(sub_df, col)

    # Apply FDR across all tests
    results = apply_fdr_to_results(results, q=fdr_q)

    return results
