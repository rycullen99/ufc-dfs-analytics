"""
Time-split validation for ROI discovery (Phase 6).

Chronological 60/40 train/test split on contest dates.
UFC has 132 dates -> ~79 train / 53 test. Rules that hold
in the out-of-sample period are much more trustworthy.

Ported from pga-dfs-analytics/src/toolkit/backtesting.py.
"""

import pandas as pd
from typing import Callable


def time_split_validation(
    df: pd.DataFrame,
    date_col: str,
    analysis_fn: Callable,
    train_frac: float = 0.60,
) -> dict:
    """
    Chronological train/test split for overfit validation.

    Parameters
    ----------
    df : DataFrame
        Full dataset with a date column.
    date_col : str
        Column name for contest date identifiers.
    analysis_fn : callable
        Function(subset_df) -> dict. Runs analysis on a subset.
    train_frac : float
        Fraction of dates for training (default 0.60).

    Returns
    -------
    dict with: train_n_dates, test_n_dates, train_n_lineups, test_n_lineups,
               train_result, test_result, split_date
    """
    dates = sorted(df[date_col].unique())
    split_idx = int(len(dates) * train_frac)
    train_dates = set(dates[:split_idx])
    test_dates = set(dates[split_idx:])

    train_df = df[df[date_col].isin(train_dates)]
    test_df = df[df[date_col].isin(test_dates)]

    train_result = analysis_fn(train_df)
    test_result = analysis_fn(test_df)

    return {
        "train_n_dates": len(train_dates),
        "test_n_dates": len(test_dates),
        "train_n_lineups": len(train_df),
        "test_n_lineups": len(test_df),
        "split_date": str(dates[split_idx]) if split_idx < len(dates) else None,
        "train_result": train_result,
        "test_result": test_result,
    }


def validate_rule(
    df: pd.DataFrame,
    date_col: str,
    group_col: str,
    target_bucket: str,
    train_frac: float = 0.60,
) -> dict:
    """
    Validate a specific rule: does the target bucket beat the rake
    in both train and test periods?

    Parameters
    ----------
    df : DataFrame with regime columns.
    date_col : str
    group_col : str — the dimension column (e.g., 'total_own_band').
    target_bucket : str — the bucket label to validate (e.g., '100-125%').
    train_frac : float

    Returns
    -------
    dict with train/test ROI, sample sizes, and a 'holds' boolean.
    """
    from .bootstrap_roi import event_bootstrap_ci_weighted, bootstrap_p_value, ROI_NULL

    dates = sorted(df[date_col].unique())
    split_idx = int(len(dates) * train_frac)
    train_dates = set(dates[:split_idx])
    test_dates = set(dates[split_idx:])

    results = {}
    for period, period_dates in [("train", train_dates), ("test", test_dates)]:
        sub = df[df[date_col].isin(period_dates)]
        bucket = sub[sub[group_col] == target_bucket]

        if len(bucket) < 100:
            results[period] = {
                "roi": None,
                "n_lineups": len(bucket),
                "n_dates": len(period_dates),
                "p_value": 1.0,
                "profitable": False,
                "note": "insufficient data",
            }
            continue

        ci = event_bootstrap_ci_weighted(
            payouts=bucket["payout"].values,
            costs=bucket["entry_cost"].values,
            weights=bucket["user_count"].values,
            event_ids=bucket[date_col].values,
            return_samples=True,
        )
        p = bootstrap_p_value(ci.pop("bootstrap_samples"), ROI_NULL)

        results[period] = {
            "roi": ci["point_estimate"],
            "ci_lower": ci["ci_lower"],
            "ci_upper": ci["ci_upper"],
            "n_lineups": ci["n_samples"],
            "n_dates": ci["n_events"],
            "p_value": p,
            "profitable": ci["ci_lower"] > ROI_NULL,
        }

    results["holds"] = (
        results.get("train", {}).get("profitable", False)
        and results.get("test", {}).get("profitable", False)
    )
    results["rule"] = f"{group_col} = {target_bucket}"

    return results
