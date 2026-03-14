"""
Event-level bootstrap confidence intervals for ROI backtesting.

Resamples whole contest dates (not individual lineups) to preserve
within-event correlation structure. UFC has ~132 dates — CIs will
be wider than PGA's 250-date dataset, which is the honest answer.

Ported from pga-dfs-analytics/src/toolkit/bootstrap.py, adapted for
UFC's weighted ROI convention: SUM(payout * user_count) / SUM(entry_cost * user_count)
centered at 1.0 (not PGA's -1 convention centered at 0.0).
"""

import numpy as np
from typing import Callable


# ─── Defaults ────────────────────────────────────────────────────────────────
BOOTSTRAP_ITERATIONS = 2_000
BOOTSTRAP_CI = 0.95
BOOTSTRAP_SEED = 42

# Null hypothesis: ROI = 1.0 (breakeven). Above 1.0 = profitable.
ROI_NULL = 1.0


def event_bootstrap_ci_weighted(
    payouts: np.ndarray,
    costs: np.ndarray,
    weights: np.ndarray,
    event_ids: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_ITERATIONS,
    ci: float = BOOTSTRAP_CI,
    seed: int = BOOTSTRAP_SEED,
    return_samples: bool = False,
) -> dict:
    """
    Entry-weighted bootstrap CI on ROI (UFC convention: centered at 1.0).

    Point estimate = SUM(payout * user_count) / SUM(entry_cost * user_count).
    Each bootstrap iteration resamples whole contest dates and computes
    the weighted ROI ratio.

    Parameters
    ----------
    payouts : array
        Raw payout per lineup row.
    costs : array
        Entry cost per lineup row.
    weights : array
        user_count per row (how many entries this lineup represents).
    event_ids : array
        Contest date identifier for each row.

    Returns
    -------
    dict with: point_estimate, ci_lower, ci_upper, std, n_events, n_samples
    """
    payouts = np.asarray(payouts, dtype=float)
    costs = np.asarray(costs, dtype=float)
    weights = np.asarray(weights, dtype=float)
    event_ids = np.asarray(event_ids)

    unique_events = np.unique(event_ids)
    n_events = len(unique_events)

    # Weighted totals per lineup
    weighted_payout = payouts * weights
    weighted_cost = costs * weights
    total_wp = weighted_payout.sum()
    total_wc = weighted_cost.sum()
    point_estimate = float(total_wp / total_wc) if total_wc > 0 else 0.0

    # Pre-compute per-event weighted sums (fast path)
    event_wp = np.zeros(n_events)
    event_wc = np.zeros(n_events)
    for i, eid in enumerate(unique_events):
        mask = event_ids == eid
        event_wp[i] = weighted_payout[mask].sum()
        event_wc[i] = weighted_cost[mask].sum()

    rng = np.random.RandomState(seed)
    alpha = (1 - ci) / 2

    bootstrap_values = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        sampled = rng.randint(0, n_events, size=n_events)
        s_wp = event_wp[sampled].sum()
        s_wc = event_wc[sampled].sum()
        bootstrap_values[b] = (s_wp / s_wc) if s_wc > 0 else 0.0

    result = {
        "point_estimate": point_estimate,
        "ci_lower": float(np.percentile(bootstrap_values, alpha * 100)),
        "ci_upper": float(np.percentile(bootstrap_values, (1 - alpha) * 100)),
        "std": float(bootstrap_values.std()),
        "n_events": n_events,
        "n_samples": len(payouts),
    }
    if return_samples:
        result["bootstrap_samples"] = bootstrap_values
    return result


def bootstrap_p_value(
    bootstrap_values: np.ndarray,
    null_value: float = ROI_NULL,
) -> float:
    """
    Two-sided p-value from bootstrap distribution.

    Tests against ROI_NULL (1.0x) by default — does this rule make money?
    """
    n = len(bootstrap_values)
    if n == 0:
        return 1.0
    point_est = float(np.mean(bootstrap_values))
    if point_est >= null_value:
        p = 2 * np.mean(bootstrap_values <= null_value)
    else:
        p = 2 * np.mean(bootstrap_values >= null_value)
    return min(float(p), 1.0)


def grouped_bootstrap_ci(
    df,
    group_col: str,
    event_col: str = "contest_date",
    n_bootstrap: int = BOOTSTRAP_ITERATIONS,
    ci: float = BOOTSTRAP_CI,
    seed: int = BOOTSTRAP_SEED,
) -> dict:
    """
    Run entry-weighted event-level bootstrap CI for each group in a DataFrame.

    Returns dict of {group_value: bootstrap_result_dict}.
    Each result also includes 'p_value' (two-sided test vs rake baseline).
    """
    results = {}
    for group_val, gdf in df.groupby(group_col, observed=True):
        res = event_bootstrap_ci_weighted(
            payouts=gdf["payout"].values,
            costs=gdf["entry_cost"].values,
            weights=gdf["user_count"].values,
            event_ids=gdf[event_col].values,
            n_bootstrap=n_bootstrap,
            ci=ci,
            seed=seed,
            return_samples=True,
        )
        res["p_value"] = bootstrap_p_value(
            res.pop("bootstrap_samples"), ROI_NULL
        )
        results[group_val] = res
    return results
