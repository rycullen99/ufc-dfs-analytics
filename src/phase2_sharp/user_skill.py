"""
Empirical Bayes beta-binomial user skill model.

Identifies sharp DFS users by their historical cash rate, with Bayesian
shrinkage toward the population mean. Users with few entries get pulled
strongly toward the prior; users with many entries let their data speak.
"""

import numpy as np
import pandas as pd
import sqlite3
import logging
from typing import Optional

from ..config import FREEROLL_CONTEST_ID

logger = logging.getLogger(__name__)


def estimate_population_prior(
    cash_counts: np.ndarray,
    entry_counts: np.ndarray,
) -> tuple:
    """
    Estimate Beta(alpha0, beta0) prior via method of moments.

    Parameters
    ----------
    cash_counts : array of int
        Number of cashes per user.
    entry_counts : array of int
        Number of entries per user.

    Returns
    -------
    (alpha0, beta0) : tuple of float
    """
    cash_counts = np.asarray(cash_counts, dtype=float)
    entry_counts = np.asarray(entry_counts, dtype=float)

    if len(cash_counts) == 0 or len(entry_counts) == 0:
        return 1.0, 1.0

    rates = cash_counts / np.maximum(entry_counts, 1)
    mu = rates.mean()
    var = rates.var()

    if var == 0 or mu == 0 or mu == 1:
        return 1.0, 1.0

    # Method of moments for Beta distribution
    common = (mu * (1 - mu) / max(var, 1e-8)) - 1
    common = max(common, 0.1)  # floor to avoid degenerate priors

    alpha0 = mu * common
    beta0 = (1 - mu) * common

    return max(alpha0, 0.1), max(beta0, 0.1)


def compute_user_skill(
    conn: sqlite3.Connection,
    min_entries: int = 10,
    contest_type: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute posterior skill estimate for each DFS user.

    Parameters
    ----------
    conn : sqlite3.Connection
        Database connection.
    min_entries : int
        Minimum entries to include a user.
    contest_type : str, optional
        Filter to specific contest type ('MME', 'SE', 'limited').
        If None, uses all contest types.

    Returns
    -------
    pd.DataFrame with columns:
        username, n_entries, n_cashes, cash_rate,
        posterior_mean, posterior_alpha, posterior_beta, is_sharp
    """
    # Build query for user-level cash stats
    type_filter = ""
    if contest_type:
        type_filter = f"AND c.contest_type = '{contest_type}'"

    query = f"""
    SELECT
        lu.username,
        COUNT(*) AS n_entries,
        SUM(CASE WHEN l.rank <= c.cash_line THEN 1 ELSE 0 END) AS n_cashes
    FROM lineup_usernames lu
    JOIN lineups l ON lu.lineup_id = l.lineup_id
    JOIN contests c ON l.contest_id = c.contest_id
    WHERE c.contest_id != {FREEROLL_CONTEST_ID}
      {type_filter}
    GROUP BY lu.username
    HAVING COUNT(*) >= {min_entries}
    """

    logger.info("Querying user cash stats (min_entries=%d) ...", min_entries)
    df = pd.read_sql_query(query, conn)
    logger.info("Found %d users with >= %d entries", len(df), min_entries)

    if len(df) == 0:
        return pd.DataFrame()

    cash_counts = df["n_cashes"].values.astype(float)
    entry_counts = df["n_entries"].values.astype(float)

    # Estimate population prior
    alpha0, beta0 = estimate_population_prior(cash_counts, entry_counts)
    logger.info("Population prior: Beta(%.2f, %.2f)", alpha0, beta0)

    # Posterior for each user
    df["cash_rate"] = cash_counts / entry_counts
    df["posterior_alpha"] = alpha0 + cash_counts
    df["posterior_beta"] = beta0 + (entry_counts - cash_counts)
    df["posterior_mean"] = df["posterior_alpha"] / (
        df["posterior_alpha"] + df["posterior_beta"]
    )

    # Sharp threshold: top 20% of posterior mean
    threshold = df["posterior_mean"].quantile(0.80)
    df["is_sharp"] = df["posterior_mean"] >= threshold

    logger.info(
        "Sharp threshold=%.4f, %d sharp users (%.1f%%)",
        threshold,
        df["is_sharp"].sum(),
        100 * df["is_sharp"].mean(),
    )

    return df
