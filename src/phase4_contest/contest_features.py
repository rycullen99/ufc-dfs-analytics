"""Contest-type feature engineering."""

import numpy as np
import pandas as pd
import sqlite3
import logging

from ..config import FREEROLL_CONTEST_ID

logger = logging.getLogger(__name__)


def infer_contest_type(row: pd.Series) -> str:
    """
    Infer contest type from contest attributes.

    Rules:
    - multi_entry_max > 1 → 'MME'
    - multi_entry_max == 1 and contest_size <= 20 → 'SE'
    - else → 'limited'
    """
    me_max = row.get("multi_entry_max", 1)
    size = row.get("contest_size", 0)

    if me_max > 1:
        return "MME"
    elif size <= 20:
        return "SE"
    else:
        return "limited"


def add_contest_type_features(
    df: pd.DataFrame,
    conn: sqlite3.Connection,
) -> pd.DataFrame:
    """
    Add contest type and related features to the DataFrame.

    Joins contest metadata and derives:
    - contest_type: 'MME', 'SE', or 'limited'
    - is_mme, is_se: binary indicators
    - log_entry_cost: log-transformed entry cost
    - field_size_bucket: small/medium/large

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'contest_id' column.
    conn : sqlite3.Connection
        Database connection for contest metadata.

    Returns
    -------
    pd.DataFrame with added contest-type features.
    """
    contest_meta = pd.read_sql_query(f"""
        SELECT
            contest_id,
            entry_cost,
            contest_size,
            multi_entry_max
        FROM contests
        WHERE contest_id != {FREEROLL_CONTEST_ID}
    """, conn)

    df = df.merge(
        contest_meta[["contest_id", "multi_entry_max"]],
        on="contest_id",
        how="left",
        suffixes=("", "_meta"),
    )

    # Infer contest type
    df["contest_type"] = df.apply(infer_contest_type, axis=1)

    # Binary indicators
    df["is_mme"] = (df["contest_type"] == "MME").astype(int)
    df["is_se"] = (df["contest_type"] == "SE").astype(int)

    # Log entry cost
    if "entry_cost" in df.columns:
        df["log_entry_cost"] = np.log1p(df["entry_cost"])

    # Field size bucket
    if "contest_size" in df.columns:
        df["field_size_bucket"] = pd.cut(
            df["contest_size"],
            bins=[0, 50, 500, 5000, float("inf")],
            labels=["small", "medium", "large", "massive"],
        )

    # Clean up merge columns
    if "multi_entry_max_meta" in df.columns:
        df.drop(columns=["multi_entry_max_meta"], inplace=True)

    logger.info("Contest types: %s", df["contest_type"].value_counts().to_dict())
    return df
