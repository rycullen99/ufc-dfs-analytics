"""Holdout manager — reserves the most recent N events for final evaluation."""

import numpy as np
import pandas as pd
from typing import Tuple


class HoldoutManager:
    """
    Splits data into development and holdout sets by event date.

    Parameters
    ----------
    n_holdout_events : int
        Number of most-recent event dates reserved for holdout.
    """

    def __init__(self, n_holdout_events: int = 10):
        self.n_holdout_events = n_holdout_events
        self._holdout_dates = None

    def split(
        self,
        df: pd.DataFrame,
        date_col: str = "date_id",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Return (dev_df, holdout_df)."""
        sorted_dates = sorted(df[date_col].unique())

        if self.n_holdout_events >= len(sorted_dates):
            raise ValueError(
                f"Cannot hold out {self.n_holdout_events} events from "
                f"{len(sorted_dates)} total events."
            )

        self._holdout_dates = set(sorted_dates[-self.n_holdout_events :])
        dev_mask = ~df[date_col].isin(self._holdout_dates)

        return df[dev_mask].copy(), df[~dev_mask].copy()

    @property
    def holdout_dates(self):
        return self._holdout_dates
