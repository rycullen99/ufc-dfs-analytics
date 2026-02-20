"""Event-grouped walk-forward cross-validation."""

import numpy as np
import pandas as pd
from typing import Iterator, Tuple


class EventWalkForwardCV:
    """
    Walk-forward CV that groups all fighters on the same event date.

    Parameters
    ----------
    min_train_events : int
        Minimum number of distinct event dates before the first test fold.
    """

    def __init__(self, min_train_events: int = 20):
        self.min_train_events = min_train_events

    def split(
        self,
        df: pd.DataFrame,
        date_col: str = "date_id",
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Yield (train_idx, test_idx) arrays.

        Each fold uses one event date as the test set and all
        prior event dates as the training set (expanding window).
        """
        sorted_dates = sorted(df[date_col].unique())

        for i in range(self.min_train_events, len(sorted_dates)):
            train_dates = set(sorted_dates[:i])
            test_date = sorted_dates[i]

            train_mask = df[date_col].isin(train_dates)
            test_mask = df[date_col] == test_date

            train_idx = np.where(train_mask)[0]
            test_idx = np.where(test_mask)[0]

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self, df: pd.DataFrame, date_col: str = "date_id") -> int:
        n_dates = df[date_col].nunique()
        return max(0, n_dates - self.min_train_events)
