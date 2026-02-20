"""
Time-Series Aware Data Structures

Based on:
- López de Prado: Purged K-Fold CV, preventing temporal leakage
- Hyndman: Time-series decomposition and proper train/test splits
- Géron: Data handling best practices

Key Principles:
1. NEVER use future data to predict past (temporal leakage)
2. Purge overlapping samples between train and validation
3. Apply embargo period after train set
4. Weight samples by uniqueness
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Iterator, Union
from dataclasses import dataclass
from sklearn.model_selection import BaseCrossValidator


@dataclass
class DatasetSplit:
    """Container for a train/test or train/val split."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    dates_train: pd.Series
    dates_test: pd.Series

    @property
    def train_size(self) -> int:
        return len(self.X_train)

    @property
    def test_size(self) -> int:
        return len(self.X_test)


class TimeSeriesDataset:
    """
    Time-series aware dataset handler.

    Ensures temporal integrity in all operations:
    - Train/test splits respect time ordering
    - No future information leaks into training
    - Proper handling of overlapping outcomes

    Based on López de Prado's methodology from "Advances in Financial ML"

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset
    date_col : str
        Name of the datetime column
    target_col : str
        Name of the target variable column
    event_end_col : str, optional
        Column indicating when the event/outcome ends (for overlap handling)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        date_col: str,
        target_col: str,
        event_end_col: Optional[str] = None
    ):
        self.df = df.copy()
        self.date_col = date_col
        self.target_col = target_col
        self.event_end_col = event_end_col

        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df[date_col]):
            self.df[date_col] = pd.to_datetime(self.df[date_col])

        # Sort by date
        self.df = self.df.sort_values(date_col).reset_index(drop=True)

        # Compute sample weights if event_end is provided
        if event_end_col:
            self._compute_sample_weights()
        else:
            self.df['_sample_weight'] = 1.0

    def _compute_sample_weights(self):
        """
        Compute sample weights based on uniqueness (López de Prado).

        Overlapping outcomes reduce sample uniqueness. We downweight
        samples that overlap with many others.
        """
        n = len(self.df)
        weights = np.ones(n)

        dates = self.df[self.date_col].values
        ends = self.df[self.event_end_col].values

        for i in range(n):
            # Count how many other samples overlap with sample i
            overlaps = np.sum(
                (dates <= ends[i]) & (ends >= dates[i])
            )
            weights[i] = 1.0 / max(1, overlaps)

        # Normalize weights to sum to n
        weights = weights * n / weights.sum()
        self.df['_sample_weight'] = weights

    def get_train_test_split(
        self,
        test_ratio: float = 0.2,
        gap_periods: int = 0
    ) -> DatasetSplit:
        """
        Split data respecting temporal order.

        Parameters
        ----------
        test_ratio : float
            Fraction of data to use for testing (from the end)
        gap_periods : int
            Number of periods to skip between train and test (embargo)

        Returns
        -------
        DatasetSplit
            Named tuple with train/test data
        """
        n = len(self.df)
        test_size = int(n * test_ratio)
        train_size = n - test_size - gap_periods

        train_df = self.df.iloc[:train_size]
        test_df = self.df.iloc[train_size + gap_periods:]

        feature_cols = [c for c in self.df.columns
                       if c not in [self.date_col, self.target_col, '_sample_weight']]

        return DatasetSplit(
            X_train=train_df[feature_cols],
            X_test=test_df[feature_cols],
            y_train=train_df[self.target_col],
            y_test=test_df[self.target_col],
            dates_train=train_df[self.date_col],
            dates_test=test_df[self.date_col]
        )

    def get_features_and_target(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Return X, y, and dates."""
        feature_cols = [c for c in self.df.columns
                       if c not in [self.date_col, self.target_col, '_sample_weight']]
        return (
            self.df[feature_cols],
            self.df[self.target_col],
            self.df[self.date_col]
        )

    def get_sample_weights(self) -> np.ndarray:
        """Return sample weights for weighted training."""
        return self.df['_sample_weight'].values

    @property
    def date_range(self) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Return the date range of the dataset."""
        return self.df[self.date_col].min(), self.df[self.date_col].max()

    @property
    def n_samples(self) -> int:
        return len(self.df)


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold Cross-Validation (López de Prado).

    Standard K-Fold is WRONG for time-series because:
    1. It allows future data to leak into training
    2. Overlapping outcomes between train/test cause information leakage

    This implementation:
    1. Respects temporal ordering (walk-forward)
    2. Purges train samples that overlap with test samples
    3. Applies an embargo period after purging

    Parameters
    ----------
    n_splits : int
        Number of folds
    purge_pct : float
        Percentage of train samples to purge before test fold
    embargo_pct : float
        Percentage of samples to embargo after train fold
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_pct: float = 0.0,
        embargo_pct: float = 0.01
    ):
        self.n_splits = n_splits
        self.purge_pct = purge_pct
        self.embargo_pct = embargo_pct
        self._actual_splits = None

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        # If we've already computed splits, return actual count
        if self._actual_splits is not None:
            return self._actual_splits
        return self.n_splits

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test indices for each fold.

        Parameters
        ----------
        X : array-like
            Features (used only for length)
        y : array-like, optional
            Target (not used)
        groups : array-like, optional
            Dates or group labels for temporal ordering

        Yields
        ------
        train_idx, test_idx : arrays
            Indices for training and testing
        """
        n = len(X)

        # If groups (dates) provided, sort by them
        if groups is not None:
            indices = np.argsort(groups)
        else:
            indices = np.arange(n)

        # Calculate sizes
        test_size = n // self.n_splits
        purge_size = int(n * self.purge_pct)
        embargo_size = int(n * self.embargo_pct)

        # Store valid splits
        splits = []

        for i in range(self.n_splits):
            # Test fold is at the end of the remaining data
            test_start = n - (self.n_splits - i) * test_size
            test_end = test_start + test_size if i < self.n_splits - 1 else n

            # Train is everything before test, minus purge and embargo
            train_end = max(0, test_start - purge_size - embargo_size)

            train_idx = indices[:train_end]
            test_idx = indices[test_start:test_end]

            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        self._actual_splits = len(splits)

        for train_idx, test_idx in splits:
            yield train_idx, test_idx


class WalkForwardCV(BaseCrossValidator):
    """
    Walk-Forward Cross-Validation (Hyndman style).

    Also known as "rolling forecast origin" or "time series CV".

    The training set grows with each fold while test remains fixed size.
    This mimics real-world deployment where you retrain on all available data.

    Parameters
    ----------
    n_splits : int
        Number of folds
    min_train_size : int
        Minimum training samples before starting CV
    test_size : int
        Fixed test size for each fold
    gap : int
        Gap between train and test (embargo period)
    expanding : bool
        If True, training window expands. If False, it slides (fixed size).
    """

    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: int = 0,
        expanding: bool = True
    ):
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.gap = gap
        self.expanding = expanding

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Optional[Union[np.ndarray, pd.Series]] = None,
        groups: Optional[Union[np.ndarray, pd.Series]] = None
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices."""
        n = len(X)

        # Defaults
        test_size = self.test_size or n // (self.n_splits + 1)
        min_train = self.min_train_size or test_size

        # Calculate step size
        remaining = n - min_train - test_size - self.gap
        step = max(1, remaining // self.n_splits)

        for i in range(self.n_splits):
            train_end = min_train + i * step
            test_start = train_end + self.gap
            test_end = min(test_start + test_size, n)

            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, train_end - min_train)

            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx


def compute_sample_uniqueness(
    dates: pd.Series,
    event_ends: pd.Series
) -> np.ndarray:
    """
    Compute sample uniqueness based on overlapping outcomes.

    From López de Prado: samples that overlap with many others
    provide less unique information.

    Parameters
    ----------
    dates : pd.Series
        Event start dates
    event_ends : pd.Series
        Event end dates

    Returns
    -------
    np.ndarray
        Uniqueness scores (0-1), higher = more unique
    """
    n = len(dates)
    concurrency = np.zeros(n)

    dates_arr = dates.values
    ends_arr = event_ends.values

    for i in range(n):
        # Count concurrent samples
        concurrent = np.sum(
            (dates_arr <= ends_arr[i]) & (ends_arr >= dates_arr[i])
        )
        concurrency[i] = concurrent

    # Uniqueness is inverse of concurrency
    uniqueness = 1.0 / np.maximum(1, concurrency)
    return uniqueness


def get_embargo_times(
    dates: pd.Series,
    pct_embargo: float = 0.01
) -> pd.Series:
    """
    Calculate embargo end times for each sample.

    Parameters
    ----------
    dates : pd.Series
        Sample dates
    pct_embargo : float
        Percentage of total time span to embargo

    Returns
    -------
    pd.Series
        Embargo end timestamps for each sample
    """
    time_span = dates.max() - dates.min()
    embargo_delta = time_span * pct_embargo
    return dates + embargo_delta
