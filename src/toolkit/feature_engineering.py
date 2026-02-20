"""
Feature Engineering Utilities

Based on:
- Zheng & Casari: Feature Engineering for ML
- Géron: Data preparation and sklearn pipelines
- López de Prado: Fractional differentiation for stationarity
- Hyndman: Time-series feature extraction

Key Principles:
1. Features > Algorithms - good features make simple models work
2. Feature engineering is iterative
3. Domain knowledge should guide feature creation
4. Always check for data leakage in temporal features
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Union, Callable
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from scipy import stats


class FeatureEngineering:
    """
    Feature engineering utilities for DFS/Betting prediction.

    Provides methods for:
    - Lag features (with proper temporal handling)
    - Rolling statistics
    - Categorical encoding
    - Numeric transformations
    - Time-series specific features

    Example
    -------
    >>> fe = FeatureEngineering()
    >>> df = fe.add_lag_features(df, cols=['points'], lags=[1, 2, 3], group_col='player_id')
    >>> df = fe.add_rolling_features(df, cols=['points'], windows=[3, 5], group_col='player_id')
    """

    def __init__(self):
        self._fitted_transformers = {}

    # =========================================================================
    # LAG FEATURES
    # =========================================================================

    def add_lag_features(
        self,
        df: pd.DataFrame,
        cols: List[str],
        lags: List[int],
        group_col: Optional[str] = None,
        date_col: Optional[str] = None,
        fill_value: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Add lagged features.

        IMPORTANT: Lags are computed within groups (e.g., per player)
        to prevent data leakage across entities.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe (must be sorted by date)
        cols : list
            Columns to lag
        lags : list
            Lag periods (e.g., [1, 2, 3] for 1, 2, 3 period lags)
        group_col : str, optional
            Column to group by (e.g., 'player_id')
        date_col : str, optional
            Date column for sorting within groups
        fill_value : float, optional
            Value to fill NaN lags (default: leave as NaN)

        Returns
        -------
        pd.DataFrame
            DataFrame with new lag columns
        """
        df = df.copy()

        if date_col:
            df = df.sort_values([group_col, date_col] if group_col else date_col)

        for col in cols:
            for lag in lags:
                col_name = f"{col}_lag{lag}"

                if group_col:
                    df[col_name] = df.groupby(group_col)[col].shift(lag)
                else:
                    df[col_name] = df[col].shift(lag)

                if fill_value is not None:
                    df[col_name] = df[col_name].fillna(fill_value)

        return df

    # =========================================================================
    # ROLLING FEATURES
    # =========================================================================

    def add_rolling_features(
        self,
        df: pd.DataFrame,
        cols: List[str],
        windows: List[int],
        group_col: Optional[str] = None,
        agg_funcs: List[str] = ['mean', 'std'],
        min_periods: int = 1
    ) -> pd.DataFrame:
        """
        Add rolling window statistics.

        Computes rolling statistics shifted by 1 to prevent leakage
        (current row is never included in its own rolling calculation).

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        cols : list
            Columns to compute rolling stats for
        windows : list
            Window sizes (e.g., [3, 5, 10])
        group_col : str, optional
            Column to group by
        agg_funcs : list
            Aggregation functions: 'mean', 'std', 'min', 'max', 'sum', 'median'
        min_periods : int
            Minimum periods required for calculation

        Returns
        -------
        pd.DataFrame
            DataFrame with new rolling columns
        """
        df = df.copy()

        for col in cols:
            for window in windows:
                for agg in agg_funcs:
                    col_name = f"{col}_roll{window}_{agg}"

                    if group_col:
                        grouped = df.groupby(group_col)[col]
                        # Shift by 1 to exclude current row
                        rolled = grouped.transform(
                            lambda x: x.shift(1).rolling(window, min_periods=min_periods).agg(agg)
                        )
                    else:
                        rolled = df[col].shift(1).rolling(window, min_periods=min_periods).agg(agg)

                    df[col_name] = rolled

        return df

    def add_expanding_features(
        self,
        df: pd.DataFrame,
        cols: List[str],
        group_col: Optional[str] = None,
        agg_funcs: List[str] = ['mean', 'std']
    ) -> pd.DataFrame:
        """
        Add expanding window (cumulative) statistics.

        Uses all historical data up to (but not including) current row.
        """
        df = df.copy()

        for col in cols:
            for agg in agg_funcs:
                col_name = f"{col}_expanding_{agg}"

                if group_col:
                    grouped = df.groupby(group_col)[col]
                    expanded = grouped.transform(
                        lambda x: x.shift(1).expanding().agg(agg)
                    )
                else:
                    expanded = df[col].shift(1).expanding().agg(agg)

                df[col_name] = expanded

        return df

    # =========================================================================
    # DIFFERENCE FEATURES
    # =========================================================================

    def add_diff_features(
        self,
        df: pd.DataFrame,
        cols: List[str],
        periods: List[int] = [1],
        group_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Add difference features (changes from previous periods).

        Useful for:
        - Momentum indicators
        - Trend detection
        - Rate of change

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        cols : list
            Columns to difference
        periods : list
            Difference periods (default [1] for first difference)
        group_col : str, optional
            Column to group by

        Returns
        -------
        pd.DataFrame
            DataFrame with difference columns
        """
        df = df.copy()

        for col in cols:
            for period in periods:
                col_name = f"{col}_diff{period}"

                if group_col:
                    df[col_name] = df.groupby(group_col)[col].diff(period)
                else:
                    df[col_name] = df[col].diff(period)

        return df

    def add_pct_change_features(
        self,
        df: pd.DataFrame,
        cols: List[str],
        periods: List[int] = [1],
        group_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Add percentage change features.

        Returns
        -------
        pd.DataFrame
            DataFrame with pct change columns
        """
        df = df.copy()

        for col in cols:
            for period in periods:
                col_name = f"{col}_pct{period}"

                if group_col:
                    df[col_name] = df.groupby(group_col)[col].pct_change(period)
                else:
                    df[col_name] = df[col].pct_change(period)

                # Handle infinities
                df[col_name] = df[col_name].replace([np.inf, -np.inf], np.nan)

        return df

    # =========================================================================
    # CATEGORICAL FEATURES
    # =========================================================================

    def add_target_encoding(
        self,
        df: pd.DataFrame,
        cat_col: str,
        target_col: str,
        group_col: Optional[str] = None,
        smoothing: float = 10.0,
        min_samples: int = 5
    ) -> pd.DataFrame:
        """
        Add target encoding for categorical variables.

        Uses smoothed mean to prevent overfitting on rare categories.
        IMPORTANT: Compute statistics only on training data to prevent leakage.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        cat_col : str
            Categorical column to encode
        target_col : str
            Target column for computing means
        group_col : str, optional
            For leave-one-out encoding within groups
        smoothing : float
            Smoothing parameter (higher = more regularization)
        min_samples : int
            Minimum samples for category-specific mean

        Returns
        -------
        pd.DataFrame
            DataFrame with target encoded column
        """
        df = df.copy()
        col_name = f"{cat_col}_target_enc"

        global_mean = df[target_col].mean()
        cat_stats = df.groupby(cat_col)[target_col].agg(['mean', 'count'])

        # Smoothed mean: weighted average of category mean and global mean
        smooth_mean = (
            cat_stats['count'] * cat_stats['mean'] + smoothing * global_mean
        ) / (cat_stats['count'] + smoothing)

        # Apply minimum samples threshold
        smooth_mean = smooth_mean.where(cat_stats['count'] >= min_samples, global_mean)

        df[col_name] = df[cat_col].map(smooth_mean)
        df[col_name] = df[col_name].fillna(global_mean)

        return df

    def add_frequency_encoding(
        self,
        df: pd.DataFrame,
        cols: List[str]
    ) -> pd.DataFrame:
        """
        Add frequency encoding for categorical variables.

        Simple encoding: category value = its frequency in the data.
        """
        df = df.copy()

        for col in cols:
            freq = df[col].value_counts(normalize=True)
            df[f"{col}_freq"] = df[col].map(freq)

        return df

    # =========================================================================
    # NUMERIC TRANSFORMATIONS
    # =========================================================================

    def add_log_transform(
        self,
        df: pd.DataFrame,
        cols: List[str],
        offset: float = 1.0
    ) -> pd.DataFrame:
        """
        Add log-transformed features.

        Useful for:
        - Right-skewed distributions
        - Multiplicative relationships
        - Percentage changes

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        cols : list
            Columns to transform
        offset : float
            Offset to add before log (handles zeros)

        Returns
        -------
        pd.DataFrame
            DataFrame with log columns
        """
        df = df.copy()

        for col in cols:
            df[f"{col}_log"] = np.log1p(df[col].clip(lower=0) + offset - 1)

        return df

    def add_power_transform(
        self,
        df: pd.DataFrame,
        cols: List[str],
        powers: List[float] = [0.5, 2.0]
    ) -> pd.DataFrame:
        """
        Add power-transformed features (square root, square, etc).
        """
        df = df.copy()

        for col in cols:
            for power in powers:
                power_name = str(power).replace('.', '_')
                if power == 0.5:
                    df[f"{col}_sqrt"] = np.sqrt(df[col].clip(lower=0))
                elif power == 2.0:
                    df[f"{col}_sq"] = df[col] ** 2
                else:
                    df[f"{col}_pow{power_name}"] = np.sign(df[col]) * np.abs(df[col]) ** power

        return df

    def add_quantile_bins(
        self,
        df: pd.DataFrame,
        cols: List[str],
        n_bins: int = 10,
        labels: bool = False
    ) -> pd.DataFrame:
        """
        Add quantile-binned features.

        From Zheng & Casari: binning can help capture non-linear relationships.
        """
        df = df.copy()

        for col in cols:
            df[f"{col}_qbin{n_bins}"] = pd.qcut(
                df[col], q=n_bins, labels=labels, duplicates='drop'
            )

        return df

    # =========================================================================
    # TIME FEATURES
    # =========================================================================

    def add_date_features(
        self,
        df: pd.DataFrame,
        date_col: str
    ) -> pd.DataFrame:
        """
        Extract date/time features from datetime column.

        Features:
        - Day of week (0-6)
        - Month (1-12)
        - Week of year
        - Is weekend
        - Day of month
        - Quarter
        """
        df = df.copy()

        dt = pd.to_datetime(df[date_col])

        df[f"{date_col}_dayofweek"] = dt.dt.dayofweek
        df[f"{date_col}_month"] = dt.dt.month
        df[f"{date_col}_weekofyear"] = dt.dt.isocalendar().week
        df[f"{date_col}_is_weekend"] = dt.dt.dayofweek.isin([5, 6]).astype(int)
        df[f"{date_col}_dayofmonth"] = dt.dt.day
        df[f"{date_col}_quarter"] = dt.dt.quarter

        return df

    def add_cyclical_encoding(
        self,
        df: pd.DataFrame,
        col: str,
        max_val: int
    ) -> pd.DataFrame:
        """
        Add cyclical (sin/cos) encoding for periodic features.

        Useful for: day of week, month, hour, etc.
        This preserves the cyclical nature (e.g., Sunday is close to Monday).
        """
        df = df.copy()

        df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / max_val)
        df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / max_val)

        return df

    # =========================================================================
    # INTERACTION FEATURES
    # =========================================================================

    def add_ratio_features(
        self,
        df: pd.DataFrame,
        numerator_cols: List[str],
        denominator_cols: List[str],
        epsilon: float = 1e-6
    ) -> pd.DataFrame:
        """
        Add ratio features.

        Creates all combinations of numerator/denominator.
        """
        df = df.copy()

        for num_col in numerator_cols:
            for denom_col in denominator_cols:
                if num_col != denom_col:
                    col_name = f"{num_col}_over_{denom_col}"
                    df[col_name] = df[num_col] / (df[denom_col] + epsilon)

        return df

    def add_interaction_features(
        self,
        df: pd.DataFrame,
        cols: List[str],
        interaction_type: str = 'multiply'
    ) -> pd.DataFrame:
        """
        Add interaction features (products or sums of pairs).
        """
        df = df.copy()

        for i, col1 in enumerate(cols):
            for col2 in cols[i+1:]:
                if interaction_type == 'multiply':
                    df[f"{col1}_x_{col2}"] = df[col1] * df[col2]
                elif interaction_type == 'add':
                    df[f"{col1}_plus_{col2}"] = df[col1] + df[col2]

        return df

    # =========================================================================
    # STATISTICAL FEATURES
    # =========================================================================

    def add_zscore(
        self,
        df: pd.DataFrame,
        cols: List[str],
        group_col: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Add z-score normalized features.

        If group_col provided, normalizes within groups.
        """
        df = df.copy()

        for col in cols:
            if group_col:
                grouped = df.groupby(group_col)[col]
                df[f"{col}_zscore"] = grouped.transform(lambda x: stats.zscore(x, nan_policy='omit'))
            else:
                df[f"{col}_zscore"] = stats.zscore(df[col], nan_policy='omit')

        return df

    def add_rank_features(
        self,
        df: pd.DataFrame,
        cols: List[str],
        group_col: Optional[str] = None,
        pct: bool = True
    ) -> pd.DataFrame:
        """
        Add rank features.

        If pct=True, returns percentile rank (0-1).
        Useful for: comparing players within a slate, comparing across time.
        """
        df = df.copy()

        for col in cols:
            col_name = f"{col}_rank" if not pct else f"{col}_pctrank"

            if group_col:
                df[col_name] = df.groupby(group_col)[col].rank(pct=pct)
            else:
                df[col_name] = df[col].rank(pct=pct)

        return df


# =============================================================================
# SKLEARN PIPELINE BUILDER
# =============================================================================

def build_preprocessing_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
    numeric_strategy: str = 'median',
    scale_numeric: bool = True
) -> ColumnTransformer:
    """
    Build sklearn preprocessing pipeline (Géron style).

    Handles:
    - Missing value imputation
    - Numeric scaling
    - Categorical encoding

    Parameters
    ----------
    numeric_cols : list
        Numeric feature columns
    categorical_cols : list
        Categorical feature columns
    numeric_strategy : str
        Imputation strategy for numerics ('median', 'mean', 'constant')
    scale_numeric : bool
        Whether to standardize numeric features

    Returns
    -------
    ColumnTransformer
        Fitted transformer pipeline
    """
    # Numeric pipeline
    numeric_steps = [('imputer', SimpleImputer(strategy=numeric_strategy))]
    if scale_numeric:
        numeric_steps.append(('scaler', StandardScaler()))
    numeric_transformer = Pipeline(steps=numeric_steps)

    # Categorical pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='drop'
    )

    return preprocessor


# =============================================================================
# FRACTIONAL DIFFERENTIATION (López de Prado)
# =============================================================================

def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Get weights for fractional differentiation (fixed-width window).

    From López de Prado: fractional differentiation balances
    stationarity (for modeling) with memory (for predictive power).

    Parameters
    ----------
    d : float
        Differentiation order (0 < d < 1 typically)
        d=0: no differentiation (keeps all memory)
        d=1: full differentiation (loses all memory)
    threshold : float
        Minimum weight magnitude to include

    Returns
    -------
    np.ndarray
        Weights for weighted moving average
    """
    weights = [1.0]
    k = 1

    while abs(weights[-1]) >= threshold:
        w = -weights[-1] * (d - k + 1) / k
        weights.append(w)
        k += 1

    return np.array(weights[::-1]).reshape(-1, 1)


def frac_diff(
    series: pd.Series,
    d: float,
    threshold: float = 1e-5
) -> pd.Series:
    """
    Apply fractional differentiation to a series.

    This transforms non-stationary series while preserving memory.
    Standard differentiation (d=1) removes all memory.
    Fractional (0 < d < 1) keeps some memory for prediction.

    Parameters
    ----------
    series : pd.Series
        Input time series
    d : float
        Differentiation order (try 0.3-0.7 range)
    threshold : float
        Weight threshold

    Returns
    -------
    pd.Series
        Fractionally differentiated series
    """
    weights = get_weights_ffd(d, threshold)
    width = len(weights)

    result = []
    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1:i + 1].values.reshape(-1, 1)
        result.append(np.dot(weights.T, window)[0, 0])

    return pd.Series(result, index=series.index[width - 1:])


def find_min_ffd(
    series: pd.Series,
    d_range: np.ndarray = np.arange(0, 1.1, 0.1),
    p_value_threshold: float = 0.05
) -> float:
    """
    Find minimum d that makes series stationary.

    Uses ADF test to check stationarity.

    Parameters
    ----------
    series : pd.Series
        Input time series
    d_range : np.ndarray
        Range of d values to try
    p_value_threshold : float
        ADF p-value threshold for stationarity

    Returns
    -------
    float
        Minimum d for stationarity
    """
    from statsmodels.tsa.stattools import adfuller

    for d in d_range:
        diff_series = frac_diff(series.dropna(), d)
        if len(diff_series) < 10:
            continue

        adf_result = adfuller(diff_series, maxlag=1, regression='c')
        p_value = adf_result[1]

        if p_value < p_value_threshold:
            return d

    return 1.0  # Full differentiation if nothing else works
