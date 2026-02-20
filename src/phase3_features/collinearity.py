"""Collinearity detection: correlation matrix, VIF, and clustering."""

import numpy as np
import pandas as pd
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def correlation_report(
    df: pd.DataFrame,
    features: List[str],
    threshold: float = 0.85,
) -> pd.DataFrame:
    """
    Find pairs of features with |correlation| > threshold.

    Returns DataFrame with columns: feature_a, feature_b, correlation.
    """
    X = df[features].select_dtypes(include=[np.number])
    corr = X.corr()

    pairs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > threshold:
                pairs.append({
                    "feature_a": cols[i],
                    "feature_b": cols[j],
                    "correlation": round(r, 4),
                })

    result = pd.DataFrame(pairs)
    if len(result) > 0:
        result = result.sort_values("correlation", key=abs, ascending=False)
    logger.info("Found %d correlated pairs (|r| > %.2f)", len(result), threshold)
    return result


def compute_vif(
    df: pd.DataFrame,
    features: List[str],
) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for each feature.

    VIF > 10 suggests problematic multicollinearity.

    Returns DataFrame with columns: feature, vif.
    """
    from sklearn.linear_model import LinearRegression

    X = df[features].select_dtypes(include=[np.number]).dropna()
    cols = X.columns.tolist()
    vif_data = []

    for i, col in enumerate(cols):
        y = X[col].values
        X_others = X.drop(columns=[col]).values

        if len(X_others) == 0 or X_others.shape[1] == 0:
            vif_data.append({"feature": col, "vif": 1.0})
            continue

        reg = LinearRegression().fit(X_others, y)
        r_squared = reg.score(X_others, y)

        vif = 1.0 / (1.0 - r_squared) if r_squared < 1.0 else float("inf")
        vif_data.append({"feature": col, "vif": round(vif, 2)})

    result = pd.DataFrame(vif_data).sort_values("vif", ascending=False)
    logger.info("VIF computed for %d features; %d with VIF > 10",
                len(result), (result["vif"] > 10).sum())
    return result


def cluster_correlated_features(
    df: pd.DataFrame,
    features: List[str],
    threshold: float = 0.85,
) -> List[List[str]]:
    """
    Group features into clusters based on correlation.

    Uses single-linkage: two features are in the same cluster if
    |corr| > threshold. Returns list of clusters (each a list of
    feature names).
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    X = df[features].select_dtypes(include=[np.number])
    corr = X.corr().abs()

    # Convert correlation to distance
    dist_arr = (1 - corr).values.copy()
    np.fill_diagonal(dist_arr, 0)

    # Ensure symmetry and no negatives
    dist_arr = np.clip(dist_arr, 0, None)
    condensed = squareform(dist_arr)

    Z = linkage(condensed, method="single")
    labels = fcluster(Z, t=1 - threshold, criterion="distance")

    clusters = {}
    for feat, label in zip(corr.columns, labels):
        clusters.setdefault(label, []).append(feat)

    result = [v for v in clusters.values() if len(v) > 1]
    logger.info("Found %d correlated clusters", len(result))
    return result
