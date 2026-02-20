"""Permutation importance with fold stability analysis."""

import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
import logging

logger = logging.getLogger(__name__)


def fold_stable_importance(
    models: List,
    X_tests: List[pd.DataFrame],
    y_tests: List[np.ndarray],
    feature_names: List[str],
    n_repeats: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute permutation importance across multiple CV folds and assess stability.

    Parameters
    ----------
    models : list of fitted sklearn estimators/pipelines
        One per fold.
    X_tests : list of pd.DataFrame
        Test feature matrices, one per fold.
    y_tests : list of array-like
        Test targets, one per fold.
    feature_names : list of str
        Feature names (must match X_tests columns).
    n_repeats : int
        Number of permutation repeats per fold.
    seed : int
        Random seed.

    Returns
    -------
    pd.DataFrame with columns:
        feature, mean_importance, std_importance,
        mean_rank, std_rank, rank_stability (1 = perfectly stable)
    """
    all_importances = []

    for fold_idx, (model, X_test, y_test) in enumerate(zip(models, X_tests, y_tests)):
        result = permutation_importance(
            model, X_test, y_test,
            scoring="neg_mean_absolute_error",
            n_repeats=n_repeats,
            random_state=seed + fold_idx,
        )

        fold_imp = pd.DataFrame({
            "feature": feature_names,
            "importance": result.importances_mean,
            "fold": fold_idx,
        })
        fold_imp["rank"] = fold_imp["importance"].rank(ascending=False)
        all_importances.append(fold_imp)

    combined = pd.concat(all_importances, ignore_index=True)

    summary = combined.groupby("feature").agg(
        mean_importance=("importance", "mean"),
        std_importance=("importance", "std"),
        mean_rank=("rank", "mean"),
        std_rank=("rank", "std"),
    ).reset_index()

    # Rank stability: 1 - normalized std of rank
    n_features = len(feature_names)
    summary["rank_stability"] = 1 - (summary["std_rank"] / n_features)
    summary = summary.sort_values("mean_importance", ascending=False)

    logger.info("Top 5 stable features: %s",
                summary.head(5)["feature"].tolist())

    return summary
