"""Event-level bootstrap confidence intervals."""

import numpy as np
import pandas as pd
from typing import Callable


def event_bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    event_ids: np.ndarray,
    metric_fn: Callable = None,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Bootstrap confidence intervals by resampling whole events.

    Parameters
    ----------
    y_true, y_pred : arrays
        True and predicted values.
    event_ids : array
        Event identifier for each sample (e.g., date_id).
    metric_fn : callable, optional
        Function(y_true, y_pred) -> float. Defaults to MAE.
    n_bootstrap : int
        Number of bootstrap iterations.
    ci : float
        Confidence level (0-1).
    seed : int
        Random seed.

    Returns
    -------
    dict with keys: point_estimate, ci_lower, ci_upper, std, bootstrap_values
    """
    if metric_fn is None:
        metric_fn = lambda yt, yp: float(np.mean(np.abs(yt - yp)))

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    event_ids = np.asarray(event_ids)

    unique_events = np.unique(event_ids)
    rng = np.random.RandomState(seed)

    point_estimate = metric_fn(y_true, y_pred)

    bootstrap_values = []
    alpha = (1 - ci) / 2

    for _ in range(n_bootstrap):
        sampled_events = rng.choice(unique_events, size=len(unique_events), replace=True)
        mask = np.isin(event_ids, sampled_events)
        if mask.sum() == 0:
            continue
        val = metric_fn(y_true[mask], y_pred[mask])
        bootstrap_values.append(val)

    bootstrap_values = np.array(bootstrap_values)

    return {
        "point_estimate": point_estimate,
        "ci_lower": float(np.percentile(bootstrap_values, alpha * 100)),
        "ci_upper": float(np.percentile(bootstrap_values, (1 - alpha) * 100)),
        "std": float(bootstrap_values.std()),
        "bootstrap_values": bootstrap_values,
    }
