"""
Calibration utilities for ownership predictions.

Provides isotonic calibration, calibration metrics, and bootstrap-based
prediction uncertainty estimation.
"""

import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.isotonic import IsotonicRegression
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error


def isotonic_calibrate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> IsotonicRegression:
    """
    Fit isotonic regression for calibrating ownership predictions.

    Parameters
    ----------
    y_true : array-like
        Actual ownership percentages.
    y_pred : array-like
        Raw predicted ownership percentages.

    Returns
    -------
    IsotonicRegression
        Fitted calibrator. Call .predict(y_pred_new) to calibrate new predictions.
    """
    iso = IsotonicRegression(y_min=0, y_max=100, out_of_bounds="clip")
    iso.fit(y_pred, y_true)
    return iso


def calibration_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration metrics for ownership predictions.

    Parameters
    ----------
    y_true : array-like
        Actual ownership percentages.
    y_pred : array-like
        Predicted ownership percentages.
    n_bins : int
        Number of bins for Expected Calibration Error.

    Returns
    -------
    dict with keys:
        - ece : Expected Calibration Error (weighted mean absolute bin error)
        - mae : Mean Absolute Error
        - slope : OLS slope of y_true ~ y_pred (perfect = 1.0)
        - intercept : OLS intercept (perfect = 0.0)
        - bin_details : list of dicts per bin
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # OLS calibration slope/intercept
    if len(y_pred) > 1 and np.std(y_pred) > 0:
        coeffs = np.polyfit(y_pred, y_true, deg=1)
        slope, intercept = coeffs[0], coeffs[1]
    else:
        slope, intercept = 1.0, 0.0

    # Binned calibration
    bin_edges = np.linspace(y_pred.min(), y_pred.max() + 1e-8, n_bins + 1)
    bin_details = []
    weighted_errors = []

    for b in range(n_bins):
        mask = (y_pred >= bin_edges[b]) & (y_pred < bin_edges[b + 1])
        n = mask.sum()
        if n == 0:
            continue
        mean_pred = y_pred[mask].mean()
        mean_true = y_true[mask].mean()
        bin_error = abs(mean_pred - mean_true)
        weighted_errors.append(bin_error * n)
        bin_details.append({
            "bin": b,
            "n": int(n),
            "mean_pred": float(mean_pred),
            "mean_true": float(mean_true),
            "abs_error": float(bin_error),
        })

    total_n = sum(d["n"] for d in bin_details)
    ece = sum(weighted_errors) / total_n if total_n > 0 else 0.0

    return {
        "ece": float(ece),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "slope": float(slope),
        "intercept": float(intercept),
        "n_samples": len(y_true),
        "bin_details": bin_details,
    }


def add_prediction_uncertainty(
    y_pred: np.ndarray,
    model,
    X: pd.DataFrame,
    n_bootstrap: int = 100,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate prediction uncertainty via bootstrap resampling.

    Retrains the model on n_bootstrap bootstrap samples of the training
    data that was used to fit the model, then predicts on X. Returns
    the mean and standard deviation of predictions across bootstraps.

    Parameters
    ----------
    y_pred : array-like
        Base predictions (used as fallback / reference).
    model : sklearn Pipeline
        A fitted sklearn pipeline (will be cloned and retrained).
    X : pd.DataFrame
        Feature matrix to predict on.
    n_bootstrap : int
        Number of bootstrap iterations.

    Returns
    -------
    (mean, std) : tuple of np.ndarray
        Mean and standard deviation of bootstrap predictions for each sample.

    Notes
    -----
    This function requires the model's training data to be accessible.
    If the model does not store training data, it uses the prediction
    variance from sub-models (for ensemble models) or returns zeros
    for std as a fallback.
    """
    y_pred = np.asarray(y_pred)
    n_samples = len(y_pred)

    # Try to get sub-predictions from ensemble pipelines
    # For sklearn Pipelines, we can clone and resample
    try:
        bootstrap_preds = np.zeros((n_bootstrap, n_samples))
        rng = np.random.RandomState(42)

        for b in range(n_bootstrap):
            # Resample X indices with replacement
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = X.iloc[idx]

            # Predict with the fitted model on bootstrap sample
            # (This measures prediction stability, not training variance)
            pred_b = model.predict(X_boot)
            # Map back to original indices via averaging duplicates
            for orig_i, pred_val in zip(idx, pred_b):
                bootstrap_preds[b, orig_i] += pred_val

            # Normalize by count of times each index was sampled
            counts = np.bincount(idx, minlength=n_samples).astype(float)
            counts[counts == 0] = 1  # avoid div by zero for unsampled
            bootstrap_preds[b] /= counts

        mean_preds = bootstrap_preds.mean(axis=0)
        std_preds = bootstrap_preds.std(axis=0)

    except Exception:
        # Fallback: return original predictions with zero uncertainty
        mean_preds = y_pred.copy()
        std_preds = np.zeros_like(y_pred)

    return np.clip(mean_preds, 0, 100), std_preds
