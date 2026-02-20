"""
Partial pooling across contest types.

Two-stage approach (no PyMC dependency):
1. Train a single pooled model with contest_type as a feature
2. Compute contest-type-specific intercept adjustments with shrinkage
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.base import clone
import logging

logger = logging.getLogger(__name__)


class PartialPoolingModel:
    """
    Partial pooling model that adjusts predictions by contest type.

    Stage 1: Fit a pooled model on all data (contest_type as feature).
    Stage 2: Compute residuals per contest type and shrink intercept
    adjustments toward zero based on sample size.
    """

    def __init__(self, base_model, shrinkage_strength: float = 10.0):
        """
        Parameters
        ----------
        base_model : sklearn estimator or Pipeline
            The pooled model to use as the base.
        shrinkage_strength : float
            Controls how much per-type adjustments shrink toward zero.
            Higher = more shrinkage (more conservative).
        """
        self.base_model = base_model
        self.shrinkage_strength = shrinkage_strength
        self.fitted_model = None
        self.type_adjustments: Dict[str, float] = {}
        self.global_residual_std: float = 0.0

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        contest_types: pd.Series,
    ) -> "PartialPoolingModel":
        """
        Fit the partial pooling model.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (should include contest_type as a feature).
        y : array-like
            Target variable.
        contest_types : pd.Series
            Contest type for each sample ('MME', 'SE', 'limited').
        """
        # Stage 1: Fit pooled model
        self.fitted_model = clone(self.base_model)
        self.fitted_model.fit(X, y)

        # Stage 2: Compute per-type residuals
        y_pred = self.fitted_model.predict(X)
        residuals = np.asarray(y) - y_pred

        self.global_residual_std = residuals.std()

        # Shrunk intercept adjustments per contest type
        self.type_adjustments = {}
        for ct in contest_types.unique():
            mask = contest_types == ct
            n = mask.sum()
            mean_resid = residuals[mask].mean()

            # Shrinkage: adjustment = mean_resid * n / (n + shrinkage_strength)
            shrinkage_factor = n / (n + self.shrinkage_strength)
            self.type_adjustments[ct] = mean_resid * shrinkage_factor

        logger.info("Type adjustments: %s", {
            k: round(v, 4) for k, v in self.type_adjustments.items()
        })

        return self

    def predict(
        self,
        X: pd.DataFrame,
        contest_types: pd.Series,
    ) -> np.ndarray:
        """Predict with contest-type-specific adjustments."""
        if self.fitted_model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        base_pred = self.fitted_model.predict(X)

        adjustments = contest_types.map(self.type_adjustments).fillna(0).values
        adjusted = base_pred + adjustments

        return np.clip(adjusted, 0, 100)

    def get_adjustment_summary(self) -> pd.DataFrame:
        """Return summary of per-type adjustments."""
        return pd.DataFrame([
            {"contest_type": k, "adjustment": v}
            for k, v in self.type_adjustments.items()
        ])
