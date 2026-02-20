"""
Walk-forward ownership prediction model.

Trains Ridge, LightGBM, and RandomForest regressors using expanding-window
cross-validation grouped by event date. All fighters on the same event
stay together in the same fold.
"""

import numpy as np
import pandas as pd
import sqlite3
import logging
from typing import Optional

from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone

from ..config import MIN_TRAINING_EVENTS, HOLDOUT_EVENTS
from .features import NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES

logger = logging.getLogger(__name__)

# Try to import LightGBM; fall back gracefully
try:
    import lightgbm as lgb
    _HAS_LGB = True
except ImportError:
    _HAS_LGB = False
    logger.warning("LightGBM not installed; skipping LGB model.")


# ---------------------------------------------------------------------------
# Preprocessing pipeline builder
# ---------------------------------------------------------------------------
def _build_preprocessor() -> ColumnTransformer:
    """Build a ColumnTransformer that scales numerics and one-hot encodes categoricals."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------
def _get_base_models() -> dict:
    """Return dict of model_name -> sklearn estimator."""
    models = {
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        ),
    }
    if _HAS_LGB:
        models["lightgbm"] = lgb.LGBMRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        )
    return models


def _score_fold(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute per-fold metrics."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
        "mean_pred": float(np.mean(y_pred)),
        "mean_actual": float(np.mean(y_true)),
        "calibration_ratio": float(np.mean(y_pred) / max(np.mean(y_true), 1e-6)),
        "n_samples": len(y_true),
    }


# ---------------------------------------------------------------------------
# OwnershipModel
# ---------------------------------------------------------------------------
class OwnershipModel:
    """Walk-forward ownership prediction model ensemble."""

    def __init__(self):
        self.preprocessor = _build_preprocessor()
        self.base_models = _get_base_models()
        self.fitted_pipelines: dict = {}
        self.cv_results: Optional[dict] = None

    # ----- walk-forward CV ------------------------------------------------
    def walk_forward_cv(
        self,
        df: pd.DataFrame,
        min_train_events: int = MIN_TRAINING_EVENTS,
    ) -> dict:
        """
        Walk-forward cross-validation grouped by event date.

        Parameters
        ----------
        df : pd.DataFrame
            Feature matrix from build_ownership_features().
        min_train_events : int
            Minimum number of distinct date_id values before first test fold.

        Returns
        -------
        dict
            Keys: model names. Values: list of per-fold metric dicts.
        """
        sorted_dates = sorted(df["date_id"].unique())
        results = {name: [] for name in self.base_models}

        for i in range(min_train_events, len(sorted_dates)):
            train_dates = sorted_dates[:i]
            test_date = sorted_dates[i]

            train_mask = df["date_id"].isin(train_dates)
            test_mask = df["date_id"] == test_date

            X_train = df.loc[train_mask, ALL_FEATURES]
            y_train = df.loc[train_mask, "ownership"]
            X_test = df.loc[test_mask, ALL_FEATURES]
            y_test = df.loc[test_mask, "ownership"]

            if len(y_test) == 0:
                continue

            for name, base_est in self.base_models.items():
                pipe = Pipeline([
                    ("prep", clone(self.preprocessor)),
                    ("model", clone(base_est)),
                ])
                pipe.fit(X_train, y_train)
                y_pred = pipe.predict(X_test)
                # Clip predictions to valid ownership range
                y_pred = np.clip(y_pred, 0, 100)
                metrics = _score_fold(y_test.values, y_pred)
                metrics["test_date_id"] = test_date
                metrics["train_events"] = len(train_dates)
                results[name].append(metrics)

        self.cv_results = results
        # Log summary
        for name, folds in results.items():
            if folds:
                avg_mae = np.mean([f["mae"] for f in folds])
                avg_rmse = np.mean([f["rmse"] for f in folds])
                logger.info("%s  avg MAE=%.3f  avg RMSE=%.3f  (%d folds)",
                            name, avg_mae, avg_rmse, len(folds))
        return results

    # ----- fit final model ------------------------------------------------
    def fit_final(
        self,
        df: pd.DataFrame,
        holdout_events: int = HOLDOUT_EVENTS,
    ) -> None:
        """
        Train final models on all data except the last holdout_events dates.

        Parameters
        ----------
        df : pd.DataFrame
            Full feature matrix.
        holdout_events : int
            Number of most-recent events to hold out for final evaluation.
        """
        sorted_dates = sorted(df["date_id"].unique())
        if holdout_events > 0:
            train_dates = sorted_dates[:-holdout_events]
        else:
            train_dates = sorted_dates

        train_mask = df["date_id"].isin(train_dates)
        X_train = df.loc[train_mask, ALL_FEATURES]
        y_train = df.loc[train_mask, "ownership"]

        self.fitted_pipelines = {}
        for name, base_est in self.base_models.items():
            pipe = Pipeline([
                ("prep", clone(self.preprocessor)),
                ("model", clone(base_est)),
            ])
            pipe.fit(X_train, y_train)
            self.fitted_pipelines[name] = pipe
            logger.info("Fitted final %s on %d rows (%d events)",
                        name, len(X_train), len(train_dates))

    # ----- predict -------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict ownership percentages (ensemble average of all fitted models).

        Parameters
        ----------
        X : pd.DataFrame
            Feature columns matching ALL_FEATURES.

        Returns
        -------
        np.ndarray
            Predicted ownership percentages, clipped to [0, 100].
        """
        if not self.fitted_pipelines:
            raise RuntimeError("No fitted models. Call fit_final() first.")

        preds = np.column_stack([
            pipe.predict(X) for pipe in self.fitted_pipelines.values()
        ])
        ensemble = preds.mean(axis=1)
        return np.clip(ensemble, 0, 100)

    # ----- save predictions to DB ----------------------------------------
    def save_predictions(
        self,
        conn: sqlite3.Connection,
        predictions_df: pd.DataFrame,
    ) -> None:
        """
        Write ownership predictions to the ownership_predictions table.

        Parameters
        ----------
        conn : sqlite3.Connection
            Database connection.
        predictions_df : pd.DataFrame
            Must contain columns: player_id, contest_id, date_id,
            full_name, predicted_ownership. Optionally: actual_ownership.
        """
        required = {"player_id", "contest_id", "date_id", "predicted_ownership"}
        missing = required - set(predictions_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        conn.execute("""
            CREATE TABLE IF NOT EXISTS ownership_predictions (
                player_id   INTEGER NOT NULL,
                contest_id  INTEGER NOT NULL,
                date_id     INTEGER NOT NULL,
                full_name   TEXT,
                predicted_ownership REAL NOT NULL,
                actual_ownership    REAL,
                model_version       TEXT DEFAULT "v1",
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (player_id, contest_id)
            )
        """)

        # Upsert via INSERT OR REPLACE
        cols = ["player_id", "contest_id", "date_id", "full_name",
                "predicted_ownership", "actual_ownership"]
        available = [c for c in cols if c in predictions_df.columns]
        predictions_df[available].to_sql(
            "ownership_predictions", conn, if_exists="replace", index=False,
        )
        conn.commit()
        logger.info("Saved %d ownership predictions to DB.", len(predictions_df))
