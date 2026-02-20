"""
Model Training and Selection

Based on:
- Géron: Model comparison, hyperparameter tuning, final evaluation
- Hastie et al.: One standard error rule, regularization
- ISLR: Practical implementation guidance
- López de Prado: Time-series aware validation

Key Principles:
1. Compare multiple models before selecting
2. Use proper cross-validation for time-series
3. Apply one standard error rule (prefer simpler models)
4. Evaluate on test set ONLY ONCE
5. Bootstrap for confidence intervals
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from scipy import stats
import warnings

from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss
)

from .data_structures import PurgedKFold, WalkForwardCV
from .feature_engineering import build_preprocessing_pipeline


@dataclass
class ModelResult:
    """Container for a single model's evaluation results."""
    model_name: str
    cv_scores: np.ndarray
    cv_mean: float
    cv_std: float
    best_params: Optional[Dict] = None
    training_time: float = 0.0

    @property
    def one_se_threshold(self) -> float:
        """One standard error rule: score must beat best - 1 SE."""
        return self.cv_mean - self.cv_std


@dataclass
class FinalResult:
    """Container for final test set evaluation."""
    model_name: str
    test_score: float
    test_metrics: Dict[str, float]
    confidence_interval: Tuple[float, float]
    predictions: np.ndarray = field(default_factory=lambda: np.array([]))


class ModelTrainer:
    """
    Model comparison and selection with time-series awareness.

    Follows Géron's methodology:
    1. Compare multiple models using CV
    2. Select best using one standard error rule
    3. Fine-tune hyperparameters
    4. Evaluate on test set ONCE

    Parameters
    ----------
    task_type : str
        'regression' or 'classification'
    cv_method : str
        'purged' (López de Prado) or 'walkforward' (Hyndman)
    n_splits : int
        Number of CV folds
    random_state : int
        Random seed for reproducibility
    """

    def __init__(
        self,
        task_type: str = 'regression',
        cv_method: str = 'purged',
        n_splits: int = 5,
        random_state: int = 42
    ):
        self.task_type = task_type
        self.cv_method = cv_method
        self.n_splits = n_splits
        self.random_state = random_state

        self._default_models = self._get_default_models()
        self._param_distributions = self._get_param_distributions()

    def _get_default_models(self) -> Dict[str, BaseEstimator]:
        """Get default models for comparison."""
        if self.task_type == 'regression':
            return {
                'ridge': Ridge(random_state=self.random_state),
                'lasso': Lasso(random_state=self.random_state),
                'elastic_net': ElasticNet(random_state=self.random_state),
                'random_forest': RandomForestRegressor(
                    n_estimators=100, random_state=self.random_state, n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingRegressor(
                    n_estimators=100, random_state=self.random_state
                )
            }
        else:
            return {
                'logistic': LogisticRegression(
                    random_state=self.random_state, max_iter=1000
                ),
                'random_forest': RandomForestClassifier(
                    n_estimators=100, random_state=self.random_state, n_jobs=-1
                ),
                'gradient_boosting': GradientBoostingClassifier(
                    n_estimators=100, random_state=self.random_state
                )
            }

    def _get_param_distributions(self) -> Dict[str, Dict]:
        """Get hyperparameter distributions for tuning."""
        if self.task_type == 'regression':
            return {
                'ridge': {
                    'alpha': stats.loguniform(1e-3, 1e3)
                },
                'lasso': {
                    'alpha': stats.loguniform(1e-4, 1e1)
                },
                'elastic_net': {
                    'alpha': stats.loguniform(1e-4, 1e1),
                    'l1_ratio': stats.uniform(0.1, 0.9)
                },
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'gradient_boosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 0.9, 1.0]
                }
            }
        else:
            return {
                'logistic': {
                    'C': stats.loguniform(1e-3, 1e3),
                    'penalty': ['l1', 'l2'],
                    'solver': ['saga']
                },
                'random_forest': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5, 10]
                },
                'gradient_boosting': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7]
                }
            }

    def _get_cv(self, dates: Optional[pd.Series] = None):
        """Get cross-validation splitter."""
        if self.cv_method == 'purged':
            return PurgedKFold(n_splits=self.n_splits)
        else:
            return WalkForwardCV(n_splits=self.n_splits)

    def _get_scorer(self) -> str:
        """Get scoring metric."""
        return 'neg_mean_squared_error' if self.task_type == 'regression' else 'roc_auc'

    def compare_models(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: Optional[pd.Series] = None,
        models: Optional[Dict[str, BaseEstimator]] = None,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None
    ) -> List[ModelResult]:
        """
        Compare multiple models using cross-validation.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        dates : pd.Series, optional
            Dates for temporal ordering
        models : dict, optional
            Models to compare (default: standard set)
        numeric_cols : list, optional
            Numeric columns (auto-detected if None)
        categorical_cols : list, optional
            Categorical columns (auto-detected if None)

        Returns
        -------
        List[ModelResult]
            Results sorted by CV mean score (best first)
        """
        if models is None:
            models = self._default_models

        # Auto-detect column types
        if numeric_cols is None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if categorical_cols is None:
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        cv = self._get_cv(dates)
        scorer = self._get_scorer()
        results = []

        # Build preprocessing pipeline
        preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)

        print(f"\n{'='*60}")
        print(f"MODEL COMPARISON ({self.task_type})")
        print(f"{'='*60}")
        print(f"  Samples: {len(X)}")
        print(f"  Features: {len(X.columns)} ({len(numeric_cols)} numeric, {len(categorical_cols)} categorical)")
        print(f"  CV Method: {self.cv_method} ({self.n_splits} folds)")
        print(f"{'='*60}\n")

        for name, model in models.items():
            import time
            start_time = time.time()

            # Manual CV to handle preprocessing properly
            scores = []
            for train_idx, val_idx in cv.split(X, y, dates):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Fit preprocessing on train only
                X_train_proc = preprocessor.fit_transform(X_train)
                X_val_proc = preprocessor.transform(X_val)

                # Fit model
                model_clone = clone(model)
                model_clone.fit(X_train_proc, y_train)

                # Score
                if self.task_type == 'regression':
                    preds = model_clone.predict(X_val_proc)
                    score = -mean_squared_error(y_val, preds)  # Negative for consistency
                else:
                    if hasattr(model_clone, 'predict_proba'):
                        preds = model_clone.predict_proba(X_val_proc)[:, 1]
                        score = roc_auc_score(y_val, preds)
                    else:
                        preds = model_clone.predict(X_val_proc)
                        score = accuracy_score(y_val, preds)

                scores.append(score)

            elapsed = time.time() - start_time
            scores = np.array(scores)

            result = ModelResult(
                model_name=name,
                cv_scores=scores,
                cv_mean=scores.mean(),
                cv_std=scores.std(),
                training_time=elapsed
            )
            results.append(result)

            # Print result
            if self.task_type == 'regression':
                rmse = np.sqrt(-result.cv_mean)
                print(f"  {name:<20} RMSE: {rmse:.4f} (+/- {np.sqrt(result.cv_std):.4f})")
            else:
                print(f"  {name:<20} AUC: {result.cv_mean:.4f} (+/- {result.cv_std:.4f})")

        # Sort by score (higher is better for our metrics)
        results.sort(key=lambda x: x.cv_mean, reverse=True)

        print(f"\n{'='*60}")
        print("ONE STANDARD ERROR RULE")
        print(f"{'='*60}")
        best = results[0]
        threshold = best.one_se_threshold
        print(f"  Best model: {best.model_name}")
        print(f"  Best score: {best.cv_mean:.4f}")
        print(f"  1 SE threshold: {threshold:.4f}")

        # Find simplest model within 1 SE
        simple_order = ['ridge', 'lasso', 'logistic', 'elastic_net',
                        'random_forest', 'gradient_boosting']
        for simple_name in simple_order:
            for r in results:
                if r.model_name == simple_name and r.cv_mean >= threshold:
                    if r.model_name != best.model_name:
                        print(f"  -> Recommend: {r.model_name} (simpler, within 1 SE)")
                    break

        return results

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_name: str,
        dates: Optional[pd.Series] = None,
        n_iter: int = 20,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None
    ) -> Tuple[BaseEstimator, Dict]:
        """
        Tune hyperparameters using RandomizedSearchCV.

        Parameters
        ----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        model_name : str
            Name of model to tune
        dates : pd.Series, optional
            Dates for temporal ordering
        n_iter : int
            Number of random parameter combinations
        numeric_cols, categorical_cols : list
            Column types

        Returns
        -------
        Tuple[BaseEstimator, Dict]
            Best model and best parameters
        """
        from sklearn.model_selection import TimeSeriesSplit

        if model_name not in self._default_models:
            raise ValueError(f"Unknown model: {model_name}")

        model = self._default_models[model_name]
        param_dist = self._param_distributions.get(model_name, {})

        if not param_dist:
            print(f"No hyperparameters to tune for {model_name}")
            return model, {}

        # Auto-detect columns
        if numeric_cols is None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if categorical_cols is None:
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Preprocess data once
        preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
        X_processed = preprocessor.fit_transform(X)

        # Use sklearn's TimeSeriesSplit for hyperparameter tuning
        # (more compatible with RandomizedSearchCV)
        cv = TimeSeriesSplit(n_splits=self.n_splits)
        scorer = self._get_scorer()

        print(f"\n{'='*60}")
        print(f"HYPERPARAMETER TUNING: {model_name}")
        print(f"{'='*60}")
        print(f"  Iterations: {n_iter}")

        search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=cv,
            scoring=scorer,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_processed, y)

        print(f"  Best params: {search.best_params_}")
        if self.task_type == 'regression':
            print(f"  Best RMSE: {np.sqrt(-search.best_score_):.4f}")
        else:
            print(f"  Best AUC: {search.best_score_:.4f}")

        return search.best_estimator_, search.best_params_

    def final_evaluation(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model: BaseEstimator,
        numeric_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        confidence: float = 0.95,
        n_bootstrap: int = 1000
    ) -> FinalResult:
        """
        Final evaluation on held-out test set.

        IMPORTANT: This should be called ONLY ONCE per project.
        Multiple test set evaluations lead to overfitting.

        Parameters
        ----------
        X_train, y_train : pd.DataFrame, pd.Series
            Training data (for fitting)
        X_test, y_test : pd.DataFrame, pd.Series
            Test data (for evaluation)
        model : BaseEstimator
            Model to evaluate
        numeric_cols, categorical_cols : list
            Column types
        confidence : float
            Confidence level for intervals
        n_bootstrap : int
            Bootstrap iterations

        Returns
        -------
        FinalResult
            Test results with confidence intervals
        """
        # Auto-detect columns
        if numeric_cols is None:
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        if categorical_cols is None:
            categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

        # Build and fit preprocessor
        preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)

        # Train final model
        model.fit(X_train_proc, y_train)

        # Predictions
        if self.task_type == 'regression':
            predictions = model.predict(X_test_proc)
            errors = predictions - y_test.values

            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                'mae': mean_absolute_error(y_test, predictions),
                'r2': r2_score(y_test, predictions),
                'mape': np.mean(np.abs(errors / (y_test.values + 1e-8))) * 100
            }

            # Bootstrap confidence interval for RMSE
            ci = self._bootstrap_ci(errors, metric='rmse', confidence=confidence, n_bootstrap=n_bootstrap)
            test_score = metrics['rmse']

        else:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_test_proc)[:, 1]
            else:
                proba = model.predict(X_test_proc)

            predictions = (proba >= 0.5).astype(int)

            metrics = {
                'accuracy': accuracy_score(y_test, predictions),
                'precision': precision_score(y_test, predictions, zero_division=0),
                'recall': recall_score(y_test, predictions, zero_division=0),
                'f1': f1_score(y_test, predictions, zero_division=0),
                'auc': roc_auc_score(y_test, proba) if len(np.unique(y_test)) > 1 else 0.5,
                'log_loss': log_loss(y_test, proba)
            }

            ci = self._bootstrap_ci(
                np.column_stack([y_test.values, proba]),
                metric='auc', confidence=confidence, n_bootstrap=n_bootstrap
            )
            test_score = metrics['auc']

        # Print results
        print(f"\n{'='*60}")
        print("FINAL TEST SET EVALUATION")
        print(f"{'='*60}")
        print(f"  Test samples: {len(y_test)}")

        if self.task_type == 'regression':
            print(f"\n  RMSE: {metrics['rmse']:.4f} ({ci[0]:.4f}, {ci[1]:.4f})")
            print(f"  MAE:  {metrics['mae']:.4f}")
            print(f"  R²:   {metrics['r2']:.4f}")
            print(f"  MAPE: {metrics['mape']:.2f}%")
        else:
            print(f"\n  AUC:       {metrics['auc']:.4f} ({ci[0]:.4f}, {ci[1]:.4f})")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  F1:        {metrics['f1']:.4f}")

        return FinalResult(
            model_name=type(model).__name__,
            test_score=test_score,
            test_metrics=metrics,
            confidence_interval=ci,
            predictions=predictions
        )

    def _bootstrap_ci(
        self,
        data: np.ndarray,
        metric: str,
        confidence: float,
        n_bootstrap: int
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval."""
        rng = np.random.default_rng(self.random_state)
        n = len(data)
        boot_scores = []

        for _ in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)

            if metric == 'rmse':
                errors = data[idx]
                score = np.sqrt(np.mean(errors ** 2))
            elif metric == 'auc':
                y_true = data[idx, 0]
                y_pred = data[idx, 1]
                if len(np.unique(y_true)) > 1:
                    score = roc_auc_score(y_true, y_pred)
                else:
                    score = 0.5
            else:
                score = np.mean(data[idx])

            boot_scores.append(score)

        alpha = (1 - confidence) / 2
        return (
            np.percentile(boot_scores, alpha * 100),
            np.percentile(boot_scores, (1 - alpha) * 100)
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def one_standard_error_rule(
    results: List[ModelResult],
    model_complexity: Dict[str, int]
) -> str:
    """
    Apply one standard error rule to select model.

    From Hastie et al.: Choose the simplest model whose
    error is within one standard error of the best.

    Parameters
    ----------
    results : List[ModelResult]
        Model comparison results
    model_complexity : Dict[str, int]
        Complexity scores (lower = simpler)

    Returns
    -------
    str
        Name of selected model
    """
    # Find best score
    best = max(results, key=lambda x: x.cv_mean)
    threshold = best.cv_mean - best.cv_std

    # Find simplest model within 1 SE
    candidates = [r for r in results if r.cv_mean >= threshold]

    if not candidates:
        return best.model_name

    # Sort by complexity
    candidates.sort(key=lambda x: model_complexity.get(x.model_name, 999))

    return candidates[0].model_name


def compute_deflated_sharpe(
    sharpe: float,
    n_trials: int,
    skew: float = 0,
    kurtosis: float = 3
) -> float:
    """
    Compute deflated Sharpe ratio (López de Prado).

    Adjusts for multiple testing. A Sharpe of 2.0 after
    testing 100 strategies is very different from after testing 1.

    Parameters
    ----------
    sharpe : float
        Reported Sharpe ratio
    n_trials : int
        Number of strategy variations tested
    skew : float
        Return skewness
    kurtosis : float
        Return kurtosis

    Returns
    -------
    float
        Deflated (realistic) Sharpe ratio
    """
    # Expected maximum Sharpe under null (no skill)
    e_max_sharpe = (1 - np.euler_gamma) * stats.norm.ppf(1 - 1/n_trials) + \
                   np.euler_gamma * stats.norm.ppf(1 - 1/(n_trials * np.e))

    # Standard error adjustment
    se = np.sqrt((1 + 0.5 * sharpe**2 - skew * sharpe +
                  (kurtosis - 3) / 4 * sharpe**2))

    # Deflated Sharpe
    deflated = stats.norm.cdf((sharpe - e_max_sharpe) / se)

    return deflated
