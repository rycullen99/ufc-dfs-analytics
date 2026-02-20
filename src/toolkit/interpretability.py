"""
Model Interpretability Tools

Based on:
- Molnar: Interpretable Machine Learning
- Domain knowledge should validate features
- Black box models are dangerous in high-stakes domains

Key Methods:
| Method | Scope | Use Case |
|--------|-------|----------|
| SHAP | Local/Global | Fair attribution, individual predictions |
| LIME | Local | Any model, lay person explanations |
| Permutation FI | Global | Feature rankings |
| PDP | Global | Marginal effects (uncorrelated features) |
| ALE | Global | Marginal effects (correlated features) |

Principles:
1. Understand what your model learned
2. Domain knowledge should validate features
3. Unexpected important features = potential data leakage
4. Interpretability enables improvement
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union, Callable
from dataclasses import dataclass
import warnings


@dataclass
class FeatureImportance:
    """Container for feature importance results."""
    feature_names: List[str]
    importances: np.ndarray
    std: Optional[np.ndarray] = None
    method: str = "unknown"

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to sorted DataFrame."""
        df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.importances
        })
        if self.std is not None:
            df['std'] = self.std
        return df.sort_values('importance', ascending=False).reset_index(drop=True)

    def top_n(self, n: int = 10) -> pd.DataFrame:
        """Get top N most important features."""
        return self.to_dataframe().head(n)


@dataclass
class LocalExplanation:
    """Container for local (single prediction) explanation."""
    feature_names: List[str]
    contributions: np.ndarray  # Feature contributions to prediction
    base_value: float  # Expected/baseline prediction
    prediction: float  # Actual prediction
    instance_values: np.ndarray  # Feature values for this instance

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame sorted by absolute contribution."""
        df = pd.DataFrame({
            'feature': self.feature_names,
            'value': self.instance_values,
            'contribution': self.contributions
        })
        df['abs_contribution'] = np.abs(df['contribution'])
        return df.sort_values('abs_contribution', ascending=False).reset_index(drop=True)


class ModelInterpreter:
    """
    Model interpretability tools.

    Provides methods for understanding what models learned
    and validating that features make domain sense.

    Parameters
    ----------
    random_state : int
        Random seed for reproducibility
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    """

    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        self.random_state = random_state
        self.n_jobs = n_jobs

    # =========================================================================
    # PERMUTATION IMPORTANCE
    # =========================================================================

    def permutation_importance(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str = 'rmse',
        n_repeats: int = 10,
        sample_weight: Optional[np.ndarray] = None
    ) -> FeatureImportance:
        """
        Calculate permutation feature importance.

        Measures how much performance degrades when a feature is shuffled.
        Model-agnostic and works with any sklearn-compatible model.

        Parameters
        ----------
        model : fitted model
            Trained model with predict method
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        metric : str
            'rmse', 'mae', 'r2' for regression; 'accuracy', 'auc' for classification
        n_repeats : int
            Number of times to permute each feature
        sample_weight : np.ndarray, optional
            Sample weights for metric calculation

        Returns
        -------
        FeatureImportance
            Feature importance scores
        """
        rng = np.random.default_rng(self.random_state)

        # Get baseline score
        baseline_score = self._score_model(model, X, y, metric, sample_weight)

        importances = []
        importances_std = []

        for col in X.columns:
            col_scores = []

            for _ in range(n_repeats):
                # Permute column
                X_permuted = X.copy()
                X_permuted[col] = rng.permutation(X_permuted[col].values)

                # Score with permuted feature
                permuted_score = self._score_model(
                    model, X_permuted, y, metric, sample_weight
                )

                # Importance = decrease in performance
                col_scores.append(baseline_score - permuted_score)

            importances.append(np.mean(col_scores))
            importances_std.append(np.std(col_scores))

        return FeatureImportance(
            feature_names=list(X.columns),
            importances=np.array(importances),
            std=np.array(importances_std),
            method='permutation'
        )

    def _score_model(
        self,
        model,
        X: pd.DataFrame,
        y: pd.Series,
        metric: str,
        sample_weight: Optional[np.ndarray] = None
    ) -> float:
        """Score model with given metric (higher is better)."""
        predictions = model.predict(X)

        if metric == 'rmse':
            return -np.sqrt(np.average((y - predictions) ** 2, weights=sample_weight))
        elif metric == 'mae':
            return -np.average(np.abs(y - predictions), weights=sample_weight)
        elif metric == 'r2':
            from sklearn.metrics import r2_score
            return r2_score(y, predictions, sample_weight=sample_weight)
        elif metric == 'accuracy':
            from sklearn.metrics import accuracy_score
            return accuracy_score(y, predictions, sample_weight=sample_weight)
        elif metric == 'auc':
            from sklearn.metrics import roc_auc_score
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X)[:, 1]
            else:
                proba = predictions
            return roc_auc_score(y, proba, sample_weight=sample_weight)
        else:
            raise ValueError(f"Unknown metric: {metric}")

    # =========================================================================
    # SHAP VALUES
    # =========================================================================

    def shap_values(
        self,
        model,
        X: pd.DataFrame,
        background: Optional[pd.DataFrame] = None,
        n_background: int = 100
    ) -> Tuple[np.ndarray, float]:
        """
        Calculate SHAP values for feature attribution.

        SHAP (SHapley Additive exPlanations) provides fair attribution
        based on game theory. Each feature gets credit proportional
        to its contribution across all possible coalitions.

        Parameters
        ----------
        model : fitted model
            Trained model
        X : pd.DataFrame
            Instances to explain
        background : pd.DataFrame, optional
            Background dataset for SHAP (default: sample from X)
        n_background : int
            Number of background samples to use

        Returns
        -------
        Tuple[np.ndarray, float]
            SHAP values array and expected value
        """
        try:
            import shap
        except ImportError:
            raise ImportError("SHAP not installed. Run: pip install shap")

        # Select background data
        if background is None:
            if len(X) > n_background:
                background = X.sample(n_background, random_state=self.random_state)
            else:
                background = X

        # Create explainer based on model type
        if hasattr(model, 'feature_importances_'):
            # Tree-based model
            explainer = shap.TreeExplainer(model, data=background)
        else:
            # Linear or other model
            explainer = shap.KernelExplainer(
                model.predict, background, link='identity'
            )

        # Calculate SHAP values
        shap_values = explainer.shap_values(X)
        expected_value = explainer.expected_value

        # Handle multi-output
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # For binary classification
        if isinstance(expected_value, np.ndarray):
            expected_value = expected_value[1]

        return shap_values, expected_value

    def shap_global_importance(
        self,
        model,
        X: pd.DataFrame,
        **kwargs
    ) -> FeatureImportance:
        """
        Get global feature importance from SHAP values.

        Uses mean absolute SHAP value per feature.
        """
        shap_values, _ = self.shap_values(model, X, **kwargs)

        importances = np.abs(shap_values).mean(axis=0)

        return FeatureImportance(
            feature_names=list(X.columns),
            importances=importances,
            method='shap'
        )

    def shap_local_explanation(
        self,
        model,
        X: pd.DataFrame,
        instance_idx: int,
        **kwargs
    ) -> LocalExplanation:
        """
        Get local explanation for a single prediction.

        Parameters
        ----------
        model : fitted model
            Trained model
        X : pd.DataFrame
            Features dataset
        instance_idx : int
            Index of instance to explain

        Returns
        -------
        LocalExplanation
            Contribution of each feature to this prediction
        """
        shap_values, expected_value = self.shap_values(model, X, **kwargs)

        instance = X.iloc[instance_idx]
        prediction = model.predict(X.iloc[[instance_idx]])[0]

        return LocalExplanation(
            feature_names=list(X.columns),
            contributions=shap_values[instance_idx],
            base_value=expected_value,
            prediction=prediction,
            instance_values=instance.values
        )

    # =========================================================================
    # PARTIAL DEPENDENCE PLOTS
    # =========================================================================

    def partial_dependence(
        self,
        model,
        X: pd.DataFrame,
        feature: str,
        grid_resolution: int = 50,
        percentile_range: Tuple[float, float] = (0.05, 0.95)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Partial Dependence Plot data.

        PDP shows the marginal effect of a feature on predictions,
        averaging over all other features.

        WARNING: PDP can be misleading with correlated features.
        Use ALE for correlated features instead.

        Parameters
        ----------
        model : fitted model
            Trained model
        X : pd.DataFrame
            Features
        feature : str
            Feature to analyze
        grid_resolution : int
            Number of points in the grid
        percentile_range : tuple
            Range of feature values to plot

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Grid values and mean predictions
        """
        # Create grid
        feature_values = X[feature].values
        lower = np.percentile(feature_values, percentile_range[0] * 100)
        upper = np.percentile(feature_values, percentile_range[1] * 100)
        grid = np.linspace(lower, upper, grid_resolution)

        # Calculate marginal predictions
        pdp_values = []
        for val in grid:
            X_modified = X.copy()
            X_modified[feature] = val
            preds = model.predict(X_modified)
            pdp_values.append(np.mean(preds))

        return grid, np.array(pdp_values)

    def partial_dependence_2d(
        self,
        model,
        X: pd.DataFrame,
        feature1: str,
        feature2: str,
        grid_resolution: int = 25
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate 2D Partial Dependence for feature interactions.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Grid for feature1, grid for feature2, and prediction matrix
        """
        grid1 = np.linspace(X[feature1].min(), X[feature1].max(), grid_resolution)
        grid2 = np.linspace(X[feature2].min(), X[feature2].max(), grid_resolution)

        pdp_matrix = np.zeros((grid_resolution, grid_resolution))

        for i, v1 in enumerate(grid1):
            for j, v2 in enumerate(grid2):
                X_modified = X.copy()
                X_modified[feature1] = v1
                X_modified[feature2] = v2
                preds = model.predict(X_modified)
                pdp_matrix[i, j] = np.mean(preds)

        return grid1, grid2, pdp_matrix

    # =========================================================================
    # ACCUMULATED LOCAL EFFECTS (ALE)
    # =========================================================================

    def ale(
        self,
        model,
        X: pd.DataFrame,
        feature: str,
        n_bins: int = 50
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Accumulated Local Effects.

        ALE is preferred over PDP when features are correlated.
        It measures the effect of a feature while accounting for
        correlations with other features.

        Parameters
        ----------
        model : fitted model
            Trained model
        X : pd.DataFrame
            Features
        feature : str
            Feature to analyze
        n_bins : int
            Number of bins for feature

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Bin centers and ALE values
        """
        feature_values = X[feature].values

        # Create bins based on quantiles
        quantiles = np.quantile(feature_values, np.linspace(0, 1, n_bins + 1))
        quantiles = np.unique(quantiles)  # Remove duplicates

        if len(quantiles) < 3:
            warnings.warn(f"Feature {feature} has too few unique values for ALE")
            return np.array([]), np.array([])

        # Calculate local effects in each bin
        local_effects = []
        bin_centers = []

        for i in range(len(quantiles) - 1):
            lower, upper = quantiles[i], quantiles[i + 1]

            # Get samples in this bin
            mask = (feature_values >= lower) & (feature_values < upper)
            if i == len(quantiles) - 2:
                mask = (feature_values >= lower) & (feature_values <= upper)

            X_bin = X[mask].copy()

            if len(X_bin) == 0:
                continue

            # Predict at bin edges
            X_lower = X_bin.copy()
            X_lower[feature] = lower
            X_upper = X_bin.copy()
            X_upper[feature] = upper

            pred_lower = model.predict(X_lower)
            pred_upper = model.predict(X_upper)

            # Local effect is average difference
            local_effect = np.mean(pred_upper - pred_lower)
            local_effects.append(local_effect)
            bin_centers.append((lower + upper) / 2)

        # Accumulate and center
        ale_values = np.cumsum(local_effects)
        ale_values = ale_values - np.mean(ale_values)  # Center at 0

        return np.array(bin_centers), ale_values

    # =========================================================================
    # FEATURE INTERACTION DETECTION
    # =========================================================================

    def detect_interactions(
        self,
        model,
        X: pd.DataFrame,
        top_n: int = 10,
        method: str = 'friedman_h'
    ) -> pd.DataFrame:
        """
        Detect feature interactions.

        Uses Friedman's H-statistic to measure interaction strength.
        H=0 means no interaction, H=1 means complete interaction.

        Parameters
        ----------
        model : fitted model
            Trained model
        X : pd.DataFrame
            Features
        top_n : int
            Number of top interactions to return
        method : str
            'friedman_h' (only option currently)

        Returns
        -------
        pd.DataFrame
            Feature pairs with interaction strength
        """
        from itertools import combinations

        features = X.columns.tolist()
        interactions = []

        for f1, f2 in combinations(features, 2):
            h_stat = self._friedman_h_statistic(model, X, f1, f2)
            interactions.append({
                'feature1': f1,
                'feature2': f2,
                'h_statistic': h_stat
            })

        df = pd.DataFrame(interactions)
        return df.nlargest(top_n, 'h_statistic')

    def _friedman_h_statistic(
        self,
        model,
        X: pd.DataFrame,
        feature1: str,
        feature2: str,
        n_samples: int = 100
    ) -> float:
        """Calculate Friedman's H-statistic for interaction."""
        # Sample for efficiency
        if len(X) > n_samples:
            X_sample = X.sample(n_samples, random_state=self.random_state)
        else:
            X_sample = X

        # Get partial dependences
        _, pd1 = self.partial_dependence(model, X_sample, feature1, grid_resolution=20)
        _, pd2 = self.partial_dependence(model, X_sample, feature2, grid_resolution=20)
        _, _, pd12 = self.partial_dependence_2d(model, X_sample, feature1, feature2, grid_resolution=20)

        # H-statistic (simplified)
        var_total = np.var(pd12)
        var_additive = np.var(pd1) + np.var(pd2)

        if var_total == 0:
            return 0.0

        h = (var_total - var_additive) / var_total
        return max(0, h)

    # =========================================================================
    # VALIDATION AND SANITY CHECKS
    # =========================================================================

    def validate_feature_importance(
        self,
        importance: FeatureImportance,
        expected_important: List[str],
        expected_unimportant: List[str],
        threshold_important: float = 0.05,
        threshold_unimportant: float = 0.01
    ) -> Dict[str, List[str]]:
        """
        Validate feature importance against domain knowledge.

        Flags:
        - Expected important features that aren't
        - Unexpected features that are highly important
        - Potential data leakage indicators

        Parameters
        ----------
        importance : FeatureImportance
            Computed importance scores
        expected_important : list
            Features you expect to be important (domain knowledge)
        expected_unimportant : list
            Features that shouldn't be important
        threshold_important : float
            Minimum importance for "important" features
        threshold_unimportant : float
            Maximum importance for "unimportant" features

        Returns
        -------
        Dict[str, List[str]]
            Validation flags
        """
        df = importance.to_dataframe()
        df['importance_normalized'] = df['importance'] / df['importance'].sum()

        flags = {
            'missing_expected': [],
            'unexpected_important': [],
            'potential_leakage': []
        }

        for feat in expected_important:
            if feat in df['feature'].values:
                imp = df.loc[df['feature'] == feat, 'importance_normalized'].values[0]
                if imp < threshold_important:
                    flags['missing_expected'].append(f"{feat} (importance={imp:.3f})")

        for feat in expected_unimportant:
            if feat in df['feature'].values:
                imp = df.loc[df['feature'] == feat, 'importance_normalized'].values[0]
                if imp > threshold_unimportant:
                    flags['unexpected_important'].append(f"{feat} (importance={imp:.3f})")

        # Check for potential leakage (features correlated with target)
        top_features = df.head(5)['feature'].tolist()
        leakage_keywords = ['actual', 'result', 'outcome', 'final', 'true', 'target']
        for feat in top_features:
            if any(kw in feat.lower() for kw in leakage_keywords):
                flags['potential_leakage'].append(feat)

        return flags

    def print_importance_report(
        self,
        importance: FeatureImportance,
        top_n: int = 15
    ):
        """Print formatted feature importance report."""
        print(f"\n{'='*60}")
        print(f"FEATURE IMPORTANCE ({importance.method.upper()})")
        print(f"{'='*60}")

        df = importance.to_dataframe()
        total = df['importance'].sum()

        for i, row in df.head(top_n).iterrows():
            pct = row['importance'] / total * 100
            bar = '█' * int(pct / 2)
            print(f"  {row['feature']:<30} {pct:>5.1f}% {bar}")

        print(f"\n  ... and {len(df) - top_n} more features")
