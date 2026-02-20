"""Tests for pipeline stage utilities."""

import pytest
import pandas as pd
import numpy as np
from src.cv.bootstrap import event_bootstrap_ci
from src.phase3_features.collinearity import correlation_report, compute_vif


class TestBootstrapCI:
    def test_returns_dict_with_ci(self):
        y_true = np.random.randn(50)
        y_pred = y_true + np.random.randn(50) * 0.1
        event_ids = np.repeat(np.arange(10), 5)

        result = event_bootstrap_ci(y_true, y_pred, event_ids, n_bootstrap=100)
        assert "ci_lower" in result
        assert "ci_upper" in result
        assert "point_estimate" in result
        assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]

    def test_ci_width_reasonable(self):
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.5
        event_ids = np.repeat(np.arange(20), 5)

        result = event_bootstrap_ci(y_true, y_pred, event_ids, n_bootstrap=500)
        ci_width = result["ci_upper"] - result["ci_lower"]
        assert ci_width > 0
        assert ci_width < 5  # shouldn't be absurdly wide


class TestCorrelationReport:
    def test_finds_correlated_pair(self):
        df = pd.DataFrame({
            "a": range(100),
            "b": range(100),
            "c": np.random.randn(100),
        })
        report = correlation_report(df, ["a", "b", "c"], threshold=0.85)
        assert len(report) >= 1
        assert "a" in report["feature_a"].values or "a" in report["feature_b"].values

    def test_no_pairs_below_threshold(self):
        rng = np.random.RandomState(42)
        df = pd.DataFrame({
            "a": rng.randn(100),
            "b": rng.randn(100),
            "c": rng.randn(100),
        })
        report = correlation_report(df, ["a", "b", "c"], threshold=0.99)
        assert len(report) == 0


class TestVIF:
    def test_collinear_features_high_vif(self):
        df = pd.DataFrame({
            "a": range(100),
            "b": [x * 2 + 1 for x in range(100)],
            "c": np.random.randn(100),
        })
        vif_df = compute_vif(df, ["a", "b", "c"])
        a_vif = vif_df.loc[vif_df["feature"] == "a", "vif"].iloc[0]
        assert a_vif > 10
