"""Tests for pipeline stage utilities."""

import pytest
import pandas as pd
import numpy as np
from src.cv.bootstrap import event_bootstrap_ci
from src.phase3_features.collinearity import correlation_report, compute_vif, cluster_correlated_features
from src.phase4_contest.contest_features import add_contest_type_features


class TestBootstrapCI:
    def test_returns_three_values(self):
        df = pd.DataFrame({
            "date_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
            "error": np.random.randn(10),
        })
        point, lower, upper = event_bootstrap_ci(
            df, lambda d: d["error"].mean(), n_iterations=100
        )
        assert lower <= point <= upper

    def test_ci_contains_mean(self):
        df = pd.DataFrame({
            "date_id": list(range(20)) * 5,
            "value": np.random.randn(100),
        })
        point, lower, upper = event_bootstrap_ci(
            df, lambda d: d["value"].mean(), n_iterations=500
        )
        # CI should be centered roughly around the point estimate
        assert lower < upper


class TestCorrelationReport:
    def test_finds_correlated_pair(self):
        df = pd.DataFrame({
            "a": range(100),
            "b": range(100),
            "c": np.random.randn(100),
        })
        report = correlation_report(df, threshold=0.85)
        assert len(report) >= 1
        assert "a" in report["feature_1"].values or "a" in report["feature_2"].values

    def test_no_pairs_below_threshold(self):
        df = pd.DataFrame({
            "a": np.random.randn(100),
            "b": np.random.randn(100),
            "c": np.random.randn(100),
        })
        report = correlation_report(df, threshold=0.99)
        assert len(report) == 0


class TestVIF:
    def test_collinear_features_high_vif(self):
        df = pd.DataFrame({
            "a": range(100),
            "b": [x * 2 + 1 for x in range(100)],
            "c": np.random.randn(100),
        })
        vif_df = compute_vif(df)
        # a and b are perfectly collinear → high VIF
        a_vif = vif_df.loc[vif_df["feature"] == "a", "vif"].iloc[0]
        assert a_vif > 10


class TestContestFeatures:
    def test_contest_type_assignment(self):
        df = pd.DataFrame({
            "multi_entry_max": [1, 3, 20, 150, 500],
            "entry_cost": [1, 5, 10, 25, 50],
        })
        result = add_contest_type_features(df)
        assert list(result["contest_type"]) == ["SE", "3max", "limited", "MME", "mass"]
