"""Tests for ownership model leakage prevention and feature validation."""

import pytest
from src.phase0_data.leakage_guard import assert_no_leakage


class TestOwnershipModelLeakageGuard:
    """Verify that the ownership model never uses post-contest features."""

    OWNERSHIP_FEATURES = [
        "salary",
        "salary_rank_overall",
        "salary_pct",
        "card_position",
        "ownership_lag1",
        "ownership_lag2",
        "ownership_rolling_3_mean",
        "historical_avg_ownership",
        "is_favorite",
        "consensus_ml_prob",
        "days_since_last_fight",
        "dfs_sample_size",
        "log_field_size",
        "contest_size",
        "entry_cost",
        "num_fighters_on_slate",
        "salary_tier",
    ]

    def test_all_features_pass_leakage_check(self):
        """The ownership model feature set should contain zero leaky features."""
        assert_no_leakage(self.OWNERSHIP_FEATURES)

    def test_adding_ownership_would_fail(self):
        """Adding raw ownership to features must raise."""
        bad_features = self.OWNERSHIP_FEATURES + ["ownership"]
        with pytest.raises(ValueError, match="Feature leakage detected"):
            assert_no_leakage(bad_features)

    def test_adding_actual_points_would_fail(self):
        """Adding actual_points to features must raise."""
        bad_features = self.OWNERSHIP_FEATURES + ["actual_points"]
        with pytest.raises(ValueError, match="Feature leakage detected"):
            assert_no_leakage(bad_features)
