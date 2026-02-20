"""Tests for feature leakage detection."""

import pytest
from src.phase0_data.leakage_guard import validate_features, assert_no_leakage, is_leaky


class TestValidateFeatures:
    def test_clean_features_pass(self):
        features = ["salary", "proj_points", "card_position", "salary_tier"]
        clean, warnings = validate_features(features)
        assert clean == features
        assert warnings == []

    def test_ownership_blocked(self):
        features = ["salary", "ownership", "proj_points"]
        clean, warnings = validate_features(features)
        assert "salary" in clean
        assert "proj_points" in clean
        assert "ownership" not in clean
        assert len(warnings) == 1
        assert "LEAKAGE" in warnings[0]

    def test_multiple_leaky_features(self):
        features = ["actual_points", "ownership", "is_cashing", "salary"]
        clean, warnings = validate_features(features)
        assert clean == ["salary"]
        assert len(warnings) == 3

    def test_all_leaky_features_caught(self):
        leaky = [
            "ownership", "actual_points", "actual_fpts", "is_cashing",
            "payout", "lineup_rank", "points_percentile", "was_optimal", "cashed",
        ]
        clean, warnings = validate_features(leaky)
        assert clean == []
        assert len(warnings) == len(leaky)

    def test_custom_blocklist(self):
        features = ["salary", "custom_bad"]
        clean, warnings = validate_features(features, blocklist=frozenset({"custom_bad"}))
        assert clean == ["salary"]
        assert len(warnings) == 1


class TestAssertNoLeakage:
    def test_clean_passes(self):
        assert_no_leakage(["salary", "proj_points", "card_position"])

    def test_leaky_raises(self):
        with pytest.raises(ValueError, match="Feature leakage detected"):
            assert_no_leakage(["salary", "ownership", "proj_points"])

    def test_actual_points_raises(self):
        with pytest.raises(ValueError):
            assert_no_leakage(["actual_points"])


class TestIsLeaky:
    def test_ownership_is_leaky(self):
        assert is_leaky("ownership") is True

    def test_salary_not_leaky(self):
        assert is_leaky("salary") is False

    def test_case_insensitive(self):
        assert is_leaky("Ownership") is True
        assert is_leaky("ACTUAL_POINTS") is True
