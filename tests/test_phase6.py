"""Smoke tests for Phase 6: ROI backtesting pipeline."""
import pytest
from src.phase6_backtesting.loader import load_contests, load_lineups, weighted_roi
from src.phase6_backtesting.discover import run_all


@pytest.fixture(scope="module")
def contests():
    return load_contests(well_scraped_only=True)


@pytest.fixture(scope="module")
def lineups():
    return load_lineups(well_scraped_only=True)


class TestContests:
    def test_loads(self, contests):
        assert len(contests) >= 50

    def test_all_150max(self, contests):
        assert contests["is_150max"].all()

    def test_regime_columns_present(self, contests):
        for col in ["slate_size", "field_size", "fee_tier"]:
            assert col in contests.columns


class TestLineups:
    def test_loads(self, lineups):
        assert len(lineups) > 10_000

    def test_true_roi_below_one(self, lineups):
        roi = weighted_roi(lineups)
        assert 0.5 < roi < 1.0, f"ROI out of expected range: {roi:.4f}"

    def test_regime_columns_present(self, lineups):
        for col in ["slate_size", "fee_tier", "total_own_band", "salary_remaining_band"]:
            assert col in lineups.columns


class TestDiscovery:
    def test_run_all(self, lineups):
        results = run_all(lineups)
        assert "ownership" in results
        assert not results["ownership"].empty

    def test_roi_values_plausible(self, lineups):
        results = run_all(lineups)
        roi_vals = results["ownership"]["weighted_roi"]
        assert roi_vals.max() < 5.0
        assert roi_vals.min() > 0.0
