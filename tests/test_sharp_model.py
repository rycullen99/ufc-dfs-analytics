"""Tests for the sharp user skill model."""

import numpy as np
import pytest
from src.phase2_sharp.user_skill import estimate_population_prior


class TestEstimatePopulationPrior:
    def test_uniform_rates(self):
        """With uniform cash rates around 0.5, prior should be near Beta(a, a)."""
        cash_counts = np.full(100, 50)
        entry_counts = np.full(100, 100)
        alpha, beta = estimate_population_prior(cash_counts, entry_counts)
        assert alpha > 0
        assert beta > 0
        # Mean should be near 0.5
        assert abs(alpha / (alpha + beta) - 0.5) < 0.1

    def test_low_variance_high_shrinkage(self):
        """Low variance in rates -> strong prior (high alpha + beta)."""
        rng = np.random.RandomState(42)
        # All users cash ~30% of the time with very little variance
        entry_counts = np.full(100, 100)
        cash_counts = (rng.normal(0.3, 0.01, 100).clip(0.05, 0.95) * entry_counts).astype(int)
        alpha, beta = estimate_population_prior(cash_counts, entry_counts)
        # Strong prior = high sum
        assert alpha + beta > 10

    def test_degenerate_returns_uninformative(self):
        """With zero variance, should return safe fallback prior."""
        cash_counts = np.array([50, 50, 50])
        entry_counts = np.array([100, 100, 100])
        alpha, beta = estimate_population_prior(cash_counts, entry_counts)
        assert alpha > 0
        assert beta > 0

    def test_empty_returns_uninformative(self):
        """With no data, should return uninformative prior."""
        cash_counts = np.array([])
        entry_counts = np.array([])
        alpha, beta = estimate_population_prior(cash_counts, entry_counts)
        assert alpha == 1.0
        assert beta == 1.0
