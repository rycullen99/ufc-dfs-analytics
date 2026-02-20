"""Tests for the sharp user skill model."""

import numpy as np
import pytest
from src.phase2_sharp.user_skill import estimate_population_prior


class TestEstimatePopulationPrior:
    def test_uniform_rates(self):
        """With uniform cash rates around 0.5, prior should be near Beta(a, a)."""
        rates = np.full(100, 0.5)
        counts = np.full(100, 50)
        alpha, beta = estimate_population_prior(rates, counts, min_entries=10)
        # Both should be positive
        assert alpha > 0
        assert beta > 0
        # Mean should be near 0.5
        assert abs(alpha / (alpha + beta) - 0.5) < 0.1

    def test_low_variance_high_shrinkage(self):
        """Low variance in rates → strong prior (high alpha + beta)."""
        rates = np.random.normal(0.3, 0.01, 100).clip(0, 1)
        counts = np.full(100, 50)
        alpha, beta = estimate_population_prior(rates, counts, min_entries=10)
        # Strong prior = high sum
        assert alpha + beta > 10

    def test_min_entries_filter(self):
        """Users below min_entries should be excluded from prior estimation."""
        rates = np.array([0.5, 0.5, 0.5, 0.9, 0.1])
        counts = np.array([100, 100, 100, 2, 2])  # last two below threshold
        alpha, beta = estimate_population_prior(rates, counts, min_entries=10)
        # Prior should be near Beta(a, a) since filtered rates are all 0.5
        assert abs(alpha / (alpha + beta) - 0.5) < 0.15

    def test_insufficient_data(self):
        """With very few users, should return uninformative prior."""
        rates = np.array([0.5])
        counts = np.array([100])
        alpha, beta = estimate_population_prior(rates, counts, min_entries=10)
        assert alpha == 1.0
        assert beta == 1.0
