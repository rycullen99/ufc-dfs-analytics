"""Tests for walk-forward cross-validation."""

import numpy as np
import pandas as pd
import pytest
from src.cv.walk_forward import EventWalkForwardCV


class TestEventWalkForwardCV:
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame with 30 events, ~10 fighters each."""
        rows = []
        for date_id in range(20220101, 20220131):
            for pid in range(1, 11):
                rows.append({
                    "date_id": date_id,
                    "player_id": pid,
                    "salary": 7000 + pid * 200,
                    "ownership": np.random.uniform(1, 30),
                })
        return pd.DataFrame(rows)

    def test_min_train_events_respected(self, sample_df):
        cv = EventWalkForwardCV(min_train_events=20)
        folds = list(cv.split(sample_df))
        # 30 dates - 20 min = 10 folds
        assert len(folds) == 10

    def test_no_overlap(self, sample_df):
        cv = EventWalkForwardCV(min_train_events=20)
        for train_idx, test_idx in cv.split(sample_df):
            train_dates = set(sample_df.iloc[train_idx]["date_id"])
            test_dates = set(sample_df.iloc[test_idx]["date_id"])
            assert train_dates.isdisjoint(test_dates), "Train and test dates overlap!"

    def test_train_before_test(self, sample_df):
        cv = EventWalkForwardCV(min_train_events=20)
        for train_idx, test_idx in cv.split(sample_df):
            max_train_date = sample_df.iloc[train_idx]["date_id"].max()
            min_test_date = sample_df.iloc[test_idx]["date_id"].min()
            assert max_train_date < min_test_date, "Train dates not before test!"

    def test_expanding_window(self, sample_df):
        cv = EventWalkForwardCV(min_train_events=20)
        train_sizes = []
        for train_idx, _ in cv.split(sample_df):
            train_sizes.append(len(train_idx))
        # Each fold should have more training data
        assert train_sizes == sorted(train_sizes)

    def test_all_test_fighters_grouped(self, sample_df):
        cv = EventWalkForwardCV(min_train_events=20)
        for _, test_idx in cv.split(sample_df):
            test_dates = sample_df.iloc[test_idx]["date_id"].unique()
            assert len(test_dates) == 1, "Test fold should contain exactly one event date"

    def test_n_splits(self, sample_df):
        cv = EventWalkForwardCV(min_train_events=20)
        assert cv.get_n_splits(sample_df) == 10

    def test_too_few_events(self):
        df = pd.DataFrame({
            "date_id": [1, 1, 2, 2],
            "player_id": [1, 2, 1, 2],
        })
        cv = EventWalkForwardCV(min_train_events=20)
        assert list(cv.split(df)) == []
