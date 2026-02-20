"""Tests for canonical fighter identity mapping."""

import pytest
from src.phase0_data.identity import normalize_name, last_name, reversed_name, match_name


class TestNormalizeName:
    def test_basic(self):
        assert normalize_name("Jon Jones") == "jon jones"

    def test_strips_periods(self):
        assert normalize_name("T.J. Dillashaw") == "tj dillashaw"

    def test_strips_hyphens(self):
        assert normalize_name("Jan Blachowicz-Smith") == "jan blachowicz smith"

    def test_collapses_spaces(self):
        assert normalize_name("  Jon   Jones  ") == "jon jones"

    def test_trailing_period(self):
        assert normalize_name("Jones.") == "jones"


class TestLastName:
    def test_basic(self):
        assert last_name("Jon Jones") == "jones"

    def test_single_name(self):
        assert last_name("Jones") == "jones"

    def test_three_parts(self):
        assert last_name("Charles Do Bronx Oliveira") == "oliveira"


class TestReversedName:
    def test_two_parts(self):
        assert reversed_name("Jon Jones") == "Jones Jon"

    def test_three_parts_unchanged(self):
        assert reversed_name("A B C") == "A B C"


class TestMatchName:
    @pytest.fixture
    def candidates(self):
        return {
            "jon jones": 1,
            "charles oliveira": 2,
            "nina ansaroff": 3,
            "alexander volkanovski": 4,
        }

    def test_exact_match(self, candidates):
        cid, method = match_name("Jon Jones", candidates)
        assert cid == 1
        assert method == "exact"

    def test_alias_match(self, candidates):
        cid, method = match_name("Nina Nunes", candidates)
        assert cid == 3
        assert method == "alias"

    def test_last_name_match(self, candidates):
        cid, method = match_name("Bones Jones", candidates)
        assert cid == 1
        assert method == "last_name"

    def test_reversed_match(self, candidates):
        cid, method = match_name("Jones Jon", candidates)
        assert cid == 1
        assert method == "reversed"

    def test_unmatched(self, candidates):
        cid, method = match_name("Completely Unknown Fighter", candidates)
        assert cid is None
        assert method == "unmatched"

    def test_fuzzy_match(self, candidates):
        cid, method = match_name("Alexander Volkanovsky", candidates)  # slight misspelling
        assert cid == 4
        assert method == "fuzzy"
