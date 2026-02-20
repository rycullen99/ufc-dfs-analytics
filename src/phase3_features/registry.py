"""Feature registry with metadata and leakage flags."""

import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import logging

from ..config import LEAKY_FEATURES

logger = logging.getLogger(__name__)


@dataclass
class FeatureMeta:
    """Metadata for a single feature."""
    name: str
    dtype: str = "float"
    source: str = "dfs"
    available_pre_contest: bool = True
    leaky: bool = False
    group: str = "general"
    description: str = ""


class FeatureRegistry:
    """
    Central registry of all features with metadata.

    Enforces that model features are registered and not leaky.
    """

    def __init__(self):
        self._features: Dict[str, FeatureMeta] = {}

    def register(self, meta: FeatureMeta) -> None:
        if meta.name in LEAKY_FEATURES:
            meta.leaky = True
            meta.available_pre_contest = False
        self._features[meta.name] = meta

    def register_many(self, metas: List[FeatureMeta]) -> None:
        for m in metas:
            self.register(m)

    def get(self, name: str) -> Optional[FeatureMeta]:
        return self._features.get(name)

    def validate(self, feature_list: List[str]) -> List[str]:
        """
        Check that all features are registered and not leaky.

        Returns list of warning messages (empty = all good).
        """
        warnings = []
        for f in feature_list:
            if f not in self._features:
                warnings.append(f"Feature '{f}' is not registered")
            elif self._features[f].leaky:
                warnings.append(f"Feature '{f}' is flagged as LEAKY")
            elif not self._features[f].available_pre_contest:
                warnings.append(f"Feature '{f}' is not available pre-contest")
        return warnings

    def get_clean_features(self, feature_list: List[str]) -> List[str]:
        """Return only registered, non-leaky, pre-contest features."""
        return [
            f for f in feature_list
            if f in self._features
            and not self._features[f].leaky
            and self._features[f].available_pre_contest
        ]

    def to_dataframe(self) -> pd.DataFrame:
        rows = [
            {
                "name": m.name,
                "dtype": m.dtype,
                "source": m.source,
                "pre_contest": m.available_pre_contest,
                "leaky": m.leaky,
                "group": m.group,
                "description": m.description,
            }
            for m in self._features.values()
        ]
        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self._features)

    def __contains__(self, name: str) -> bool:
        return name in self._features


def build_default_registry() -> FeatureRegistry:
    """Build the default feature registry for the ownership model."""
    reg = FeatureRegistry()
    reg.register_many([
        FeatureMeta("salary", "int", "dfs", True, False, "salary", "DraftKings salary"),
        FeatureMeta("salary_rank_overall", "int", "dfs", True, False, "salary", "Salary rank within contest"),
        FeatureMeta("salary_pct", "float", "dfs", True, False, "salary", "Salary as % of cap"),
        FeatureMeta("salary_tier", "category", "dfs", True, False, "salary", "Binned salary tier"),
        FeatureMeta("card_position", "int", "dfs", True, False, "card", "Card position (1=main event)"),
        FeatureMeta("card_section", "category", "dfs", True, False, "card", "Card section label"),
        FeatureMeta("ownership_lag1", "float", "dfs", True, False, "ownership_history", "Ownership from previous appearance"),
        FeatureMeta("ownership_lag2", "float", "dfs", True, False, "ownership_history", "Ownership from 2 appearances ago"),
        FeatureMeta("ownership_rolling_3_mean", "float", "dfs", True, False, "ownership_history", "Rolling 3-event mean ownership"),
        FeatureMeta("historical_avg_ownership", "float", "dfs", True, False, "ownership_history", "Expanding mean ownership"),
        FeatureMeta("is_favorite", "bool", "odds", True, False, "odds", "Whether fighter is betting favorite"),
        FeatureMeta("consensus_ml_prob", "float", "odds", True, False, "odds", "Implied probability from closing odds"),
        FeatureMeta("days_since_last_fight", "float", "dfs", True, False, "recency", "Days since last DFS appearance"),
        FeatureMeta("dfs_sample_size", "int", "dfs", True, False, "recency", "Number of prior DFS appearances"),
        FeatureMeta("log_field_size", "float", "dfs", True, False, "contest", "Log of contest field size"),
        FeatureMeta("contest_size", "int", "dfs", True, False, "contest", "Contest field size"),
        FeatureMeta("entry_cost", "float", "dfs", True, False, "contest", "Contest entry cost"),
        FeatureMeta("num_fighters_on_slate", "int", "dfs", True, False, "contest", "Fighters available on slate"),
        # Leaky features (for documentation)
        FeatureMeta("ownership", "float", "dfs", False, True, "leaky", "ACTUAL ownership (post-contest)"),
        FeatureMeta("actual_points", "float", "dfs", False, True, "leaky", "Actual fantasy points scored"),
    ])
    return reg
