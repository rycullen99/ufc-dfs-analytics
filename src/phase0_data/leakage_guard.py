"""
Feature leakage detection and prevention.

Catches the known bug where post-contest `ownership` was used as a prediction
feature, and provides a reusable guard for all model training.
"""

from ..config import LEAKY_FEATURES


def validate_features(
    feature_list: list[str],
    blocklist: frozenset[str] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Validate a feature list against the leakage blocklist.

    Parameters
    ----------
    feature_list : list[str]
        Proposed features for model training.
    blocklist : frozenset[str], optional
        Custom blocklist. Defaults to LEAKY_FEATURES from config.

    Returns
    -------
    clean_list : list[str]
        Features with leaky columns removed.
    warnings : list[str]
        Human-readable warnings for each removed feature.
    """
    if blocklist is None:
        blocklist = LEAKY_FEATURES

    clean = []
    warnings = []

    for feat in feature_list:
        # Check exact match and common suffixed variants
        base = feat.lower().strip()
        if base in blocklist:
            warnings.append(
                f"LEAKAGE: '{feat}' is post-contest data and cannot be used as a feature."
            )
        else:
            clean.append(feat)

    return clean, warnings


def assert_no_leakage(feature_list: list[str]) -> None:
    """
    Assert that no leaky features are present. Raises ValueError if any found.

    Use this as a pre-training assertion in model pipelines.
    """
    _, warnings = validate_features(feature_list)
    if warnings:
        raise ValueError(
            "Feature leakage detected! The following features are post-contest data "
            "and must not be used for pre-contest prediction:\n"
            + "\n".join(f"  - {w}" for w in warnings)
        )


def is_leaky(feature_name: str) -> bool:
    """Check if a single feature name is in the leakage blocklist."""
    return feature_name.lower().strip() in LEAKY_FEATURES
