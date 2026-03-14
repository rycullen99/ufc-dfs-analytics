"""
False Discovery Rate correction for ROI discovery (Phase 6).

UFC runs ~60-80 implicit hypothesis tests across 12+ analyses.
Without FDR correction, expect ~6-8 false positives at alpha=0.05.
BH procedure at q=0.10 controls the expected proportion of false
discoveries among rejected hypotheses.

Ported from pga-dfs-analytics/src/toolkit/backtesting.py.
"""

import numpy as np


def benjamini_hochberg(p_values: np.ndarray, q: float = 0.10) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : array of p-values from bootstrap tests.
    q : float
        Target false discovery rate (default 0.10).

    Returns
    -------
    array of bool — True if the test survives FDR correction.
    """
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=bool)

    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]

    # BH critical values: (rank / n) * q
    bh_critical = np.arange(1, n + 1) / n * q

    # Find largest k where p_(k) <= BH critical value
    surviving = sorted_p <= bh_critical
    if not surviving.any():
        return np.zeros(n, dtype=bool)

    max_k = np.max(np.where(surviving)[0])
    result = np.zeros(n, dtype=bool)
    result[sorted_idx[: max_k + 1]] = True
    return result


def apply_fdr_to_results(
    analysis_results: dict,
    q: float = 0.10,
) -> dict:
    """
    Apply BH FDR correction across all p-values from grouped bootstrap results.

    Parameters
    ----------
    analysis_results : dict
        {analysis_name: {group_value: {p_value: float, ...}, ...}, ...}
        As returned by discover.run_all_with_ci().
    q : float
        Target FDR.

    Returns
    -------
    dict — same structure with 'survives_fdr' bool added to each group result.
    Also adds '_fdr_summary' key with counts.
    """
    # Collect all p-values with their locations
    locations = []
    p_vals = []

    for analysis_name, groups in analysis_results.items():
        if analysis_name.startswith("_"):
            continue
        if isinstance(groups, dict):
            for group_val, result in groups.items():
                if isinstance(result, dict) and "p_value" in result:
                    locations.append((analysis_name, group_val))
                    p_vals.append(result["p_value"])

    if not p_vals:
        return analysis_results

    p_array = np.array(p_vals)
    survives = benjamini_hochberg(p_array, q=q)

    # Write results back
    for (analysis_name, group_val), survived in zip(locations, survives):
        analysis_results[analysis_name][group_val]["survives_fdr"] = bool(survived)

    # Summary
    analysis_results["_fdr_summary"] = {
        "total_tests": len(p_vals),
        "surviving": int(survives.sum()),
        "rejected": int((~survives).sum()),
        "q": q,
        "min_p": float(p_array.min()) if len(p_array) > 0 else None,
        "max_surviving_p": float(p_array[survives].max()) if survives.any() else None,
    }

    return analysis_results
