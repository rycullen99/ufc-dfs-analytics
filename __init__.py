"""
UFC DFS Backtesting & Analytics System
=======================================

Analyzes ~1.85M DraftKings UFC GPP lineups across 359 contests (2022-2025)
to discover ROI-validated rules for lineup construction.

Module Pipeline:
    config.py      →  Constants, paths, thresholds, regime definitions
    data_loader.py →  SQL queries → pandas DataFrames with derived features
    discover.py    →  ROI analysis across dimensions (ownership, salary, odds, etc.)
    validate.py    →  Statistical significance testing (bootstrap, FDR)
    reports.py     →  Human-readable markdown reports
    export.py      →  Skill-safe JSON rules for the lineup review skill

Usage:
    from ufc_dfs_analytics.data_loader import load_contests, load_lineups
    from ufc_dfs_analytics.discover import analyze_ownership_by_regime
"""
