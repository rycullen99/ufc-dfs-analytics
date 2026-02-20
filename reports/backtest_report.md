# UFC DFS Analytics — Backtest Report

## Data Summary

| Metric | Value |
|--------|-------|
| Total events | 132 |
| Total fighter-event rows | 12,405 |
| Dev events | 122 |
| Holdout events | 10 |
| Holdout samples | 964 |
| Features | 18 (16 numeric + 2 categorical) |
| Leakage check | PASSED |

## Walk-Forward Cross-Validation (Dev Set, 102 Folds)

| Model | MAE | Status |
|-------|-----|--------|
| Ridge | 5.65 | Baseline |
| **Random Forest** | **4.27** | Best |
| LightGBM | 4.40 | Runner-up |
| Ensemble (mean) | ~4.5 | Used for final |

All models use expanding-window walk-forward CV grouped by event date.
No within-event splitting. Minimum 20 training events before first fold.

## Holdout Evaluation (Final, Evaluated Once)

| Metric | Point Estimate | 95% CI |
|--------|---------------|--------|
| MAE | 5.71 pp | [5.20, 6.15] |
| RMSE | 9.56 pp | [8.86, 10.24] |
| Calibration slope | 0.907 | — |
| Calibration intercept | 3.23 | — |
| ECE | 1.61 pp | — |

Bootstrap CIs computed via event-level resampling (1,000 iterations).

## Sharp User Signals

| Metric | Value |
|--------|-------|
| Users scored | 33,121 |
| Sharp users (top 20%) | 6,630 |
| Population prior | Beta(4.11, 14.43) |
| Fighter-events with signals | 2,726 |
| Positive sharp delta | 44.3% |

## Feature Hygiene

### Correlated Clusters (|r| > 0.85)

1. **Salary cluster**: salary, salary_rank_overall, salary_pct, is_favorite, consensus_ml_prob
2. **Ownership history**: ownership_lag1, ownership_lag2, ownership_rolling_3_mean, historical_avg_ownership

### High VIF Features (VIF > 10)

| Feature | VIF |
|---------|-----|
| salary / salary_pct | inf (perfectly collinear) |
| salary_rank_overall | 195.8 |
| ownership_rolling_3_mean | 52.9 |
| ownership_lag2 | 18.5 |
| consensus_ml_prob | 12.9 |
| ownership_lag1 | 10.5 |

## Contest-Type Partial Pooling

| Contest Type | N Samples | Adjustment (pp) |
|-------------|-----------|----------------|
| Limited | 9,352 | +0.008 |
| MME | 2,944 | -0.025 |
| SE | 109 | -0.034 |

Partial pooling improvement over fully pooled: <0.001 pp (negligible).
Contest-type effects already captured by contest_size and entry_cost features.

## Data Reliability (Phase 0)

| Component | Result |
|-----------|--------|
| Canonical fighters | 1,008 |
| Fighter aliases | 1,011 |
| Unmatched names | 0 |
| Card positions backfilled | 3,439 |
| Odds quality (high/medium/none) | 455 / 394 / 2,590 |
| Odds coverage | 24.7% of fighter-events |
