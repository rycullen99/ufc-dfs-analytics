# UFC DFS Analytics

Historical backtesting + build-day intelligence for UFC DFS on DraftKings.

## Database

**Location:** `/Users/ryancullen/Desktop/DFS/Databases/resultsdb_ufc.db` (1.5 GB)
- 529 contests, 2.38M lineups, 138 dates (2022-01 through 2026-03)
- 12,437 Odds API fighter-events, 881K lineups with 6/6 odds coverage
- Configured in `src/config.py` → `DB_PATH`

**Exclude:** `FREEROLL_CONTEST_ID = 142690805` in all analyses.

## Pipeline Phases

| Phase | Module | Purpose |
|-------|--------|---------|
| 0 | `src/phase0_data/` | ETL, identity resolution, card position, odds quality |
| 1 | `src/phase1_ownership/` | Ownership prediction (walk-forward CV: Ridge + RF + LightGBM) |
| 2 | `src/phase2_sharp/` | Sharp vs field exposure signals (Bayesian user skill model) |
| 3 | `src/phase3_features/` | Feature registry, collinearity checks, importance tracking |
| 4 | `src/phase4_contest/` | Contest regime classification |
| 5 | `src/phase5_pipeline/` | Orchestration, Supabase export, idempotent stage runner |
| 6 | `src/phase6_backtesting/` | ROI discovery, bootstrap CIs, FDR correction, time-split validation |

## Key Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_pipeline.py` | Run full pipeline (phases 0-5) |
| `scripts/run_discovery.py` | Run Phase 6 ROI discovery |
| `scripts/resolve_incomplete_lineups.py` | Recover lineups with missing player IDs |
| `scripts/pull_odds_api.py` | Pull historical MMA odds from The Odds API |
| `scripts/match_odds_to_fighters.py` | Map odds names → canonical DFS fighters |
| `scripts/prebuild_signals.py` | Pre-build intelligence report for a slate |
| `scripts/export_skill_tables.py` | Export ROI tables for skill updates |

## Running

```bash
# Generate validation report
python3 -m src.phase6_backtesting.report

# Resolve incomplete lineups
python3 scripts/resolve_incomplete_lineups.py --dry-run

# Pull odds and match
python3 scripts/pull_odds_api.py && python3 scripts/match_odds_to_fighters.py
```

## Conventions

- **Odds source:** Odds API consensus closing lines only (BFO deprecated)
- **ROI metric:** Weighted ROI = SUM(payout × user_count) / SUM(entry_cost × user_count)
- **Well-scraped filter:** Contests within 6% payout tolerance (`PAYOUT_TOLERANCE` in `regime.py`)
- **Validation tiers:** GOLD (FDR + time-split), SILVER (FDR only), BRONZE (directional)
- **Leakage blocklist:** `config.py` → `LEAKY_FEATURES` — never use post-contest features in models

## Companion Skills

- `ufc-dfs-lineup-review` (v2.4) — quantitative construction rules
- `ufc-dfs-strategy-guide` (v1.1) — qualitative process + game theory
