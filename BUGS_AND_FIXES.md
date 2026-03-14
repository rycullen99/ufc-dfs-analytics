# UFC DFS Analytics — Bugs & Fixes Backlog

## Odds Coverage Pipeline

### BUG: 6/6 lineup coverage dropped after pipeline rebuild (730K → 705K)
- **Status:** Open
- **Severity:** Medium
- **Context:** Re-running `match_odds_to_fighters.py` rebuilt `odds_api_matched` and `fighter_odds_enriched` tables. Then re-running `integrate_odds_api.py` produced 705,265 lineups with 6/6 coverage, down from 730,869.
- **Root cause:** Likely a subtle change in how `odds_api_matched` was rebuilt — either fight deduplication changed or the matching pipeline resolved some fights differently. The `prob_lookup` went from 4,050 → 4,060 entries (gained fighters) but some existing fighters may have shifted probabilities or lost matches.
- **Fix needed:** Compare old vs new `odds_api_matched` row-by-row to find which fights dropped. May need to snapshot the table before rebuilds. Consider making the matching pipeline idempotent (UPDATE instead of DROP+CREATE).
- **Files:** `scripts/match_odds_to_fighters.py`, `Desktop/UFC DFS/scripts/integrate_odds_api.py`

### BUG: 36K lineups with 0/6 odds coverage
- **Status:** Diagnosed, partially fixable
- **Breakdown:**
  - 28,218 lineups on date 20220604 (UFC 275) — Odds API has zero data for this event. **Unfixable** without a different data source.
  - 4,647 lineups with no `lineup_players` entries — likely overlay/empty seats. **Expected**, not a bug.
  - 3,437 lineups with fighters + odds data on date but no fighter matched — dates with sparse API coverage (1-5 fights vs 12+ on card). **Partially fixable** by re-pulling from API.
- **Files:** `Desktop/UFC DFS/scripts/integrate_odds_api.py` (line 175: `has_odds` check)

### BUG: 331K lineups at 5/6 coverage
- **Status:** Diagnosed
- **Root cause:** Per-fight API gaps. Example: Andre Fialho vs Cameron VanCamp on 20220507 — the API returned 15 other fights but not this one. Each missing fight causes all lineups containing either fighter to lose 1 coverage point.
- **Fix:** Re-pull historical odds for specific missing fights, or accept as data limitation.

### FIX APPLIED: Added missing fighter aliases
- **Status:** Done (2026-03-07)
- **Details:** Added 4 aliases to `fighter_alias`:
  - `khaos williams` → canonical_id 524 (Kalinn Williams) — 5+ dates affected
  - `dong hoon choi` → canonical_id 283 (Donghun Choi)
  - `ravena oliveira` → canonical_id 796 (Ravena Oliveira Morais)
  - `bobby king` → canonical_id 553 (King Green)
- **Result:** `odds_api_fighter_map` match rate improved 938→966 (78.3%), unmatched dropped 296→268

### BUG: Duplicate fighter_alias entries
- **Status:** Open
- **Details:** `khaos williams` has 2 rows for canonical_id 524 (one from `manual` insert, one from `odds_api` re-registration in `match_odds_to_fighters.py` line 106-116). The UNIQUE constraint is on `(alias_name, alias_source)`, so different sources create duplicates.
- **Impact:** Low — the integration script picks the first match. But creates noise.
- **Fix:** Add dedup logic or change UNIQUE constraint to just `alias_name`.

### IMPROVEMENT: `fighter_odds_enriched` JOIN should use `fighter_alias`
- **Status:** Open
- **Details:** The enrichment SQL in `match_odds_to_fighters.py` (lines 207-210) joins `players.full_name` against `canonical_fighter.canonical_name` only. It doesn't use `fighter_alias`. This means fighters whose DFS name differs from canonical name get NULL `player_id`.
- **Impact:** 355 fighter-events on DFS dates have NULL player_id (966 non-DFS dates excluded). Of these, ~39 are truly fixable by using aliases, rest are non-card fighters the API returned.
- **Fix:** Rewrite the JOIN to: `LEFT JOIN players p ON LOWER(p.full_name) IN (SELECT alias_name FROM fighter_alias WHERE canonical_id = m.canonical_id_a) AND p.contest_id IN (...)`

### IMPROVEMENT: Re-pull sparse Odds API dates
- **Status:** Open
- **Details:** 8 dates have <5 fights in `odds_api_raw` vs 10-15 on the DFS card:
  - 20220604: 0 fights (UFC 275 — no data exists)
  - 20220430, 20230520, 20241012: 1 fight each
  - 20230114: 2 fights
  - 20220115: 3 fights
  - 20231104: 5 fights
  - 20250517: 11 fights (close to full)
- **Fix:** Use different snapshot timestamps when pulling. The current strategy uses `date + 26 hours` (02:00 UTC next day). Some events may have different commence times. Try pulling at the exact `commence_time` of each event.
- **Cost:** ~8 API calls × 10 credits = 80 credits

## Discovery Pipeline

### FIX APPLIED: Favorite count binning bug
- **Status:** Done (2026-03-07)
- **Details:** `pd.cut` with `right=False` on `FAV_COUNT_BINS = [-1, 0, 1, 2, 3, 4, 5, 6]` caused `favorite_count=6` to fall outside all bins (NaN). Got merged with fav=5 in the report, showing inflated 1.95x ROI for "6". SQL validation showed fav=6 is actually 0.72x (672 lineups).
- **Fix:** Changed to `right=True` in `src/phase6_backtesting/regime.py`.
- **File:** `src/phase6_backtesting/regime.py` line 93

## Skill (SKILL.md) Updates Pending

### Discrepancies found between fresh report and SKILL.md v2.1
- **Status:** Done (2026-03-07) — SKILL.md updated to v2.2
- **Details:** All discrepancies addressed with weighted ROI cross-validation notes throughout SKILL.md v2.2:
  - SHORT ownership peak: Added TIER 2 caveat — simple ROI shows 175-200% (2.10x), weighted shows 150-175% (1.27x). Both methods kept, difference documented.
  - Toss-up ROI: Added cross-validation — simple 3.53x vs weighted 1.66x. Direction holds (0 toss-ups best), magnitude differs.
  - Favorite count: Confirmed 5 favs = 1.95x weighted. Documented pd.cut binning bug fix (fav=6 was merged with fav=5).
  - own_prob_ratio <0.50 = 2.13x weighted confirmed.
  - Added new "Optimal Lineup Benchmarks" section from 132-date analysis.
  - Fixed SaberSim Rule 12 internal inconsistency (had wrong toss-up ROI value).
  - Documented coverage regression (724K → ~705K) in Data Sources section.
