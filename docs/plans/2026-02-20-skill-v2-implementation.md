# UFC DFS Skill v2 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Overhaul `ufc-dfs-lineup-review` skill with 5 contest archetypes, fighter selection rules (weight class, gender, line movement, probability bands), and expanded odds data from The Odds API.

**Architecture:** Three analysis scripts mine the DB → results inform a restructured 16-section skill file. Scripts output human-readable tables + JSON summaries. The skill is a single markdown document (~1,200-1,400 lines) with ROI-validated rules organized by contest archetype → fighter selection → entry fee.

**Tech Stack:** Python 3.12, SQLite (`resultsdb_ufc.db`), pandas, numpy. Skill is markdown.

**DB path:** `/Users/ryancullen/Desktop/resultsdb_ufc.db`
**Skill path:** `/Users/ryancullen/.claude/skills/ufc-dfs-lineup-review/SKILL.md`
**Scripts dir:** `/Users/ryancullen/ufc-dfs-analytics/scripts/`

---

## Critical Context

### DB Schema (key tables)

```
contests: contest_id, date_id, entry_cost, contest_size, multi_entry_max, cash_line, total_prizes
lineups: id, contest_id, lineup_hash, lineup_rank, points, total_salary, total_ownership, min_ownership, max_ownership, is_cashing, payout, lineup_percentile, lineup_count, favorite_count, underdog_count, odds_coverage
lineup_players: id, lineup_id, player_id, roster_slot
players: player_id, contest_id, full_name, salary, ownership, actual_points, roster_position
fighter_odds_enriched: date_id, canonical_id, player_id, dfs_name, salary, actual_ownership, actual_points, open_prob, close_prob, line_move, is_favorite, close_n_books
canonical_fighter: canonical_id, canonical_name
fighter_card_position: date_id, player_id, card_section
player_features: player_id, contest_id, weight_class (+ many feature columns)
```

### Contest Type Distribution (from DB)

| multi_entry_max | n_contests | avg_fee | avg_field | Archetype |
|----------------|------------|---------|-----------|-----------|
| NULL | 358 | $10.3 | 29,473 | UNLIMITED |
| 1 | 8 | $3,710 | 13 | SE |
| 3-18 | 139 | $555 | ~300 | LIMITED |
| 20 | 3 | $371 | 4,408 | 20-Max |
| 150 | 4 | $8.5 | 22,275 | 150-Max |

**Sample size reality:** UNLIMITED=TIER 1, LIMITED=TIER 2, SE/20-Max/150-Max=TIER 3. The skill must annotate confidence tiers.

### Freeroll Exclusion

Always exclude contest_id 142690805 (freeroll) from all queries.

### ROI Calculation

```python
# ROI = avg payout / entry cost
roi = lineups_df.groupby(bucket)['payout'].mean() / entry_cost
```

### Existing Skill Location

`/Users/ryancullen/.claude/skills/ufc-dfs-lineup-review/SKILL.md` (~888 lines, v1)

---

## Task 1: Contest Archetype Analysis Script

**Files:**
- Create: `/Users/ryancullen/ufc-dfs-analytics/scripts/contest_archetype_analysis.py`

**Step 1: Write the script**

This script classifies contests into 5 archetypes and mines ROI across all dimensions for each. It should:

1. **Load all lineups with contest metadata** — JOIN lineups + contests. Exclude freeroll.
2. **Classify each contest** into archetype:
   ```python
   def classify_contest(multi_entry_max, entry_cost, contest_size):
       if multi_entry_max == 1:
           return "SE"
       elif multi_entry_max is not None and 3 <= multi_entry_max <= 18:
           return "LIMITED"
       elif multi_entry_max == 20:
           return "20-Max"
       elif multi_entry_max == 150:
           return "150-Max"
       else:  # NULL = unlimited
           return "UNLIMITED"
   ```
3. **Compute effective entries**: `min(multi_entry_max or 999, contest_size) / contest_size` — for UNLIMITED, use 999 as proxy.
4. **For each archetype, compute:**
   - ROI by total ownership band (<100%, 100-125%, 125-150%, 150-175%, 175-200%, 200%+)
   - ROI by salary remaining (<$500, $500-999, $1K-2K, $2K+)
   - ROI by salary spread (using max - min salary of 6 fighters)
   - ROI by min ownership fighter (the contrarian dimension)
   - ROI by max ownership fighter (the chalk anchor dimension)
   - ROI by lineup_count (duplication: 1, 2-5, 6-20, 21-100, 101+)
   - Scoring thresholds: avg, cash line, top-5%, top-1%, winner by archetype
   - ROI by $8.5K+ fighter count (0-6)
5. **Within UNLIMITED, also split by:**
   - Entry fee tier: MICRO ($1-3), LOW ($5-8), MID ($15-20), HIGH ($25-30)
   - Field size bucket: Small (<5K), Medium (5K-20K), Large (20K-50K), Mega (50K+)
6. **Effective entries analysis**: group by effective_entries bands and compute ROI
7. **Output:** Print formatted tables AND save JSON to `data/analysis/archetype_results.json`

**Key SQL for salary spread (requires lineup_players JOIN):**
```sql
SELECT l.id as lineup_id,
       MAX(p.salary) - MIN(p.salary) as salary_spread,
       SUM(CASE WHEN p.salary >= 8500 THEN 1 ELSE 0 END) as high_salary_count
FROM lineups l
JOIN lineup_players lp ON l.id = lp.lineup_id
JOIN players p ON lp.player_id = p.player_id AND p.contest_id = l.contest_id
GROUP BY l.id
```

**Performance note:** lineup_players has 12M rows. Use batch processing by contest_id. Pre-compute salary stats per lineup in a temp table, then aggregate.

**Step 2: Run and review output**

```bash
cd /Users/ryancullen/ufc-dfs-analytics && python scripts/contest_archetype_analysis.py
```

Verify: each archetype has ROI tables. Check that UNLIMITED matches existing skill numbers approximately (sanity check). Flag any archetype with <1,000 lineups as TIER 3.

**Step 3: Commit**

```bash
git add scripts/contest_archetype_analysis.py data/analysis/
git commit -m "feat: add contest archetype analysis (5 archetypes, ROI by all dimensions)"
```

---

## Task 2: Odds API Lineup Composition Analysis Script

**Files:**
- Create: `/Users/ryancullen/ufc-dfs-analytics/scripts/odds_api_lineup_composition.py`

**Step 1: Write the script**

This replaces the BFO-based Section 11 analysis with the Odds API dataset (10,485 fighter-events vs 849). It should:

1. **Load fighter_odds_enriched** with player_id NOT NULL and close_prob NOT NULL
2. **Fighter-level analysis (all fighter-events):**
   - Performance by close_prob band: Heavy Fav (70%+), Mod Fav (55-70%), Toss-up (45-55%), Mod Dog (30-45%), Heavy Dog (<30%)
   - Metrics: avg FPTS, FPTS/$1K, avg salary, avg ownership, 100+ FPTS%, ceiling uplift (top-1% FPTS vs baseline)
   - DFS outscore rate vs implied fight win probability
3. **Lineup composition (requires matching odds to lineups):**
   - For each lineup, count favorites (close_prob > 0.5) and heavy favorites (close_prob >= 0.7)
   - Filter to lineups where ALL 6 fighters have odds data (odds_coverage = 6, or compute from lineup_players JOIN)
   - ROI by favorite count (0-6)
   - ROI by heavy favorite count (0-4)
   - Cross-tab with entry fee and archetype
4. **New cross-tabulations not in v1:**
   - Prob band × weight class (from player_features)
   - Prob band × line movement direction (steamed/stable/drifted)
   - Prob band × card position
   - Prob band × gender
5. **Output:** Print tables + save JSON to `data/analysis/odds_composition_results.json`

**Key: linking odds to lineups**
```sql
-- For each lineup, count how many fighters have odds data
SELECT l.id as lineup_id, l.contest_id,
       COUNT(foe.canonical_id) as odds_coverage,
       SUM(CASE WHEN foe.is_favorite = 1 THEN 1 ELSE 0 END) as fav_count,
       SUM(CASE WHEN foe.close_prob >= 0.7 THEN 1 ELSE 0 END) as heavy_fav_count
FROM lineups l
JOIN lineup_players lp ON l.id = lp.lineup_id
JOIN players p ON lp.player_id = p.player_id AND p.contest_id = l.contest_id
JOIN contests c ON l.contest_id = c.contest_id
LEFT JOIN canonical_fighter cf ON cf.canonical_name = p.full_name
LEFT JOIN fighter_odds_enriched foe
    ON foe.date_id = c.date_id AND foe.canonical_id = cf.canonical_id
GROUP BY l.id
```

**Performance:** This query touches 12M lineup_players rows. Process by date_id batch. Only compute for dates that have odds data (131 events). Pre-compute the per-lineup aggregates in a temp table.

**Step 2: Run and review**

```bash
python scripts/odds_api_lineup_composition.py
```

Verify: fighter-level tables should have ~10,485 rows. Compare heavy fav FPTS/$1K against existing skill (should be similar direction). Check lineup-level coverage — how many lineups have 6/6 odds? Expect significantly more than the BFO-based 413K since Odds API covers 131 events vs 55.

**Step 3: Commit**

```bash
git add scripts/odds_api_lineup_composition.py data/analysis/
git commit -m "feat: odds API lineup composition analysis (10K+ fighter-events, replaces BFO)"
```

---

## Task 3: Fighter Rules by Archetype Analysis Script

**Files:**
- Create: `/Users/ryancullen/ufc-dfs-analytics/scripts/fighter_rules_by_archetype.py`

**Step 1: Write the script**

Tests whether fighter-level construction rules interact with contest archetype. This is the "do weight class tiers change depending on contest type?" analysis.

1. **Load lineups + odds + weight class + contest archetype** — big JOIN
2. **For each archetype, compute fighter-level metrics:**
   - Weight class ROI/FPTS contribution (does WW underdog edge hold in LIMITED?)
   - Gender effect by archetype (are women more valuable in small fields?)
   - Line movement signal by archetype (does steam matter more in 150-Max?)
   - Probability band by archetype
3. **Key interaction tests:**
   - Archetype × weight class tier → FPTS/$1K
   - Archetype × gender → FPTS/$1K
   - Archetype × line movement → ROI
   - Archetype × prob band → ROI
   - Field size × line movement (do steamed fighters matter more in big fields?)
4. **Effective entries interactions:**
   - Does optimal ownership shift with effective entries?
   - Does fighter selection matter more when effective entries are low (SE) vs high (150-Max)?
5. **Output:** Print tables + save JSON to `data/analysis/fighter_archetype_results.json`

**Approach:** Since we can't get per-fighter ROI directly (ROI is lineup-level), use two methods:
- **Method A (fighter-level):** FPTS, ceiling rate, dud rate by weight class × archetype (pooled across all contests of that archetype on each date)
- **Method B (lineup-level):** Tag each lineup's fighters by weight class/gender/line movement, then compute ROI by the tag composition

Method A is simpler and likely sufficient for most rules. Method B is needed for "do lineups with WW underdogs produce higher ROI in UNLIMITED vs LIMITED?"

**Step 2: Run and review**

```bash
python scripts/fighter_rules_by_archetype.py
```

Verify: check if any archetype-specific reversals appear (e.g., weight class tier F becomes tier A in some archetype). If not, the fighter rules are universal — simpler skill.

**Step 3: Commit**

```bash
git add scripts/fighter_rules_by_archetype.py data/analysis/
git commit -m "feat: fighter rules by archetype cross-tabulation"
```

---

## Task 4: Review Analysis Results and Identify Key Rules

**Files:**
- Read: `data/analysis/archetype_results.json`
- Read: `data/analysis/odds_composition_results.json`
- Read: `data/analysis/fighter_archetype_results.json`

**Step 1: Review all three analysis outputs**

Synthesize findings into a rules summary. For each rule, note:
- The data that supports it (n, ROI, FPTS/$1K)
- Confidence tier (TIER 1/2/3)
- Whether it's universal or archetype-specific
- Whether it confirms, extends, or contradicts existing v1 rules

**Step 2: Create rules summary**

Write a brief `data/analysis/rules_summary.md` capturing the top findings that will go into the skill. Focus on:
- Which of the 5 archetypes have meaningfully different rules?
- Which fighter selection rules are universal vs archetype-dependent?
- Any surprising reversals or non-obvious interactions?
- Sample size warnings

**Step 3: Commit**

```bash
git add data/analysis/rules_summary.md
git commit -m "docs: analysis rules summary for skill v2"
```

---

## Task 5: Write the Skill v2

**Files:**
- Modify: `/Users/ryancullen/.claude/skills/ufc-dfs-lineup-review/SKILL.md`

**Step 1: Back up current skill**

```bash
cp /Users/ryancullen/.claude/skills/ufc-dfs-lineup-review/SKILL.md \
   /Users/ryancullen/.claude/skills/ufc-dfs-lineup-review/SKILL.md.v1.backup
```

**Step 2: Write new SKILL.md**

Structure (16 sections):

```markdown
# UFC DFS Lineup Review & Construction v2

Review and build UFC DFS lineups using ROI-validated rules derived from
**505 contests, 2.3M unique lineups** (2022-2025) plus **10,485 fighter-events
with Odds API closing lines** (131 events). All contests are GPPs on DraftKings.

## Data Coverage (v2 — 2026-02-20)
[Updated table with Odds API coverage]

## 1. REGIME IDENTIFICATION
[5 contest archetypes table]
[Slate Size → Archetype → Entry Fee hierarchy]
[Effective entries concept]

## 2. TOTAL OWNERSHIP (by Archetype)
[Ownership bands × archetype ROI table]
[Key reversals between archetypes]

## 3. SALARY RULES (Universal)
[Kept from v1 — spend cap, salary remaining, total salary floor]

## 4. SALARY STRUCTURE (Universal)
[Kept from v1 — stars & scrubs, $8.5K+ count, salary spread]

## 5. OWNERSHIP COMPOSITION (by Archetype)
[Min/max ownership × archetype]
[Ideal profiles per archetype]

## 6. DUPLICATION (Universal)
[Kept from v1]

## 7. FIGHTER SELECTION: WEIGHT CLASS & GENDER (NEW)
[Weight class tiers S/A/B/F]
[Gender value differential]
[Weight class × fav/dog]
[Best underdog weight classes]

## 8. FIGHTER SELECTION: LINE MOVEMENT (NEW)
[Steamed/stable/drifted rules]
[Movement magnitude (20pp+ vs 5-10pp trap)]
[Line movement × weight class]
[Line movement × gender]

## 9. FIGHTER SELECTION: PROBABILITY BANDS (NEW)
[Prob band performance]
[Sweet spots: slight fav 50-65%, toss-up 45-55%]
[Prob × salary tier]
[Chalk analysis: heavy favs deliver?]

## 10. FIGHTER SELECTION: CARD POSITION (Revised)
[Card position × fav/dog]
[Card position × gender]
[Card position × line movement]

## 11. SCORING PATTERNS & DUD AVOIDANCE (NEW)
[Dud rate by weight class × fav/dog]
[Floor analysis by prob tier]
[Cross-cut profiles: Cash Lock, GPP Leverage, Trap Play, Dud Risk]

## 12. ODDS COMPOSITION (Revised — 12x more data)
[Fighter eval by win probability (Odds API)]
[DFS outscore rate]
[Ceiling uplift]
[Lineup composition by favorite count]
[Prob × weight class × line movement cross-tabs]

## 13. CONTEST ARCHETYPES: DETAILED PROFILES (NEW)
[One subsection per archetype with its specific rules]
[Entry strategy, lineup count, build method, unique rules]

## 14. LINEUP REVIEW WORKFLOW (Revised)
[Updated review template incorporating fighter rules]
[Weight class check, line movement check, dud risk check]

## 15. QUICK REFERENCE (Revised)
[All rules at a glance — one page cheat sheet]

## 16. DATA QUALITY NOTES (Revised)
[Updated coverage, sample sizes, confidence tiers]
[Odds API coverage vs BFO coverage]
```

**Key principles for writing:**
- Every table gets a confidence tier annotation (TIER 1/2/3)
- Every rule gets a 📊 (validated) or 💡 (directional/expert) badge
- Tables are compact (no more than 8 columns)
- Cross-references between sections use `(See Section X)`
- Fighter selection rules include "REVERSAL ALERT" callouts when archetype matters
- Keep the Quick Reference (Section 15) as a standalone cheat sheet that works without reading the full skill

**Step 3: Commit**

```bash
git add /Users/ryancullen/.claude/skills/ufc-dfs-lineup-review/SKILL.md
git add /Users/ryancullen/.claude/skills/ufc-dfs-lineup-review/SKILL.md.v1.backup
git commit -m "feat: UFC DFS skill v2 — contest archetypes, fighter selection rules, Odds API"
```

---

## Task 6: Validate Skill Against Known Results

**Files:**
- Read: `/Users/ryancullen/.claude/skills/ufc-dfs-lineup-review/SKILL.md`

**Step 1: Spot-check consistency**

- Verify no contradictions between sections (e.g., Section 2 ownership target vs Section 5 composition)
- Verify fighter rules don't conflict with contest rules
- Check that all v1 rules that were "universal" are still present
- Verify TIER annotations match sample sizes

**Step 2: Check against v1**

Compare key numbers between v1 and v2:
- UNLIMITED 100-125% ROI should still be ~1.53x
- Stars & scrubs ROI should still be ~1.35x
- Unique lineup ROI should still be ~1.46x

If numbers shift significantly, investigate whether it's a data issue or a genuine improvement from larger odds dataset.

**Step 3: Final commit**

```bash
git add -A && git commit -m "docs: skill v2 validation complete"
```

---

## Task 7: Update Strategy Guide Companion Skill

**Files:**
- Modify: `/Users/ryancullen/.claude/skills/ufc-dfs-strategy-guide/SKILL.md`

**Step 1: Update cross-references**

The strategy guide references specific sections of the lineup review skill. Update all "Data checkpoint" references to point to the new section numbers.

**Step 2: Add fighter selection guidance**

Add a brief subsection to the strategy guide's matchup analysis framework (Section 10) that references the new weight class tiers and line movement rules.

**Step 3: Commit**

```bash
git add /Users/ryancullen/.claude/skills/ufc-dfs-strategy-guide/SKILL.md
git commit -m "docs: update strategy guide cross-references for skill v2"
```

---

## Execution Order & Dependencies

```
Task 1 (archetype analysis) ─┐
Task 2 (odds composition)  ──┼── Task 4 (review results) → Task 5 (write skill) → Task 6 (validate) → Task 7 (update strategy guide)
Task 3 (fighter × archetype) ┘
```

Tasks 1-3 are **independent** and can run in parallel.
Tasks 4-7 are **sequential** (each depends on the previous).

## Estimated Time

| Task | Complexity | Notes |
|------|-----------|-------|
| 1 | High | 12M row JOINs, batch processing needed |
| 2 | High | Odds matching to lineups, large JOINs |
| 3 | Medium | Reuses Task 1-2 infrastructure |
| 4 | Low | Reading and synthesizing |
| 5 | High | ~1,200 lines of structured markdown |
| 6 | Low | Spot-check validation |
| 7 | Low | Cross-reference updates |
