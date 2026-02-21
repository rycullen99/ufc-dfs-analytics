# UFC DFS Lineup Review Skill v2 — Design Document

**Date:** 2026-02-20
**Status:** Approved

## Problem

The existing `ufc-dfs-lineup-review` skill (v1, ~888 lines) has two gaps:

1. **Contest types too coarse.** Two buckets (UNLIMITED vs LIMITED) miss meaningful differences between Single Entry, 3-Max, 20-Max, and 150-Max contests. The DB contains 505 contests across 5 distinct archetypes.

2. **No fighter-level selection rules.** The skill tells you *how much* ownership and salary to use but not *which fighters* to target. A new construction rules analysis (33,431 fighter-events with Odds API closing lines, 131 events) reveals weight class, gender, line movement, and probability band rules that are not captured anywhere.

## Approach

Single mega-skill. Expand `ufc-dfs-lineup-review` from ~888 to ~1,200-1,400 lines. One reference document with a clear hierarchy:

```
Slate Size → Contest Archetype (5 types) → Fighter Selection Rules → Entry Fee Adjustments
```

## Deliverables

### A. Three Analysis Scripts

Run against `resultsdb_ufc.db` to mine the data needed for the skill.

#### 1. `scripts/contest_archetype_analysis.py`

Classifies contests into 5 archetypes and mines ROI by:
- Total ownership band
- Salary remaining / salary structure
- Ownership composition (min/max owned fighter)
- Duplication count
- Favorite count (using Odds API data)
- Scoring thresholds (cash line, top-1%, winner)
- Effective entries (your_entries / contest_size)

**Archetype definitions:**

| Archetype | Max Entries | Typical Field | Typical Fee |
|-----------|------------|---------------|-------------|
| SE | 1 | 100-300 | $5-$25 |
| 3-Max | 3 | 100-1,000 | $5-$555 |
| 20-Max | 20 | 1K-10K | $5-$20 |
| 150-Max Flagship | 150 | 15K-70K | $15-$25 |
| 150-Max Mini | 150 | 3K-15K | $1-$5 |

#### 2. `scripts/odds_api_lineup_composition.py`

Re-does the existing Section 11 (Odds) analysis with the Odds API dataset instead of BFO:
- Fighter evaluation by win probability (10,485 fighter-events vs 849)
- DFS outscore rate vs implied probability
- Ceiling uplift by prob band
- Lineup composition by favorite count (needs lineup-level odds matching)
- Cross-tabulation with new fighter rules (weight class × prob band × line movement)

#### 3. `scripts/fighter_rules_by_archetype.py`

Tests whether fighter-level rules (weight class tiers, line movement, gender) interact with contest archetype:
- Do weight class tiers change by contest type?
- Does line movement signal strength vary by field size?
- Are women's fighters more valuable in specific archetypes?
- Cross-cuts: weight class × archetype × prob band

### B. Revised Skill Structure (16 sections)

```
 1. REGIME IDENTIFICATION (revised — 5 contest archetypes)
 2. TOTAL OWNERSHIP (revised — by archetype)
 3. SALARY RULES (kept — universal)
 4. SALARY STRUCTURE (kept — stars & scrubs universal)
 5. OWNERSHIP COMPOSITION (revised — by archetype)
 6. DUPLICATION (kept — universal)
 7. FIGHTER SELECTION: WEIGHT CLASS & GENDER (NEW)
 8. FIGHTER SELECTION: LINE MOVEMENT (NEW)
 9. FIGHTER SELECTION: PROBABILITY BANDS (NEW)
10. FIGHTER SELECTION: CARD POSITION (revised + expanded)
11. SCORING PATTERNS & DUD AVOIDANCE (NEW)
12. ODDS COMPOSITION (revised — Odds API dataset, 12x more data)
13. CONTEST ARCHETYPES: DETAILED PROFILES (NEW)
14. LINEUP REVIEW WORKFLOW (revised — incorporate fighter rules)
15. QUICK REFERENCE (revised)
16. DATA QUALITY NOTES (revised)
```

### C. New Concepts

**Effective Entries:** `min(your_entries, max_entries) / contest_size`. Captures field coverage percentage. Drives diversification vs concentration strategy.

**Weight Class Tiers:**
- Tier S: W-Strawweight, W-Flyweight, Featherweight
- Tier A: Welterweight, Heavyweight, Middleweight
- Tier B: Lightweight, Light Heavyweight
- Tier F: Bantamweight, Flyweight (60%+ dud rate)

**Line Movement Filter (from Odds API):**
- Stable: safest, best floor
- Steamed 20pp+: screaming plays (8.8 pts/$1K, 64% ceiling)
- Steamed 2-5pp: moderate signal
- Steamed 5-10pp: TRAP (2.5 pts/$1K, 73% dud)
- Drifted: fade

**Fighter Profile Tags:**
- Cash Lock: 25%+ owned + stable/steamed + favorite + Tier S/A
- GPP Leverage: 5-15% owned + slight dog (35-50%) + Tier S/A
- Trap Play: 20%+ owned + drifted line
- Dud Risk: BW/FLW underdog

## Data Sources

| Source | Records | Coverage |
|--------|---------|----------|
| Lineup data (existing) | 2.3M lineups, 505 contests | Jan 2022 - Dec 2025 |
| Odds API (new) | 10,485 fighter-events with closing lines | 131 events, 84% DFS match rate |
| Construction rules analysis | 33,431 fighter-events (with weight class) | Same 131 events |

## Key Questions the Analysis Must Answer

1. Does 20-Max behave more like SE/3-Max (chalk-friendly) or 150-Max (contrarian)?
2. Do fighter-level rules (weight class, line movement) interact with contest archetype?
3. Does the Odds API data (10,485 events) confirm or change the BFO-based rules (849 events)?
4. What is the optimal favorite count by archetype?
5. Do effective entries predict optimal ownership strategy?

## Risk

- **Sample size for SE/3-Max:** These have fewer lineups. Rules may be Tier 2/3.
- **Odds API row count inflation:** The construction analysis shows 33,431 rows vs expected 10,485 due to JOINs creating duplicates. Must fix before trusting relative comparisons.
- **Skill length:** ~1,400 lines may cause context window pressure. Mitigate by keeping tables compact and using the Quick Reference as a standalone cheat sheet.

## Success Criteria

1. Each of the 5 contest archetypes has at least 3 distinct, data-backed rules that differ from other archetypes
2. Fighter selection rules improve upon "play favorites" baseline by identifying specific weight class/gender/line movement targets
3. No contradictions between new fighter rules and existing contest-level rules
4. All rules have sample size and confidence tier annotations
