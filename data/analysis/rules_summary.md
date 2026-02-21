# Skill v2 Rules Summary — Synthesized from 3 Analysis Scripts

**Date:** 2026-02-21
**Sources:** archetype_results.json, odds_composition_results.json, fighter_archetype_results.json

---

## 1. Contest Archetype Conclusions

### Data Reality
| Archetype | Contests | Lineups | Confidence |
|-----------|----------|---------|------------|
| UNLIMITED | 358 | 2,278,873 | TIER 1 |
| LIMITED | 131 | 28,946 | TIER 2 |
| SE | 8 | 87 | TIER 3 (directional only) |
| 20-Max | 3 | 1,702 | TIER 3 (directional only) |
| 150-Max | 4 | 2,876 | TIER 3 (directional only) |

### Key Archetype Differences (UNLIMITED vs LIMITED)

**OWNERSHIP — GENUINE REVERSAL:**
- UNLIMITED: Best ROI at 100-125% (1.53x). Moderate chalk.
- LIMITED: Best ROI at 175-200% (1.31x). Heavy chalk.
- This is the most important archetype-specific rule.

**MIN OWNERSHIP FIGHTER:**
- UNLIMITED: Sweet spot 5-8% (1.57x ROI). Need a contrarian piece.
- LIMITED: Sweet spot 8-12% (1.05x ROI). Less contrarian room.

**MAX OWNERSHIP FIGHTER:**
- UNLIMITED: Best 20-30% max (1.92x). Moderate chalk anchor.
- LIMITED: Flat across 40-60% range (~0.95x). Higher chalk OK.

**SALARY REMAINING — UNIVERSAL:**
- Both: <$500 remaining is best (UNLIMITED 1.84x, LIMITED 1.41x).

**SALARY SPREAD — UNIVERSAL:**
- Both: $2K-3K spread is optimal (UNLIMITED 1.40x, LIMITED 1.25x).

**$8.5K+ COUNT — UNIVERSAL:**
- Both: 3 fighters at $8.5K+ is sweet spot (UNLIMITED 1.68x, LIMITED 1.62x).
- 4 fighters also strong (UNLIMITED 1.76x, LIMITED 1.59x).

**DUPLICATION — UNIVERSAL:**
- Both: Unique lineups best (UNLIMITED 1.46x, LIMITED 1.01x).
- But limited has less duplication penalty overall.

### UNLIMITED Sub-Splits
- Entry fee: HIGH ($25-30) = 1.47x ROI best, then MID ($15-20) = 1.23x.
- Field size: Large (20K-50K) = 1.29x best. Mega 50K+ slightly lower.

### Scoring Thresholds
| Metric | UNLIMITED | LIMITED | SE |
|--------|-----------|---------|------|
| Avg Points | 282 | 272 | 325 |
| Cash Line | 331 | 347 | 427 |
| Top 5% | 413 | 424 | 480 |
| Top 1% | 470 | 489 | 480 |
| Winner | 556 | 503 | 480 |

---

## 2. Fighter Selection Rules — ALL UNIVERSAL

**Tier rankings are IDENTICAL between UNLIMITED and LIMITED.** No archetype-specific fighter rules found.

### Weight Class Tiers (by FPTS/$1K, UNLIMITED data)
| Tier | Weight Classes | Avg Val | Ceil% | Dud% |
|------|---------------|---------|-------|------|
| S | W-Strawweight (4.64), Welterweight (4.33), W-Flyweight (4.25) | 4.41 | 26.8 | 37.8 |
| A | W-Bantamweight (4.17), Lightweight (4.11) | 4.14 | 26.5 | 39.3 |
| B | Featherweight (3.93), Light Heavyweight (3.92), Middleweight (3.86) | 3.90 | 24.8 | 42.8 |
| C | Heavyweight (3.69) | 3.69 | 23.0 | 45.4 |
| F | Flyweight (3.40), Bantamweight (3.31) | 3.36 | 22.5 | 52.3 |

**NOTE:** Tier assignments shifted from the initial detailed analysis:
- Welterweight moved UP to S (was A) — consistently top-3 value
- Lightweight moved UP to A (was B) — strong value
- Featherweight moved DOWN to B (was S) — middle tier in current data
- Heavyweight moved DOWN to C (was A) — lowest non-F dud rate

### Gender — Women Consistently Undervalued
| Gender | Val | Ceil% | Dud% | Avg Own% |
|--------|-----|-------|------|----------|
| Women | 4.38 | 27.7 | 38.5 | 14.6% |
| Men | 3.89 | 24.4 | 43.7 | 17.3% |

Women: +0.49 val advantage, +3.3% ceiling edge, -5.2% dud edge, 2.7% lower ownership.
**Universal across UNLIMITED and LIMITED.**

### Line Movement (from fighter_odds_enriched, 10K+ events)
| Direction | Val | Ceil% | Dud% | Avg Own% |
|-----------|-----|-------|------|----------|
| Steamed (>2pp toward) | 4.84 | 33.5 | 42.1 | 25.1% |
| Stable (<2pp) | 4.91 | 30.6 | 31.6 | 17.8% |
| Drifted (>2pp away) | 3.71 | 19.8 | 37.9 | 19.9% |

Steamed-Drifted delta: +1.13 val (universal UNLIMITED/LIMITED).

**Steam magnitude (from odds composition analysis):**
| Magnitude | N | FPTS/$1K | Ceil% |
|-----------|---|----------|-------|
| Small (2-5pp) | 271 | 5.29 | 30.6% |
| Medium (5-10pp) | 139 | 4.42 | 24.5% |
| Large (10-20pp) | 41 | 5.33 | 34.1% |
| Huge (20pp+) | 15 | 9.00 | 60.0% |

**TRAP CONFIRMED:** 5-10pp steam is WORST — lower than small steam. The 20pp+ steam is elite but tiny sample.

**Field size interaction (UNLIMITED only):**
- Large fields (20K-50K): Steam delta +1.32
- Mega fields (50K+): Steam delta +0.90
- Steam signal matters more in Large fields.

### Probability Bands (from fighter_odds_enriched)
| Band | N | FPTS/$1K | Ceil% | Dud% | Avg Own% |
|------|---|----------|-------|------|----------|
| Heavy Fav (70%+) | 598 | 5.90 | 31.3 | 27.4 | 21.8 |
| Mod Fav (55-70%) | 1,177 | 5.57 | 24.9 | 23.4 | 21.4 |
| Toss-up (45-55%) | 493 | 5.23 | 23.5 | 26.8 | 21.7 |
| Mod Dog (30-45%) | 1,183 | 4.81 | 18.1 | 25.6 | 18.4 |
| Heavy Dog (<30%) | 593 | 2.78 | 6.9 | 35.2 | 11.6 |

Outscore rate delta (outscore% - implied prob):
- Heavy Fav: -38.4% (outscore 38.5% vs implied 76.8%)
- Mod Dog: -8.7% (outscore 28.7% vs implied 37.4%)
- **Favorites are overpriced relative to implied probability but still produce best DFS value.**

### Card Position
| Position | Val | Ceil% | Dud% |
|----------|-----|-------|------|
| Main Event | 6.01 | 50.7 | 41.7 |
| Co-Main | 5.85 | 47.6 | 40.8 |
| Main Card | 5.25 | 35.8 | 37.0 |
| Prelim | 4.51 | 26.4 | 32.9 |

**Universal across archetypes. Main card fighters have higher ceiling but also higher dud rate (binary outcome of higher-profile fights).**

---

## 3. Odds Composition — Lineup Level

### Favorite Count (649K lineups with 6/6 odds coverage)
| Fav Count | N | ROI | Cash% | AvgPts |
|-----------|---|-----|-------|--------|
| 0 | 1,005 | 0.37 | 14.6 | 251 |
| 1 | 8,644 | 0.50 | 17.3 | 312 |
| 2 | 71,036 | 1.05 | 21.5 | 340 |
| 3 | 288,886 | 1.39 | 24.1 | 346 |
| 4 | 249,875 | 1.84 | 27.4 | 353 |
| 5 | 29,371 | 2.34 | 30.4 | 356 |
| 6 | 470 | 1.32 | 36.6 | 340 |

**STRONG monotonic: more favorites = higher ROI up to 5. Six favorites drops off.**

### Heavy Favorite Count (70%+)
| Heavy Fav | N | ROI |
|-----------|---|-----|
| 0 | 196,658 | 1.65 |
| 1 | 296,878 | 1.31 |
| 2 | 139,083 | 1.82 |
| 3 | 16,528 | 2.74 |
| 4 | 140 | 0.53 |

**Sweet spot: 2-3 heavy favorites. 4+ collapses.**

### Favorite Count × Fee Tier
5 favorites at MID ($15-20): 3.04x ROI
5 favorites at HIGH ($25-30): 3.00x ROI
Favorite-loading matters MORE at higher entry fees.

---

## 4. Lineup Composition — Weight Class (Lineup-Level ROI)

### Tier S Count (UNLIMITED)
| S Count | N | ROI |
|---------|---|-----|
| 0 | 1,369,868 | 0.92 |
| 1 | 527,544 | 1.41 |
| 2 | 290,157 | 2.34 |
| 3 | 80,604 | 2.93 |
| 4 | 10,125 | 3.29 |

**More Tier S fighters = dramatically higher ROI.** Loading 3-4 Tier S fighters is optimal.

### Tier F Count (UNLIMITED)
| F Count | N | ROI |
|---------|---|-----|
| 0 | 1,561,578 | 1.17 |
| 1 | 481,901 | 1.40 |
| 2 | 180,898 | 2.16 |
| 3 | 46,528 | 1.66 |
| 4+ | 7,845 | 0.71 |

**Surprising: 1-2 Tier F fighters BOOST ROI** (leveraging low ownership + occasional ceiling). But 4+ is bad.

### Women Count (UNLIMITED)
| Women | N | ROI |
|-------|---|-----|
| 0 | 1,569,703 | 1.00 |
| 1 | 506,899 | 1.95 |
| 2 | 174,370 | 1.80 |
| 3 | 26,479 | 3.31 |

**Women fighters dramatically boost ROI.** 1-3 women's fighters optimal. Under-owned leverage.

---

## 5. Effective Entries

### ROI by Effective Entries Band
| EE Band | N | ROI |
|---------|---|-----|
| 0.1-1% | 2,225 | 6.87 |
| 1-5% | 1,841,633 | 1.19 |
| 5-10% | 285,061 | 1.18 |
| 10-50% | 182,196 | 1.02 |
| 50%+ | 1,369 | 0.82 |

Lower effective entries = higher ROI. This is mostly a size/fee proxy.

### EE × Ownership
- Low EE (0.1-1%): Optimal at 175-200% ownership (extreme chalk)
- Mid EE (1-5%): Optimal at 100-125% (moderate chalk)
- This confirms: small-field contests → chalk heavy; large-field → moderate.

---

## 6. Key Reversals & Non-Obvious Findings

1. **Ownership reversal**: UNLIMITED wants 100-125% total; LIMITED wants 175-200%. Only genuine archetype difference.
2. **Tier F surprise**: 1-2 BW/FLW fighters HELP lineups despite being individually bad — leverage play.
3. **Women's boost**: Largest single ROI lever. 3 women = 3.31x ROI (UNLIMITED).
4. **Steam trap**: 5-10pp steam is WORSE than 2-5pp — classic trap zone.
5. **6 favorites collapses**: ROI drops from 2.34 (5 favs) to 1.32 (6 favs).
6. **Fighter rules are universal**: No archetype-specific fighter selection rules found. All weight class/gender/line movement/prob band rules hold identically across UNLIMITED and LIMITED.

---

## 7. What Changed from v1

| Rule | v1 | v2 | Changed? |
|------|----|----|----------|
| Ownership sweet spot (UNLIMITED) | 100-125% (1.53x) | 100-125% (1.53x) | Same |
| Salary remaining | <$500 best | <$500 best (1.84x) | Confirmed |
| Unique lineups | 1.46x ROI | 1.46x | Confirmed |
| $8.5K+ count | 3+ fighters | 3-4 fighters (1.68-1.76x) | Refined |
| Weight class tiers | Not in v1 | S/A/B/C/F tiers | NEW |
| Gender value | Not in v1 | Women +0.49 val | NEW |
| Line movement | Not in v1 | Steamed/stable/drifted | NEW |
| Line movement trap | Not in v1 | 5-10pp trap zone | NEW |
| Prob bands | BFO (849 events) | Odds API (4,044 events) | EXPANDED |
| Favorite count (lineup) | 2-3 max | 4-5 optimal (2.34x at 5) | REVISED |
| Heavy fav count | max 2 (70%+) | 2-3 optimal (2.74x at 3) | REVISED |
| LIMITED ownership | Same as UNLIMITED | 175-200% (reversal) | NEW |
| Lineup composition | Not in v1 | Tier S/F/Women counts | NEW |
