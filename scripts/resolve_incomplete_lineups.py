"""
Resolve incomplete lineups in resultsdb_ufc.db.

FantasyLabs CDN returns -1 for player IDs it can't resolve. This script
recovers the missing fighter using three resolution methods:

1. SALARY DIFF: total_salary - sum(known fighters' salary) = missing fighter's salary.
   Cross-contest: same lineup_hash in another contest may store the full total.
2. POINTS + OWNERSHIP + CAP: Derive missing fighter's actual_points and ownership
   from lineup totals minus known fighters. Combined with salary cap constraint,
   this uniquely identifies the fighter.
3. (Future) 2-missing lineups could use combinatorial matching.

Usage:
    python resolve_incomplete_lineups.py [--dry-run] [--verbose]
"""

import argparse
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "Desktop" / "DFS" / "Databases" / "resultsdb_ufc.db"


def load_player_data(conn):
    """Load player salary, points, ownership indexed by (contest_id, player_id)."""
    c = conn.cursor()
    c.execute("SELECT contest_id, player_id, actual_points, ownership, salary FROM players")
    data = {}
    for cid, pid, pts, own, sal in c.fetchall():
        data[(cid, pid)] = (pts or 0.0, own or 0.0, sal or 0)
    return data


def load_date_players(conn):
    """Build date -> list of (player_id, pts, own, sal) for candidate matching."""
    c = conn.cursor()
    c.execute("""
        SELECT p.player_id, p.actual_points, p.ownership, p.salary, c.contest_date
        FROM players p
        JOIN contests c ON c.contest_id = p.contest_id
    """)
    date_players = defaultdict(list)
    date_sal_to_players = defaultdict(set)
    for pid, pts, own, sal, dt in c.fetchall():
        date_str = dt[:10] if dt else ""
        date_players[date_str].append((pid, pts or 0.0, own or 0.0, sal or 0))
        date_sal_to_players[(date_str, sal or 0)].add(pid)
    return date_players, date_sal_to_players


def load_contest_dates(conn):
    """Map contest_id -> date string (YYYY-MM-DD)."""
    c = conn.cursor()
    c.execute("SELECT contest_id, contest_date FROM contests")
    return {cid: (dt[:10] if dt else "") for cid, dt in c.fetchall()}


def load_incomplete_lineups(conn, n_missing=1):
    """Load lineups with exactly n_missing -1 slots."""
    c = conn.cursor()
    neg1_chars = n_missing * 2  # each '-1' is 2 chars in the replacement diff
    c.execute(f"""
        SELECT l.id, l.lineup_hash, l.contest_id, l.total_salary, l.points,
               l.total_ownership
        FROM lineups l
        WHERE l.lineup_hash LIKE '%-1%'
          AND LENGTH(l.lineup_hash) - LENGTH(REPLACE(l.lineup_hash, '-1', '')) = {neg1_chars}
    """)
    return c.fetchall()


def resolve_by_salary(hash_data, player_data, contest_dates, date_sal_to_players):
    """Method 1: Salary diff resolution (cross-contest)."""
    resolved = {}

    for lh, appearances in hash_data.items():
        parts = lh.split(":")
        known_pids = set(int(p) for p in parts if p != "-1")

        for cid, tsal, pts, town in appearances:
            known_sal = 0
            all_found = True
            for pid in known_pids:
                d = player_data.get((cid, pid))
                if d:
                    known_sal += d[2]
                else:
                    all_found = False
                    break

            if not all_found:
                continue

            diff = tsal - known_sal
            if 6000 <= diff <= 10600:
                dt = contest_dates.get(cid, "")
                candidates = date_sal_to_players.get((dt, diff), set()) - known_pids
                if len(candidates) == 1:
                    resolved[lh] = next(iter(candidates))
                    break

    return resolved


def resolve_by_pts_own_cap(hash_data, resolved_already, player_data,
                           contest_dates, date_players):
    """Method 2: Points + ownership + salary cap constraint."""
    resolved = {}

    for lh in hash_data:
        if lh in resolved_already:
            continue

        appearances = hash_data[lh]
        parts = lh.split(":")
        known_pids = set(int(p) for p in parts if p != "-1")

        for cid, tsal, lineup_pts, lineup_town in appearances:
            if not lineup_pts or not lineup_town:
                continue

            known_pts = 0.0
            known_own = 0.0
            known_sal = 0
            all_found = True
            for pid in known_pids:
                d = player_data.get((cid, pid))
                if d:
                    known_pts += d[0]
                    known_own += d[1]
                    known_sal += d[2]
                else:
                    all_found = False
                    break

            if not all_found:
                continue

            missing_pts = round(lineup_pts - known_pts, 2)
            missing_own = round(lineup_town - known_own, 2)
            max_sal = 50000 - known_sal
            dt = contest_dates.get(cid, "")

            candidates = set()
            for pid, pts, own, sal in date_players.get(dt, []):
                if pid in known_pids:
                    continue
                if (round(pts, 2) == missing_pts
                        and round(own, 2) == missing_own
                        and sal <= max_sal):
                    candidates.add(pid)

            if len(candidates) == 1:
                resolved[lh] = next(iter(candidates))
                break

    return resolved


def apply_fixes(conn, hash_data, resolution_map, player_data, dry_run=False):
    """
    Apply fixes to the database:
    1. Update lineup_hash (replace -1 with resolved player_id)
    2. Insert into lineup_players
    3. Recalculate total_salary/total_ownership if needed
    """
    c = conn.cursor()

    rows_updated = 0
    rows_merged = 0
    lp_inserted = 0

    for lh, missing_pid in resolution_map.items():
        parts = lh.split(":")
        neg1_idx = parts.index("-1")
        new_parts = parts.copy()
        new_parts[neg1_idx] = str(missing_pid)
        new_hash = ":".join(new_parts)

        for cid, tsal, pts, town in hash_data[lh]:
            c.execute(
                "SELECT id FROM lineups WHERE lineup_hash = ? AND contest_id = ?",
                (lh, cid),
            )
            lineup_ids = [row[0] for row in c.fetchall()]

            for lid in lineup_ids:
                if dry_run:
                    rows_updated += 1
                    lp_inserted += 1
                    continue

                # Check if the resolved hash already exists in this contest
                c.execute(
                    "SELECT id FROM lineups WHERE lineup_hash = ? AND contest_id = ?",
                    (new_hash, cid),
                )
                existing = c.fetchone()
                if existing:
                    # Complete version already exists — just delete the incomplete
                    # Don't merge user_count: the FantasyLabs API already counted
                    # these users in the complete version's user_count. Adding
                    # would double-count and inflate scraped_prizes.
                    c.execute(
                        "DELETE FROM lineup_players WHERE lineup_id = ?", (lid,)
                    )
                    c.execute("DELETE FROM lineups WHERE id = ?", (lid,))
                    rows_merged += 1
                    continue

                # Update lineup_hash
                c.execute(
                    "UPDATE lineups SET lineup_hash = ? WHERE id = ?",
                    (new_hash, lid),
                )
                rows_updated += 1

                # Insert into lineup_players
                c.execute(
                    "SELECT roster_slot FROM lineup_players WHERE lineup_id = ?",
                    (lid,),
                )
                existing_slots = {row[0] for row in c.fetchall()}
                all_slots = {str(i) for i in range(6)}
                missing_slots = all_slots - existing_slots
                if not missing_slots:
                    all_slots = {f"F{i}" for i in range(6)}
                    missing_slots = all_slots - existing_slots
                if not missing_slots:
                    slot = str(neg1_idx)
                else:
                    slot = min(missing_slots)

                c.execute(
                    """INSERT OR IGNORE INTO lineup_players
                       (lineup_id, player_id, roster_slot) VALUES (?, ?, ?)""",
                    (lid, missing_pid, slot),
                )
                lp_inserted += 1

                # Fix total_salary if it only had 5 fighters' worth
                d = player_data.get((cid, missing_pid))
                if d:
                    missing_sal = d[2]
                    known_pids = [int(p) for p in parts if p != "-1"]
                    known_sal = sum(
                        player_data.get((cid, pid), (0, 0, 0))[2]
                        for pid in known_pids
                    )
                    if tsal > 0 and abs(tsal - known_sal) < 100:
                        c.execute(
                            "UPDATE lineups SET total_salary = ? WHERE id = ?",
                            (tsal + missing_sal, lid),
                        )

        # Commit in batches to avoid huge transactions
        if rows_updated % 10000 == 0 and rows_updated > 0 and not dry_run:
            conn.commit()

    if not dry_run:
        conn.commit()

    return rows_updated, rows_merged, lp_inserted


def main():
    parser = argparse.ArgumentParser(description="Resolve incomplete UFC DFS lineups")
    parser.add_argument("--dry-run", action="store_true", help="Don't modify database")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print(f"Database: {DB_PATH}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")

    conn = sqlite3.connect(str(DB_PATH), timeout=60)

    # Pre-counts
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM lineups WHERE lineup_hash NOT LIKE '%-1%'")
    complete_before = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM lineups")
    total = c.fetchone()[0]
    print(f"\nBefore: {complete_before:,} / {total:,} complete ({complete_before/total*100:.1f}%)")

    # Load data
    t0 = time.time()
    print("\nLoading player data...")
    player_data = load_player_data(conn)
    print(f"  {len(player_data):,} (contest, player) entries")

    print("Loading date-indexed player maps...")
    date_players, date_sal_to_players = load_date_players(conn)

    print("Loading contest dates...")
    contest_dates = load_contest_dates(conn)

    print("Loading 1-missing lineups...")
    raw = load_incomplete_lineups(conn, n_missing=1)
    print(f"  {len(raw):,} lineup rows")

    # Group by hash
    hash_data = defaultdict(list)
    for lid, lh, cid, tsal, pts, town in raw:
        hash_data[lh].append((cid, tsal or 0, pts or 0, town or 0))
    print(f"  {len(hash_data):,} unique hashes")

    # Method 1: Salary diff
    print("\n--- Method 1: Salary Diff (cross-contest) ---")
    salary_map = resolve_by_salary(hash_data, player_data, contest_dates,
                                   date_sal_to_players)
    salary_rows = sum(len(hash_data[lh]) for lh in salary_map)
    print(f"  Resolved: {len(salary_map):,} hashes ({salary_rows:,} rows)")

    # Method 2: Points + ownership + cap
    print("\n--- Method 2: Points + Ownership + Salary Cap ---")
    pts_map = resolve_by_pts_own_cap(hash_data, salary_map, player_data,
                                     contest_dates, date_players)
    pts_rows = sum(len(hash_data[lh]) for lh in pts_map)
    print(f"  Resolved: {len(pts_map):,} hashes ({pts_rows:,} rows)")

    # Combined
    combined = {**salary_map, **pts_map}
    combined_rows = sum(len(hash_data[lh]) for lh in combined)
    print(f"\n=== TOTAL: {len(combined):,} hashes, {combined_rows:,} lineup rows ===")

    # Apply
    print(f"\nApplying fixes ({'DRY RUN' if args.dry_run else 'LIVE'})...")
    t1 = time.time()
    updated, merged, inserted = apply_fixes(conn, hash_data, combined, player_data,
                                            dry_run=args.dry_run)
    print(f"  Lineup rows updated: {updated:,}")
    print(f"  Lineup rows merged (duplicate removed): {merged:,}")
    print(f"  Lineup_players inserted: {inserted:,}")
    print(f"  Time: {time.time()-t1:.1f}s")

    # Post-counts
    if not args.dry_run:
        c.execute("SELECT COUNT(*) FROM lineups WHERE lineup_hash NOT LIKE '%-1%'")
        complete_after = c.fetchone()[0]
        print(f"\nAfter: {complete_after:,} / {total:,} complete ({complete_after/total*100:.1f}%)")
        print(f"Recovery: +{complete_after - complete_before:,} lineups")
    else:
        print(f"\nProjected: {complete_before + combined_rows:,} / {total:,} complete "
              f"({(complete_before + combined_rows)/total*100:.1f}%)")

    print(f"\nTotal time: {time.time()-t0:.1f}s")
    conn.close()


if __name__ == "__main__":
    main()
