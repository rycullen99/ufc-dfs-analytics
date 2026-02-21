"""
Pull historical UFC/MMA moneyline odds from The Odds API.

Strategy:
  1. Pull one snapshot per event date (near fight time) — 132 requests
  2. Each snapshot contains ALL upcoming MMA events, not just that night's fights
  3. Opening lines are extracted for free: for each event, the earliest snapshot
     where it appears (from a prior event date's pull) is its opening line

Cost: 132 requests × 10 credits = 1,320 credits total.

Usage:
    python scripts/pull_odds_api.py              # full run
    python scripts/pull_odds_api.py --dry-run    # show plan, no API calls
"""

import os
import json
import time
import copy
import sqlite3
import argparse
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SPORT = "mma_mixed_martial_arts"
BASE_URL = "https://api.the-odds-api.com/v4/historical/sports/{sport}/odds"
DB_PATH = os.path.expanduser("~/Desktop/resultsdb_ufc.db")
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "odds_api"
CHECKPOINT_PATH = OUTPUT_DIR / "checkpoint.json"

REQUEST_DELAY = 0.5  # seconds between API calls

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def load_api_key() -> str:
    """Load API key from environment or .env files."""
    for env_path in [".env", os.path.expanduser("~/nhl_model/.env")]:
        if os.path.exists(env_path):
            load_dotenv(env_path)

    key = os.getenv("ODDS_API_KEY")
    if not key:
        raise SystemExit(
            "ODDS_API_KEY not found. Set it in environment or .env file."
        )
    return key


def get_snapshot(
    date_iso: str,
    api_key: str,
    *,
    credits_tracker: dict | None = None,
) -> dict:
    """
    Pull a single historical snapshot from The Odds API.

    Returns the full response payload with keys:
      timestamp, previous_timestamp, next_timestamp, data
    """
    url = BASE_URL.format(sport=SPORT)
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
        "date": date_iso,
    }

    r = requests.get(url, params=params, timeout=60)

    if credits_tracker is not None:
        remaining = r.headers.get("x-requests-remaining")
        used = r.headers.get("x-requests-used")
        if remaining:
            credits_tracker["remaining"] = int(remaining)
        if used:
            credits_tracker["used"] = int(used)

    if r.status_code == 422:
        return {"data": [], "timestamp": None}

    r.raise_for_status()
    time.sleep(REQUEST_DELAY)
    return r.json()


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def date_id_to_closing_iso(date_id: int) -> str:
    """
    Convert date_id (YYYYMMDD int) to ISO timestamp for closing line.

    UFC events in the US typically have main cards starting ~10pm ET (03:00 UTC next day).
    Request at 02:00 UTC next day to get odds close to lock time.
    """
    s = str(date_id)
    dt = datetime(int(s[:4]), int(s[4:6]), int(s[6:8]), tzinfo=timezone.utc)
    closing_dt = dt + timedelta(hours=26)
    return closing_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Phase 1: Pull all snapshots
# ---------------------------------------------------------------------------


def pull_snapshots(
    date_ids: list[int],
    api_key: str,
    checkpoint: dict,
    credits: dict,
) -> dict:
    """
    Pull one snapshot per event date. Each snapshot contains all upcoming
    MMA events at that point in time.

    Returns dict mapping date_id (str) -> list of events.
    """
    snapshots = checkpoint.get("snapshots", {})

    for i, date_id in enumerate(date_ids):
        date_key = str(date_id)
        if date_key in snapshots:
            log.info("  [%d/%d] date %s — cached", i + 1, len(date_ids), date_id)
            continue

        iso = date_id_to_closing_iso(date_id)
        log.info(
            "  [%d/%d] date %s → %s (credits remaining: %s)",
            i + 1, len(date_ids), date_id, iso, credits.get("remaining", "?"),
        )

        snapshot = get_snapshot(iso, api_key, credits_tracker=credits)
        events = snapshot.get("data", [])

        # Tag each event with the snapshot metadata
        for ev in events:
            ev["_snapshot_date_id"] = date_id
            ev["_snapshot_timestamp"] = snapshot.get("timestamp")

        snapshots[date_key] = events
        checkpoint["snapshots"] = snapshots
        save_checkpoint(checkpoint)

    return snapshots


# ---------------------------------------------------------------------------
# Phase 2: Extract opening + closing lines from snapshots
# ---------------------------------------------------------------------------


def extract_opening_and_closing(
    snapshots: dict,
    date_ids: list[int],
) -> tuple[dict, dict]:
    """
    From the pulled snapshots, extract opening and closing lines for each event.

    Opening = earliest snapshot where an event appears (from a prior event date's pull).
    Closing = the snapshot from the event's own date_id (fight night).

    Returns (closing_events, opening_events) where each maps
    event_id -> event data dict.
    """
    # Build index: for each event_id, collect all (snapshot_date_id, event_data)
    event_appearances: dict[str, list[tuple[int, dict]]] = {}

    for date_key, events in snapshots.items():
        snapshot_date_id = int(date_key)
        for ev in events:
            eid = ev.get("id")
            if not eid:
                continue
            event_appearances.setdefault(eid, []).append((snapshot_date_id, ev))

    closing_events = {}
    opening_events = {}

    for event_id, appearances in event_appearances.items():
        # Sort by snapshot date
        appearances.sort(key=lambda x: x[0])

        # The event's own date: use commence_time to determine which date_id
        # it belongs to, or just find the snapshot closest to commence_time
        commence = appearances[0][1].get("commence_time", "")

        # Determine the event's actual fight date_id
        fight_date_id = _commence_to_date_id(commence)

        # Opening = earliest appearance
        earliest_date, earliest_ev = appearances[0]
        opening = copy.deepcopy(earliest_ev)
        opening["_line_type"] = "opening"
        opening["_date_id"] = fight_date_id
        opening_events[event_id] = opening

        # Closing = latest appearance (ideally from fight night or closest to it)
        latest_date, latest_ev = appearances[-1]
        closing = copy.deepcopy(latest_ev)
        closing["_line_type"] = "closing"
        closing["_date_id"] = fight_date_id
        closing_events[event_id] = closing

        # Log line movement if both exist
        if len(appearances) > 1:
            open_odds = _extract_consensus_prob(opening)
            close_odds = _extract_consensus_prob(closing)
            if open_odds and close_odds:
                home = opening.get("home_team", "?")
                move = close_odds - open_odds
                log.debug(
                    "  %s: open=%.1f%% close=%.1f%% move=%+.1f%% (%d snapshots)",
                    home, open_odds * 100, close_odds * 100, move * 100,
                    len(appearances),
                )

    log.info(
        "Extracted %d closing + %d opening lines from %d unique events",
        len(closing_events), len(opening_events), len(event_appearances),
    )

    # Count events with line movement data (appeared in >1 snapshot)
    multi_snap = sum(1 for apps in event_appearances.values() if len(apps) > 1)
    log.info(
        "Events with line movement data (>1 snapshot): %d/%d (%.0f%%)",
        multi_snap, len(event_appearances),
        100 * multi_snap / max(len(event_appearances), 1),
    )

    return closing_events, opening_events


def _commence_to_date_id(commence_iso: str) -> int | None:
    """Convert commence_time ISO string to date_id (YYYYMMDD int)."""
    if not commence_iso:
        return None
    try:
        dt = datetime.fromisoformat(commence_iso.replace("Z", "+00:00"))
        # UFC events often start late evening US time, which is next day UTC.
        # Subtract 6 hours to approximate US Eastern fight date.
        dt_eastern_approx = dt - timedelta(hours=6)
        return int(dt_eastern_approx.strftime("%Y%m%d"))
    except (ValueError, TypeError):
        return None


def _extract_consensus_prob(event: dict) -> float | None:
    """Quick extraction of home fighter's average implied prob from an event."""
    probs = []
    for bm in event.get("bookmakers", []):
        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue
            outcomes = market.get("outcomes", [])
            if len(outcomes) == 2:
                price = outcomes[0].get("price")
                if price is not None:
                    probs.append(american_to_implied(price))
    return sum(probs) / len(probs) if probs else None


# ---------------------------------------------------------------------------
# Parsing & DB storage
# ---------------------------------------------------------------------------


def parse_event_odds(event: dict) -> list[dict]:
    """
    Parse a single event's bookmaker odds into rows.

    Returns list of dicts with:
        event_id, commence_time, fighter_a, fighter_b,
        bookmaker, odds_a, odds_b, implied_prob_a, implied_prob_b,
        snapshot_timestamp, line_type, date_id
    """
    if event is None:
        return []

    rows = []
    event_id = event.get("id")
    commence = event.get("commence_time")
    line_type = event.get("_line_type", "unknown")
    snapshot_ts = event.get("_snapshot_timestamp")
    date_id = event.get("_date_id")

    for bm in event.get("bookmakers", []):
        bm_key = bm.get("key", "")
        bm_title = bm.get("title", "")

        for market in bm.get("markets", []):
            if market.get("key") != "h2h":
                continue

            outcomes = market.get("outcomes", [])
            if len(outcomes) != 2:
                continue

            o1, o2 = outcomes
            rows.append({
                "event_id": event_id,
                "commence_time": commence,
                "fighter_a": o1.get("name", ""),
                "fighter_b": o2.get("name", ""),
                "bookmaker_key": bm_key,
                "bookmaker": bm_title,
                "odds_a": o1.get("price"),
                "odds_b": o2.get("price"),
                "implied_prob_a": american_to_implied(o1.get("price")),
                "implied_prob_b": american_to_implied(o2.get("price")),
                "line_type": line_type,
                "snapshot_timestamp": snapshot_ts,
                "date_id": date_id,
            })

    return rows


def american_to_implied(odds: int | float | None) -> float | None:
    """Convert American odds to implied probability."""
    if odds is None:
        return None
    if odds > 0:
        return 100.0 / (odds + 100.0)
    elif odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    else:
        return 0.5


def build_consensus_odds(all_rows: list[dict]) -> list[dict]:
    """
    Build consensus (market average) odds per fighter per event per line_type.

    Averages implied probabilities across all bookmakers, then normalizes
    to remove vig.
    """
    import pandas as pd

    if not all_rows:
        return []

    df = pd.DataFrame(all_rows)

    consensus = (
        df.groupby(["event_id", "date_id", "commence_time", "fighter_a", "fighter_b", "line_type"])
        .agg(
            implied_prob_a=("implied_prob_a", "mean"),
            implied_prob_b=("implied_prob_b", "mean"),
            n_books=("bookmaker_key", "nunique"),
            snapshot_timestamp=("snapshot_timestamp", "first"),
        )
        .reset_index()
    )

    # Remove vig: normalize so probs sum to 1
    total = consensus["implied_prob_a"] + consensus["implied_prob_b"]
    consensus["implied_prob_a_fair"] = consensus["implied_prob_a"] / total
    consensus["implied_prob_b_fair"] = consensus["implied_prob_b"] / total

    # Compute line movement (closing - opening) where both exist
    opening = consensus[consensus["line_type"] == "opening"][
        ["event_id", "fighter_a", "implied_prob_a_fair", "implied_prob_b_fair"]
    ].rename(columns={
        "implied_prob_a_fair": "open_prob_a",
        "implied_prob_b_fair": "open_prob_b",
    })

    closing = consensus[consensus["line_type"] == "closing"][
        ["event_id", "fighter_a", "implied_prob_a_fair", "implied_prob_b_fair"]
    ].rename(columns={
        "implied_prob_a_fair": "close_prob_a",
        "implied_prob_b_fair": "close_prob_b",
    })

    if len(opening) > 0 and len(closing) > 0:
        movement = opening.merge(closing, on=["event_id", "fighter_a"], how="inner")
        movement["line_move_a"] = movement["close_prob_a"] - movement["open_prob_a"]
        movement["line_move_b"] = movement["close_prob_b"] - movement["open_prob_b"]

        # Merge movement back into consensus
        consensus = consensus.merge(
            movement[["event_id", "fighter_a", "line_move_a", "line_move_b"]],
            on=["event_id", "fighter_a"],
            how="left",
        )
    else:
        consensus["line_move_a"] = None
        consensus["line_move_b"] = None

    return consensus.to_dict("records")


def save_to_database(all_rows: list[dict], consensus: list[dict], db_path: str):
    """Save raw bookmaker odds and consensus odds to the database."""
    import pandas as pd

    conn = sqlite3.connect(db_path)

    # Raw bookmaker odds (per-book, per-line-type)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS odds_api_raw (
            event_id TEXT NOT NULL,
            date_id INTEGER,
            commence_time TEXT,
            fighter_a TEXT,
            fighter_b TEXT,
            bookmaker_key TEXT,
            bookmaker TEXT,
            odds_a REAL,
            odds_b REAL,
            implied_prob_a REAL,
            implied_prob_b REAL,
            line_type TEXT NOT NULL,
            snapshot_timestamp TEXT,
            PRIMARY KEY (event_id, bookmaker_key, line_type)
        )
    """)

    # Consensus odds (market average, vig-removed, with line movement)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS odds_api_consensus (
            event_id TEXT NOT NULL,
            date_id INTEGER,
            commence_time TEXT,
            fighter_a TEXT,
            fighter_b TEXT,
            line_type TEXT NOT NULL,
            implied_prob_a REAL,
            implied_prob_b REAL,
            implied_prob_a_fair REAL,
            implied_prob_b_fair REAL,
            n_books INTEGER,
            snapshot_timestamp TEXT,
            line_move_a REAL,
            line_move_b REAL,
            PRIMARY KEY (event_id, line_type)
        )
    """)

    raw_df = pd.DataFrame(all_rows)
    if len(raw_df) > 0:
        raw_df.to_sql("odds_api_raw", conn, if_exists="replace", index=False)
        log.info("Saved %d raw odds rows", len(raw_df))

    cons_df = pd.DataFrame(consensus)
    if len(cons_df) > 0:
        cons_df.to_sql("odds_api_consensus", conn, if_exists="replace", index=False)
        log.info("Saved %d consensus odds rows", len(cons_df))

    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Checkpoint management
# ---------------------------------------------------------------------------


def load_checkpoint() -> dict:
    """Load progress checkpoint from disk."""
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            return json.load(f)
    return {}


def save_checkpoint(checkpoint: dict):
    """Save progress checkpoint to disk."""
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump(checkpoint, f, default=str)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def get_event_dates(db_path: str) -> list[int]:
    """Get all unique event date_ids from the database."""
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT DISTINCT date_id FROM contests ORDER BY date_id"
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def main():
    parser = argparse.ArgumentParser(description="Pull UFC odds from The Odds API")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without API calls")
    parser.add_argument("--db", default=DB_PATH, help="Path to SQLite database")
    args = parser.parse_args()

    api_key = load_api_key()
    date_ids = get_event_dates(args.db)
    log.info("Found %d event dates (%d – %d)", len(date_ids), date_ids[0], date_ids[-1])

    if args.dry_run:
        log.info("DRY RUN — no API calls will be made")
        log.info("Snapshots to pull: %d × 10 credits = %d credits", len(date_ids), len(date_ids) * 10)
        log.info("Opening lines: extracted from snapshots (0 additional credits)")
        log.info("Total: %d credits", len(date_ids) * 10)
        return

    checkpoint = load_checkpoint()
    credits = {"remaining": "?", "used": "?"}

    # Phase 1: Pull all snapshots
    log.info("=" * 60)
    log.info("PHASE 1: Pulling snapshots for %d event dates", len(date_ids))
    log.info("=" * 60)
    snapshots = pull_snapshots(date_ids, api_key, checkpoint, credits)
    total_events = sum(len(v) for v in snapshots.values())
    log.info("Snapshots complete: %d events across %d dates", total_events, len(snapshots))

    # Phase 2: Extract opening + closing from snapshots
    log.info("=" * 60)
    log.info("PHASE 2: Extracting opening + closing lines (0 API calls)")
    log.info("=" * 60)
    closing_events, opening_events = extract_opening_and_closing(snapshots, date_ids)

    # Phase 3: Parse and save
    log.info("=" * 60)
    log.info("PHASE 3: Parsing odds and saving to database")
    log.info("=" * 60)

    all_rows = []
    for ev in closing_events.values():
        all_rows.extend(parse_event_odds(ev))
    for ev in opening_events.values():
        all_rows.extend(parse_event_odds(ev))

    log.info("Parsed %d total raw odds rows", len(all_rows))

    consensus = build_consensus_odds(all_rows)
    log.info("Built %d consensus odds rows", len(consensus))

    save_to_database(all_rows, consensus, args.db)

    # Save raw JSON backup
    raw_json_path = OUTPUT_DIR / "odds_api_raw.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(raw_json_path, "w") as f:
        json.dump({"snapshots": snapshots}, f, default=str)
    log.info("Raw JSON saved to %s", raw_json_path)

    log.info("=" * 60)
    log.info("DONE — Credits used: %s, remaining: %s",
             credits.get("used", "?"), credits.get("remaining", "?"))
    log.info("=" * 60)


if __name__ == "__main__":
    main()
