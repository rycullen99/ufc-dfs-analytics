"""
Canonical fighter identity mapping.

Resolves name mismatches between DFS platforms, odds sources, and MMA databases
into a single canonical identity per fighter. Uses multi-tier matching adapted
from integrate_odds.py.
"""

import re
import sqlite3
from collections import defaultdict
from difflib import SequenceMatcher

from ..config import KNOWN_ALIASES


# ─── Name normalization ──────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """Lowercase, strip whitespace/periods/trailing punctuation, collapse spaces."""
    n = name.lower().strip().rstrip(".")
    n = n.replace(".", "").replace("-", " ")
    n = re.sub(r"\s+", " ", n).strip()
    return n


def last_name(name: str) -> str:
    """Extract last name (last word)."""
    parts = name.strip().split()
    return parts[-1].lower() if parts else ""


def reversed_name(name: str) -> str:
    """'Firstname Lastname' → 'Lastname Firstname'."""
    parts = name.strip().split()
    if len(parts) == 2:
        return f"{parts[1]} {parts[0]}"
    return name


# ─── Multi-tier matching ─────────────────────────────────────────────────────

def match_name(
    query: str,
    candidates: dict[str, int],
    *,
    fuzzy_threshold: float = 0.85,
) -> tuple[int | None, str]:
    """
    Match a query name against a dict of {normalized_name: canonical_id}.

    Returns (canonical_id, match_method) or (None, 'unmatched').

    Tiers:
      1. Exact normalized match
      2. Known alias lookup
      3. Last-name match (unambiguous)
      4. Reversed name order
      5. Fuzzy (SequenceMatcher > threshold)
    """
    norm = normalize_name(query)

    # Tier 1: exact
    if norm in candidates:
        return candidates[norm], "exact"

    # Tier 2: known alias
    alias = KNOWN_ALIASES.get(norm)
    if alias and alias in candidates:
        return candidates[alias], "alias"

    # Tier 3: unambiguous last name
    ln = last_name(query)
    ln_matches = [
        (cname, cid) for cname, cid in candidates.items() if last_name(cname) == ln
    ]
    if len(ln_matches) == 1:
        return ln_matches[0][1], "last_name"

    # Tier 4: reversed
    rev = normalize_name(reversed_name(query))
    if rev in candidates:
        return candidates[rev], "reversed"

    # Tier 5: fuzzy
    best_ratio, best_id = 0.0, None
    for cname, cid in candidates.items():
        ratio = SequenceMatcher(None, norm, cname).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_id = cid
    if best_ratio > fuzzy_threshold:
        return best_id, "fuzzy"

    return None, "unmatched"


# ─── Table creation ──────────────────────────────────────────────────────────

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS canonical_fighter (
    canonical_id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_name TEXT NOT NULL UNIQUE,
    weight_class TEXT
);

CREATE TABLE IF NOT EXISTS fighter_alias (
    alias_id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_id INTEGER NOT NULL REFERENCES canonical_fighter(canonical_id),
    alias_name TEXT NOT NULL,
    alias_source TEXT NOT NULL,  -- 'dfs', 'odds', 'mma_db', 'manual'
    confidence REAL DEFAULT 1.0,
    UNIQUE(alias_name, alias_source)
);

CREATE INDEX IF NOT EXISTS idx_fighter_alias_name ON fighter_alias(alias_name);
CREATE INDEX IF NOT EXISTS idx_fighter_alias_canonical ON fighter_alias(canonical_id);
"""


def create_tables(conn: sqlite3.Connection) -> None:
    """Create canonical_fighter and fighter_alias tables."""
    conn.executescript(SCHEMA_SQL)


def build_canonical_fighters(conn: sqlite3.Connection) -> dict[str, int]:
    """
    Build the canonical fighter identity table from existing DFS data.

    Process:
      1. Collect all unique fighter names from `players` (DFS source)
      2. Collect all unique names from `fighter_odds` (odds source, if exists)
      3. Normalize and deduplicate via multi-tier matching
      4. Insert into canonical_fighter + fighter_alias

    Returns mapping of normalized_name → canonical_id.
    """
    create_tables(conn)

    # Gather all unique DFS fighter names
    dfs_names = set()
    for row in conn.execute(
        "SELECT DISTINCT full_name FROM players WHERE full_name IS NOT NULL"
    ):
        dfs_names.add(row[0])

    # Also check fighters table if it exists
    try:
        for row in conn.execute(
            "SELECT DISTINCT full_name FROM fighters WHERE full_name IS NOT NULL"
        ):
            dfs_names.add(row[0])
    except sqlite3.OperationalError:
        pass

    # Build canonical set: normalized_name → canonical_name (display form)
    canonical_map: dict[str, str] = {}  # norm → display
    for name in sorted(dfs_names):
        norm = normalize_name(name)
        if norm not in canonical_map:
            canonical_map[norm] = name  # keep original casing for display

    # Handle known aliases — merge into existing canonical entries
    alias_targets: dict[str, str] = {}  # alias_norm → canonical_norm
    for alias_norm, canon_norm in KNOWN_ALIASES.items():
        if canon_norm in canonical_map:
            alias_targets[alias_norm] = canon_norm
        elif alias_norm in canonical_map:
            # The alias form is what we have; add canonical as alternate
            alias_targets[canon_norm] = alias_norm

    # Insert canonical fighters
    name_to_id: dict[str, int] = {}
    for norm, display in canonical_map.items():
        conn.execute(
            "INSERT OR IGNORE INTO canonical_fighter (canonical_name) VALUES (?)",
            (display,),
        )

    # Fetch back IDs
    for row in conn.execute("SELECT canonical_id, canonical_name FROM canonical_fighter"):
        name_to_id[normalize_name(row[1])] = row[0]

    # Insert DFS aliases
    for norm, display in canonical_map.items():
        cid = name_to_id.get(norm)
        if cid:
            conn.execute(
                "INSERT OR IGNORE INTO fighter_alias (canonical_id, alias_name, alias_source) "
                "VALUES (?, ?, 'dfs')",
                (cid, norm),
            )

    # Insert known alias mappings
    for alias_norm, canon_norm in alias_targets.items():
        cid = name_to_id.get(canon_norm)
        if cid:
            conn.execute(
                "INSERT OR IGNORE INTO fighter_alias (canonical_id, alias_name, alias_source) "
                "VALUES (?, ?, 'manual')",
                (cid, alias_norm),
            )

    conn.commit()

    # Build lookup from aliases
    lookup: dict[str, int] = {}
    for row in conn.execute(
        "SELECT alias_name, canonical_id FROM fighter_alias"
    ):
        lookup[row[0]] = row[1]

    # Also include canonical names directly
    for norm, cid in name_to_id.items():
        lookup[norm] = cid

    return lookup


def resolve_name(name: str, lookup: dict[str, int]) -> int | None:
    """Resolve a fighter name to its canonical_id using the lookup table."""
    cid, _ = match_name(name, lookup)
    return cid


def get_unmatched_report(
    conn: sqlite3.Connection,
    lookup: dict[str, int],
) -> list[dict]:
    """Find fighter names that don't resolve to any canonical_id."""
    unmatched = []

    # Check odds sources
    try:
        for row in conn.execute(
            "SELECT DISTINCT full_name, date_id FROM fighter_odds"
        ):
            norm = normalize_name(row[0])
            if norm not in lookup:
                cid, method = match_name(row[0], lookup)
                if cid is None:
                    unmatched.append({
                        "name": row[0],
                        "source": "fighter_odds",
                        "date_id": row[1],
                    })
    except sqlite3.OperationalError:
        pass

    return unmatched
