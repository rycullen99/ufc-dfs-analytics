"""Idempotent stage definitions for the analytics pipeline."""

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Optional

from ..config import REPORTS_DIR

logger = logging.getLogger(__name__)

STAGE_ORDER = ["ingest", "normalize", "features", "score", "export"]

# Track stage checksums for idempotency
_CHECKSUM_FILE = REPORTS_DIR / "stage_checksums.json"


def _load_checksums() -> dict:
    if _CHECKSUM_FILE.exists():
        return json.loads(_CHECKSUM_FILE.read_text())
    return {}


def _save_checksums(checksums: dict) -> None:
    _CHECKSUM_FILE.write_text(json.dumps(checksums, indent=2))


def stage_checksum(conn: sqlite3.Connection, table_name: str) -> str:
    """Compute a checksum for a table based on row count and schema."""
    try:
        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        schema = conn.execute(
            f"SELECT sql FROM sqlite_master WHERE name='{table_name}'"
        ).fetchone()
        schema_str = schema[0] if schema else ""
        content = f"{table_name}:{row_count}:{schema_str}"
        return hashlib.md5(content.encode()).hexdigest()
    except Exception:
        return ""


def should_run_stage(
    stage_name: str,
    conn: sqlite3.Connection,
    output_table: str,
) -> bool:
    """Check if a stage needs to run by comparing checksums."""
    checksums = _load_checksums()
    current = stage_checksum(conn, output_table)
    stored = checksums.get(stage_name, "")
    return current != stored or current == ""


def _mark_stage_done(stage_name: str, conn: sqlite3.Connection, output_table: str):
    """Record that a stage has completed."""
    checksums = _load_checksums()
    checksums[stage_name] = stage_checksum(conn, output_table)
    _save_checksums(checksums)


def run_stage(
    stage_name: str,
    conn: sqlite3.Connection,
    holdout_events: int = 10,
) -> None:
    """
    Run a single pipeline stage.

    Parameters
    ----------
    stage_name : str
        One of: ingest, normalize, features, score, export
    conn : sqlite3.Connection
        Database connection.
    holdout_events : int
        Number of holdout events (used by score stage).
    """
    if stage_name == "ingest":
        _stage_ingest(conn)
    elif stage_name == "normalize":
        _stage_normalize(conn)
    elif stage_name == "features":
        _stage_features(conn)
    elif stage_name == "score":
        _stage_score(conn, holdout_events)
    elif stage_name == "export":
        _stage_export(conn)
    else:
        raise ValueError(f"Unknown stage: {stage_name}")


def _stage_ingest(conn: sqlite3.Connection) -> None:
    """Verify source data is accessible and tables exist."""
    required_tables = ["players", "contests", "lineups", "lineup_players", "lineup_usernames"]
    for table in required_tables:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        logger.info("Table %s: %d rows", table, count)
    logger.info("Ingest stage: all source tables verified.")


def _stage_normalize(conn: sqlite3.Connection) -> None:
    """Run Phase 0 data normalization."""
    from ..phase0_data.identity import build_canonical_fighters
    from ..phase0_data.card_position import backfill_card_position
    from ..phase0_data.odds_quality import write_fighter_odds_quality

    build_canonical_fighters(conn)
    backfill_card_position(conn)
    write_fighter_odds_quality(conn)
    logger.info("Normalize stage complete.")


def _stage_features(conn: sqlite3.Connection) -> None:
    """Build ownership prediction features."""
    from ..phase1_ownership.features import build_ownership_features

    df = build_ownership_features(conn)
    logger.info("Feature matrix: %d rows, %d cols", df.shape[0], df.shape[1])


def _stage_score(conn: sqlite3.Connection, holdout_events: int) -> None:
    """Train and evaluate ownership model."""
    from ..phase1_ownership.features import build_ownership_features
    from ..phase1_ownership.model import OwnershipModel

    df = build_ownership_features(conn)
    model = OwnershipModel()

    results = model.walk_forward_cv(df)
    model.fit_final(df, holdout_events=holdout_events)
    logger.info("Score stage complete. CV results computed.")


def _stage_export(conn: sqlite3.Connection) -> None:
    """Export curated tables (placeholder for Supabase export)."""
    logger.info("Export stage: would push to Supabase here.")
    logger.info("See export_supabase.py for full implementation.")
