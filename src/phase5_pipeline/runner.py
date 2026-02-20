"""Pipeline CLI runner — orchestrates all stages."""

import argparse
import logging
import sys

from ..db import get_connection
from .stages import run_stage, STAGE_ORDER

logger = logging.getLogger(__name__)


def run_pipeline(stages=None, holdout_events=10):
    """
    Run the analytics pipeline.

    Parameters
    ----------
    stages : list of str, optional
        Stages to run. If None, runs all stages in order.
    holdout_events : int
        Number of events to hold out for final evaluation.
    """
    if stages is None:
        stages = STAGE_ORDER

    conn = get_connection()

    for stage_name in stages:
        logger.info("=" * 60)
        logger.info("Running stage: %s", stage_name)
        logger.info("=" * 60)
        try:
            run_stage(stage_name, conn, holdout_events=holdout_events)
            logger.info("Stage '%s' completed successfully.", stage_name)
        except Exception as e:
            logger.error("Stage '%s' failed: %s", stage_name, e)
            raise

    conn.close()
    logger.info("Pipeline completed.")


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="UFC DFS Analytics Pipeline")
    parser.add_argument(
        "--stage",
        default="all",
        help="Stage to run: all, ingest, normalize, features, score, export",
    )
    parser.add_argument(
        "--holdout",
        type=int,
        default=10,
        help="Number of holdout events (default: 10)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    stages = None if args.stage == "all" else [args.stage]
    run_pipeline(stages=stages, holdout_events=args.holdout)


if __name__ == "__main__":
    main()
