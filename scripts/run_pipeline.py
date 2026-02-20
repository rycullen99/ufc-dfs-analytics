#!/usr/bin/env python3
"""CLI entrypoint for the UFC DFS analytics pipeline."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phase5_pipeline.runner import main

if __name__ == "__main__":
    main()
