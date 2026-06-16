"""
CLI entry point for the full StructFetch benchmark.

Runs candidate fetching + rawfield alignment for every CSV row, then computes
ChimeraX Q-scores and renders the GT-vs-identified scatter plot. Defaults
come from ``configs/benchmark_all.json``; CLI flags override the config.
"""
from __future__ import annotations

import sys
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = METHOD_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from benchmark import main  # noqa: E402


if __name__ == "__main__":
    main()
