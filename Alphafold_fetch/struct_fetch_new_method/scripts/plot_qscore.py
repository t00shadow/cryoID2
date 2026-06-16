"""
CLI helper: render the GT-vs-identified Q-score scatter plot from an existing CSV.

Usage::

    python scripts/plot_qscore.py --csv <benchmark_qscores.csv> --out <plot.png>

Delegates to :func:`plot_qscore.main` in ``src/``.
"""
from __future__ import annotations

import sys
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = METHOD_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from plot_qscore import main  # noqa: E402


if __name__ == "__main__":
    main()
