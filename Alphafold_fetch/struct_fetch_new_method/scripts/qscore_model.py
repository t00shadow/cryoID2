"""
CLI helper: compute the ChimeraX Q-score for one (model, map) pair.

Usage::

    python scripts/qscore_model.py --model <model.cif> --map <map.map> [--chimerax <path>]

Prints ``<qscore><tab><elapsed_s>s`` to stdout on success, or exits with the
ChimeraX error message on failure.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = METHOD_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from qscore import compute_model_qscore, find_chimerax  # noqa: E402


def main() -> None:
    """Parse args, locate ChimeraX, and print the mean per-atom Q-score for the model/map pair."""
    parser = argparse.ArgumentParser(description="Compute ChimeraX Q-score for one model/map pair.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--map", required=True)
    parser.add_argument("--chimerax", default="")
    parser.add_argument("--timeout-sec", type=int, default=1800)
    parser.add_argument("--work-dir", default=str(METHOD_ROOT / "runs" / "qscore_single" / "data" / "_tmp"))
    args = parser.parse_args()

    chimerax_path = find_chimerax(args.chimerax or None)
    score, error, elapsed = compute_model_qscore(
        model_path=Path(args.model).resolve(),
        map_path=Path(args.map).resolve(),
        chimerax_path=chimerax_path,
        work_root=Path(args.work_dir).resolve(),
        timeout_sec=int(args.timeout_sec),
    )
    if error:
        raise SystemExit(error)
    print(f"{float(score):.9f}\t{elapsed:.3f}s")


if __name__ == "__main__":
    main()
