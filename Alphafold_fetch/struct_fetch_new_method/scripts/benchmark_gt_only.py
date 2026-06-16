"""
CLI entry point for the GT-only StructFetch benchmark.

Skips candidate fetching and aligns each row's deposited ground-truth model
directly to its cryo-EM map, then computes Q-scores and renders the plot.
Defaults come from ``configs/benchmark_gt_only.json``.

This script uses ``importlib`` to load ``src/benchmark_gt_only.py`` under a
distinct module name so it doesn't collide with the same-name script module
(this CLI shim) when both happen to be on ``sys.path``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = METHOD_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

spec = importlib.util.spec_from_file_location("struct_fetch_gt_only_benchmark", SRC_ROOT / "benchmark_gt_only.py")
if spec is None or spec.loader is None:
    raise RuntimeError("Could not load src/benchmark_gt_only.py")
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
main = module.main


if __name__ == "__main__":
    main(sys.argv[1:])
