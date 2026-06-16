"""
CLI entry point for the production StructFetch pipeline.

Thin wrapper that adds ``src/`` to ``sys.path`` and delegates to
:func:`pipeline.main`. Defaults are read from ``configs/pipeline.json`` and
can be overridden on the command line. The pool FASTA is auto-downloaded on
first use and deleted at the end of the run.
"""
from __future__ import annotations

import sys
from pathlib import Path


METHOD_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = METHOD_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pipeline import main  # noqa: E402


if __name__ == "__main__":
    main()
