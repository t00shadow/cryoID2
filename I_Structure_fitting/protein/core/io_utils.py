"""
Shared I/O + formatting helpers used by both the pipeline and benchmark workflows.

Consolidates the small JSON-serialization, JSON-file-writer, and numeric-formatter
helpers that used to be defined separately in ``aligner.py``, ``benchmark.py``,
``benchmark_gt_only.py``, and ``pipeline.py``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def to_jsonable(value: Any) -> Any:
    """Recursively coerce a value into something ``json.dumps`` can serialize.

    Handles ``Path``, ``numpy.ndarray``, nested lists/tuples/dicts. Anything
    else falls back to its ``str()`` representation so serialization never
    raises on unexpected input.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as pretty-printed JSON, creating parent directories.

    Values are first run through :func:`to_jsonable` for serialization.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def format_float(value: Any, *, fmt: str = ".9f", fallback: str = "") -> str:
    """Coerce ``value`` to ``float`` and format with ``fmt``; return ``fallback`` on failure or non-finite."""
    try:
        number = float(value)
    except Exception:
        return fallback
    if not math.isfinite(number):
        return fallback
    return f"{number:{fmt}}"
