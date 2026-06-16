"""
Package-root path constants and a generic path resolver.

``METHOD_ROOT`` is the absolute path to ``struct_fetch_new_method/``; every
other module derives its data, runs, and config directories from it so the
package is location-independent (works from any working directory).
"""
from __future__ import annotations

from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parent
METHOD_ROOT = SRC_ROOT.parent
DEFAULT_RUNS_DIR = METHOD_ROOT / "runs"


def resolve_path(value: str | Path, *, base: Path = METHOD_ROOT, must_exist: bool = False) -> Path:
    """Resolve ``value`` to an absolute path, trying base-relative, cwd-relative, then bare.

    The first candidate that exists wins. When none exist and ``must_exist``
    is True the function raises ``FileNotFoundError`` on the base-relative
    candidate; otherwise it returns the base-relative candidate unchanged.
    """
    path = Path(value)
    candidates = [path] if path.is_absolute() else [base / path, Path.cwd() / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    resolved = candidates[0]
    if must_exist:
        raise FileNotFoundError(resolved)
    return resolved
