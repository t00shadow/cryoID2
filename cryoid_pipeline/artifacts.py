"""Safe filesystem helpers for large, immutable input artifacts."""

from __future__ import annotations

import os
from pathlib import Path


def link_input_file(source: Path, destination: Path) -> Path:
    """Expose an immutable input by hard link or symlink, never by copying."""

    source = Path(source).expanduser().resolve()
    destination = Path(destination)
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return destination
        raise FileExistsError(f"Refusing to replace existing path: {destination}")
    try:
        os.link(source, destination)
    except OSError as hardlink_error:
        try:
            destination.symlink_to(source)
        except OSError as symlink_error:
            raise OSError(
                f"Could not link {source} to {destination}; CryoID does not copy "
                "large immutable inputs. Hard-link error: {hardlink_error}; "
                f"symlink error: {symlink_error}"
            ) from symlink_error
    return destination
