from __future__ import annotations

import os
from pathlib import Path
import sys


_LINUX_QT_LIBRARIES = (
    "libxcb-icccm.so.4",
    "libxcb-image.so.0",
    "libxcb-keysyms.so.1",
    "libxcb-render-util.so.0",
    "libxkbcommon-x11.so.0",
)
_LINUX_LIBRARY_PATH_READY = "CRYOID_GUI_LINUX_LIBRARY_PATH_READY"


def _conda_qt_library_dir() -> Path | None:
    """Return the active environment's Qt support library directory."""

    if not sys.platform.startswith("linux"):
        return None

    library_dir = Path(sys.prefix) / "lib"
    if all((library_dir / name).exists() for name in _LINUX_QT_LIBRARIES):
        return library_dir
    return None


def _ensure_linux_qt_library_path() -> None:
    """Make Conda's XCB libraries visible before Qt loads its platform plugin.

    Pip's PyQt Linux wheels do not search ``$CONDA_PREFIX/lib`` for XCB helper
    libraries.  The dynamic loader reads ``LD_LIBRARY_PATH`` at process start,
    so changing it in-place is too late; restart the launcher once instead.
    """

    library_dir = _conda_qt_library_dir()
    if library_dir is None:
        return

    library_dir_text = str(library_dir)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    current_entries = [entry for entry in current.split(":") if entry]
    if library_dir_text in current_entries:
        return
    if os.environ.get(_LINUX_LIBRARY_PATH_READY) == "1":
        return

    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(
        [library_dir_text, *current_entries]
    )
    environment[_LINUX_LIBRARY_PATH_READY] = "1"
    os.execve(
        sys.executable,
        [sys.executable, "-m", "G_GUI.launcher", *sys.argv[1:]],
        environment,
    )


def main() -> None:
    """Launch the GUI from the installed console command."""

    _ensure_linux_qt_library_path()

    from .main_gui import main as launch

    launch()


if __name__ == "__main__":
    main()
