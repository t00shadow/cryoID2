from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable

from cryoid_pipeline.startup_config import chimerax_executable, find_chimerax


def launch_chimerax(files: Iterable[str | Path] | None = None, *, chimerax_path: str | Path | None = None) -> list[str]:
    executable = find_chimerax(chimerax_path)
    command = [str(executable)]
    if files:
        command.extend(str(Path(path)) for path in files)

    subprocess.Popen(command)
    return command
