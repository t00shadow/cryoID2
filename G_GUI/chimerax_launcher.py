from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


def _candidate_chimerax_paths() -> Iterable[Path]:
    env_path = os.environ.get("CHIMERAX_PATH")
    if env_path:
        yield Path(env_path).expanduser()

    for executable in ("ChimeraX", "ChimeraX.exe", "chimerax", "ucsf-chimerax"):
        resolved = shutil.which(executable)
        if resolved:
            yield Path(resolved)

    if os.name == "nt":
        for base_text in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
            if not base_text:
                continue
            base = Path(base_text)
            for pattern in ("ChimeraX*/bin/ChimeraX.exe", "ChimeraX*/bin/ChimeraX-console.exe"):
                yield from sorted(base.glob(pattern), reverse=True)


def find_chimerax(explicit_path: str | Path | None = None) -> Path:
    paths: list[Path] = []
    if explicit_path:
        paths.append(Path(explicit_path).expanduser())
    paths.extend(_candidate_chimerax_paths())

    seen: set[str] = set()
    for path in paths:
        key = str(path).lower()
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path

    raise FileNotFoundError(
        "ChimeraX executable not found. Set CHIMERAX_PATH or set the ChimeraX path manually in Settings > Configurations."
    )


def launch_chimerax(files: Iterable[str | Path] | None = None, *, chimerax_path: str | Path | None = None) -> list[str]:
    executable = find_chimerax(chimerax_path)
    command = [str(executable)]
    if files:
        command.extend(str(Path(path)) for path in files)

    subprocess.Popen(command)
    return command
