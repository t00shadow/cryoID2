"""Discover installed resources without importing GUI or numerical libraries."""
from __future__ import annotations

import json
import os
import platform
import shutil
from pathlib import Path
from typing import Iterable

from .user_config import ENV_CHIMERAX_PATH, ENV_WEIGHTS_DIR


def find_weights_dir() -> Path | None:
    candidates = [os.environ.get(ENV_WEIGHTS_DIR), Path.home() / "cryoid_weights" / "train_val_data",
                  Path.home() / "cryoid_weights", Path(__file__).resolve().parents[1] / "train_val_data"]
    for value in candidates:
        if not value:
            continue
        root = Path(value).expanduser().resolve()
        try:
            epochs = json.loads((root / "epoch_num_path.txt").read_text(encoding="utf-8"))
            weights = json.loads((root / "class_weight_for_training.txt").read_text(encoding="utf-8"))
            models = ("secondary_strctures", "key_atoms", "residue_types_6letter")
            if all(model in weights and (root / "checkpoint" / model / f"{model}.epoch{int(epochs[model])}.pth.tar").is_file() for model in models):
                return root
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def _candidate_chimerax_paths() -> Iterable[Path]:
    env_path = os.environ.get(ENV_CHIMERAX_PATH)
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
    if platform.system().lower() == "darwin":
        for base in (Path("/Applications"), Path.home() / "Applications"):
            yield from sorted(base.glob("ChimeraX*.app/Contents/MacOS/ChimeraX"), reverse=True)


def chimerax_executable(path: str | Path) -> Path:
    """Accept a macOS application bundle as well as a program file."""
    candidate = Path(path).expanduser()
    if candidate.suffix.lower() == ".app" and candidate.is_dir():
        candidate = candidate / "Contents" / "MacOS" / "ChimeraX"
    return candidate


def find_chimerax(explicit_path: str | Path | None = None) -> Path:
    paths: list[Path] = []
    if explicit_path:
        paths.append(Path(explicit_path).expanduser())
    paths.extend(_candidate_chimerax_paths())

    seen: set[str] = set()
    for path in paths:
        path = chimerax_executable(path)
        key = str(path).lower() if os.name == "nt" else str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path

    raise FileNotFoundError(
        "ChimeraX executable not found. Choose its path in Settings > Configurations "
        "or set CRYOID_CHIMERAX_PATH."
    )


