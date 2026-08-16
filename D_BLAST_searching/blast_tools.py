from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Mapping


BLAST_PROGRAMS = ("blastp", "blastn", "makeblastdb")
ENV_BLAST_BIN_DIR = "CRYOID_BLAST_BIN_DIR"


def _program_filename(program: str, system: str | None = None) -> str:
    current = (system or platform.system()).lower()
    return f"{program}.exe" if current == "windows" else program


def validate_blast_bin_dir(
    value: str | Path,
    *,
    system: str | None = None,
) -> Path:
    directory = Path(value).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"BLAST+ bin folder does not exist: {directory}")
    current_system = (system or platform.system()).lower()
    missing: list[str] = []
    for program in BLAST_PROGRAMS:
        filename = _program_filename(program, current_system)
        executable = directory / filename
        if not executable.is_file() or (
            current_system != "windows" and not os.access(executable, os.X_OK)
        ):
            missing.append(filename)
    if missing:
        raise ValueError(
            f"BLAST+ bin folder is missing {', '.join(missing)}: {directory}"
        )
    return directory


def _common_candidates(
    *,
    system: str,
    environ: Mapping[str, str],
    python_prefix: Path,
) -> list[Path]:
    candidates = [python_prefix / ("Scripts" if system == "windows" else "bin")]
    conda_prefix = str(environ.get("CONDA_PREFIX") or "").strip()
    if conda_prefix:
        candidates.append(Path(conda_prefix) / ("Scripts" if system == "windows" else "bin"))
    if system == "windows":
        roots = [
            Path(str(environ.get("SystemDrive") or "C:")) / "Tools",
            Path(str(environ.get("USERPROFILE") or Path.home())) / "Tools",
            Path(str(environ.get("ProgramFiles") or r"C:\Program Files")),
            Path(str(environ.get("ProgramFiles(x86)") or r"C:\Program Files (x86)")),
        ]
        for root in roots:
            candidates.extend(sorted(root.glob("ncbi-blast*/bin"), reverse=True))
            candidates.extend(sorted(root.glob("NCBI/blast*/bin"), reverse=True))
    else:
        candidates.extend(
            [
                Path("/usr/local/bin"),
                Path("/usr/bin"),
                Path.home() / "envs" / "cryoid_py310" / "bin",
            ]
        )
        candidates.extend(
            sorted((Path.home() / "envs").glob("*blast*/bin"), reverse=True)
        )
        candidates.extend(sorted(Path("/opt").glob("ncbi-blast*/bin"), reverse=True))
    return candidates


def autodetect_blast_bin_dir(
    *,
    environ: Mapping[str, str] | None = None,
    system: str | None = None,
    python_prefix: Path | None = None,
) -> Path | None:
    environment = os.environ if environ is None else environ
    current_system = (system or platform.system()).lower()
    prefix = Path(sys.prefix) if python_prefix is None else Path(python_prefix)

    env_value = str(environment.get(ENV_BLAST_BIN_DIR) or "").strip()
    if env_value:
        try:
            return validate_blast_bin_dir(env_value, system=current_system)
        except ValueError:
            pass

    path_value = str(environment.get("PATH") or "")
    resolved = [shutil.which(program, path=path_value) for program in BLAST_PROGRAMS]
    if all(resolved):
        parents = {Path(str(path)).resolve().parent for path in resolved}
        if len(parents) == 1:
            return next(iter(parents))

    seen: set[str] = set()
    for candidate in _common_candidates(
        system=current_system,
        environ=environment,
        python_prefix=prefix,
    ):
        key = str(candidate).lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            return validate_blast_bin_dir(candidate, system=current_system)
        except ValueError:
            continue
    return None


def resolve_blast_bin_dir(
    argument: str | Path | None,
    config: Mapping[str, object] | None = None,
) -> Path | None:
    if argument not in (None, ""):
        return validate_blast_bin_dir(argument)
    env_value = str(os.environ.get(ENV_BLAST_BIN_DIR) or "").strip()
    if env_value:
        return validate_blast_bin_dir(env_value)
    config_value = str((config or {}).get("blast_bin_dir") or "").strip()
    if config_value:
        return validate_blast_bin_dir(config_value)
    return autodetect_blast_bin_dir()


def blast_executable(program: str, blast_bin_dir: Path | None = None) -> str:
    if program not in BLAST_PROGRAMS:
        raise ValueError(f"Unsupported BLAST+ program: {program}")
    if blast_bin_dir is not None:
        directory = validate_blast_bin_dir(blast_bin_dir)
        return str(directory / _program_filename(program))
    detected = autodetect_blast_bin_dir()
    if detected is not None:
        return str(detected / _program_filename(program))
    raise FileNotFoundError(
        f"NCBI BLAST+ executable not found: {program}. Configure the BLAST+ bin folder "
        "in Advanced Settings or run `cryoid config set blast-bin-dir PATH`."
    )
