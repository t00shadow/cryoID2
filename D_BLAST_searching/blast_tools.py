"""Resolve CryoID's private, pinned BLAST+ tools without external configuration."""
from __future__ import annotations

import os
import platform
import stat
from pathlib import Path

BLAST_PROGRAMS = ("blastp", "blastn", "makeblastdb")
BLAST_VERSION = "2.17.0"
BUNDLE_ROOT = Path(__file__).resolve().parent / "bundled"


def bundled_platform_key(*, system: str | None = None, machine: str | None = None) -> str:
    """Select by the running Python architecture (including Rosetta Python)."""
    current = (system or platform.system()).lower()
    architecture = (machine or platform.machine()).lower()
    if architecture in {"amd64", "x86_64", "x64"}:
        architecture = "x86_64"
    elif architecture in {"arm64", "aarch64"}:
        architecture = "arm64"
    key = f"{current}-{architecture}"
    if key not in {"windows-x86_64", "linux-x86_64", "darwin-x86_64", "darwin-arm64"}:
        raise RuntimeError(
            "The bundled BLAST+ tools support Windows/Linux x86-64 and macOS Intel/Apple Silicon. "
            f"This platform is {current}/{architecture}."
        )
    return key


def _program_filename(program: str, system: str | None = None) -> str:
    return f"{program}.exe" if (system or platform.system()).lower() == "windows" else program


def validate_blast_bin_dir(value: str | Path, *, system: str | None = None) -> Path:
    """Validate an internal tool directory (also used by the comparison harness)."""
    directory = Path(value).expanduser().resolve()
    if not directory.is_dir():
        raise ValueError(f"BLAST+ bin folder does not exist: {directory}")
    current = (system or platform.system()).lower()
    missing = []
    for program in BLAST_PROGRAMS:
        filename = _program_filename(program, current)
        executable = directory / filename
        if not executable.is_file() or executable.stat().st_size == 0 or (
            current != "windows" and not os.access(executable, os.X_OK)
        ):
            missing.append(filename)
    if missing:
        raise ValueError(f"BLAST+ bin folder is missing {', '.join(missing)}: {directory}")
    return directory


def bundled_blast_bin_dir(*, system: str | None = None, machine: str | None = None) -> Path:
    current = (system or platform.system()).lower()
    directory = BUNDLE_ROOT / bundled_platform_key(system=current, machine=machine) / "bin"
    if current != "windows" and os.name != "nt":
        # Editable installs checked out on Windows may lose executable modes.
        for program in BLAST_PROGRAMS:
            executable = directory / program
            if executable.is_file() and not os.access(executable, os.X_OK):
                executable.chmod(executable.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    try:
        return validate_blast_bin_dir(directory, system=current)
    except ValueError as exc:
        raise FileNotFoundError("CryoID's bundled BLAST+ tools are missing or damaged. Reinstall CryoID.") from exc


def blast_executable(program: str, blast_bin_dir: Path | None = None) -> str:
    if program not in BLAST_PROGRAMS:
        raise ValueError(f"Unsupported BLAST+ program: {program}")
    directory = validate_blast_bin_dir(blast_bin_dir) if blast_bin_dir is not None else bundled_blast_bin_dir()
    return str(directory / _program_filename(program))
