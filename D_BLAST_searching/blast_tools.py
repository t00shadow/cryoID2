"""Resolve CryoID's private, pinned BLAST+ tools without external configuration."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import sys
from pathlib import Path

BLAST_PROGRAMS = ("blastp", "blastn", "makeblastdb")
BLAST_VERSION = "2.17.0"
BUNDLE_ROOT = Path(__file__).resolve().parent / "bundled"
LINUX_MIN_GLIBC = (2, 28)
MANAGED_BLAST_BUILD = "hb02a186_1"
MANAGED_BLAST_PACKAGE = f"blast-{BLAST_VERSION}-{MANAGED_BLAST_BUILD}"
MANAGED_BLAST_SHA256 = "d506d022ca8ab1346dc816cbcbe3e798d7f104714e14dea1d750db9dec47c205"
MANAGED_BLAST_URL = f"https://conda.anaconda.org/bioconda/linux-64/{MANAGED_BLAST_PACKAGE}.conda"


def needs_managed_linux_blast() -> bool:
    """The tested Conda build supports glibc 2.17 through 2.27."""
    if platform.system().lower() != "linux":
        return False
    name, release = platform.libc_ver()
    try:
        installed = tuple(int(part) for part in release.split(".")[:2])
    except ValueError:
        return False
    return name == "glibc" and (2, 17) <= installed < LINUX_MIN_GLIBC


def linux_runtime_issue() -> str | None:
    """Check the ABI required by the pinned upstream Linux payload."""
    if platform.system().lower() != "linux":
        return None
    name, release = platform.libc_ver()
    try:
        installed = tuple(int(part) for part in release.split(".")[:2])
    except ValueError:
        installed = ()
    if name == "glibc" and installed >= LINUX_MIN_GLIBC:
        return None
    found = f"{name} {release}".strip() or "an unidentified C runtime"
    return (
        f"CryoID's bundled BLAST+ {BLAST_VERSION} Linux executables require glibc 2.28 or newer; "
        f"this host provides {found}. Use a Linux host/container with glibc >=2.28 and zlib >=1.2.9, "
        "or rebuild the CryoID distribution with a verified BLAST+ 2.17.0 payload compatible with this host. "
        "Conda cannot upgrade the host's glibc."
    )


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
        directory = validate_blast_bin_dir(directory, system=current)
    except ValueError as exc:
        raise FileNotFoundError("CryoID's bundled BLAST+ tools are missing or damaged. Reinstall CryoID.") from exc
    # Explicit system/machine arguments also support inspecting other-platform
    # payloads in the packaging harness without executing them on this host.
    if system is None and machine is None:
        issue = linux_runtime_issue()
        if issue:
            raise RuntimeError(issue)
    return directory


def blast_executable(program: str, blast_bin_dir: Path | None = None) -> str:
    if program not in BLAST_PROGRAMS:
        raise ValueError(f"Unsupported BLAST+ program: {program}")
    directory = validate_blast_bin_dir(blast_bin_dir) if blast_bin_dir is not None else runtime_blast_bin_dir()
    return str(directory / _program_filename(program))


def managed_blast_payload(directory: Path) -> list[tuple[Path, int, str]]:
    """Read the pinned package's installed-file hashes from this Python's Conda prefix.

    Conda records hashes after prefix relocation, which changes binary RPATHs.
    No PATH lookup, saved BLAST directory, or other environment is accepted.
    """
    prefix = Path(sys.prefix).resolve()
    if Path(directory).resolve() != prefix / "bin":
        raise ValueError("Managed BLAST must belong to the active Python environment")
    record = json.loads((prefix / "conda-meta" / f"{MANAGED_BLAST_PACKAGE}.json").read_text(encoding="utf-8"))
    if (record.get("name") != "blast" or record.get("version") != BLAST_VERSION
            or record.get("build") != MANAGED_BLAST_BUILD
            or record.get("sha256") != MANAGED_BLAST_SHA256 or record.get("url") != MANAGED_BLAST_URL):
        raise ValueError("Managed BLAST package does not match CryoID's pinned Bioconda build")
    payload = []
    required = {f"bin/{program}" for program in BLAST_PROGRAMS}
    found = set()
    for entry in record.get("paths_data", {}).get("paths", []):
        relative = entry.get("_path", "")
        if relative not in required and not relative.startswith("lib/ncbi-blast+/"):
            continue
        path = (prefix / relative).resolve()
        digest = entry.get("sha256_in_prefix") or entry.get("sha256")
        if not path.is_relative_to(prefix) or not digest or len(digest) != 64:
            raise ValueError(f"Managed BLAST has no verifiable installed-file record: {relative}")
        payload.append((path, int(entry["size_in_bytes"]), digest))
        found.add(relative)
    if not required <= found:
        raise ValueError("Managed BLAST metadata is missing required executables")
    return payload


def verify_managed_blast(directory: Path) -> None:
    for path, size, expected in managed_blast_payload(directory):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size != size or digest.hexdigest() != expected:
            raise ValueError(f"Managed BLAST file failed integrity validation: {path.name}")


def runtime_blast_bin_dir() -> Path:
    """Use the shipped payload, or the installer-managed legacy Linux build."""
    if needs_managed_linux_blast():
        try:
            directory = validate_blast_bin_dir(Path(sys.prefix) / "bin")
            managed_blast_payload(directory)
            return directory
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(
                f"This Linux host needs CryoID's pinned Conda-managed BLAST+ {BLAST_VERSION} "
                f"({MANAGED_BLAST_BUILD}). Create its environment with "
                "`python -m cryoid_pipeline.create_conda_env` using supported Miniforge. "
                f"Managed BLAST validation failed: {exc}"
            ) from exc
    return bundled_blast_bin_dir()
