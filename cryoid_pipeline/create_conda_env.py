"""Create CryoID's shared Conda environment with portable platform selectors.

Run with Conda's base Python: python -m cryoid_pipeline.create_conda_env.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import struct
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]
SELECTOR = re.compile(r"^(.*?)\s+#\s*\[([^\]]+)\]\s*$")


def select_environment(source: str, platform: str) -> str:
    os_name = platform.split("-", 1)[0]
    enabled = {"unix": os_name in {"linux", "osx"}, "linux": os_name == "linux",
               "osx": os_name == "osx", "win": os_name == "win"}
    lines = []
    for line in source.splitlines():
        match = SELECTOR.match(line)
        if match:
            value, selector = match.groups()
            if selector not in enabled:
                raise ValueError(f"Unsupported environment selector: {selector}")
            if not enabled[selector]:
                continue
            line = value
        lines.append(line)
    return "\n".join(lines) + "\n"


def installation_environment(source: str, platform: str, *, managed_blast: bool = False, cuda118: bool = False) -> str:
    selected = select_environment(source, platform)
    if managed_blast:
        from D_BLAST_searching.blast_tools import BLAST_VERSION, MANAGED_BLAST_BUILD
        selected = selected.replace("dependencies:\n", f"dependencies:\n  - blast={BLAST_VERSION}={MANAGED_BLAST_BUILD}\n", 1)
    if cuda118:
        lines = [line for line in selected.splitlines() if not re.match(r"\s*- pytorch(?:[<>=]|\s|$)", line)]
        index = lines.index("  - pip:") + 1
        lines[index:index] = ["      - --extra-index-url https://download.pytorch.org/whl/cu118",
                              "      - torch==2.5.1+cu118"]
        selected = "\n".join(lines) + "\n"
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--prefix", type=Path)
    destination.add_argument("--name")
    parser.add_argument("--platform", choices=("linux-64", "win-64", "osx-64", "osx-arm64"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--cuda118", action="store_true",
                        help="Install PyTorch 2.5.1 CUDA 11.8 wheels for compatible NVIDIA GPUs on Linux/Windows x86-64.")
    args = parser.parse_args()
    host = {"win32": "win-64", "linux": "linux-64", "darwin": "osx"}.get(sys.platform)
    if host is None:
        parser.error(f"Unsupported operating system: {sys.platform}")
    target = args.platform or host
    if args.cuda118 and target.startswith("osx"):
        parser.error("--cuda118 supports Linux/Windows x86-64; Macs use CPU.")
    if args.platform and target.split("-")[0] != host.split("-")[0] and not args.dry_run:
        parser.error("Cross-platform creation is supported only with --dry-run.")
    from D_BLAST_searching.blast_tools import bundled_platform_key, linux_runtime_issue, needs_managed_linux_blast
    managed_blast = needs_managed_linux_blast() and target == "linux-64"
    if not args.dry_run:
        if struct.calcsize("P") != 8:
            parser.error("CryoID requires 64-bit Python. Install 64-bit Miniforge before creating the environment.")
        try:
            bundled_platform_key()
        except RuntimeError as exc:
            parser.error(str(exc))
        issue = linux_runtime_issue()
        if issue and not managed_blast:
            parser.error(issue)
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        parser.error("Conda was not found. Run this script with conda run -n base python.")
    try:
        result = subprocess.run([conda, "--version"], capture_output=True, text=True, check=True, timeout=15)
        match = re.search(r"\bconda\s+(\d+)\.(\d+)\.(\d+)", result.stdout)
        if not match or tuple(map(int, match.groups())) < (25, 5, 1):
            parser.error("CryoID setup requires Conda >=25.5.1. Install Miniforge >=25.9.1-0 and use its base Python.")
    except (OSError, subprocess.SubprocessError) as exc:
        parser.error(f"Could not check Conda: {exc}. Use a working Miniforge >=25.9.1-0 installation.")
    content = installation_environment((ROOT / "environment.yml").read_text(encoding="utf-8"), target,
                                       managed_blast=managed_blast, cuda118=args.cuda118)
    if managed_blast:
        print("Using the pinned Conda-managed BLAST+ 2.17.0 build for this older Linux host.", file=sys.stderr)
    # Conda resolves pip's editable path relative to the YAML location. Keep
    # the temporary file beside environment.yml, then remove it after setup.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=ROOT,
                                     prefix=".cryoid-env-", suffix=".yml", delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    command = [conda, "env", "create", "--file", str(temporary)]
    if args.prefix:
        command.extend(["--prefix", str(args.prefix.expanduser().resolve())])
    if args.name:
        command.extend(["--name", args.name])
    for option in ("platform", "dry_run", "json", "yes"):
        value = getattr(args, option)
        if value:
            command.append("--" + option.replace("_", "-"))
            if isinstance(value, str):
                command.append(value)
    try:
        result = subprocess.run(command, cwd=ROOT).returncode
        if result or args.dry_run:
            return result
        target_env = (["--prefix", str(args.prefix.expanduser().resolve())] if args.prefix
                      else ["--name", args.name or "cryoid"])
        check_command = [conda, "run", *target_env, "python", "-m", "cryoid_pipeline.preflight", "--gui"]
        if args.cuda118:
            check_command.extend(["--device", "cuda:0"])
        return subprocess.run(
            check_command,
            cwd=ROOT,
        ).returncode
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
