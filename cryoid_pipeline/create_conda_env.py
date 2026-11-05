"""Create CryoID's shared Conda environment with portable platform selectors.

Run with Conda's base Python: python -m cryoid_pipeline.create_conda_env.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import struct
import sys
import tempfile

from .install_hardware import TorchProfile, clean_environment, cpu_profile, probe_cuda, select_torch_profile


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


def installation_environment(source: str, platform: str, *, managed_blast: bool = False) -> str:
    """Create native dependencies first; install selected Torch and the app later.

    This separates CUDA download/driver failures from Conda solve failures and
    keeps CPU recovery from overwriting a Conda-owned PyTorch installation.
    """
    selected = select_environment(source, platform)
    if managed_blast:
        from D_BLAST_searching.blast_tools import BLAST_VERSION, MANAGED_BLAST_BUILD
        selected = selected.replace("dependencies:\n", f"dependencies:\n  - blast={BLAST_VERSION}={MANAGED_BLAST_BUILD}\n", 1)
    lines = selected.splitlines()
    # The shared template's pip section is last. The app is installed only
    # after the selected Torch build has executed successfully.
    if "  - pip:" in lines:
        lines = lines[:lines.index("  - pip:")]
    if not platform.startswith("osx"):
        lines = [line for line in lines if not re.match(r"\s*- pytorch(?:[<>=]|\s|$)", line)]
    return "\n".join(lines) + "\n"


def install_runtime(conda: str, target_env: list[str], target: str,
                    profile: TorchProfile, hardware: dict) -> int:
    environment = clean_environment()
    prefix = [conda, "run", "--no-capture-output", *target_env, "python"]

    def run(arguments, **kwargs):
        return subprocess.run([*prefix, *arguments], cwd=ROOT, env=environment, **kwargs).returncode

    def install(profile):
        if profile.requirement is None:
            return 0  # macOS Torch was installed by Conda with the native packages.
        return run(["-m", "pip", "install", "--only-binary=:all:", "--upgrade",
                    "--index-url", profile.index, "--extra-index-url", "https://pypi.org/simple",
                    profile.requirement])

    def check(profile):
        code = ("import torch; from cryoid_pipeline.preflight import exercise_torch; "
                "torch.set_num_threads(2); ")
        if profile.cuda:
            code += ("assert torch.cuda.is_available(), 'CUDA unavailable'; "
                     "assert torch.cuda.device_count()>0, 'No visible CUDA devices'; "
                     "[(exercise_torch('cuda:'+str(i)), print('CUDA execution passed:', i, "
                     "torch.cuda.get_device_name(i), flush=True)) for i in range(torch.cuda.device_count())]")
        else:
            code += "exercise_torch('cpu'); print('CPU execution passed.', flush=True)"
            if profile.requirement:
                code += "; assert torch.version.cuda is None, 'Expected CPU-only PyTorch'"
        # Driver initialization and native kernels can crash or hang: isolate
        # the check and recover only the new environment, never the host driver.
        supervisor = ("import subprocess,sys\n"
                      "try:\n"
                      " result=subprocess.run([sys.executable, '-c', sys.argv[1]], timeout=90)\n"
                      " sys.exit(result.returncode)\n"
                      "except subprocess.TimeoutExpired:\n"
                      " print('PyTorch execution timed out.', file=sys.stderr)\n"
                      " sys.exit(1)\n")
        try:
            return run(["-c", supervisor, code], timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            print(f"PyTorch execution check failed: {exc}", file=sys.stderr)
            return 1

    selected = profile
    result = install(selected)
    if not result:
        result = check(selected)
    fallback = ""
    if result and selected.cuda:
        fallback = f"{selected.name} installation or execution failed (exit {result})."
        print(f"{fallback} Installing and validating CPU support automatically.", file=sys.stderr)
        selected = cpu_profile(target, fallback)
        result = install(selected)
        if not result:
            result = check(selected)
    if result:
        return result
    result = run(["-m", "pip", "install", "--only-binary=:all:", "-e", ".[gui,dev]"])
    if result:
        return result
    result = run(["-m", "cryoid_pipeline.preflight", "--gui", "--device", "cuda:0" if selected.cuda else "cpu"])
    if result:
        return result
    report = json.dumps({"platform": target, "hardware": hardware, "requested_profile": profile.name,
                         "installed_profile": selected.name, "reason": selected.reason, "fallback": fallback})
    result = run(["-c", "import sys; from pathlib import Path; "
                  "Path(sys.prefix, 'cryoid-installation.json').write_text(sys.argv[1], encoding='utf-8')", report])
    if not result:
        print(f"CryoID setup passed. Compute: {selected.name}. Runtime device selection remains Auto.", file=sys.stderr)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--prefix", type=Path)
    destination.add_argument("--name")
    parser.add_argument("--platform", choices=("linux-64", "win-64", "osx-64", "osx-arm64"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    host = {"win32": "win-64", "linux": "linux-64", "darwin": "osx"}.get(sys.platform)
    if host is None:
        parser.error(f"Unsupported operating system: {sys.platform}")
    target = args.platform or host
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
    # Do not use the host GPU when previewing a different platform.
    hardware = (probe_cuda() if target in {"linux-64", "win-64"} and target == host
                else {"devices": [], "reason": "CPU selected for macOS or a cross-platform preview."})
    profile = select_torch_profile(target, hardware)
    print(f"Automatic compute selection: {profile.name}. {profile.reason}", file=sys.stderr)
    content = installation_environment((ROOT / "environment.yml").read_text(encoding="utf-8"), target,
                                       managed_blast=managed_blast)
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
        result = subprocess.run(command, cwd=ROOT, env=clean_environment()).returncode
        if result or args.dry_run:
            return result
        target_env = (["--prefix", str(args.prefix.expanduser().resolve())] if args.prefix
                      else ["--name", args.name or "cryoid"])
        return install_runtime(conda, target_env, target, profile, hardware)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
