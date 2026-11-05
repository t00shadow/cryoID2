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
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not conda:
        parser.error("Conda was not found. Run this script with conda run -n base python.")
    content = select_environment((ROOT / "environment.yml").read_text(encoding="utf-8"), target)
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
        return subprocess.run(
            [conda, "run", *target_env, "python", "-m", "cryoid_pipeline.preflight", "--gui"],
            cwd=ROOT,
        ).returncode
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
