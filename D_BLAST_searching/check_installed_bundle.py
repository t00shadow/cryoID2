"""Smoke-check an isolated wheel installation without scientific dependencies."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from zipfile import ZipFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, type=Path)
    parser.add_argument("--wheel", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    site = args.site.resolve()
    sys.path.insert(0, str(site))
    tools = importlib.import_module("D_BLAST_searching.blast_tools")
    assert Path(tools.__file__).resolve().is_relative_to(site)
    manifest = json.loads((tools.BUNDLE_ROOT / "manifest.json").read_text())
    with ZipFile(args.wheel) as wheel:
        assert wheel.testzip() is None
        for payload in manifest.values():
            for item in payload["files"]:
                path = tools.BUNDLE_ROOT / item["path"]
                assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
                name = "D_BLAST_searching/bundled/" + item["path"]
                assert wheel.read(name) == path.read_bytes()
                if "/bin/" in name and "/windows-" not in name:
                    assert (wheel.getinfo(name).external_attr >> 16) & 0o111 == 0o111
        assert not any("/validation/work/" in name for name in wheel.namelist())
    # No PATH discovery, saved BLAST configuration, or scientific libraries.
    os.environ["PATH"] = ""
    os.environ["CRYOID_BLAST_BIN_DIR"] = "invalid-legacy-override"
    directory = tools.bundled_blast_bin_dir()
    versions = {}
    for program in tools.BLAST_PROGRAMS:
        executable = Path(tools.blast_executable(program))
        assert executable.parent == directory
        if sys.platform != "win32":
            assert executable.stat().st_mode & 0o111 == 0o111
        result = subprocess.check_output([str(executable), "-version"], text=True)
        assert f"{program}: {tools.BLAST_VERSION}+" in result
        versions[program] = result.strip()
    report = {"platform": platform.platform(), "wheel": args.wheel.name,
              "wheel_sha256": hashlib.sha256(args.wheel.read_bytes()).hexdigest(),
              "versions": versions, "payload_hashes_match": True,
              "unix_wheel_execute_bits": True, "empty_path_smoke_test": True}
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wheel bundle verified on {platform.system()}: all three tools run without external BLAST.")


if __name__ == "__main__":
    main()
