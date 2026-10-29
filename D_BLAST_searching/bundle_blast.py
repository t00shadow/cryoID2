"""Reproduce the pinned, private BLAST+ payloads shipped with CryoID.

This is a maintainer build tool. Normal installations never download BLAST.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

VERSION = "2.17.0"
ROOT = Path(__file__).resolve().parent
SOURCES = {
    "windows-x86_64": f"ncbi-blast-{VERSION}+-x64-win64.tar.gz",
    "linux-x86_64": f"ncbi-blast-{VERSION}+-x64-linux.tar.gz",
    "darwin-x86_64": f"ncbi-blast-{VERSION}+-x64-macosx.tar.gz",
    "darwin-arm64": f"ncbi-blast-{VERSION}+-aarch64-macosx.tar.gz",
}
BASE_URL = f"https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/{VERSION}/"
PROGRAMS = {"blastp", "blastn", "makeblastdb"}


def digest(path: Path, algorithm: str = "sha256") -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def prepare(platform: str) -> dict:
    filename = SOURCES[platform]
    downloads = ROOT / ".downloads"
    downloads.mkdir(exist_ok=True)
    archive = downloads / filename
    url = BASE_URL + filename
    with urllib.request.urlopen(url + ".md5", timeout=60) as response:
        expected_md5 = response.read().decode("ascii").split()[0]
    if not archive.exists() or digest(archive, "md5") != expected_md5:
        temporary = archive.with_suffix(archive.suffix + ".part")
        print(f"Downloading {platform}: {url}", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle, 1024 * 1024)
        if digest(temporary, "md5") != expected_md5:
            raise ValueError(f"NCBI checksum mismatch: {filename}")
        temporary.replace(archive)
    manifest_path = ROOT / "bundled" / "manifest.json"
    pinned = json.loads(manifest_path.read_text()).get(platform, {}) if manifest_path.exists() else {}
    if pinned.get("archive_sha256") and digest(archive) != pinned["archive_sha256"]:
        raise ValueError(f"Pinned SHA-256 mismatch: {filename}")
    destination = ROOT / "bundled" / platform
    files = []
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            path = PurePosixPath(member.name)
            relative = PurePosixPath(*path.parts[1:])
            if not member.isfile() or ".." in relative.parts:
                continue
            name = relative.name
            binary = relative.parts[:1] == ("bin",) and (
                name in PROGRAMS or name.removesuffix(".exe") in PROGRAMS
                or name.removesuffix(".exe.manifest") in PROGRAMS or name.endswith(".dll")
            )
            support = relative.parts[:1] in {("lib",), ("data",)}
            notice = len(relative.parts) == 1 and name.lower().startswith(("license", "copying", "readme"))
            if not (binary or support or notice):
                continue
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = source.extractfile(member)
            assert stream is not None
            with stream, target.open("wb") as handle:
                shutil.copyfileobj(stream, handle)
            if binary and not name.endswith((".manifest", ".dll")):
                target.chmod(0o755)
            files.append({"path": target.relative_to(ROOT / "bundled").as_posix(),
                          "size": target.stat().st_size, "sha256": digest(target)})
    found = {Path(item["path"]).name.removesuffix(".exe") for item in files}
    if not PROGRAMS <= found:
        raise ValueError(f"Incomplete BLAST archive: {platform}")
    print(f"Prepared {platform}: {len(files)} files", flush=True)
    return {"version": VERSION, "source": url, "archive_md5": expected_md5,
            "archive_sha256": digest(archive), "files": sorted(files, key=lambda item: item["path"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=SOURCES, action="append")
    args = parser.parse_args()
    selected = args.platform or list(SOURCES)
    manifest_path = ROOT / "bundled" / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    with ThreadPoolExecutor(max_workers=2) as executor:
        for platform, result in zip(selected, executor.map(prepare, selected)):
            manifest[platform] = result
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
