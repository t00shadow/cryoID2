"""Legacy Linux uses only the pinned BLAST package in the active Conda prefix."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from D_BLAST_searching import blast_tools as tools
from G_GUI import diagnostics


@pytest.fixture
def managed_prefix(monkeypatch, tmp_path):
    monkeypatch.setattr(tools.sys, "prefix", str(tmp_path))
    monkeypatch.setattr(tools.platform, "system", lambda: "Linux")
    monkeypatch.setattr(tools.platform, "libc_ver", lambda: ("glibc", "2.17"))
    entries = []
    for relative in [*(f"bin/{name}" for name in tools.BLAST_PROGRAMS), "lib/ncbi-blast+/libtest.so"]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        data = relative.encode()
        path.write_bytes(data)
        path.chmod(0o755)
        entries.append({"_path": relative, "size_in_bytes": len(data),
                        "sha256": "0" * 64, "sha256_in_prefix": hashlib.sha256(data).hexdigest()})
    record = {"name": "blast", "version": tools.BLAST_VERSION, "build": tools.MANAGED_BLAST_BUILD,
              "url": tools.MANAGED_BLAST_URL, "sha256": tools.MANAGED_BLAST_SHA256,
              "paths_data": {"paths": entries}}
    metadata = tmp_path / "conda-meta" / f"{tools.MANAGED_BLAST_PACKAGE}.json"
    metadata.parent.mkdir()
    metadata.write_text(json.dumps(record))
    return tmp_path, metadata, record


def test_only_active_prefix_is_used_despite_path_and_retired_override(managed_prefix, monkeypatch):
    prefix, _, _ = managed_prefix
    monkeypatch.setenv("PATH", "unrelated-tools")
    monkeypatch.setenv("CRYOID_BLAST_BIN_DIR", "unrelated-tools")
    assert tools.runtime_blast_bin_dir() == prefix / "bin"
    for name in tools.BLAST_PROGRAMS:
        assert Path(tools.blast_executable(name)) == prefix / "bin" / name
    tools.verify_managed_blast(prefix / "bin")


@pytest.mark.parametrize("field,value", [
    ("version", "2.12.0"), ("build", "other-build"), ("sha256", "0" * 64),
    ("url", "https://example.com/other-package.conda"),
])
def test_different_conda_package_is_rejected(managed_prefix, field, value):
    prefix, metadata, record = managed_prefix
    record[field] = value
    metadata.write_text(json.dumps(record))
    with pytest.raises(RuntimeError, match="pinned Conda-managed BLAST"):
        tools.runtime_blast_bin_dir()


@pytest.mark.parametrize("relative", ["bin/blastp", "lib/ncbi-blast+/libtest.so"])
def test_corrupt_executable_or_support_library_fails_integrity(managed_prefix, relative):
    prefix, _, _ = managed_prefix
    (prefix / relative).write_bytes(b"damaged")
    with pytest.raises(ValueError, match="integrity"):
        tools.verify_managed_blast(prefix / "bin")


def test_missing_package_never_uses_another_blast_on_path(managed_prefix, monkeypatch):
    _, metadata, _ = managed_prefix
    metadata.unlink()
    monkeypatch.setenv("PATH", "unrelated-valid-blast")
    with pytest.raises(RuntimeError, match="create_conda_env"):
        tools.runtime_blast_bin_dir()


def test_modern_linux_keeps_using_the_shipped_payload(managed_prefix, monkeypatch):
    prefix, _, _ = managed_prefix
    monkeypatch.setattr(tools.platform, "libc_ver", lambda: ("glibc", "2.28"))
    expected = tools.BUNDLE_ROOT / "linux-x86_64" / "bin"
    assert tools.runtime_blast_bin_dir() == expected
    assert expected != prefix / "bin"


def test_setup_checks_hashes_versions_and_searches_for_managed_package(managed_prefix, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        name = Path(command[0]).name
        return SimpleNamespace(stdout=f"{name}: {tools.BLAST_VERSION}+\n" if "-version" in command else "cryoid_self_test\n", stderr="")
    monkeypatch.setattr(diagnostics.subprocess, "run", run)
    report = diagnostics.check_blast_tools()
    assert report.ok, report
    assert report.summary == "Conda / CPU"
    assert len(calls) == 7
    (managed_prefix[0] / "lib/ncbi-blast+/libtest.so").write_bytes(b"corrupt")
    calls.clear()
    report = diagnostics.check_blast_tools()
    assert not report.ok
    assert "Recreate" in report.detail
    assert not calls
