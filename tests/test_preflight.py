"""Failure-path checks for the environment gate, independent of installed versions."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from cryoid_pipeline import preflight
from G_GUI import diagnostics
from D_BLAST_searching import blast_tools


@pytest.fixture
def environment(monkeypatch):
    packages = {
        "numpy": SimpleNamespace(version="1.26.4", metadata={}, requires=[]),
        "matplotlib": SimpleNamespace(version="3.9.0", metadata={}, requires=["pillow>=8"]),
        "pillow": SimpleNamespace(version="11.0", metadata={}, requires=[]),
        "PyQt6": SimpleNamespace(version="6.7.0", metadata={}, requires=[]),
    }
    monkeypatch.setattr(preflight, "project_requirements", lambda: (
        ">=3.10,<3.14", ["numpy>=1.23,<3", "matplotlib>=3.7,<4"],
        {"gui-pyqt6": ["PyQt6>=6.7,<7"]},
    ))
    def distribution(name):
        if name not in packages:
            raise preflight.metadata.PackageNotFoundError(name)
        return packages[name]
    monkeypatch.setattr(preflight.metadata, "distribution", distribution)
    monkeypatch.setattr(preflight.importlib, "import_module", lambda name: SimpleNamespace(new=lambda *args: SimpleNamespace(close=lambda: None)))
    return packages


def test_valid_versions_and_transitive_pillow_are_checked(environment):
    lines, issues = preflight.check_environment("PyQt6")
    assert not issues
    assert any("pillow: OK" in line for line in lines)
    assert any("pyqt6: OK" in line for line in lines)


@pytest.mark.parametrize("package,bad_version", [("numpy", "3.0"), ("pillow", "7.0"), ("PyQt6", "6.6")])
def test_out_of_range_versions_block(environment, package, bad_version):
    environment[package].version = bad_version
    _, issues = preflight.check_environment("PyQt6")
    assert any(package in issue and "requires" in issue for issue in issues)


def test_cli_does_not_require_qt_but_gui_does(environment):
    del environment["PyQt6"]
    assert not preflight.check_environment()[1]
    assert any("PyQt6" in issue for issue in preflight.check_environment("PyQt6")[1])


def test_missing_transitive_dependency_blocks(environment):
    del environment["pillow"]
    assert any("pillow" in issue and "matplotlib" in issue for issue in preflight.check_environment()[1])


def test_native_import_failure_blocks(environment, monkeypatch):
    def broken(name):
        raise OSError("wrong ELF class")
    monkeypatch.setattr(preflight.importlib, "import_module", broken)
    lines, issues = preflight.check_environment()
    assert any("numpy: ERROR" in line for line in lines)
    assert any("wrong ELF class" in issue for issue in issues)


@pytest.mark.parametrize("python_version,bits", [("3.14.0", 8), ("3.11.0", 4)])
def test_unsupported_python_version_and_bitness_block(environment, monkeypatch, python_version, bits):
    monkeypatch.setattr(preflight.platform, "python_version", lambda: python_version)
    monkeypatch.setattr(preflight.struct, "calcsize", lambda fmt: bits)
    assert any("Python" in issue for issue in preflight.check_environment()[1])


def test_requirement_markers_and_transitive_extras_are_respected(environment):
    environment["numpy"].requires = ["absent; extra == 'accelerated'", "absent; python_version < '3.0'"]
    assert not preflight.check_environment()[1]
    environment["matplotlib"].requires.append("numpy[accelerated]>=1.23")
    assert any("absent" in issue for issue in preflight.check_environment()[1])


def test_torch_numpy_abi_failure_is_reported(monkeypatch):
    import torch
    monkeypatch.setattr(torch, "from_numpy", lambda array: (_ for _ in ()).throw(RuntimeError("NumPy ABI unavailable")))
    monkeypatch.setattr(diagnostics, "check_environment", lambda *args: ([], []))
    monkeypatch.setattr(diagnostics, "check_blast_tools", lambda: diagnostics.Check("OK", "OK", True))
    report = diagnostics.collect_diagnostics("", "cpu")
    assert not report.compute.ok
    assert "NumPy ABI unavailable" in report.compute.technical_detail


@pytest.fixture
def blast_bundle(tmp_path, monkeypatch):
    directory = tmp_path / "test-platform" / "bin"
    directory.mkdir(parents=True)
    executable = directory / "tool"
    executable.write_bytes(b"test-binary")
    manifest = {"test-platform": {"version": diagnostics.BLAST_VERSION, "files": [{
        "path": "test-platform/bin/tool", "size": executable.stat().st_size,
        "sha256": hashlib.sha256(executable.read_bytes()).hexdigest(),
    }]}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(diagnostics, "BUNDLE_ROOT", tmp_path)
    monkeypatch.setattr(diagnostics, "bundled_platform_key", lambda: "test-platform")
    monkeypatch.setattr(diagnostics, "runtime_blast_bin_dir", lambda: directory)
    monkeypatch.setattr(diagnostics, "blast_executable", lambda program, directory: program)
    return executable


def blast_result(command, **kwargs):
    program = command[0]
    output = f"{program}: {diagnostics.BLAST_VERSION}+\n" if "-version" in command else "cryoid_self_test\n"
    return SimpleNamespace(stdout=output, stderr="")


def test_blast_integrity_and_functional_searches(blast_bundle, monkeypatch):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return blast_result(command, **kwargs)
    monkeypatch.setattr(diagnostics.subprocess, "run", run)
    assert diagnostics.check_blast_tools().ok
    assert sum("-version" in command for command in calls) == 3
    assert sum("-dbtype" in command for command in calls) == 2
    assert sum("-query" in command for command in calls) == 2


def test_corrupted_blast_is_blocked_before_execution(blast_bundle, monkeypatch):
    blast_bundle.write_bytes(b"bad-binary!")
    monkeypatch.setattr(diagnostics.subprocess, "run", lambda *args, **kwargs: pytest.fail("Corrupt tool executed"))
    assert not diagnostics.check_blast_tools().ok


@pytest.mark.parametrize("output", ["blastp: 2.16.0+", "unrelated output", "", "blastn: 2.17.0+"])
def test_wrong_blast_version_or_identity_blocks(blast_bundle, monkeypatch, output):
    monkeypatch.setattr(diagnostics.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=output, stderr=""))
    report = diagnostics.check_blast_tools()
    assert not report.ok
    assert "Expected" in report.technical_detail


def test_failed_blast_search_blocks_even_when_versions_match(blast_bundle, monkeypatch):
    def run(command, **kwargs):
        result = blast_result(command, **kwargs)
        if "-query" in command:
            result.stdout = ""
        return result
    monkeypatch.setattr(diagnostics.subprocess, "run", run)
    report = diagnostics.check_blast_tools()
    assert not report.ok
    assert "self-test" in report.detail


def test_blast_timeout_blocks(blast_bundle, monkeypatch):
    def run(command, **kwargs):
        raise diagnostics.subprocess.TimeoutExpired(command, 10)
    monkeypatch.setattr(diagnostics.subprocess, "run", run)
    report = diagnostics.check_blast_tools()
    assert not report.ok
    assert "did not respond" in report.detail


@pytest.mark.parametrize("issues,expected", [([], 0), (["pandas: missing"], 1)])
def test_headless_command_needs_no_qt_and_returns_failure_status(monkeypatch, capsys, issues, expected):
    import builtins
    original_import = builtins.__import__
    def without_qt(name, *args, **kwargs):
        if name.startswith(("PyQt", "G_GUI.qt_compat")):
            raise AssertionError("CLI-only validation imported Qt")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_qt)
    monkeypatch.setattr(preflight, "check_environment", lambda qt: (["environment checked"], issues))
    monkeypatch.setattr(preflight, "exercise_torch", lambda device: None)
    monkeypatch.setattr(diagnostics, "check_blast_tools", lambda: diagnostics.Check("OK", "OK", True))
    assert preflight.main([]) == expected
    assert ("Setup checks passed" if expected == 0 else "pandas: missing") in capsys.readouterr().out


def test_installed_wheel_requirements_come_from_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(preflight, "__file__", str(tmp_path / "cryoid_pipeline" / "preflight.py"))
    required = ["numpy>=1.23,<3", 'PyQt6>=6.7,<7; extra == "gui-pyqt6"']
    monkeypatch.setattr(preflight.metadata, "distribution", lambda name: SimpleNamespace(
        metadata={"Requires-Python": ">=3.10,<3.14"}, requires=required,
    ))
    assert preflight.project_requirements() == (">=3.10,<3.14", required, {})


@pytest.mark.parametrize("name,release,blocked", [
    ("glibc", "2.17", True), ("glibc", "2.28", False),
    ("glibc", "2.36", False), ("musl", "1.2.5", True), ("", "", True),
])
def test_linux_runtime_requirement_matches_bundled_blast(monkeypatch, name, release, blocked):
    monkeypatch.setattr(blast_tools.platform, "system", lambda: "Linux")
    monkeypatch.setattr(blast_tools.platform, "libc_ver", lambda: (name, release))
    issue = blast_tools.linux_runtime_issue()
    assert bool(issue) == blocked
    if blocked:
        assert "glibc 2.28" in issue
        assert "host/container" in issue


def test_gui_reports_incompatible_host_runtime_instead_of_reinstall_advice(monkeypatch):
    monkeypatch.setattr(diagnostics, "runtime_blast_bin_dir", lambda: (_ for _ in ()).throw(
        RuntimeError("Bundled BLAST requires glibc 2.28; host has glibc 2.17. Use a supported host/container.")))
    report = diagnostics.check_blast_tools()
    assert not report.ok
    assert report.summary == "Unsupported runtime"
    assert "glibc 2.17" in report.detail
    assert "Reinstall" not in report.detail
