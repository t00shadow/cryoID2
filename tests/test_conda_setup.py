from pathlib import Path

import pytest
from types import SimpleNamespace

from cryoid_pipeline import create_conda_env
from D_BLAST_searching import blast_tools
from cryoid_pipeline.create_conda_env import select_environment
from cryoid_pipeline.install_hardware import cpu_profile, select_torch_profile


@pytest.fixture(autouse=True)
def no_host_gpu_dependency(monkeypatch):
    monkeypatch.setattr(create_conda_env, "probe_cuda", lambda: {"devices": [], "reason": "Test CPU host"})


@pytest.mark.parametrize("platform", ["linux-64", "osx-64", "osx-arm64", "win-64"])
def test_shared_environment_selects_a_supported_parasail_install(platform):
    source = (Path(__file__).parents[1] / "environment.yml").read_text(encoding="utf-8")
    selected = select_environment(source, platform)
    assert ("parasail-python=1.3.4" in selected) == (platform != "win-64")
    assert ("  - bioconda" in selected) == (platform != "win-64")
    assert "-e .[gui,dev]" in selected
    assert "# [unix]" not in selected
    assert source.count("parasail-python=1.3.4") == 1


def test_unknown_selector_fails_instead_of_silently_dropping_dependencies():
    with pytest.raises(ValueError, match="Unsupported environment selector"):
        select_environment("dependencies:\n  - example  # [typo]\n", "win-64")


@pytest.mark.parametrize("args,create_code,check_code", [
    ([], 0, 0), (["--name", "custom"], 0, 1), (["--prefix", "custom-prefix"], 0, 0),
    (["--dry-run"], 0, 0), ([], 1, 0),
])
def test_conda_creation_validates_target_environment_and_propagates_failures(
    monkeypatch, tmp_path, args, create_code, check_code,
):
    (tmp_path / "environment.yml").write_text("name: cryoid\ndependencies:\n  - python=3.11\n")
    monkeypatch.setattr(create_conda_env, "ROOT", tmp_path)
    monkeypatch.setenv("CONDA_EXE", "test-conda")
    monkeypatch.setattr(blast_tools, "linux_runtime_issue", lambda: None)
    monkeypatch.setattr(create_conda_env.sys, "argv", ["create_conda_env", *args])
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[-1] == "--version":
            return SimpleNamespace(returncode=0, stdout="conda 25.9.1\n")
        return SimpleNamespace(returncode=create_code if command[1:3] == ["env", "create"] else check_code)
    monkeypatch.setattr(create_conda_env.subprocess, "run", run)
    result = create_conda_env.main()
    assert result == (create_code or check_code)
    assert calls[0] == ["test-conda", "--version"]
    calls = calls[1:]
    if create_code or "--dry-run" in args:
        assert len(calls) == 1
    else:
        assert any("cryoid_pipeline.preflight" in call for call in calls) == (check_code == 0)
        target_call = calls[1]
        if "--prefix" in args:
            assert target_call[3:5] == ["--prefix", str(create_conda_env.Path("custom-prefix").resolve())]
        else:
            assert target_call[3:5] == ["--name", "custom" if "--name" in args else "cryoid"]
    assert not list(tmp_path.glob(".cryoid-env-*.yml"))


def test_incompatible_linux_runtime_stops_before_creating_environment(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(create_conda_env, "ROOT", tmp_path)
    monkeypatch.setattr(create_conda_env.sys, "argv", ["create_conda_env"])
    monkeypatch.setattr(blast_tools, "linux_runtime_issue", lambda: "Bundled BLAST requires glibc 2.28; host has glibc 2.12.")
    monkeypatch.setattr(blast_tools, "needs_managed_linux_blast", lambda: False)
    monkeypatch.setattr(create_conda_env.subprocess, "run", lambda *args, **kwargs: pytest.fail("Installation started on unsupported host"))
    with pytest.raises(SystemExit) as failure:
        create_conda_env.main()
    assert failure.value.code == 2
    assert "glibc 2.28" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_32_bit_python_is_rejected_before_installation(monkeypatch, capsys):
    monkeypatch.setattr(create_conda_env.sys, "argv", ["create_conda_env"])
    monkeypatch.setattr(create_conda_env.struct, "calcsize", lambda fmt: 4)
    monkeypatch.setattr(create_conda_env.subprocess, "run", lambda *args, **kwargs: pytest.fail("Installation started with 32-bit Python"))
    with pytest.raises(SystemExit):
        create_conda_env.main()
    assert "64-bit" in capsys.readouterr().err


@pytest.mark.parametrize("installed", ["conda 4.5.11", "conda 25.5.0", "unrecognized version"])
def test_old_or_unrecognized_conda_is_rejected_before_installation(monkeypatch, tmp_path, capsys, installed):
    monkeypatch.setattr(create_conda_env, "ROOT", tmp_path)
    monkeypatch.setattr(create_conda_env.sys, "argv", ["create_conda_env"])
    monkeypatch.setenv("CONDA_EXE", "test-conda")
    monkeypatch.setattr(blast_tools, "linux_runtime_issue", lambda: None)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert command == ["test-conda", "--version"]
        return SimpleNamespace(returncode=0, stdout=installed)
    monkeypatch.setattr(create_conda_env.subprocess, "run", run)
    with pytest.raises(SystemExit):
        create_conda_env.main()
    assert "Conda >=25.5.1" in capsys.readouterr().err
    assert len(calls) == 1
    assert not list(tmp_path.iterdir())


def test_native_stage_defers_torch_and_app_until_hardware_is_selected():
    source = (Path(__file__).parents[1] / "environment.yml").read_text()
    selected = create_conda_env.installation_environment(source, "linux-64", managed_blast=True)
    assert f"blast={blast_tools.BLAST_VERSION}={blast_tools.MANAGED_BLAST_BUILD}" in selected
    assert "  - pytorch" not in selected
    assert "  - pip\n" in selected
    assert "  - pip:" not in selected
    assert "-e .[gui,dev]" not in selected
    mac = create_conda_env.installation_environment(source, "osx-arm64")
    assert "  - pytorch" in mac
    assert "  - pip:" not in mac


def test_legacy_linux_installer_selects_verified_conda_blast(monkeypatch, tmp_path):
    (tmp_path / "environment.yml").write_text("name: cryoid\ndependencies:\n  - python=3.11\n")
    monkeypatch.setattr(create_conda_env, "ROOT", tmp_path)
    monkeypatch.setattr(create_conda_env.sys, "argv", ["create_conda_env"])
    monkeypatch.setattr(create_conda_env.sys, "platform", "linux")
    monkeypatch.setattr(blast_tools.platform, "system", lambda: "Linux")
    monkeypatch.setattr(blast_tools.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(blast_tools.platform, "libc_ver", lambda: ("glibc", "2.17"))
    monkeypatch.setenv("CONDA_EXE", "test-conda")
    def run(command, **kwargs):
        if command[-1] == "--version":
            return SimpleNamespace(stdout="conda 25.9.1", returncode=0)
        assert command[1:3] == ["env", "create"]
        yaml = Path(command[command.index("--file") + 1]).read_text()
        assert f"blast=2.17.0={blast_tools.MANAGED_BLAST_BUILD}" in yaml
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(create_conda_env.subprocess, "run", run)
    assert create_conda_env.main() == 1


def test_gpu_is_detected_without_installation_flags_and_checked_before_app_install(monkeypatch, tmp_path):
    source = (Path(__file__).parents[1] / "environment.yml").read_text()
    (tmp_path / "environment.yml").write_text(source)
    monkeypatch.setattr(create_conda_env, "ROOT", tmp_path)
    monkeypatch.setattr(create_conda_env.sys, "argv", ["create_conda_env"])
    monkeypatch.setattr(create_conda_env.sys, "platform", "linux")
    monkeypatch.setattr(blast_tools.platform, "system", lambda: "Linux")
    monkeypatch.setattr(blast_tools.platform, "machine", lambda: "x86_64")
    hardware = {"driver_api": 11020, "devices": [{"capability": [6, 1], "name": "GTX 1080 Ti"}]}
    monkeypatch.setattr(create_conda_env, "probe_cuda", lambda: hardware)
    monkeypatch.setattr(blast_tools, "linux_runtime_issue", lambda: None)
    monkeypatch.setenv("CONDA_EXE", "test-conda")
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="conda 25.9.1", returncode=0)
    monkeypatch.setattr(create_conda_env.subprocess, "run", run)
    assert create_conda_env.main() == 0
    assert any(command[-1] == "torch==2.5.1+cu118" for command in commands)
    probe = next(i for i, command in enumerate(commands) if any("CUDA execution passed" in argument for argument in command))
    app = next(i for i, command in enumerate(commands) if ".[gui,dev]" in command)
    assert probe < app
    assert any(command[-2:] == ["--device", "cuda:0"] for command in commands)
    assert '"installed_profile": "cu118"' in commands[-1][-1]


@pytest.mark.parametrize("failure", ["download", "execution", "crash", "timeout"])
def test_failed_cuda_automatically_installs_cpu_and_never_reports_gpu_success(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(create_conda_env, "ROOT", tmp_path)
    hardware = {"driver_api": 11020, "devices": [{"capability": [6, 1]}]}
    profile = select_torch_profile("linux-64", hardware)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        if command[-1] == "torch==2.5.1+cu118" and failure == "download":
            return SimpleNamespace(returncode=1)
        if any("CUDA execution passed" in argument for argument in command):
            if failure == "timeout":
                raise create_conda_env.subprocess.TimeoutExpired(command, 120)
            return SimpleNamespace(returncode=-11 if failure == "crash" else 1)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(create_conda_env.subprocess, "run", run)
    assert create_conda_env.install_runtime("conda", ["--prefix", str(tmp_path)], "linux-64", profile, hardware) == 0
    assert any(command[-1] == "torch==2.5.1+cpu" for command in calls)
    assert any(command[-2:] == ["--device", "cpu"] for command in calls)
    assert not any(command[-2:] == ["--device", "cuda:0"] for command in calls)
    assert '"installed_profile": "cpu"' in calls[-1][-1]
    assert "failed" in calls[-1][-1]


def test_cpu_failure_is_not_reported_as_success(monkeypatch, tmp_path):
    monkeypatch.setattr(create_conda_env, "ROOT", tmp_path)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(create_conda_env.subprocess, "run", run)
    assert create_conda_env.install_runtime("conda", ["--name", "new"], "win-64", cpu_profile("win-64", "No GPU"), {}) == 1
    assert len(calls) == 1


def test_failed_blast_or_dependency_validation_is_not_hidden_by_cpu_fallback(monkeypatch, tmp_path):
    monkeypatch.setattr(create_conda_env, "ROOT", tmp_path)
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if "cryoid_pipeline.preflight" in command else 0)
    monkeypatch.setattr(create_conda_env.subprocess, "run", run)
    hardware = {"driver_api": 11020, "devices": [{"capability": [6, 1]}]}
    assert create_conda_env.install_runtime("conda", ["--name", "new"], "linux-64", select_torch_profile("linux-64", hardware), hardware) == 1
    assert not any(command[-1] == "torch==2.5.1+cpu" for command in calls)
    assert "cryoid_pipeline.preflight" in calls[-1]


def test_mac_uses_native_conda_torch_and_never_installs_cuda_wheel(monkeypatch, tmp_path):
    monkeypatch.setattr(create_conda_env, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(create_conda_env.subprocess, "run", lambda command, **kwargs: calls.append(command) or SimpleNamespace(returncode=0))
    assert create_conda_env.install_runtime("conda", ["--name", "new"], "osx-arm64", cpu_profile("osx-arm64", "Mac"), {}) == 0
    assert not any(any("torch==" in argument for argument in command) for command in calls)
    assert any(command[-2:] == ["--device", "cpu"] for command in calls)


def test_removed_cuda_flag_is_not_required_or_advertised(monkeypatch, capsys):
    monkeypatch.setattr(create_conda_env.sys, "argv", ["create_conda_env", "--help"])
    with pytest.raises(SystemExit) as exit:
        create_conda_env.main()
    assert exit.value.code == 0
    assert "--cuda118" not in capsys.readouterr().out
