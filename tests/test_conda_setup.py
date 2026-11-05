from pathlib import Path

import pytest
from types import SimpleNamespace

from cryoid_pipeline import create_conda_env
from D_BLAST_searching import blast_tools
from cryoid_pipeline.create_conda_env import select_environment


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
        assert calls[1][-4:] == ["python", "-m", "cryoid_pipeline.preflight", "--gui"]
        if "--prefix" in args:
            assert calls[1][2:4] == ["--prefix", str(create_conda_env.Path("custom-prefix").resolve())]
        else:
            assert calls[1][2:4] == ["--name", "custom" if "--name" in args else "cryoid"]
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


def test_cuda_profile_uses_one_pinned_torch_provider_and_managed_blast():
    source = (Path(__file__).parents[1] / "environment.yml").read_text()
    selected = create_conda_env.installation_environment(source, "linux-64", managed_blast=True, cuda118=True)
    assert f"blast={blast_tools.BLAST_VERSION}={blast_tools.MANAGED_BLAST_BUILD}" in selected
    assert "  - pytorch" not in selected
    assert "torch==2.5.1+cu118" in selected
    assert "--extra-index-url https://download.pytorch.org/whl/cu118" in selected
    assert "-e .[gui,dev]" in selected


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


def test_cuda_profile_checks_cuda_before_installation_is_accepted(monkeypatch, tmp_path):
    source = (Path(__file__).parents[1] / "environment.yml").read_text()
    (tmp_path / "environment.yml").write_text(source)
    monkeypatch.setattr(create_conda_env, "ROOT", tmp_path)
    monkeypatch.setattr(create_conda_env.sys, "argv", ["create_conda_env", "--cuda118"])
    monkeypatch.setattr(blast_tools, "linux_runtime_issue", lambda: None)
    monkeypatch.setenv("CONDA_EXE", "test-conda")
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout="conda 25.9.1", returncode=0)
    monkeypatch.setattr(create_conda_env.subprocess, "run", run)
    assert create_conda_env.main() == 0
    assert commands[-1][-2:] == ["--device", "cuda:0"]
