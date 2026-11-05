from pathlib import Path

import pytest
from types import SimpleNamespace

from cryoid_pipeline import create_conda_env
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
    monkeypatch.setattr(create_conda_env.sys, "argv", ["create_conda_env", *args])
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=create_code if len(calls) == 1 else check_code)
    monkeypatch.setattr(create_conda_env.subprocess, "run", run)
    result = create_conda_env.main()
    assert result == (create_code or check_code)
    if create_code or "--dry-run" in args:
        assert len(calls) == 1
    else:
        assert calls[1][-4:] == ["python", "-m", "cryoid_pipeline.preflight", "--gui"]
        if "--prefix" in args:
            assert calls[1][2:4] == ["--prefix", str(create_conda_env.Path("custom-prefix").resolve())]
        else:
            assert calls[1][2:4] == ["--name", "custom" if "--name" in args else "cryoid"]
    assert not list(tmp_path.glob(".cryoid-env-*.yml"))
