"""CLI project settings must match GUI settings without rewriting saved files."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from cryoid_pipeline import cli
from cryoid_pipeline.models import PipelineResult, StageResult
from cryoid_pipeline.user_config import load_config, results_project_dir, save_config


@pytest.fixture
def captured_pipeline(monkeypatch, tmp_path):
    captured = []

    class Pipeline:
        def __init__(self, **kwargs):
            captured.append(kwargs)

        def run(self, *args, **kwargs):
            return PipelineResult(tmp_path, [StageResult("Run", "ok", "Test pipeline")])

    monkeypatch.setattr("cryoid_pipeline.orchestrator.CryoIDPipeline", Pipeline)
    return captured


def test_cli_switches_between_project_configs_instead_of_global_defaults(tmp_path, captured_pipeline):
    save_config({"rcl": "9", "weights_dir": str(tmp_path / "global-weights"), "sequence_mode": "auto"})
    projects = [results_project_dir(tmp_path / name) for name in ("A", "B")]
    original = {}
    for index, project in enumerate(projects):
        path = project / "config.toml"
        save_config({"rcl": str(0.25 + index), "weights_dir": str(tmp_path / f"weights-{index}"),
                     "sequence_mode": "protein" if index == 0 else "rna",
                     "cpu_threads": "1", "device": "cpu", "keep_tmp": "true",
                     "protein_min_trace_length": str(15 + index), "structure_selection": "3"}, path)
        original[path] = path.read_bytes()
    for index in (0, 1, 0):
        project = projects[index]
        # Both parent-directory and cryoid_results arguments must work.
        location = project.parent if index == 0 else project
        assert cli.main(["run", "--map", "map.mrc", "--pool", "pool.fa", "--project", str(location)]) == 0
        actual = captured_pipeline[-1]
        assert actual["project_dir"] == project
        assert actual["unet_rcl"] == 0.25 + index
        assert actual["unet_weights_dir"] == tmp_path / "global-weights"
        assert actual["sequence_mode"] == ("protein" if index == 0 else "rna")
        assert actual["protein_min_trace_length"] == 15 + index
        assert actual["structure_selection"] == "3"
        assert actual["cpu_threads"] == 1
        assert actual["unet_device"] == "cpu"
        assert actual["keep_tmp"] is True
    assert all(path.read_bytes() == data for path, data in original.items())
    assert load_config()["rcl"] == "9"


def test_cli_flags_override_environment_then_project_then_user_defaults(monkeypatch, tmp_path, captured_pipeline):
    save_config({"rcl": "9", "weights_dir": "global-weights"})
    project = results_project_dir(tmp_path)
    path = project / "config.toml"
    save_config({"rcl": "0.25", "weights_dir": "project-weights", "device": "cpu",
                 "cpu_threads": "1", "sequence_mode": "rna", "keep_tmp": "true"}, path)
    before = path.read_bytes()
    monkeypatch.setenv("CRYOID_RCL", "0.5")
    monkeypatch.setenv("CRYOID_UNET_WEIGHTS_DIR", str(tmp_path / "environment-weights"))
    monkeypatch.setenv("CRYOID_CPU_THREADS", "1")
    monkeypatch.setenv("CRYOID_KEEP_TMP", "true")
    base = ["run", "--map", "map.mrc", "--pool", "pool.fa", "--project", str(tmp_path)]
    assert cli.main(base) == 0
    assert captured_pipeline[-1]["unet_rcl"] == 0.5
    assert captured_pipeline[-1]["unet_weights_dir"] == tmp_path / "environment-weights"
    assert cli.main([*base, "--rcl", "0", "--weights-dir", str(tmp_path / "explicit-weights"),
                     "--device", "auto", "--cpu-threads", "auto", "--mode", "protein", "--no-keep-tmp"]) == 0
    actual = captured_pipeline[-1]
    assert actual["unet_rcl"] == 0
    assert actual["unet_weights_dir"] == tmp_path / "explicit-weights"
    assert actual["unet_device"] == actual["cpu_threads"] == "auto"
    assert actual["sequence_mode"] == "protein"
    assert actual["keep_tmp"] is False
    assert path.read_bytes() == before


@pytest.mark.parametrize("location", ["argument", "current-directory", "current-directory-with-env"])
def test_cli_resolves_selected_project_and_partial_config(monkeypatch, tmp_path, captured_pipeline, location):
    project = results_project_dir(tmp_path / "selected")
    save_config({"rcl": "9", "weights_dir": str(tmp_path / "user-weights"), "sequence_mode": "protein",
                 "last_density_map": "other-project.mrc", "last_candidate_pool": "other-project.fa",
                 "project_dir": str(tmp_path / "unrelated-project")})
    project.mkdir(parents=True)
    # Fields omitted from a hand-written project config inherit user defaults.
    (project / "config.toml").write_text('[defaults]\nrcl = "0.125"\n', encoding="utf-8")
    args = ["run", "--map", "map.mrc", "--pool", "pool.fa"]
    if location == "argument":
        args += ["--project", str(project.parent)]
    else:
        monkeypatch.chdir(project.parent)
        if location == "current-directory-with-env":
            monkeypatch.setenv("CRYOID_PROJECT_DIR", str(tmp_path / "unrelated-environment-project"))
    assert cli.main(args) == 0
    actual = captured_pipeline[-1]
    assert actual["project_dir"] == project
    assert actual["unet_rcl"] == 0.125
    assert actual["unet_weights_dir"] == tmp_path / "user-weights"
    assert actual["sequence_mode"] == "protein"
    assert actual["user_config"]["last_density_map"] == ""
    assert actual["user_config"]["last_candidate_pool"] == ""


def test_cli_without_project_config_uses_user_defaults(tmp_path, captured_pipeline):
    save_config({"rcl": "0.75", "sequence_mode": "rna"})
    assert cli.main(["run", "--map", "map.mrc", "--pool", "pool.fa", "--project", str(tmp_path)]) == 0
    assert captured_pipeline[-1]["unet_rcl"] == 0.75
    assert captured_pipeline[-1]["sequence_mode"] == "rna"
    saved = load_config(results_project_dir(tmp_path) / "config.toml")
    assert saved["rcl"] == "0.75"
    assert saved["sequence_mode"] == "rna"


def test_project_without_config_never_restores_a_historical_run(tmp_path, captured_pipeline):
    from cryoid_pipeline.user_config import load_effective_config
    project = results_project_dir(tmp_path)
    save_config({"rcl": "0.5"})
    for index, name in enumerate(("run_20261001_120000", "run_20261002_120000")):
        run = project / "runs" / name
        run.mkdir(parents=True)
        (run / "run_config.json").write_text(json.dumps({"configured_rcl": 0.25 + index,
                                                        "requested_mode": "protein"}), encoding="utf-8")
    base = ["run", "--map", "map.mrc", "--pool", "pool.fa", "--project", str(tmp_path)]
    assert cli.main(base) == 0
    assert captured_pipeline[-1]["unet_rcl"] == 0.5
    assert load_config(project / "config.toml")["rcl"] == "0.5"
    save_config({"rcl": "0.75"}, project / "config.toml")
    assert cli.main(base) == 0
    assert captured_pipeline[-1]["unet_rcl"] == 0.75
    loaded = load_effective_config(project)
    assert loaded.values["rcl"] == "0.75"
    assert load_config(project / "config.toml")["rcl"] == "0.75"


def test_alternate_user_defaults_are_overlaid_by_the_selected_project(tmp_path, captured_pipeline):
    base = tmp_path / "other-defaults.toml"
    save_config({"rcl": "9", "sequence_mode": "rna"}, base)
    project = results_project_dir(tmp_path)
    project.mkdir(parents=True)
    (project / "config.toml").write_text('[defaults]\nrcl = "0.25"\n', encoding="utf-8")
    assert cli.main(["run", "--map", "map.mrc", "--pool", "pool.fa", "--config", str(base), "--project", str(tmp_path)]) == 0
    assert captured_pipeline[-1]["unet_rcl"] == 0.25
    assert captured_pipeline[-1]["sequence_mode"] == "rna"


def test_malformed_project_config_has_a_clear_subprocess_error(tmp_path):
    path = results_project_dir(tmp_path) / "config.toml"
    path.parent.mkdir(parents=True)
    path.write_text("broken = [", encoding="utf-8")
    result = subprocess.run([sys.executable, "-m", "cryoid_pipeline", "run", "--map", "map.mrc", "--pool", "pool.fa",
                             "--project", str(tmp_path)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert str(path) in result.stderr
    assert "Traceback" not in result.stderr
    assert not (path.parent / "runs").exists()


@pytest.mark.parametrize("content", ['broken = [', 'defaults = []', '[defaults]\nrcl = "invalid"'])
def test_invalid_project_config_stops_cli_without_rewriting_it(tmp_path, captured_pipeline, capsys, content):
    path = results_project_dir(tmp_path) / "config.toml"
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    assert cli.main(["run", "--map", "map.mrc", "--pool", "pool.fa", "--project", str(tmp_path)]) == 2
    assert not captured_pipeline
    assert "Cannot continue:" in capsys.readouterr().err
    assert path.read_text(encoding="utf-8") == content


def test_real_cli_records_effective_project_settings_in_run_snapshot(tmp_path):
    save_config({"rcl": "9", "weights_dir": str(tmp_path / "global-weights")})
    project = results_project_dir(tmp_path)
    config = project / "config.toml"
    save_config({"rcl": "0.125", "weights_dir": str(tmp_path / "project-weights"),
                 "sequence_mode": "protein", "structure_selection": "3", "keep_tmp": "true"}, config)
    before = config.read_bytes()
    result = subprocess.run([sys.executable, "-m", "cryoid_pipeline", "run", "--map", "missing.mrc", "--pool", "missing.fa",
                             "--project", str(tmp_path), "--event-stream"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2, result.stdout + result.stderr  # Missing inputs, without running inference.
    event = next(event for line in result.stdout.splitlines()
                 if (event := cli.decode_event(line)) and event["type"] == "result")
    snapshot = json.loads((Path(event["result"]["run_dir"]) / "run_config.json").read_text())
    assert snapshot["configured_rcl"] == 0.125
    assert snapshot["weights_path"] == str(tmp_path / "global-weights")
    assert snapshot["requested_mode"] == "protein"
    assert snapshot["configured_structure_selection"] == "3"
    assert snapshot["keep_tmp"] is True
    assert config.read_bytes() == before
