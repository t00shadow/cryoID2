"""Short project workflows and compatibility with the GUI's real CLI worker."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from cryoid_pipeline import cli
from cryoid_pipeline.models import PipelineResult, StageResult
from cryoid_pipeline.user_config import load_config, results_project_dir, save_config


@pytest.fixture
def recorded_runs(monkeypatch, tmp_path):
    records = []

    class Pipeline:
        def __init__(self, **settings):
            self.settings = settings

        def run(self, density, pool, **kwargs):
            records.append((self.settings, density, pool))
            return PipelineResult(tmp_path, [StageResult("Run", "ok", "Test pipeline")])

    monkeypatch.setattr("cryoid_pipeline.orchestrator.CryoIDPipeline", Pipeline)
    return records


def test_init_run_and_switch_projects_without_retyping_saved_inputs(tmp_path, recorded_runs):
    save_config({"weights_dir": str(tmp_path / "default weights"), "rcl": "9",
                 "last_density_map": "unrelated.mrc", "last_candidate_pool": "unrelated.fa"})
    original = {}
    for index, name in enumerate(("new project", "old project")):
        parent = tmp_path / name
        density, pool = tmp_path / f"map {index}.mrc", tmp_path / f"pool {index}.fa"
        assert cli.main(["init", str(parent), "--map", str(density), "--pool", str(pool),
                         "--rcl", str(0.25 + index), "--mode", "protein"]) == 0
        config = results_project_dir(parent) / "config.toml"
        saved = load_config(config)
        assert saved["weights_dir"] == ""
        assert saved["last_density_map"] == str(density)
        assert saved["last_candidate_pool"] == str(pool)
        original[config] = config.read_bytes()
    for index in (0, 1, 0):
        parent = tmp_path / ("new project" if index == 0 else "old project")
        assert cli.main(["run", "--project", str(parent)]) == 0
        settings, density, pool = recorded_runs[-1]
        assert density == tmp_path / f"map {index}.mrc"
        assert pool == tmp_path / f"pool {index}.fa"
        assert settings["unet_rcl"] == 0.25 + index
    assert all(path.read_bytes() == contents for path, contents in original.items())
    assert load_config()["rcl"] == "9"


@pytest.mark.parametrize("target_after_action", [False, True])
def test_config_project_selector_edits_settings_and_inputs(tmp_path, recorded_runs, capsys, target_after_action):
    parent = tmp_path / "project"
    save_config({"weights_dir": str(tmp_path / "weights"), "rcl": "0.75",
                 "last_density_map": "unrelated.mrc", "last_candidate_pool": "unrelated.fa"})

    def command(*action):
        target = ["--project", str(parent)]
        args = ["config", *action, *target] if target_after_action else ["config", *target, *action]
        return cli.main(args)

    assert command("path") == 0
    config = results_project_dir(parent) / "config.toml"
    assert str(config) in capsys.readouterr().out
    assert not config.exists()
    assert command("set", "rcl", "0.5") == 0
    assert load_config(config)["last_density_map"] == load_config(config)["last_candidate_pool"] == ""
    assert command("set", "map", str(tmp_path / "saved.mrc")) == 0
    assert command("set", "pool", str(tmp_path / "saved.fa")) == 0
    saved = load_config(config)
    assert saved["weights_dir"] == ""
    assert saved["rcl"] == "0.5"
    assert command("show") == 0
    shown = capsys.readouterr().out
    assert "rcl = 0.5" in shown and "map = " in shown and "pool = " in shown
    assert cli.main(["config", "--project", str(parent)]) == 0
    assert "rcl = 0.5" in capsys.readouterr().out
    assert cli.main(["run", "--project", str(parent)]) == 0
    assert recorded_runs[-1][0]["unet_rcl"] == 0.5
    assert command("unset", "rcl") == 0
    assert load_config(config)["rcl"] == ""


def test_run_input_overrides_are_temporary(tmp_path, recorded_runs):
    assert cli.main(["init", str(tmp_path), "--map", "saved.mrc", "--pool", "saved.fa", "--rcl", "0.25"]) == 0
    config = results_project_dir(tmp_path) / "config.toml"
    before = config.read_bytes()
    assert cli.main(["run", "--project", str(tmp_path), "--map", "different.mrc", "--rcl", "0.5"]) == 0
    assert recorded_runs[-1][1] == Path("different.mrc")
    assert recorded_runs[-1][2] == Path("saved.fa").resolve()
    assert recorded_runs[-1][0]["unet_rcl"] == 0.5
    assert cli.main(["run", "--map", "old-map.mrc", "--pool", "old-pool.fa", "--project", str(tmp_path)]) == 0
    assert recorded_runs[-1][1:] == (Path("old-map.mrc"), Path("old-pool.fa"))
    assert config.read_bytes() == before


@pytest.mark.parametrize("arguments", [
    ["run", "map.mrc", "pool.fa"],
    ["run", "--project-dir", "project", "--map", "map.mrc", "--pool", "pool.fa"],
    ["config", "--project-dir", "project"],
    ["config", "set", "project-dir", "project"],
])
def test_retired_cli_forms_are_rejected(arguments, tmp_path):
    config = tmp_path / "user-settings" / "config.toml"
    save_config({"rcl": "0.25"}, config)
    before = config.read_bytes()
    with pytest.raises(SystemExit) as exited:
        cli.main(arguments)
    assert exited.value.code == 2
    assert config.read_bytes() == before


def test_relative_gui_inputs_survive_the_child_process_working_directory(tmp_path, monkeypatch):
    from G_GUI import main_gui as gui
    monkeypatch.setattr(gui, "REPO_ROOT", Path(__file__).resolve().parents[1])
    working = tmp_path / "gui working directory"
    working.mkdir()
    monkeypatch.chdir(working)
    project = results_project_dir(tmp_path / "output")
    worker = gui.PipelineWorker("missing.mrc", "missing.fa", project, "", 0, 10,
                                "auto", "", False, device="cpu")
    results = []
    worker.finished.connect(results.append)
    worker.run()
    assert len(results) == 1
    snapshot = json.loads((results[0].run_dir / "run_config.json").read_text())
    assert snapshot["density_map"] == str(working / "missing.mrc")
    assert snapshot["candidate_pool"] == str(working / "missing.fa")
    saved = load_config(project / "config.toml")
    assert saved["last_density_map"] == snapshot["density_map"]
    assert saved["last_candidate_pool"] == snapshot["candidate_pool"]


def test_init_preserves_existing_projects_and_rejects_invalid_settings(tmp_path, capsys):
    parent = tmp_path / "existing"
    assert cli.main(["init", str(parent)]) == 0
    config = results_project_dir(parent) / "config.toml"
    saved = load_config(config)
    assert saved["last_density_map"] == saved["last_candidate_pool"] == ""
    before = config.read_bytes()
    assert cli.main(["init", str(parent), "--rcl", "0.5"]) == 2
    assert "already" in capsys.readouterr().err.lower()
    assert config.read_bytes() == before
    invalid = tmp_path / "invalid"
    assert cli.main(["init", str(invalid), "--rcl", "-1"]) == 2
    assert not (results_project_dir(invalid) / "config.toml").exists()


def test_missing_saved_inputs_produce_an_actionable_error(tmp_path, recorded_runs, capsys):
    assert cli.main(["init", str(tmp_path)]) == 0
    assert cli.main(["run", "--project", str(tmp_path)]) == 2
    assert "--map" in capsys.readouterr().err
    assert not recorded_runs


def test_first_run_in_current_directory_creates_reusable_config(tmp_path, recorded_runs, monkeypatch):
    project = tmp_path / "new_project"
    project.mkdir()
    monkeypatch.chdir(project)
    save_config({"project_dir": str(tmp_path / "global-project"), "rcl": "9"})
    monkeypatch.setenv("CRYOID_PROJECT_DIR", str(tmp_path / "environment-project"))
    assert cli.main(["run", "--map", "map.mrc", "--pool", "pool.fa", "--weights-dir", "weights",
                     "--rcl", "0.25", "--mode", "protein", "--cpu-threads", "auto", "--keep-tmp"]) == 0
    config = results_project_dir(project) / "config.toml"
    saved = load_config(config)
    assert saved["project_dir"] == str(results_project_dir(project))
    assert saved["last_density_map"] == str(project / "map.mrc")
    assert saved["last_candidate_pool"] == str(project / "pool.fa")
    assert saved["weights_dir"] == ""
    assert recorded_runs[-1][0]["unet_weights_dir"] == project / "weights"
    assert saved["rcl"] == "0.25"
    assert saved["sequence_mode"] == "protein" and saved["keep_tmp"] == "true"
    assert saved["cpu_threads"] == "auto"
    assert not results_project_dir(tmp_path / "global-project").exists()
    assert not results_project_dir(tmp_path / "environment-project").exists()
    before = config.read_bytes()
    assert cli.main(["run"]) == 0
    assert recorded_runs[-1][1:] == (project / "map.mrc", project / "pool.fa")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["run", "--project", "new_project", "--rcl", "0.5"]) == 0
    assert recorded_runs[-1][1:] == (project / "map.mrc", project / "pool.fa")
    assert recorded_runs[-1][0]["unet_rcl"] == 0.5
    assert cli.main(["run", "--project", "new_project"]) == 0
    assert recorded_runs[-1][0]["unet_rcl"] == 0.25
    assert config.read_bytes() == before
    assert load_config()["rcl"] == "9"


@pytest.mark.parametrize("invalid", ["missing-input", "invalid-setting"])
def test_incomplete_first_run_does_not_initialize_a_project(tmp_path, recorded_runs, invalid):
    args = ["run", "--project", str(tmp_path), "--map", "map.mrc"]
    if invalid == "invalid-setting":
        args += ["--pool", "pool.fa", "--rcl", "-1"]
    assert cli.main(args) == 2
    assert not (results_project_dir(tmp_path) / "config.toml").exists()
    assert not recorded_runs


def test_real_first_run_creates_config_before_pipeline_and_can_run_again(tmp_path):
    project = tmp_path / "new_project"
    project.mkdir()
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]))
    first = subprocess.run([sys.executable, "-m", "cryoid_pipeline", "run", "--map", "missing.mrc",
                            "--pool", "missing.fa", "--weights-dir", "weights", "--rcl", "0.5", "--event-stream"],
                           cwd=project, env=env, capture_output=True, text=True, timeout=30)
    assert first.returncode == 2, first.stdout + first.stderr  # The pipeline reports missing files.
    config = results_project_dir(project) / "config.toml"
    assert config.is_file()
    saved = load_config(config)
    assert saved["last_density_map"] == str(project / "missing.mrc")
    assert saved["last_candidate_pool"] == str(project / "missing.fa")
    assert saved["rcl"] == "0.5"
    before = config.read_bytes()
    second = subprocess.run([sys.executable, "-m", "cryoid_pipeline", "run", "--project", "new_project", "--event-stream"],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert second.returncode == 2, second.stdout + second.stderr
    event = next(event for line in second.stdout.splitlines()
                 if (event := cli.decode_event(line)) and event["type"] == "result")
    snapshot = json.loads((Path(event["result"]["run_dir"]) / "run_config.json").read_text())
    assert snapshot["density_map"] == saved["last_density_map"]
    assert snapshot["candidate_pool"] == saved["last_candidate_pool"]
    assert snapshot["configured_rcl"] == 0.5
    assert config.read_bytes() == before


def test_saved_relative_inputs_resolve_beside_project_config(tmp_path, recorded_runs):
    project = results_project_dir(tmp_path)
    save_config({"last_density_map": "inputs/map.mrc", "last_candidate_pool": "inputs/pool.fa"}, project / "config.toml")
    assert cli.main(["run", "--project", str(tmp_path)]) == 0
    assert recorded_runs[-1][1:] == (project / "inputs/map.mrc", project / "inputs/pool.fa")


def test_short_run_with_no_project_config_does_not_restore_old_inputs(tmp_path, recorded_runs):
    run = results_project_dir(tmp_path) / "runs" / "run_20261001_120000"
    run.mkdir(parents=True)
    (run / "run_config.json").write_text(json.dumps({
        "density_map": str(tmp_path / "legacy.mrc"), "candidate_pool": str(tmp_path / "legacy.fa"),
        "configured_rcl": 0.5,
    }), encoding="utf-8")
    assert cli.main(["run", "--project", str(tmp_path)]) == 2
    assert not recorded_runs
    assert not (results_project_dir(tmp_path) / "config.toml").exists()


def test_init_and_project_config_edits_ignore_environment_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("CRYOID_RCL", "invalid")
    assert cli.main(["init", str(tmp_path), "--rcl", "0.25"]) == 0
    path = results_project_dir(tmp_path) / "config.toml"
    assert load_config(path)["rcl"] == "0.25"
    assert cli.main(["config", "set", "rcl", "0.5", "--project", str(tmp_path)]) == 0
    assert load_config(path)["rcl"] == "0.5"


def test_conflicting_selectors_and_input_arguments_are_rejected(tmp_path):
    config = tmp_path / "config.toml"
    with pytest.raises(SystemExit) as exited:
        cli.main(["config", "--project", str(tmp_path), "--config", str(config), "set", "rcl", "0.5"])
    assert exited.value.code == 2
    with pytest.raises(SystemExit) as exited:
        cli.main(["run", "--map", "map.mrc", "--pool", "pool.fa", "--project", str(tmp_path), "--no-config", "--config", str(config)])
    assert exited.value.code == 2
    with pytest.raises(SystemExit) as exited:
        cli.main(["config", "--project", str(tmp_path), "set", "rcl", "0.5", "--config", str(config)])
    assert exited.value.code == 2
    with pytest.raises(SystemExit) as exited:
        cli.main(["run", "positional.mrc", "pool.fa", "--map", "named.mrc", "--project", str(tmp_path)])
    assert exited.value.code == 2
    assert not config.exists()


@pytest.mark.parametrize("cancelled", [False, True])
@pytest.mark.parametrize("selection", ["", "3"])
@pytest.mark.parametrize("stored", ["conflicting", "malformed"])
def test_gui_worker_still_runs_real_cli_events_and_cancellation(tmp_path, monkeypatch, cancelled, selection, stored):
    from G_GUI import main_gui as gui

    monkeypatch.setattr(gui, "REPO_ROOT", Path(__file__).resolve().parents[1])
    project = results_project_dir(tmp_path)
    config = project / "config.toml"
    save_config({"rcl": "9", "sequence_mode": "rna", "structure_selection": "9",
                 "weights_dir": "unwanted-weights", "chimerax_path": "unwanted-ChimeraX"}, config)
    if stored == "malformed":
        config.write_text("broken = [", encoding="utf-8")
    for key in ("CRYOID_RCL", "CRYOID_UNET_DEVICE", "CRYOID_CPU_THREADS", "CRYOID_KEEP_TMP"):
        monkeypatch.setenv(key, "invalid")
    monkeypatch.setenv("CRYOID_UNET_WEIGHTS_DIR", "unwanted-environment-weights")
    monkeypatch.setenv("CRYOID_STRUCTURE_SELECTION", "9")
    monkeypatch.setenv("CRYOID_CHIMERAX_PATH", "unwanted-environment-ChimeraX")
    before = config.read_bytes()
    density, pool = tmp_path / "gui.mrc", tmp_path / "gui.fa"
    if cancelled:
        density.write_bytes(b"input validation only")
        pool.write_text(">candidate\nACDEFGHIK\n")
    worker = gui.PipelineWorker(str(density), str(pool), project, "", 0.25, 15, "protein", selection, False,
                                device="cpu", cpu_threads="auto", chimerax_path=str(tmp_path / "GUI-ChimeraX"))
    results, progress, statuses, logs = [], [], [], []
    worker.finished.connect(results.append)
    worker.progress.connect(progress.append)
    worker.status.connect(statuses.append)
    worker.log.connect(logs.append)
    if cancelled:
        worker.cancel()
    worker.run()
    assert len(results) == 1
    result = results[0]
    assert any(stage.status == ("skipped" if cancelled else "blocked_missing_input") for stage in result.stages)
    assert progress and statuses
    command = next(log for log in logs if log.startswith("Command:"))
    assert "--project" in command and "--event-stream" in command and "--cancel-file" in command
    assert "--no-config" in command
    snapshot = json.loads((result.run_dir / "run_config.json").read_text())
    assert snapshot["configured_rcl"] == 0.25
    assert snapshot["requested_mode"] == "protein"
    assert snapshot["density_map"] == str(density)
    assert snapshot["candidate_pool"] == str(pool)
    assert snapshot["configured_structure_selection"] == selection
    assert snapshot["weights_path"] == ""
    assert snapshot["chimerax_path"] == str(tmp_path / "GUI-ChimeraX")
    assert snapshot["unet_device"] == "cpu"
    assert snapshot["cpu_threads"] == "auto"
    assert snapshot["keep_tmp"] is False
    assert not worker._cancel_file.exists()
    assert config.read_bytes() == before


@pytest.mark.parametrize("stored", ["conflicting", "malformed"])
def test_no_config_uses_builtin_defaults_without_reading_saved_values(tmp_path, monkeypatch, recorded_runs, stored):
    from cryoid_pipeline.user_config import default_config_path
    global_path = default_config_path()
    global_path.parent.mkdir(parents=True)
    global_path.write_text("broken global config = [", encoding="utf-8")
    project_path = results_project_dir(tmp_path) / "config.toml"
    save_config({"rcl": "9", "keep_tmp": "true", "structure_selection": "9", "sequence_mode": "rna"}, project_path)
    if stored == "malformed":
        project_path.write_text("broken project config = [", encoding="utf-8")
    before = project_path.read_bytes()
    for key in ("CRYOID_RCL", "CRYOID_CPU_THREADS", "CRYOID_KEEP_TMP", "CRYOID_STRUCTURE_SELECTION", "CRYOID_UNET_DEVICE"):
        monkeypatch.setenv(key, "invalid")
    assert cli.main(["run", "--project", str(tmp_path), "--map", "map.mrc", "--pool", "pool.fa", "--no-config"]) == 0
    values = recorded_runs[-1][0]
    assert values["unet_rcl"] == 0 and values["protein_min_trace_length"] == 10
    assert values["unet_device"] == values["cpu_threads"] == "auto"
    assert values["sequence_mode"] == "auto" and values["structure_selection"] is None
    assert values["unet_weights_dir"] == values["chimerax_path"] == ""
    assert values["keep_tmp"] is False
    assert project_path.read_bytes() == before


def test_explicit_auto_selection_and_empty_paths_override_saved_values(tmp_path, monkeypatch, recorded_runs):
    path = results_project_dir(tmp_path) / "config.toml"
    save_config({"structure_selection": "9", "weights_dir": "saved-weights", "chimerax_path": "saved-ChimeraX"}, path)
    monkeypatch.setenv("CRYOID_STRUCTURE_SELECTION", "7")
    monkeypatch.setenv("CRYOID_UNET_WEIGHTS_DIR", "environment-weights")
    monkeypatch.setenv("CRYOID_CHIMERAX_PATH", "environment-ChimeraX")
    assert cli.main(["run", "--map", "map.mrc", "--pool", "pool.fa", "--project", str(tmp_path), "--structure-selection", "auto",
                     "--weights-dir", "", "--chimerax-path", ""]) == 0
    values = recorded_runs[-1][0]
    assert values["structure_selection"] is None
    assert values["unet_weights_dir"] == values["chimerax_path"] == ""
