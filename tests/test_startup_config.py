from pathlib import Path
import json

import pytest

from cryoid_pipeline import cli, startup_config
from cryoid_pipeline.startup_config import find_weights_dir as discover_weights
from cryoid_pipeline.user_config import (
    DEFAULT_CONFIG, STARTUP_KEYS, default_config_path, ensure_startup_config,
    load_config, load_effective_config, results_project_dir, save_config,
)
from G_GUI.local_config import gui_config_view, save_project_config


def test_first_startup_detects_and_persists_installation_paths_once(tmp_path, monkeypatch):
    weights, executable = tmp_path / "weights", tmp_path / "ChimeraX.exe"
    calls = []
    monkeypatch.setattr(startup_config, "find_weights_dir", lambda: calls.append("weights") or weights)
    monkeypatch.setattr(startup_config, "find_chimerax", lambda: calls.append("chimerax") or executable)
    saved = ensure_startup_config()
    assert (saved["weights_dir"], saved["chimerax_path"]) == (str(weights), str(executable))
    text = default_config_path().read_text()
    assert "[startup]" in text
    assert text.count("weights_dir =") == text.count("chimerax_path =") == 1
    assert ensure_startup_config() == saved
    assert calls == ["weights", "chimerax"]


@pytest.mark.parametrize("source", ["global", "last-project"])
def test_existing_paths_migrate_to_global_startup_section(tmp_path, monkeypatch, source):
    project = results_project_dir(tmp_path / "selected")
    config = default_config_path()
    config.parent.mkdir(parents=True)
    values = 'weights_dir = "relative-weights"\n[gui]\nchimerax_path = "relative-ChimeraX"\n'
    if source == "global":
        config.write_text('[defaults]\nrcl = "0.5"\n' + values)
        origin = config.parent
    else:
        config.write_text('[defaults]\nproject_dir = ' + json.dumps(str(project)) + '\n')
        project.mkdir(parents=True)
        (project / "config.toml").write_text('[defaults]\n' + values)
        origin = project
    monkeypatch.setattr(startup_config, "find_weights_dir", lambda: pytest.fail("Saved weights were replaced by detection"))
    monkeypatch.setattr(startup_config, "find_chimerax", lambda: pytest.fail("Saved ChimeraX was replaced by detection"))
    saved = ensure_startup_config()
    assert saved["weights_dir"] == str(origin / "relative-weights")
    assert saved["chimerax_path"] == str(origin / "relative-ChimeraX")
    assert "[startup]" in config.read_text()
    assert load_config()["weights_dir"] == saved["weights_dir"]
    if source == "global":
        assert load_config()["rcl"] == "0.5"


def test_saved_empty_paths_stay_empty_after_restart(tmp_path, monkeypatch):
    save_config({})
    monkeypatch.setattr(startup_config, "find_weights_dir", lambda: pytest.fail("Explicit blank was replaced"))
    monkeypatch.setattr(startup_config, "find_chimerax", lambda: pytest.fail("Explicit blank was replaced"))
    assert all(ensure_startup_config()[key] == "" for key in STARTUP_KEYS)


def test_project_paths_cannot_override_or_be_written_over_startup_paths(tmp_path):
    global_values = dict(DEFAULT_CONFIG, weights_dir=str(tmp_path / "shared-weights"),
                         chimerax_path=str(tmp_path / "shared-ChimeraX"))
    save_config(global_values)
    project = results_project_dir(tmp_path)
    save_config(dict(global_values, rcl="0.75", weights_dir="obsolete-project-weights", chimerax_path="obsolete-project-ChimeraX"), project / "config.toml")
    loaded = load_effective_config(project).values
    assert all(loaded[key] == global_values[key] for key in STARTUP_KEYS)
    save_project_config(gui_config_view(loaded), project, baseline=loaded, initialize=True)
    text = (project / "config.toml").read_text()
    assert "[startup]" not in text
    assert all(key not in text for key in STARTUP_KEYS)
    save_project_config(dict(gui_config_view(loaded), rcl="1.25"), project, baseline=loaded)
    text = (project / "config.toml").read_text()
    assert "[startup]" not in text
    assert all(key not in text for key in STARTUP_KEYS)
    assert all(load_config()[key] == global_values[key] for key in STARTUP_KEYS)


@pytest.mark.parametrize("key", ["weights-dir", "chimerax-path"])
def test_cli_installation_paths_are_global_and_not_project_settings(tmp_path, key):
    assert cli.main(["config", "set", key, str(tmp_path / "shared")]) == 0
    before = default_config_path().read_bytes()
    assert cli.main(["config", "--project", str(tmp_path), "set", key, "project-value"]) == 2
    assert cli.main(["config", "--project", str(tmp_path), "unset", key]) == 2
    assert default_config_path().read_bytes() == before
    assert not (results_project_dir(tmp_path) / "config.toml").exists()


def test_weights_discovery_checks_environment_and_standard_installation(tmp_path, monkeypatch):
    import json
    models = ("secondary_strctures", "key_atoms", "residue_types_6letter")
    home = tmp_path / "home"
    weights = home / "cryoid_weights" / "train_val_data"
    weights.mkdir(parents=True)
    (weights / "epoch_num_path.txt").write_text(json.dumps({model: 1 for model in models}))
    (weights / "class_weight_for_training.txt").write_text(json.dumps({model: [1, 1] for model in models}))
    for model in models:
        path = weights / "checkpoint" / model / f"{model}.epoch1.pth.tar"
        path.parent.mkdir(parents=True)
        path.touch()
    monkeypatch.setattr(startup_config.Path, "home", lambda: home)
    assert discover_weights() == weights
    monkeypatch.setenv("CRYOID_UNET_WEIGHTS_DIR", str(weights))
    assert discover_weights() == weights
    next((weights / "checkpoint").glob("*/*.pth.tar")).unlink()
    assert discover_weights() is None
