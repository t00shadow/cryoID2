"""Config persistence must survive external edits and interrupted writes."""
import subprocess
import sys

import pytest

from cryoid_pipeline import user_config as config
from G_GUI.local_config import gui_config_view, save_project_config


def test_relative_paths_resolve_at_the_config_that_contains_them(tmp_path):
    global_path = tmp_path / "user" / "config.toml"
    config.save_config({"weights_dir": "weights", "chimerax_path": "ChimeraX"}, global_path)
    project = config.results_project_dir(tmp_path / "project")
    project.mkdir(parents=True)
    (project / "config.toml").write_text('[gui]\nlast_density_map = "map.mrc"\nlast_candidate_pool = "pool.fa"\n')
    saved = config.load_effective_config(project, config_path=global_path).values
    assert saved["last_density_map"] == str(project / "map.mrc")
    assert saved["last_candidate_pool"] == str(project / "pool.fa")
    assert saved["weights_dir"] == str(global_path.parent / "weights")
    assert saved["chimerax_path"] == str(global_path.parent / "ChimeraX")


def test_gui_merges_only_edited_fields_and_preserves_external_cli_edits(tmp_path):
    project = config.results_project_dir(tmp_path)
    path = project / "config.toml"
    config.save_config({"project_dir": str(project), "rcl": "0.25"}, path, project=True)
    baseline = config.load_config(path)
    config.save_config_changes({"rcl": "1.5"}, path, project=True)
    before = path.read_bytes()
    saved = save_project_config(gui_config_view(baseline), project, baseline=baseline)
    assert saved["rcl"] == "1.5" and path.read_bytes() == before
    edited = dict(gui_config_view(baseline), cpu_threads="2")
    saved = save_project_config(edited, project, baseline=baseline)
    assert saved["rcl"] == "1.5" and saved["cpu_threads"] == "2"


def test_only_an_explicit_save_recreates_a_deleted_config(tmp_path):
    project = config.results_project_dir(tmp_path)
    path = project / "config.toml"
    config.save_config({"project_dir": str(project), "rcl": "0.25"}, path, project=True)
    baseline = config.load_config(path)
    path.unlink()
    save_project_config(gui_config_view(baseline), project, baseline=baseline)
    assert not path.exists()
    save_project_config(gui_config_view(baseline), project, baseline=baseline, initialize=True)
    assert config.load_config(path)["rcl"] == "0.25"


@pytest.mark.parametrize("snapshot", [False, True])
def test_same_field_edits_and_run_snapshots_require_conflict_resolution(tmp_path, snapshot):
    project = config.results_project_dir(tmp_path)
    path = project / "config.toml"
    config.save_config({"project_dir": str(project), "rcl": "0.25"}, path, project=True)
    baseline = config.load_config(path)
    config.save_config_changes({"rcl": "1.5", "cpu_threads": "3"}, path, project=True)
    edited = gui_config_view(baseline)
    if not snapshot:
        edited["rcl"] = "0.75"
    before = path.read_bytes()
    with pytest.raises(config.ConfigConflictError) as error:
        save_project_config(edited, project, baseline=baseline, snapshot=snapshot)
    assert set(error.value.keys) == ({"rcl", "cpu_threads"} if snapshot else {"rcl"})
    assert path.read_bytes() == before
    saved = save_project_config(edited, project, baseline=baseline, force=True, snapshot=snapshot)
    assert saved["rcl"] == edited["rcl"]
    assert saved["cpu_threads"] == ("auto" if snapshot else "3")


def test_failed_atomic_replace_leaves_existing_config_and_no_partial_temp_files(tmp_path, monkeypatch):
    path = tmp_path / "config.toml"
    config.save_config({"rcl": "0.25"}, path)
    before = path.read_bytes()
    def fail(*args):
        raise PermissionError("Simulated interrupted replace")
    monkeypatch.setattr(config.os, "replace", fail)
    with pytest.raises(PermissionError):
        config.save_config_changes({"rcl": "1.5"}, path, project=True)
    assert path.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_independent_process_writers_merge_updates(tmp_path):
    path = tmp_path / "config.toml"
    config.save_config({}, path)
    script = "from cryoid_pipeline.user_config import update_config; import sys; update_config(sys.argv[1], **{sys.argv[2]:sys.argv[3]})"
    writers = [subprocess.Popen([sys.executable, "-c", script, str(path), key, value],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
               for key, value in (("rcl", "0.5"), ("cpu_threads", "2"), ("sequence_mode", "rna"),
                                  ("keep_tmp", "true"))]
    for writer in writers:
        output, error = writer.communicate(timeout=30)
        assert writer.returncode == 0, output + error
    saved = config.load_config(path)
    assert (saved["rcl"], saved["cpu_threads"], saved["sequence_mode"], saved["keep_tmp"]) == ("0.5", "2", "rna", "true")
