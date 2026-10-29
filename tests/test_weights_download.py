"""Small deterministic Hub bundles exercise the production resolution path."""
import io
import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout

import pytest

from cryoid_pipeline import weights, cli
from cryoid_pipeline.orchestrator import CryoIDPipeline, PipelineCancelled
from cryoid_pipeline.models import PredictionBundle, QueryBundle, SearchHit, StageResult
from cryoid_pipeline.user_config import load_config, save_config


@pytest.fixture
def hub(monkeypatch, tmp_path, request):
    import huggingface_hub
    from huggingface_hub import constants
    contents = {
        "class_weight_for_training.txt": json.dumps({model: [1, 1] for model in weights.MODELS}).encode(),
        "epoch_num_path.txt": json.dumps({model: 1 for model in weights.MODELS}).encode(),
        **{f"checkpoint/{model}/{model}.epoch1.pth.tar": b"test-checkpoint" * 10 for model in weights.MODELS},
    }
    monkeypatch.setattr(weights, "PUBLISHED_FILES", {name: len(data) for name, data in contents.items()})
    # Keep the fake filesystem below Windows MAX_PATH, like the default Hub
    # cache; pytest's descriptive per-test directory otherwise adds ~40 chars.
    cache = tmp_path / ("long-cache-" * 8) if getattr(request, "param", False) else tmp_path.parent / ("h" + uuid.uuid4().hex[:6])
    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(cache))
    directory = weights._weights_directory(Path(constants.HF_HUB_CACHE) / ("models--" + weights.PUBLISHED_REPO.replace("/", "--"))
                                           / "snapshots" / weights.PUBLISHED_REVISION / "train_val_data")
    state = SimpleNamespace(directory=directory, contents=contents, calls=[], failure=None, on_chunk=lambda: None)

    def download(*, repo_id, revision, filename, tqdm_class, force_download=False):
        assert repo_id == weights.PUBLISHED_REPO and revision == weights.PUBLISHED_REVISION
        name = filename.removeprefix("train_val_data/")
        state.calls.append((name, force_download))
        if state.failure and state.failure in name:
            raise OSError("Network disconnected")
        data = contents[name]
        with tqdm_class(total=len(data), initial=0, unit="B") as bar:
            for _ in data:
                state.on_chunk()
                bar.update(1)
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return str(target)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    return state


def resolve(value="", *, check_cancel=lambda: None):
    logs, events = [], []
    path = weights.resolve_weights(value, log=logs.append, status=events.append, check_cancel=check_cancel)
    return path, logs, events


@pytest.mark.parametrize("configured", ["", "missing-custom-folder"])
def test_download_then_offline_reuse_and_cleared_cache_recovery(hub, tmp_path, configured):
    configured = str(tmp_path / configured) if configured else ""
    path, logs, events = resolve(configured)
    assert path == hub.directory and not weights.weights_problems(path)
    assert len(hub.calls) == 5
    assert events[-1]["weights_path"] == str(path)
    assert events[-1]["automatic"] is True
    assert "source:" in "\n".join(logs)
    byte_events = [e for e in events if e["name"] == "Downloading weights"]
    assert byte_events[0]["completed_bytes"] == 0
    assert byte_events[-1]["completed_bytes"] == byte_events[-1]["total_bytes"]
    assert all(e["remaining_seconds"] is None or e["remaining_seconds"] >= 0 for e in byte_events)
    assert sorted(e["completed_bytes"] for e in byte_events) == [e["completed_bytes"] for e in byte_events]
    hub.calls.clear()
    assert resolve(str(path))[0] == path
    assert resolve("")[0] == path  # cache discovery with no saved setting
    assert not hub.calls
    # Clear only this test's isolated cache, then reuse its now-stale saved path.
    assert hub.directory.is_relative_to(tmp_path.parent)
    shutil.rmtree(hub.directory)
    assert resolve(str(path))[0] == path
    assert len(hub.calls) == 5


def test_custom_weights_are_preserved_and_invalid_custom_folder_is_not_replaced(hub, tmp_path):
    directory = tmp_path / "custom-weights"
    for name, data in hub.contents.items():
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    assert resolve(str(directory))[0] == directory
    assert not hub.calls
    (directory / "epoch_num_path.txt").write_text("bad metadata")
    with pytest.raises(ValueError, match="Invalid custom weights"):
        resolve(str(directory))
    assert not hub.calls


@pytest.mark.parametrize("hub", [True], indirect=True)
def test_long_cache_paths_can_be_downloaded_validated_and_reused(hub):
    path, _, _ = resolve()
    assert path == hub.directory
    assert not weights.weights_problems(path)
    hub.calls.clear()
    assert resolve(str(path))[0] == path
    assert not hub.calls


def test_provenance_accepts_hub_symlinks_outside_snapshot(hub, tmp_path):
    from C_Unet_model_predicting.prediction_provenance import weights_fingerprint

    plain, snapshot, blobs = (tmp_path / name for name in ("plain", "snapshot", "blobs"))
    blobs.mkdir()
    for name, data in hub.contents.items():
        physical, linked = plain / name, snapshot / name
        physical.parent.mkdir(parents=True, exist_ok=True)
        linked.parent.mkdir(parents=True, exist_ok=True)
        physical.write_bytes(data)
        blob = blobs / hashlib.sha256(data).hexdigest()
        blob.write_bytes(data)
        try:
            linked.symlink_to(blob)
        except OSError as exc:
            if getattr(exc, "winerror", None) == 1314:
                pytest.skip("Windows symlinks require Developer Mode or administrator privileges")
            raise
    expected = weights_fingerprint(plain)
    actual = weights_fingerprint(snapshot)
    assert actual["sha256"] == expected["sha256"]
    assert actual["files"] == expected["files"]
    assert all(not Path(item["path"]).is_absolute() for item in actual["files"])
    checkpoint = next(name for name in hub.contents if name.startswith("checkpoint/"))
    (snapshot / checkpoint).resolve().write_bytes(b"changed checkpoint contents")
    assert weights_fingerprint(snapshot)["sha256"] != expected["sha256"]


@pytest.mark.parametrize("damage", ["missing", "truncated", "metadata"])
def test_incomplete_published_cache_is_repaired(hub, damage):
    path, _, _ = resolve()
    hub.calls.clear()
    checkpoint = path / next(name for name in hub.contents if name.startswith("checkpoint/"))
    if damage == "missing":
        checkpoint.unlink()
    elif damage == "truncated":
        checkpoint.write_bytes(b"bad")
    else:
        meta = path / "epoch_num_path.txt"
        meta.write_bytes(b"x" * len(hub.contents[meta.name]))  # same length, invalid JSON
    assert weights.auto_download_needed(path)
    assert resolve(str(path))[0] == path
    assert not weights.weights_problems(path)
    assert len(hub.calls) == (2 if damage == "metadata" else 3)


def test_download_failure_is_logged_and_never_publishes_a_ready_path(hub):
    hub.failure = "checkpoint/"
    logs, events = [], []
    with pytest.raises(RuntimeError, match="Network disconnected"):
        weights.resolve_weights("", log=logs.append, status=events.append, check_cancel=lambda: None)
    assert any("Weights download failed" in line for line in logs)
    assert not any("weights_path" in event for event in events)
    hub.failure = None
    assert resolve()[0] == hub.directory


def test_cancellation_during_transfer_is_not_reported_as_failure(hub):
    cancelled = [False]
    def check():
        if cancelled[0]:
            raise PipelineCancelled("cancelled")
    hub.on_chunk = lambda: cancelled.__setitem__(0, True)
    logs, events = [], []
    with pytest.raises(PipelineCancelled):
        weights.resolve_weights("", log=logs.append, status=events.append, check_cancel=check)
    assert not any("failed" in line for line in logs)
    assert not any("weights_path" in event for event in events)
    cancelled[0] = False
    hub.on_chunk = lambda: None
    assert resolve()[0] == hub.directory


@pytest.fixture
def science(monkeypatch, tmp_path):
    """Keep CLI/config/events real while replacing expensive scientific stages."""
    seen = []
    def prediction(self, density, run_dir, progress, percent, check_cancel, mode):
        assert not weights.weights_problems(self.unet_weights_dir)
        seen.append(self.unet_weights_dir)
        return StageResult("Prediction maps", "ok", "Test inference"), PredictionBundle(
            "test", density, run_dir, None, [])
    def query(self, **kwargs):
        query_path = kwargs["run_dir"] / "query.fasta"
        query_path.write_text(">query\nACDEFGHIK\n")
        return StageResult("Queries", "ok", "Test queries"), QueryBundle(query_path, "test")
    def search(self, **kwargs):
        return StageResult("Search", "ok", "Test search"), SearchHit("query", "candidate", 1, "ACDEFGHIK"), [], None
    monkeypatch.setattr(CryoIDPipeline, "_run_unet_predictions", prediction)
    monkeypatch.setattr(CryoIDPipeline, "_run_protein_query_generation", query)
    monkeypatch.setattr(CryoIDPipeline, "_run_protein_blast", search)
    monkeypatch.setattr(CryoIDPipeline, "_run_struct_fetch", lambda self, **kwargs: (StageResult("Structure", "ok", "Test fitting"), []))
    density, pool = tmp_path / "input.mrc", tmp_path / "pool.fa"
    density.write_bytes(b"test map")
    pool.write_text(">candidate\nACDEFGHIK\n")
    return SimpleNamespace(seen=seen, density=density, pool=pool)


def cli_args(science, tmp_path):
    return ["run", "--map", str(science.density), "--pool", str(science.pool), "--project", str(tmp_path),
            "--event-stream", "--stop-after-identification"]


@pytest.mark.parametrize("saved", ["", "missing"])
def test_cli_downloads_persists_and_recovers_saved_cache(hub, science, tmp_path, capsys, saved):
    config = tmp_path / "selected-settings.toml"
    save_config({"weights_dir": str(tmp_path / saved) if saved else ""}, config)
    args = cli_args(science, tmp_path) + ["--config", str(config)]
    for clear in (False, True):
        if clear:
            assert hub.directory.is_relative_to(tmp_path.parent)
            shutil.rmtree(hub.directory)
        assert cli.main(args) == 0
        assert load_config(config)["weights_dir"] == str(hub.directory)
        output = capsys.readouterr().out
        events = [cli.decode_event(line) for line in output.splitlines()]
        events = [event for event in events if event]
        assert any(e.get("name") == "Downloading weights" for e in events)
        percents = [e["value"] for e in events if e["type"] == "progress"]
        assert percents == sorted(percents) and percents[-1] == 100
        result = next(e["result"] for e in events if e["type"] == "result")
        snapshot = json.loads((Path(result["run_dir"]) / "run_config.json").read_text())
        assert snapshot["weights_path"] == str(hub.directory)
        assert "Weights download:" in (Path(result["run_dir"]) / "pipeline.log").read_text()
    assert science.seen == [hub.directory, hub.directory]


def test_cli_no_config_downloads_without_changing_saved_settings(hub, science, tmp_path, capsys):
    save_config({"weights_dir": "other-custom-weights"})
    before = load_config()
    assert cli.main(cli_args(science, tmp_path) + ["--no-config"]) == 0
    assert load_config() == before
    assert science.seen == [hub.directory]


def test_cli_explicit_automatic_override_does_not_replace_global_custom_path(hub, science, tmp_path):
    save_config({"weights_dir": "my-custom-weights"})
    before = load_config()
    assert cli.main(cli_args(science, tmp_path) + ["--weights-dir", ""]) == 0
    assert load_config() == before


def test_cli_keeps_settings_changed_during_download(hub, science, tmp_path):
    save_config({"weights_dir": ""})
    changed = [False]
    def change_setting():
        if not changed[0]:
            changed[0] = True
            save_config({"weights_dir": str(tmp_path / "new-user-choice")})
    hub.on_chunk = change_setting
    assert cli.main(cli_args(science, tmp_path)) == 0
    assert load_config()["weights_dir"] == str(tmp_path / "new-user-choice")


def test_cli_reports_download_error_in_result_and_pipeline_log(hub, science, tmp_path, capsys):
    hub.failure = "checkpoint/"
    assert cli.main(cli_args(science, tmp_path)) == 1
    events = [cli.decode_event(line) for line in capsys.readouterr().out.splitlines()]
    result = next(e["result"] for e in events if e and e["type"] == "result")
    assert any(s["status"] == "error" and "Could not download" in s["message"] for s in result["stages"])
    assert "Network disconnected" in (Path(result["run_dir"]) / "pipeline.log").read_text()
    assert not load_config()["weights_dir"]
    assert not science.seen


def test_cli_cancellation_during_download_returns_130(hub, science, tmp_path, capsys):
    cancel = tmp_path / "cancel.signal"
    hub.on_chunk = cancel.touch
    assert cli.main(cli_args(science, tmp_path) + ["--cancel-file", str(cancel)]) == 130
    assert "Run status: cancelled" in capsys.readouterr().out
    assert not cancel.exists()
    assert not science.seen


def test_worker_forwards_real_cli_download_progress_and_weights_path(hub, science, tmp_path, monkeypatch):
    from G_GUI.main_gui import PipelineWorker
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert cli.main(cli_args(science, tmp_path) + ["--no-config"]) == 0
    output = buffer.getvalue().splitlines()
    monkeypatch.setattr("G_GUI.main_gui.subprocess.Popen", lambda *args, **kwargs: SimpleNamespace(stdout=output, wait=lambda: 0))
    worker = PipelineWorker(str(science.density), str(science.pool), tmp_path, "", 0, 10, "protein", "", False)
    events, percents, results, logs = [], [], [], []
    worker.status.connect(events.append)
    worker.progress.connect(percents.append)
    worker.finished.connect(results.append)
    worker.log.connect(logs.append)
    worker.run()
    assert any(e.get("name") == "Downloading weights" for e in events)
    assert any(e.get("weights_path") == str(hub.directory) for e in events)
    assert percents == sorted(percents)
    assert results[0].ok
    assert any("download complete" in line for line in logs)


def test_download_status_eta_and_history_exclude_download_time(tmp_path):
    from G_GUI.run_status import RunStatus
    now = [0.0]
    run = RunStatus(lambda: now[0], history_path=tmp_path / "history.json")
    run.start("cpu")
    now[0] = 2
    run.set_stage({"name": "Downloading weights", "completed_bytes": 0, "total_bytes": 100, "remaining_seconds": None})
    assert run.percent == weights.DOWNLOAD_START
    assert "Estimating" in run.eta and run.device == "Network"
    now[0] = 102
    run.set_stage({"name": "Downloading weights", "completed_bytes": 50, "total_bytes": 100, "remaining_seconds": 100})
    assert run.percent == 7 and "50.0%" in run.description
    assert run.remaining_seconds == pytest.approx(100 + sum(run._budgets[1:]))
    now[0] = 202
    run.set_stage({"name": "Weights ready", "device": "cpu"})
    run.set_stage({"name": "U-Net prediction", "device": "cpu"})
    assert run.percent == weights.DOWNLOAD_END
    assert run._durations[0] == pytest.approx(2)
    run.update_progress(weights.pipeline_percent(29, True))
    assert run.percent > weights.DOWNLOAD_END
    run.consume_log("UNet key_atoms: predicted cube 20/20")
    assert run.percent > weights.DOWNLOAD_END
    run.finish("Complete")
    assert run.percent == 100 and run.eta == "00:00:00"


@pytest.mark.parametrize("outcome", ["ok", "error", "skipped"])
def test_gui_download_status_and_populated_path_survive_restart(hub, science, tmp_path, monkeypatch, outcome):
    import sys
    from G_GUI import main_gui as gui
    from G_GUI.diagnostics import Check, Diagnostics, check_weights
    from cryoid_pipeline.models import PipelineResult
    from cryoid_pipeline.user_config import results_project_dir

    app = gui.QtWidgets.QApplication.instance() or gui.QtWidgets.QApplication([])
    project = results_project_dir(tmp_path)
    values = dict(project_dir=str(project), last_density_map=str(science.density), last_candidate_pool=str(science.pool))
    save_config(values)
    save_config(values, project / "config.toml", project=True)
    ready = Check("OK", "Available", True)
    monkeypatch.setattr(gui, "collect_diagnostics", lambda path, device: Diagnostics(ready, ready, check_weights(path), "cpu"))
    monkeypatch.setattr(gui, "check_blast_tools", lambda: ready)
    errors, warnings, workers = [], [], []
    monkeypatch.setattr(sys, "excepthook", lambda kind, error, traceback: errors.append(error))
    monkeypatch.setattr(gui.QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args))

    class Worker(gui.QtCore.QObject):
        log = gui.QtCore.pyqtSignal(str)
        progress = gui.QtCore.pyqtSignal(int)
        status = gui.QtCore.pyqtSignal(object)
        finished = gui.QtCore.pyqtSignal(object)
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.weights_path = args[3]
            self._cancel_requested = False
            workers.append(self)
        def run(self):
            pass  # Deliver the real worker's event contract below.
        def cancel(self):
            self._cancel_requested = True
    monkeypatch.setattr(gui, "PipelineWorker", Worker)
    expected_path = [""]

    def inspect(dialog):
        assert isinstance(dialog, gui.ConfigurationsWindow)
        assert dialog.weights_path() == expected_path[0]
        assert dialog.weights_path_edit.placeholderText() == "Default: Autodownload published weights."
        labels = [label.text() for label in dialog.findChildren(gui.QtWidgets.QLabel)]
        assert "Custom weights path" in labels
        return 0
    monkeypatch.setattr(gui, "_exec_dialog", inspect)
    original_windows = set(app.topLevelWidgets())
    restarted = [False]

    def execute(app):
        window = next(w for w in app.topLevelWidgets() if isinstance(w, gui.CryoIDMainWindow) and w not in original_windows)
        run = window.findChild(gui.QtWidgets.QPushButton, "run_button")
        bar = window.findChild(gui.QtWidgets.QProgressBar, "progress_bar")
        label = window.findChild(gui.QtWidgets.QLabel, "stage_label")
        eta = window.findChild(gui.QtWidgets.QLabel, "eta_label")
        logs = window.findChild(gui.QtWidgets.QPlainTextEdit, "log_edit")
        configure = next(a for a in window.findChildren(gui.QtGui.QAction) if a.text() == "Configurations")
        def settle(condition=lambda: True):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                app.processEvents()
                if not errors and window.isVisible() and not run.property("checking") and condition():
                    return
                if errors:
                    raise errors[0]
                time.sleep(.001)
            pytest.fail("GUI download test did not settle")
        try:
            settle()
            configure.trigger()
            if restarted[0]:
                return 0
            assert not run.property("blocked")  # Empty weights must allow Run.
            run.click()
            settle(lambda: bool(workers))
            worker = workers[-1]
            worker.status.emit({"name": "Downloading weights", "device": "Network", "completed_bytes": 0,
                                "total_bytes": 100, "remaining_seconds": None})
            settle(lambda: "Downloading weights" in label.text())
            assert "Estimating" in eta.text() and bar.value() == 2
            worker.status.emit({"name": "Downloading weights", "device": "Network", "completed_bytes": 50,
                                "total_bytes": 100, "remaining_seconds": 60})
            settle(lambda: bar.value() == 7)
            assert "50.0%" in label.text() and eta.text().startswith("~")
            if outcome == "ok":
                path, download_logs, events = resolve()
                for line in download_logs:
                    worker.log.emit(line)
                worker.status.emit(events[-1])
                expected_path[0] = str(path)
                settle(lambda: load_config()["weights_dir"] == str(path))
                configure.trigger()
                worker.status.emit({"name": "U-Net prediction", "device": "cpu"})
            elif outcome == "error":
                worker.log.emit("Weights download failed: simulated network error")
            else:
                run.click()
                assert not run.isEnabled()
            worker.finished.emit(PipelineResult(project / "runs" / "test", [StageResult("Run", outcome, "Test outcome")]))
            settle(lambda: run.text() == "Run" and not any(t.isRunning() for t in window.findChildren(gui.QtCore.QThread)))
            assert label.text() == {"ok": "Complete", "error": "Failed", "skipped": "Cancelled"}[outcome]
            assert bar.value() == {"ok": 100, "error": 7, "skipped": 0}[outcome]
            assert eta.text() == "00:00:00"
            if outcome == "error":
                assert "network error" in logs.toPlainText()
            assert load_config()["weights_dir"] == expected_path[0]
            assert not warnings and not errors
        finally:
            settle(lambda: not any(t.isRunning() for t in window.findChildren(gui.QtCore.QThread)))
            window.close()
            window.deleteLater()
            app.sendPostedEvents(None, gui.QtCore.QEvent.Type.DeferredDelete)
        return 0

    monkeypatch.setattr(gui.QtWidgets.QApplication, "exec", execute)
    for is_restart in (False, True):
        restarted[0] = is_restart
        with pytest.raises(SystemExit) as exited:
            gui.main()
        assert exited.value.code == 0
