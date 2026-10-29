from __future__ import annotations

import json
import os
import pytest
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from G_GUI import diagnostics
from G_GUI.gui_skin import CryoIDUI
from G_GUI.main_gui import PipelineWorker, apply_theme
from G_GUI.qt_compat import QtWidgets
from G_GUI.run_status import RunStatus


def test_weights_check_rejects_missing_and_empty_checkpoints(tmp_path):
    assert diagnostics.check_weights("").ok
    assert diagnostics.check_weights(str(tmp_path / "missing-cache")).ok
    assert not diagnostics.check_weights(str(tmp_path)).ok
    from C_Unet_model_predicting.unet_inference import MODEL_TYPES
    (tmp_path / "class_weight_for_training.txt").write_text(json.dumps({m: [1, 1] for m in MODEL_TYPES}))
    (tmp_path / "epoch_num_path.txt").write_text(json.dumps({m: 1 for m in MODEL_TYPES}))
    assert not diagnostics.check_weights(str(tmp_path)).ok
    for model in MODEL_TYPES:
        checkpoint = tmp_path / "checkpoint" / model / f"{model}.epoch1.pth.tar"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
    assert diagnostics.check_weights(str(tmp_path)).ok
    checkpoint.write_bytes(b"")
    result = diagnostics.check_weights(str(tmp_path))
    assert not result.ok
    assert "checkpoint is empty" in result.detail


def test_diagnostics_distinguishes_cpu_selection_from_available_cuda(monkeypatch, tmp_path):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "Test GPU")
    monkeypatch.setattr(diagnostics, "check_blast_tools", lambda *args, **kwargs: diagnostics.Check("OK", str(tmp_path), True))
    monkeypatch.setattr(diagnostics, "blast_executable", lambda name, path: name)
    monkeypatch.setattr(diagnostics.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout="blast 2.16", stderr=""))
    report = diagnostics.collect_diagnostics("", "cpu")
    assert report.compute.summary == "CPU"
    assert report.device == "cpu"
    assert report.blast.ok
    assert report.weights.ok
    assert "CUDA: 0: Test GPU" in report.lines
    assert any("PyTorch:" in line for line in report.lines)
    assert any("Operating system:" in line for line in report.lines)
    report = diagnostics.collect_diagnostics("", "cuda:0")
    assert report.compute.ok
    assert report.compute.summary == "GPU / CUDA"
    assert report.device == "cuda:0"
    report = diagnostics.collect_diagnostics("", "cuda:2")
    assert not report.compute.ok
    assert "unavailable" in report.compute.detail


def test_broken_blast_executable_is_reported(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnostics, "check_blast_tools", lambda *args, **kwargs: diagnostics.Check("OK", str(tmp_path), True))
    monkeypatch.setattr(diagnostics, "blast_executable", lambda name, path: name)
    def fail(*args, **kwargs):
        raise OSError("invalid executable")
    monkeypatch.setattr(diagnostics.subprocess, "run", fail)
    report = diagnostics.collect_diagnostics("", "cpu")
    assert not report.blast.ok
    assert "invalid executable" in report.blast.technical_detail
    assert all(f"{name} could not run" in report.blast.detail for name in ("blastp", "blastn", "makeblastdb"))
    assert "OSError" not in report.blast.detail


def test_stage_description_and_elapsed_reset_after_finish():
    now = [0.0]
    status = RunStatus(clock=lambda: now[0])
    status.start("cuda:0")
    status.set_stage({"name": "U-Net prediction", "device": "cuda:0"})
    now[0] = 12.0
    status.consume_log("UNet key_atoms: predicted cube 6/20")
    assert status.progress == "key_atoms: 6/20 cubes (30%)"
    assert status.description == "U-Net prediction \u00b7 key atoms: 6/20 cubes (30%)"
    assert status.eta.startswith("~")
    status.set_stage({"name": "BLAST+ sequence search", "device": "cpu"})
    assert status.device == "cpu"
    assert status.eta.startswith("~")
    status.finish("Cancelled")
    elapsed = status.elapsed
    now[0] += 100
    assert status.elapsed == elapsed
    assert status.eta == "00:00:00"
    assert status.stage == "Cancelled"
    status.start("cpu")
    assert status.elapsed == "00:00:00"
    assert status.progress == "Working\u2026"


def test_logs_expand_and_clipboard_matches_visible_output():
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = QtWidgets.QMainWindow()
    ui = CryoIDUI()
    ui.setup(window)
    try:
        apply_theme(app, window, ui, "light")
        window.show()
        app.processEvents()
        original_height, original_width = ui.log_edit.height(), ui.log_edit.width()
        window.resize(window.width() + 200, window.height() + 200)
        app.processEvents()
        assert ui.log_edit.height() >= original_height + 190
        assert ui.log_edit.width() >= original_width + 190
        ui.log_edit.setPlainText("CryoID diagnostics\nCPU: test\nError: example\nUnicode: Î±Î²Î³")
        ui.copy_logs_button.click()
        assert app.clipboard().text() == ui.log_edit.toPlainText()
        expanded_height = window.height()
        ui.log_section.setChecked(False)
        app.processEvents()
        assert not ui.log_edit.isVisible()
        assert not ui.copy_logs_button.isVisible()
        ui.log_section.setChecked(True)
        app.processEvents()
        assert window.height() == expanded_height
        assert ui.log_edit.isVisible()
    finally:
        window.close()


def test_worker_forwards_stage_events_and_selected_device(monkeypatch, tmp_path):
    from cryoid_pipeline.cli import encode_event
    from cryoid_pipeline.models import PipelineResult, StageResult, pipeline_result_to_dict
    result = PipelineResult(tmp_path, stages=[StageResult("Run", "ok", "Done")])
    output = [encode_event("status", name="U-Net prediction", device="cuda:0"),
              "Checkpoint load error details", encode_event("result", result=pipeline_result_to_dict(result))]
    command = []
    environment = {}
    def popen(arguments, **kwargs):
        command.extend(arguments)
        environment.update(kwargs["env"])
        return SimpleNamespace(stdout=output, wait=lambda: 0)
    monkeypatch.setenv("CRYOID_STRUCTURE_SELECTION", "invalid-inherited-value")
    monkeypatch.setattr("G_GUI.main_gui.subprocess.Popen", popen)
    worker = PipelineWorker("map.mrc", "pool.fa", tmp_path, "", 0, 10, "auto", "", False,
                            device="cuda:0", cpu_threads="4")
    events, logs, results = [], [], []
    worker.status.connect(events.append)
    worker.log.connect(logs.append)
    worker.finished.connect(results.append)
    worker.run()
    assert events[0]["name"] == "U-Net prediction"
    assert events[0]["device"] == "cuda:0"
    assert command[command.index("--device") + 1] == "cuda:0"
    assert command[command.index("--cpu-threads") + 1] == "4"
    assert command[command.index("--structure-selection") + 1] == "auto"
    assert command[command.index("--weights-dir") + 1] == ""
    assert command[command.index("--chimerax-path") + 1] == ""
    assert "--no-config" in command
    assert "--repo-root" not in command
    assert "--no-keep-tmp" in command
    assert "CRYOID_STRUCTURE_SELECTION" not in environment
    assert "Checkpoint load error details" in logs
    assert len(results) == 1


def test_gui_can_report_missing_scientific_dependencies_instead_of_failing_at_import():
    import subprocess
    import sys
    script = '''
import importlib.abc
import sys
class MissingLibraries(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'mrcfile'}:
            raise ImportError('simulated missing dependency: ' + fullname)
sys.meta_path.insert(0, MissingLibraries())
import G_GUI.main_gui
from G_GUI.diagnostics import collect_diagnostics
report = collect_diagnostics('', None)
assert not report.compute.ok
assert any('mrcfile: ERROR' in line for line in report.lines)
assert any('PyTorch/device check: ERROR' in line for line in report.lines)
print('Missing libraries reported')
'''
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Missing libraries reported' in result.stdout


def test_blocked_run_reveals_issues_and_check_setup_restores_availability(monkeypatch, tmp_path):
    import time
    import pytest
    from G_GUI import main_gui as gui
    from cryoid_pipeline.user_config import DEFAULT_CONFIG
    from G_GUI.diagnostics import Check, Diagnostics

    density, pool = tmp_path / "input.mrc", tmp_path / "pool.fasta"
    density.write_bytes(b"input")
    pool.write_text(">candidate\nMAGST\n")
    config = dict(DEFAULT_CONFIG, project_dir=str(tmp_path),
                  last_density_map=str(density), last_candidate_pool=str(pool))
    from cryoid_pipeline.user_config import save_config, results_project_dir
    save_config(config, results_project_dir(tmp_path) / "config.toml")
    save_config(config)
    ready = Check("OK", "Available", True)
    report = [Diagnostics(ready, ready, Check("Missing", "Choose the U-Net weights folder.", False), "cpu",
                          dependency_issues=["mrcfile: Install this missing Python dependency."])]
    monkeypatch.setattr(gui, "collect_diagnostics", lambda *args: report[0])
    monkeypatch.setattr(gui, "PipelineWorker", lambda *args, **kwargs: pytest.fail("Blocked Run started the CLI"))
    dialogs = []
    summaries = []
    action = ["close"]
    def inspect_dialog(dialog):
        assert isinstance(dialog, gui.SetupIssuesDialog)
        assert "Recheck" not in [button.text() for button in dialog.findChildren(QtWidgets.QPushButton)]
        deadline = time.monotonic() + 5
        while "checks are still running" in dialog.details.toPlainText() and time.monotonic() < deadline:
            QtWidgets.QApplication.instance().processEvents()
            time.sleep(.01)
        assert "checks are still running" not in dialog.details.toPlainText()
        dialogs.append(dialog.details.toPlainText())
        summaries.append(dialog.summary.text())
        dialog.action = action[0]
        return 0
    monkeypatch.setattr(gui, "_exec_dialog", inspect_dialog)

    def execute(app):
        window = next(w for w in app.topLevelWidgets() if isinstance(w, gui.CryoIDMainWindow))
        assert not window.isVisible()
        assert not any(w.isVisible() for w in app.topLevelWidgets())
        button = window.findChild(QtWidgets.QPushButton, "run_button")
        log = window.findChild(QtWidgets.QPlainTextEdit, "log_edit")
        def settle():
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                app.processEvents()
                if not button.property("checking") and not any(t.isRunning() for t in window.findChildren(gui.QtCore.QThread)):
                    return
                time.sleep(.005)
            pytest.fail("Checks did not finish")
        try:
            settle()
            assert window.isVisible()
            assert button.isEnabled()  # Remains clickable to explain why execution is blocked.
            assert button.property("blocked")
            assert not button.icon().isNull()
            assert log.toPlainText() == "CryoID2 version 1.0.0"
            assert not any(box.title() == "Environment" for box in window.findChildren(QtWidgets.QGroupBox))
            assert window.findChild(QtWidgets.QLabel, "stage_progress_label") is None
            button.click()
            assert "U-Net weights" in dialogs[-1]
            assert "mrcfile" in dialogs[-1]
            run_message = dialogs[-1]
            check_setup = next(a for a in window.findChildren(gui.QtGui.QAction) if a.text() == "Check setup")
            check_setup.trigger()
            assert dialogs[-1] == run_message
            assert log.toPlainText() == "CryoID2 version 1.0.0"
            report[0] = Diagnostics(ready, ready, ready, "cpu")
            check_setup.trigger()
            settle()
            assert not button.property("blocked")
            assert button.icon().isNull()
            assert log.toPlainText() == "CryoID2 version 1.0.0"
            action[0] = "close"
            check_setup.trigger()
            assert summaries[-1].startswith("Good")
            window.findChild(QtWidgets.QLineEdit, "density_map_edit").clear()
            assert button.property("blocked")
            action[0] = "close"
            button.click()
            assert "Density map" in dialogs[-1] and "select an input file" in dialogs[-1]
        finally:
            window.close()
        return 0
    monkeypatch.setattr(QtWidgets.QApplication, "exec", execute)
    with pytest.raises(SystemExit) as exited:
        gui.main()
    assert exited.value.code == 0


def make_valid_config_paths(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    models = ("secondary_strctures", "key_atoms", "residue_types_6letter")
    (weights / "class_weight_for_training.txt").write_text(json.dumps({name: [1, 1] for name in models}))
    (weights / "epoch_num_path.txt").write_text(json.dumps({name: 1 for name in models}))
    for name in models:
        checkpoint = weights / "checkpoint" / name / f"{name}.epoch1.pth.tar"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
    blast = tmp_path / "bin"
    blast.mkdir()
    for name in ("blastp", "blastn", "makeblastdb"):
        executable = blast / (name + (".exe" if os.name == "nt" else ""))
        executable.write_bytes(b"binary")
        executable.chmod(0o755)
    return weights, blast


def test_gui_opens_with_invalid_saved_options_and_requires_review(monkeypatch, tmp_path):
    import time
    from G_GUI import main_gui as gui
    from cryoid_pipeline.user_config import save_config, load_config

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    existing_windows = set(app.topLevelWidgets())
    for name in ("CRYOID_RCL", "CRYOID_KEEP_TMP", "CRYOID_STRUCTURE_SELECTION", "CRYOID_PROTEIN_MIN_TRACE_LENGTH"):
        monkeypatch.delenv(name, raising=False)
    config = tmp_path / "config.toml"
    density, pool = tmp_path / "input.mrc", tmp_path / "pool.fa"
    density.write_bytes(b"input")
    pool.write_text(">candidate\nMAGST\n")
    save_config({"project_dir": str(tmp_path / "project"), "weights_dir": str(tmp_path),
                 "last_density_map": str(density), "last_candidate_pool": str(pool),
                 "rcl": "bad", "sequence_mode": "bad", "protein_min_trace_length": "1.5",
                 "structure_selection": "2.5", "keep_tmp": "bad"}, config)
    from cryoid_pipeline.user_config import results_project_dir
    save_config(load_config(config), results_project_dir(tmp_path / "project") / "config.toml")
    before = config.read_bytes()
    monkeypatch.setenv("CRYOID_CONFIG", str(config))
    ready = diagnostics.Check("OK", "Available", True)
    monkeypatch.setattr(gui, "collect_diagnostics", lambda *args: diagnostics.Diagnostics(ready, ready, ready, "cpu"))
    monkeypatch.setattr(gui, "PipelineWorker", lambda *args, **kwargs: pytest.fail("Invalid options started a run"))
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path / "new-project"))
    dialogs, save_options = [], [False]

    def inspect(dialog):
        if isinstance(dialog, gui.SetupIssuesDialog):
            dialogs.append(dialog.details.toPlainText())
            dialog.action = "close"
            return 0
        assert isinstance(dialog, gui.OptionsWindow)
        assert dialog.sequence_mode() == "auto" and dialog.rcl() == 0
        if not save_options[0]:
            return 0
        dialog.sequence_mode_combo.setCurrentIndex(dialog.sequence_mode_combo.findData("protein"))
        dialog.rcl_edit.setText("0.05")
        dialog.device_combo.setCurrentIndex(dialog.device_combo.findData("cpu"))
        dialog.cpu_threads_edit.setText("4")
        dialog.accept()
        return gui._dialog_accepted_code()

    monkeypatch.setattr(gui, "_exec_dialog", inspect)

    def execute(app):
        window = next(w for w in app.topLevelWidgets()
                      if isinstance(w, gui.CryoIDMainWindow) and w not in existing_windows)
        run = window.findChild(QtWidgets.QPushButton, "run_button")
        options = window.findChild(QtWidgets.QPushButton, "options_button")
        log = window.findChild(QtWidgets.QPlainTextEdit, "log_edit")

        def settle():
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                app.processEvents()
                if window.isVisible() and not run.property("checking"):
                    return
                time.sleep(.001)
            pytest.fail("Startup checks did not settle")

        try:
            settle()
            assert config.read_bytes() == before
            assert run.property("blocked")
            run.click()
            assert all(label in dialogs[-1] for label in (
                "RCL", "Sequence mode", "Protein trace length", "Structure candidates", "Temporary files"))
            assert "Traceback" not in dialogs[-1]
            assert log.toPlainText() == "CryoID2 version 1.0.0"
            options.click()  # Cancelling review must leave the file and block intact.
            assert config.read_bytes() == before and run.property("blocked")
            next(a for a in window.findChildren(gui.QtGui.QAction) if a.text() == "New").trigger()
            assert load_config(results_project_dir(tmp_path / "new-project") / "config.toml")["sequence_mode"] == "auto"
            assert float(load_config(results_project_dir(tmp_path / "new-project") / "config.toml")["rcl"]) == 0
            window.findChild(QtWidgets.QLineEdit, "density_map_edit").setText(str(density))
            window.findChild(QtWidgets.QLineEdit, "candidate_pool_edit").setText(str(pool))
            save_options[0] = True
            options.click()
            settle()
            assert not run.property("blocked")
            assert load_config(results_project_dir(tmp_path / "new-project") / "config.toml")["sequence_mode"] == "protein"
            assert load_config(results_project_dir(tmp_path / "new-project") / "config.toml")["rcl"] == "0.05"
            assert load_config(results_project_dir(tmp_path / "new-project") / "config.toml")["device"] == "cpu"
            assert load_config(results_project_dir(tmp_path / "new-project") / "config.toml")["cpu_threads"] == "4"
        finally:
            window.close()
            window.deleteLater()
            app.sendPostedEvents(None, gui.QtCore.QEvent.Type.DeferredDelete)
        return 0

    monkeypatch.setattr(QtWidgets.QApplication, "exec", execute)
    with pytest.raises(SystemExit) as exited:
        gui.main()
    assert exited.value.code == 0


@pytest.mark.parametrize("outcome", ["ok", "skipped", "error"])
def test_whole_pipeline_timing_and_progress_reach_gui(monkeypatch, tmp_path, outcome):
    import time
    from G_GUI import main_gui as gui
    from cryoid_pipeline.user_config import DEFAULT_CONFIG
    from cryoid_pipeline.models import PipelineResult, StageResult

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    existing_windows = set(app.topLevelWidgets())

    density, pool = tmp_path / "input.mrc", tmp_path / "pool.fa"
    density.write_bytes(b"map")
    pool.write_text(">candidate\nMAGST\n")
    config = dict(DEFAULT_CONFIG, project_dir=str(tmp_path / "project"), weights_dir=str(tmp_path),
                  weights_path=str(tmp_path), last_density_map=str(density), last_candidate_pool=str(pool))
    from cryoid_pipeline.user_config import save_config, results_project_dir
    save_config(config, results_project_dir(tmp_path / "project") / "config.toml")
    save_config(config)
    monkeypatch.setattr(gui, "default_config_path", lambda: tmp_path / "config.toml")
    ready = diagnostics.Check("OK", "Available", True)
    monkeypatch.setattr(gui, "collect_diagnostics", lambda *args: diagnostics.Diagnostics(ready, ready, ready, "cpu"))
    monkeypatch.setattr(gui, "check_weights", lambda path: ready)
    monkeypatch.setattr(gui, "check_blast_tools", lambda: ready)
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *args: str(tmp_path / "new-project"))
    now = [0.0]
    monkeypatch.setattr(gui, "RunStatus", lambda **kwargs: RunStatus(clock=lambda: now[0], **kwargs))
    workers = []

    class Worker(gui.QtCore.QObject):
        log = gui.QtCore.pyqtSignal(str)
        progress = gui.QtCore.pyqtSignal(int)
        status = gui.QtCore.pyqtSignal(object)
        finished = gui.QtCore.pyqtSignal(object)

        def __init__(self, *args, **kwargs):
            super().__init__()
            self._cancel_requested = False
            workers.append(self)

        def run(self):
            pass  # The test delivers the real worker signals below.

        def cancel(self):
            self._cancel_requested = True

    monkeypatch.setattr(gui, "PipelineWorker", Worker)

    def execute(app):
        window = next(w for w in app.topLevelWidgets()
                      if isinstance(w, gui.CryoIDMainWindow) and w not in existing_windows)
        button = window.findChild(QtWidgets.QPushButton, "run_button")
        bar = window.findChild(QtWidgets.QProgressBar, "progress_bar")
        eta = window.findChild(QtWidgets.QLabel, "eta_label")
        elapsed = window.findChild(QtWidgets.QLabel, "elapsed_label")

        def settle_until(condition):
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                app.processEvents()
                if condition():
                    return
                time.sleep(0.001)
            pytest.fail("GUI did not settle")

        try:
            settle_until(lambda: window.isVisible() and not button.property("checking"))
            assert eta.text() == "00:00:00"
            button.click()
            settle_until(lambda: bool(workers))
            worker = workers[-1]
            assert eta.text().startswith("~")
            assert button.text() == "Abort"
            now[0] = 1
            worker.status.emit({"name": "U-Net prediction", "device": "cpu"})
            worker.log.emit("UNet secondary_strctures: predicted cube 1/4")
            settle_until(lambda: bar.value() == 12)
            assert eta.text().startswith("~")
            for stage, progress in (("Backbone tracing / query generation", 55),
                                    ("BLAST+ sequence search", 57),
                                    ("Structure fetching / fitting", 80)):
                now[0] += 1
                worker.status.emit({"name": stage, "device": "cpu"})
                worker.progress.emit(progress)
                settle_until(lambda: bar.value() == progress)
                assert eta.text().startswith("~")
            if outcome == "skipped":
                button.click()
                assert button.text() == "Abort" and not button.isEnabled()
                worker.progress.emit(98)
                app.processEvents()
                assert bar.value() == 80
                assert eta.text().startswith("~")
            elif outcome == "ok":
                now[0] += 1
                worker.status.emit({"name": "Finalizing outputs", "device": "cpu"})
                worker.progress.emit(98)
                worker.progress.emit(100)
                settle_until(lambda: bar.value() == 98)
                assert eta.text().startswith("~")
            worker.finished.emit(PipelineResult(tmp_path / "run", stages=[StageResult("Run", outcome, "Done")]))
            settle_until(lambda: button.text() == "Run" and not any(
                thread.isRunning() for thread in window.findChildren(gui.QtCore.QThread)))
            assert eta.text() == "00:00:00"
            assert bar.value() == {"ok": 100, "skipped": 0, "error": 80}[outcome]
            new_action = next(action for action in window.findChildren(gui.QtGui.QAction) if action.text() == "New")
            new_action.trigger()
            settle_until(lambda: not button.property("checking") and not any(
                thread.isRunning() for thread in window.findChildren(gui.QtCore.QThread)))
            assert bar.value() == 0
            assert eta.text() == elapsed.text() == "00:00:00"
        finally:
            window.close()
            window.deleteLater()
            app.sendPostedEvents(None, gui.QtCore.QEvent.Type.DeferredDelete)
        return 0

    monkeypatch.setattr(QtWidgets.QApplication, "exec", execute)
    with pytest.raises(SystemExit) as exited:
        gui.main()
    assert exited.value.code == 0


@pytest.mark.parametrize("field_name", ["weights_path_edit"])
def test_save_allows_cleared_weights_for_autodownload(monkeypatch, tmp_path, field_name):
    from G_GUI import settings_dialogs
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    weights, blast = make_valid_config_paths(tmp_path)
    dialog = settings_dialogs.ConfigurationsWindow(weights_path=str(weights))
    messages, accepted = [], []
    monkeypatch.setattr(settings_dialogs, "show_validation_issues", lambda issues, parent: messages.append(issues))
    dialog.accepted.connect(lambda: accepted.append(True))
    field = getattr(dialog, field_name)
    original = field.text()
    field.clear()
    dialog.accept()
    assert accepted == [True]
    assert not messages
    assert field.text() == ""
    field.setText(original)
    dialog.accept()
    assert accepted == [True, True]
    dialog.close()


def test_configuration_save_reports_all_errors_together(monkeypatch, tmp_path):
    from G_GUI import settings_dialogs
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    messages = []
    monkeypatch.setattr(settings_dialogs, "show_validation_issues", lambda issues, parent: messages.append(issues))
    dialog = settings_dialogs.ConfigurationsWindow(weights_path=str(tmp_path))
    dialog.accept()
    assert len(messages) == 1
    text = "\n".join(messages[0])
    assert all(name in text for name in ("U-Net weights",))
    assert "Error:" not in text
    dialog.close()


def test_weight_check_lists_all_missing_files_without_python_errors(tmp_path):
    result = diagnostics.check_weights(str(tmp_path))
    assert "class_weight_for_training.txt" in result.detail
    assert "epoch_num_path.txt" in result.detail
    assert "FileNotFoundError" not in result.detail
    models = ("secondary_strctures", "key_atoms", "residue_types_6letter")
    (tmp_path / "class_weight_for_training.txt").write_text(json.dumps({name: [1] for name in models}))
    (tmp_path / "epoch_num_path.txt").write_text(json.dumps({name: 1 for name in models}))
    result = diagnostics.check_weights(str(tmp_path))
    assert all(f"Missing {name} checkpoint" in result.detail for name in ("secondary structures", "key atoms", "residue types"))


def test_missing_bundle_has_repair_message(monkeypatch, tmp_path):
    monkeypatch.setattr("D_BLAST_searching.blast_tools.BUNDLE_ROOT", tmp_path)
    blast = diagnostics.check_blast_tools()
    assert not blast.ok
    assert "Reinstall CryoID" in blast.detail
    assert "Configurations" not in blast.detail
