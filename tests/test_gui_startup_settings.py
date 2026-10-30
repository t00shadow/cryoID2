"""Installation settings persist independently of project and run settings."""
import sys
import time

import pytest

from G_GUI import main_gui as gui
from G_GUI.diagnostics import Check, Diagnostics
from cryoid_pipeline import cli
from cryoid_pipeline.models import PipelineResult, StageResult
from cryoid_pipeline.user_config import load_config, results_project_dir, save_config


def test_gui_global_paths_survive_selection_restart_and_job_completion(monkeypatch, tmp_path):
    app = gui.QtWidgets.QApplication.instance() or gui.QtWidgets.QApplication([])
    errors, warnings, workers = [], [], []
    monkeypatch.setattr(sys, "excepthook", lambda kind, error, tb: errors.append(error))
    monkeypatch.setattr(gui.QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    ready = Check("OK", "Available", True)
    monkeypatch.setattr(gui, "collect_diagnostics", lambda *args: Diagnostics(ready, ready, ready, "cpu"))
    monkeypatch.setattr(gui, "check_weights", lambda *args: ready)
    monkeypatch.setattr(gui, "check_blast_tools", lambda *args: ready)
    paths = {letter: dict(weights_dir=str(tmp_path / f"weights-{letter}"),
                          chimerax_path=str(tmp_path / f"ChimeraX-{letter}")) for letter in "ABC"}
    expected, editing, conflicts = dict(paths["A"]), {}, []
    conflict_action = ["cancel"]
    project_a, project_b = [results_project_dir(tmp_path / letter) for letter in "ab"]
    density, pool = tmp_path / "input.mrc", tmp_path / "pool.fa"
    density.touch()
    pool.write_text(">candidate\nACDEFGHIK\n")
    save_config(paths["A"])
    save_config(dict(rcl="1.5", last_density_map=str(density), last_candidate_pool=str(pool),
                     weights_dir="obsolete-weights", chimerax_path="obsolete-ChimeraX"), project_b / "config.toml")
    selection = [str(tmp_path)]
    monkeypatch.setattr(gui.QtWidgets.QFileDialog, "getExistingDirectory", lambda *args: selection[0])

    class Worker(gui.QtCore.QObject):
        log = gui.QtCore.pyqtSignal(str)
        progress = gui.QtCore.pyqtSignal(int)
        status = gui.QtCore.pyqtSignal(object)
        finished = gui.QtCore.pyqtSignal(object)
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.args, self.kwargs = args, kwargs
            self._cancel_requested = False
            workers.append(self)
        def run(self):
            pass
        def cancel(self):
            self._cancel_requested = True
    monkeypatch.setattr(gui, "PipelineWorker", Worker)

    def inspect(dialog):
        if isinstance(dialog, gui.NewProjectDialog):
            dialog.name_edit.setText("a")
            dialog.location_edit.setText(str(tmp_path))
            return gui._dialog_accepted_code()
        if isinstance(dialog, gui.ConfigConflictDialog):
            conflicts.append(dialog)
            dialog.choose(conflict_action[0])
            return gui._dialog_accepted_code()
        if isinstance(dialog, gui.ConfigurationsWindow):
            assert dialog.weights_path() == expected["weights_dir"]
            assert dialog.chimerax_path() == expected["chimerax_path"]
            if editing:
                dialog.weights_path_edit.setText(editing["weights_dir"])
                dialog.chimerax_path_edit.setText(editing["chimerax_path"])
                return gui._dialog_accepted_code()
            return 0
        assert isinstance(dialog, gui.OptionsWindow)
        if editing.get("rcl"):
            dialog.rcl_edit.setText(editing["rcl"])
            return gui._dialog_accepted_code()
        return 0
    monkeypatch.setattr(gui, "_exec_dialog", inspect)

    def launch(restarted=False):
        existing = set(app.topLevelWidgets())
        def execute(app):
            window = next(w for w in app.topLevelWidgets() if isinstance(w, gui.CryoIDMainWindow) and w not in existing)
            run = window.findChild(gui.QtWidgets.QPushButton, "run_button")
            options = window.findChild(gui.QtWidgets.QPushButton, "options_button")
            actions = {a.text(): a for a in window.findChildren(gui.QtGui.QAction)}
            def settle():
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    app.processEvents()
                    if window.isVisible() and not run.property("checking") and not any(t.isRunning() for t in window.findChildren(gui.DiagnosticsWorker)):
                        assert not errors, errors
                        return
                    time.sleep(.001)
                pytest.fail("GUI startup settings did not settle")
            def configure(values):
                editing.update(values)
                actions["Configurations"].trigger()
                expected.update(values)
                editing.clear()
                settle()
                assert all(load_config()[key] == expected[key] for key in expected)
            try:
                settle()
                actions["Configurations"].trigger()
                if restarted:
                    assert expected == paths["C"]
                    assert window.findChild(gui.QtWidgets.QLineEdit, "density_map_edit").text() == str(density)
                    return 0
                # Global Save must work even before a project has been selected.
                configure(paths["B"])
                assert not (results_project_dir(tmp_path) / "config.toml").exists()
                actions["New"].trigger()
                settle()
                assert window.findChild(gui.QtWidgets.QLineEdit, "density_map_edit").text() == ""
                editing["rcl"] = "0.25"
                options.click()
                editing.clear()
                selection[0] = str(project_b.parent)
                actions["Open"].trigger()
                settle()
                assert load_config(project_a / "config.toml")["rcl"] == "0.25"
                assert load_config(project_b / "config.toml")["rcl"] == "1.5"
                actions["Configurations"].trigger()
                assert all(key not in (project_b / "config.toml").read_text() for key in paths["B"])
                # An external startup edit conflicts before Run; Cancel starts no worker.
                assert cli.main(["config", "set", "weights-dir", paths["C"]["weights_dir"]]) == 0
                run.click()
                assert len(conflicts) == 1 and not workers
                conflict_action[0] = "keep"
                run.click()
                assert len(conflicts) == 2 and len(workers) == 1
                worker = workers[-1]
                assert worker.args[3] == paths["B"]["weights_dir"]
                assert worker.kwargs["chimerax_path"] == paths["B"]["chimerax_path"]
                # Saving shared paths during a run cannot be undone by completion.
                configure(paths["C"])
                worker.finished.emit(PipelineResult(project_b / "runs" / "probe", [StageResult("Run", "ok", "Done")]))
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    app.processEvents()
                    if run.text() == "Run" and not any(t.isRunning() for t in window.findChildren(gui.QtCore.QThread)):
                        break
                    time.sleep(.001)
                assert run.text() == "Run"
                assert all(load_config()[key] == expected[key] for key in expected)
                assert all(key not in (project_b / "config.toml").read_text() for key in expected)
                assert not warnings and not errors
            finally:
                settle()
                window.close()
                window.deleteLater()
                app.sendPostedEvents(None, gui.QtCore.QEvent.Type.DeferredDelete)
            return 0
        monkeypatch.setattr(gui.QtWidgets.QApplication, "exec", execute)
        with pytest.raises(SystemExit) as exited:
            gui.main()
        assert exited.value.code == 0
    launch()
    launch(restarted=True)
