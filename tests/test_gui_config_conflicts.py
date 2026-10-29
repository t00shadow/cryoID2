"""Exercise live project selection, settings dialogs, and conflicting CLI edits."""
import sys
import time
from pathlib import Path

import pytest

from G_GUI import main_gui as gui
from G_GUI.diagnostics import Check, Diagnostics
from G_GUI.qt_compat import QtWidgets
from cryoid_pipeline import cli
from cryoid_pipeline.user_config import load_config, results_project_dir, save_config


@pytest.mark.parametrize("scenario", ["switch", "merge", "keep", "reload", "reload-partial", "cancel", "run-cancel", "run-reload", "paths", "unselected", "open-failure"])
def test_gui_preserves_external_edits_and_uses_only_selected_project(monkeypatch, tmp_path, scenario):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    existing = set(app.topLevelWidgets())
    errors, warnings, conflicts, seen = [], [], [], []
    monkeypatch.setattr(sys, "excepthook", lambda kind, error, tb: errors.append(error))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    project = results_project_dir(tmp_path / "A")
    other = results_project_dir(tmp_path / "B")
    density, pool = tmp_path / "map.mrc", tmp_path / "pool.fa"
    density.touch()
    pool.write_text(">candidate\nACDEFGHIK\n")
    values = dict(project_dir=str(project), rcl="0.25", weights_dir=str(tmp_path / "weights"),
                  last_density_map=str(density), last_candidate_pool=str(pool))
    save_config(dict(values, project_dir="") if scenario == "reload-partial" else values, project / "config.toml", project=True)
    save_config(dict(values, project_dir="") if scenario == "unselected" else values)
    save_config({"project_dir": str(other), "rcl": "2"}, other / "config.toml", project=True)
    monkeypatch.setenv("CRYOID_PROJECT_DIR", str(other))
    other_before = (other / "config.toml").read_bytes()
    ready = Check("OK", "Available", True)
    monkeypatch.setattr(gui, "collect_diagnostics", lambda *args: Diagnostics(ready, ready, ready, "cpu"))
    monkeypatch.setattr(gui, "check_weights", lambda *args: ready)
    monkeypatch.setattr(gui, "check_blast_tools", lambda *args: ready)
    monkeypatch.setattr(gui, "PipelineWorker", lambda *args, **kwargs: pytest.fail("Unresolved conflict started a run"))
    selected = [str(other.parent)]
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *args: selected[0])
    editing = [True]
    opened = []
    monkeypatch.setattr(gui.QtGui.QDesktopServices, "openUrl", lambda url: opened.append(Path(url.toLocalFile()).resolve()) or True)

    def inspect(dialog):
        if isinstance(dialog, gui.SetupIssuesDialog):
            assert scenario == "unselected"
            assert "File > New or Open" in dialog.details.toPlainText()
            seen.append("project required")
            return 0
        if isinstance(dialog, gui.ConfigConflictDialog):
            conflicts.append(dialog)
            dialog.choose("reload" if scenario == "reload-partial" else scenario.removeprefix("run-"))
            return gui._dialog_accepted_code()
        if isinstance(dialog, gui.ConfigurationsWindow):
            assert not hasattr(dialog, "results_directory_edit")
            assert not hasattr(dialog, "browse_results_button")
            assert not hasattr(dialog, "results_directory")
            return 0
        assert isinstance(dialog, gui.OptionsWindow)
        seen.append(dialog.rcl())
        if editing[0]:
            if scenario == "merge":
                dialog.cpu_threads_edit.setText("2")
            else:
                dialog.rcl_edit.setText("0.75")
            return gui._dialog_accepted_code()
        return 0
    monkeypatch.setattr(gui, "_exec_dialog", inspect)

    def execute(app):
        window = next(w for w in app.topLevelWidgets() if isinstance(w, gui.CryoIDMainWindow) and w not in existing)
        run = window.findChild(QtWidgets.QPushButton, "run_button")
        options = window.findChild(QtWidgets.QPushButton, "options_button")
        density_edit = window.findChild(QtWidgets.QLineEdit, "density_map_edit")
        actions = {a.text(): a for a in window.findChildren(gui.QtGui.QAction)}
        def settle():
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                app.processEvents()
                if window.isVisible() and not run.property("checking") and not any(t.isRunning() for t in window.findChildren(gui.QtCore.QThread)):
                    assert not errors, errors
                    return
                time.sleep(.001)
            pytest.fail("GUI checks did not finish")
        try:
            settle()
            actions["Configurations"].trigger()
            if scenario == "unselected":
                density_edit.setText(str(density))
                window.findChild(QtWidgets.QLineEdit, "candidate_pool_edit").setText(str(pool))
                run.click()
                assert seen == ["project required"]
                assert not (results_project_dir(tmp_path) / "config.toml").exists()
                assert load_config()["project_dir"] == ""
                actions["Results folder"].trigger()
                assert warnings and "File > New or Open" in warnings[-1]
                assert not opened
                return 0
            if scenario == "paths":
                monkeypatch.chdir(tmp_path)
                density_edit.setText("map.mrc")
                density_edit.editingFinished.emit()
                assert density_edit.text() == str(density)
                assert load_config(project / "config.toml")["last_density_map"] == str(density)
                actions["Open"].trigger()
                selected[0] = str(project.parent)
                actions["Open"].trigger()
                assert density_edit.text() == str(density)
                return 0
            assert cli.main(["config", "--project", str(project), "set", "rcl", "1.5"]) == 0
            before = (project / "config.toml").read_bytes()
            if scenario == "switch":
                actions["Open"].trigger()
                assert (project / "config.toml").read_bytes() == before
                selected[0] = str(project.parent)
                actions["Open"].trigger()
            elif scenario == "open-failure":
                save_project = gui.save_project_config
                def fail_incoming(values, selected_project, **kwargs):
                    if selected_project == other:
                        raise PermissionError("Read-only destination")
                    return save_project(values, selected_project, **kwargs)
                with monkeypatch.context() as patch:
                    patch.setattr(gui, "save_project_config", fail_incoming)
                    actions["Open"].trigger()
                assert len(warnings) == 1 and "config.toml" in warnings[0]
                warnings.clear()
            elif scenario.startswith("run-"):
                run.click()
            else:
                options.click()
            settle()
            saved = load_config(project / "config.toml")
            assert saved["rcl"] == ("0.75" if scenario == "keep" else "1.5")
            if scenario == "merge":
                assert saved["cpu_threads"] == "2"
            elif scenario != "keep":
                assert (project / "config.toml").read_bytes() == before
            assert bool(conflicts) == (scenario not in {"switch", "merge", "open-failure"})
            editing[0] = False
            options.click()
            expected = 0.25 if scenario in {"cancel", "run-cancel"} else float(saved["rcl"])
            assert seen[-1] == expected
            actions["Results folder"].trigger()
            assert opened[-1] == project
            assert (other / "config.toml").read_bytes() == other_before
            assert not warnings and not errors
        finally:
            settle()
            window.close()
            window.deleteLater()
            app.sendPostedEvents(None, gui.QtCore.QEvent.Type.DeferredDelete)
        return 0
    monkeypatch.setattr(QtWidgets.QApplication, "exec", execute)
    with pytest.raises(SystemExit) as exited:
        gui.main()
    assert exited.value.code == 0
