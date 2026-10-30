"""Exercise File > New / Open through their actual Qt actions."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

from G_GUI import main_gui as gui
from G_GUI.diagnostics import Check, Diagnostics
from cryoid_pipeline.user_config import DEFAULT_CONFIG, load_config, results_project_dir, save_config


@pytest.mark.parametrize("scenario", [
    "new", "new_nested", "existing_project", "existing_nonproject", "open_empty",
    "cancel_new", "close_new", "save_failure", "creation_failure",
    "invalid_empty", "invalid_parent", "invalid_separator", "invalid_absolute",
    "open_unrelated_runs", "open_unrelated_predictions", "open_unrelated_best",
    "open_current_dirty", "open_current_dirty_external", "cancel_open", "initial_save_failure",
    "subfolder_creation_failure", "initial_save_failure_with_data",
])
def test_new_creates_and_open_only_resumes(monkeypatch, tmp_path, scenario):
    app = gui.QtWidgets.QApplication.instance() or gui.QtWidgets.QApplication([])
    previous = results_project_dir(tmp_path / "previous")
    gui.ensure_project_dirs(previous)
    old_values = dict(DEFAULT_CONFIG, project_dir=str(previous), rcl="1.5",
                      last_density_map=str(tmp_path / "previous.mrc"),
                      last_candidate_pool=str(tmp_path / "previous.fasta"),
                      last_run_dir=str(previous / "runs/run_old"),
                      sequence_ranking_run_dir=str(previous / "runs/run_old"))
    save_config(old_values, previous / "config.toml")
    save_config(dict(old_values, weights_dir=str(tmp_path / "custom-weights")))
    old_result = previous / "runs/run_old/sequence_ranking.tsv"
    old_result.parent.mkdir()
    old_result.write_text("rank\tcandidate_id\n1\texisting\n")
    old_bytes = old_result.read_bytes()
    location = previous if scenario == "new_nested" else tmp_path
    root = location / "new-project"
    project = results_project_dir(root)
    sentinel = None
    if scenario == "existing_project":
        gui.ensure_project_dirs(project)
        save_config(dict(old_values, project_dir=str(project)), project / "config.toml")
        sentinel = project / "runs/keep.txt"
    elif scenario == "existing_nonproject":
        root.mkdir()
        sentinel = root / "keep.txt"
    elif scenario == "open_empty":
        root.mkdir()
    elif scenario.startswith("open_unrelated_"):
        marker = {"open_unrelated_runs": "runs", "open_unrelated_predictions": "unetpredictions",
                  "open_unrelated_best": "best_overall"}[scenario]
        (project / marker).mkdir(parents=True)
    if sentinel:
        sentinel.write_bytes(b"Do not overwrite existing content")
        destination_before = {str(path.relative_to(root)): path.read_bytes()
                              for path in root.rglob("*") if path.is_file()}

    names = {"invalid_empty": " ", "invalid_parent": "..", "invalid_separator": "../escape",
             "invalid_absolute": "/absolute"}
    name_requests, warnings, errors = [], [], []
    monkeypatch.setattr(sys, "excepthook", lambda kind, error, tb: errors.append(error))
    monkeypatch.setattr(gui.QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    opening = scenario.startswith("open_") or scenario == "cancel_open"
    selected = previous.parent if scenario.startswith("open_current_") else root if opening else location
    selection = ["" if scenario == "cancel_open" else str(selected)]
    monkeypatch.setattr(gui.QtWidgets.QFileDialog, "getExistingDirectory", lambda *args: selection[0])

    def enter_project(dialog):
        assert isinstance(dialog, gui.NewProjectDialog)
        name_requests.append(dialog)
        if scenario in {"cancel_new", "close_new"}:
            return 0
        dialog.name_edit.setText(names.get(scenario, "new-project"))
        dialog.location_edit.setText(str(location))
        return gui._dialog_accepted_code()
    monkeypatch.setattr(gui, "_exec_dialog", enter_project)
    ready = Check("OK", "Available", True)
    monkeypatch.setattr(gui, "collect_diagnostics", lambda *args: Diagnostics(ready, ready, ready, "cpu"))
    existing_windows = set(app.topLevelWidgets())

    def execute(app):
        window = next(w for w in app.topLevelWidgets()
                      if isinstance(w, gui.CryoIDMainWindow) and w not in existing_windows)
        actions = {action.text(): action for action in window.findChildren(gui.QtGui.QAction)}
        run = window.findChild(gui.QtWidgets.QPushButton, "run_button")
        density = window.findChild(gui.QtWidgets.QLineEdit, "density_map_edit")
        pool = window.findChild(gui.QtWidgets.QLineEdit, "candidate_pool_edit")

        def settle():
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                app.processEvents()
                if not run.property("checking") and not any(
                    thread.isRunning() for thread in window.findChildren(gui.DiagnosticsWorker)
                ):
                    assert not errors, errors
                    return
                time.sleep(.001)
            pytest.fail("GUI diagnostics did not finish")

        try:
            settle()
            with monkeypatch.context() as context:
                if scenario == "save_failure":
                    def cannot_save(*args, **kwargs):
                        raise PermissionError("Read-only outgoing project")
                    context.setattr(gui, "save_project_config", cannot_save)
                elif scenario.startswith("initial_save_failure"):
                    original_save = gui.save_project_config
                    def cannot_initialize(values, selected_project, **kwargs):
                        if selected_project == project:
                            (project / ".config.toml.lock").touch()
                            if scenario.endswith("with_data"):
                                (root / "keep.txt").write_text("Preserve newly added data")
                            raise PermissionError("Could not initialize new project")
                        return original_save(values, selected_project, **kwargs)
                    context.setattr(gui, "save_project_config", cannot_initialize)
                elif scenario in {"creation_failure", "subfolder_creation_failure"}:
                    original_mkdir = type(root).mkdir
                    def cannot_create(path, *args, **kwargs):
                        if path == (project / "runs" if scenario == "subfolder_creation_failure" else root):
                            raise PermissionError("Read-only project location")
                        return original_mkdir(path, *args, **kwargs)
                    context.setattr(type(root), "mkdir", cannot_create)
                if scenario.startswith("open_current_"):
                    density.setText(str(tmp_path / "typed-unsaved.mrc"))
                    pool.setText(str(tmp_path / "typed-unsaved.fasta"))
                    if scenario == "open_current_dirty_external":
                        save_config(dict(old_values, rcl="2.5"), previous / "config.toml", project=True)
                actions["Open" if opening else "New"].trigger()
                settle()

            assert old_result.read_bytes() == old_bytes
            if scenario in {"new", "new_nested"}:
                assert not warnings
                assert window.windowTitle() == "CryoIDv2 — new-project"
                assert f"Created project: {project}" in window.findChild(
                    gui.QtWidgets.QPlainTextEdit, "log_edit"
                ).toPlainText()
                assert density.text() == pool.text() == ""
                assert load_config()["project_dir"] == str(project)
                assert load_config()["weights_dir"] == str(tmp_path / "custom-weights")
                saved = load_config(project / "config.toml")
                assert saved["rcl"] == DEFAULT_CONFIG["rcl"]
                for key in ("last_density_map", "last_candidate_pool", "last_run_dir", "sequence_ranking_run_dir"):
                    assert saved[key] == ""
                for directory in ("runs", "best_overall", "unetpredictions"):
                    assert (project / directory).is_dir()
                    assert not list((project / directory).iterdir())
                assert not window.findChild(gui.QtWidgets.QPushButton, "ranking_button").isEnabled()
                assert not window.findChild(gui.QtWidgets.QPushButton, "model_button").isEnabled()
                selection[0] = str(root)
                actions["Open"].trigger()
                settle()
                assert load_config()["project_dir"] == str(project)
                assert density.text() == pool.text() == ""
            elif scenario.startswith("open_current_"):
                assert not warnings
                assert density.text() == str(tmp_path / "typed-unsaved.mrc")
                assert pool.text() == str(tmp_path / "typed-unsaved.fasta")
                saved = load_config(previous / "config.toml")
                assert saved["last_density_map"] == density.text()
                assert saved["last_candidate_pool"] == pool.text()
                assert saved["rcl"] == ("2.5" if scenario.endswith("external") else "1.5")
            else:
                assert window.windowTitle() == "CryoIDv2 — previous"
                assert density.text() == old_values["last_density_map"]
                assert pool.text() == old_values["last_candidate_pool"]
                assert load_config()["project_dir"] == str(previous)
                if sentinel:
                    assert {str(path.relative_to(root)): path.read_bytes()
                            for path in root.rglob("*") if path.is_file()} == destination_before
                    assert "different project name" in warnings[-1]
                elif scenario == "open_empty":
                    assert not project.exists()
                    assert "File > New" in warnings[-1]
                elif scenario.startswith("open_unrelated_"):
                    assert not (project / "config.toml").exists()
                    assert "File > New" in warnings[-1]
                elif scenario == "initial_save_failure_with_data":
                    assert (root / "keep.txt").read_text() == "Preserve newly added data"
                    assert not (project / "config.toml").exists()
                    assert warnings
                else:
                    assert not root.exists()
                    assert bool(warnings) == (scenario not in {"cancel_new", "close_new", "cancel_open"})
                if scenario == "cancel_open":
                    assert not name_requests
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


def test_create_dialog_browses_location_and_rejects_invalid_targets(monkeypatch, tmp_path):
    app = gui.QtWidgets.QApplication.instance() or gui.QtWidgets.QApplication([])
    dialog = gui.NewProjectDialog(location=tmp_path)
    alternate = tmp_path / "other-location"
    alternate.mkdir()
    selected = [str(alternate)]
    warnings = []
    monkeypatch.setattr(gui.QtWidgets.QFileDialog, "getExistingDirectory", lambda *args: selected[0])
    monkeypatch.setattr(gui.QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    try:
        dialog.show()
        app.processEvents()
        assert dialog.create_button.text() == "Create"
        assert not dialog.create_button.isEnabled()
        dialog.name_edit.setText("My project")
        dialog.browse_button.click()
        assert dialog.location_edit.text() == str(alternate)
        assert dialog.project_root() == alternate / "My project"
        assert not dialog.project_root().exists()
        selected[0] = ""
        dialog.browse_button.click()
        assert dialog.location_edit.text() == str(alternate)
        target = alternate / "My project"
        target.mkdir()
        (target / "keep.txt").write_text("Preserve existing folder")
        dialog.create_button.click()
        assert dialog.isVisible() and "File > Open" in warnings[-1]
        assert (target / "keep.txt").read_text() == "Preserve existing folder"
        dialog.name_edit.setText("../escape")
        dialog.create_button.click()
        assert dialog.isVisible() and "project name" in warnings[-1]
        dialog.name_edit.setText("Valid name")
        dialog.location_edit.setText(str(tmp_path / "missing-location"))
        dialog.create_button.click()
        assert dialog.isVisible() and "existing parent folder" in warnings[-1]
    finally:
        dialog.reject()
        dialog.deleteLater()
        app.sendPostedEvents(None, gui.QtCore.QEvent.Type.DeferredDelete)


def test_real_dialogs_shortcuts_create_open_and_cancel(monkeypatch, tmp_path):
    """Use actual modal Qt dialogs and keyboard events, with isolated settings."""
    from importlib import import_module
    from G_GUI.qt_compat import QT_API
    QtTest = import_module(f"{QT_API}.QtTest")
    app = gui.QtWidgets.QApplication.instance() or gui.QtWidgets.QApplication([])
    ready = Check("OK", "Available", True)
    monkeypatch.setattr(gui, "collect_diagnostics", lambda *args: Diagnostics(ready, ready, ready, "cpu"))
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda kind, error, tb: errors.append(error))
    existing_windows = set(app.topLevelWidgets())
    original_picker = gui.QtWidgets.QFileDialog.getExistingDirectory
    options = getattr(gui.QtWidgets.QFileDialog, "Option", gui.QtWidgets.QFileDialog)
    picker_calls = []

    def real_picker(parent, title, directory):
        picker_calls.append(title)
        # Native OS dialogs cannot be driven by QtTest; use the real Qt
        # directory picker with exactly the same selection/accept semantics.
        chosen = original_picker(parent, title, directory,
                                 options.ShowDirsOnly | options.DontUseNativeDialog)
        if chosen and not Path(chosen).resolve().is_relative_to(tmp_path.resolve()):
            errors.append(AssertionError(f"Picker left the test folder: {chosen}"))
            return ""
        return chosen
    monkeypatch.setattr(gui.QtWidgets.QFileDialog, "getExistingDirectory", real_picker)

    def execute(app):
        window = next(widget for widget in app.topLevelWidgets()
                      if isinstance(widget, gui.CryoIDMainWindow) and widget not in existing_windows)
        actions = {action.text(): action for action in window.findChildren(gui.QtGui.QAction)}
        standard = getattr(gui.QtGui.QKeySequence, "StandardKey", gui.QtGui.QKeySequence)
        steps = []
        deadline = [0.0]
        timer = gui.QtCore.QTimer(window)
        timer.setInterval(20)

        def drive_dialog():
            dialog = app.activeModalWidget()
            if dialog is None:
                return
            if time.monotonic() > deadline[0]:
                errors.append(TimeoutError(f"Dialog did not finish: {dialog.windowTitle()}"))
                dialog.reject()
                return
            if not steps:
                detail = dialog.text() if isinstance(dialog, gui.QtWidgets.QMessageBox) else ""
                errors.append(AssertionError(f"Unexpected dialog: {dialog.windowTitle()} {detail}"))
                dialog.reject()
                return
            kind, value = steps[0]
            if isinstance(dialog, gui.QtWidgets.QFileDialog) and kind.startswith("folder"):
                steps.pop(0)
                if kind == "folder_cancel":
                    dialog.reject()
                else:
                    edit = dialog.findChild(gui.QtWidgets.QLineEdit, "fileNameEdit")
                    edit.setFocus()
                    edit.selectAll()
                    edit.setText(str(value))
                    selected = dialog.selectedFiles()
                    if len(selected) != 1 or Path(selected[0]).resolve() != Path(value).resolve():
                        errors.append(AssertionError(f"Unexpected folder selection: {selected}"))
                        dialog.reject()
                        return
                    box = dialog.findChild(gui.QtWidgets.QDialogButtonBox)
                    roles = getattr(gui.QtWidgets.QDialogButtonBox, "ButtonRole", gui.QtWidgets.QDialogButtonBox)
                    button = next(button for button in box.buttons()
                                  if box.buttonRole(button) == roles.AcceptRole)
                    mouse = getattr(gui.QtCore.Qt, "MouseButton", gui.QtCore.Qt)
                    QtTest.QTest.mouseClick(button, mouse.LeftButton)
            elif isinstance(dialog, gui.NewProjectDialog) and kind == "project":
                steps.pop(0)
                assert dialog.create_button.text() == "Create"
                if value is None:
                    dialog.cancel_button.click()
                else:
                    name, location = value
                    QtTest.QTest.keyClicks(dialog.name_edit, name)
                    dialog.location_edit.setText(str(location))
                    if not dialog.project_root().resolve().is_relative_to(tmp_path.resolve()):
                        errors.append(AssertionError("New project left the test folder"))
                        dialog.reject()
                        return
                    mouse = getattr(gui.QtCore.Qt, "MouseButton", gui.QtCore.Qt)
                    QtTest.QTest.mouseClick(dialog.create_button, mouse.LeftButton)

        timer.timeout.connect(drive_dialog)
        timer.start()

        def settle():
            limit = time.monotonic() + 5
            while time.monotonic() < limit:
                app.processEvents()
                if not any(thread.isRunning() for thread in window.findChildren(gui.DiagnosticsWorker)):
                    assert not errors, errors
                    return
                time.sleep(.001)
            pytest.fail("GUI diagnostics did not finish")

        def press(action, plan):
            settle()
            assert not steps
            steps.extend(plan)
            deadline[0] = time.monotonic() + 10
            window.activateWindow()
            window.setFocus()
            app.processEvents()
            QtTest.QTest.keySequence(window, gui.QtGui.QKeySequence(getattr(standard, action)))
            settle()
            assert not steps, f"{action} shortcut did not complete its dialogs"

        first, second = [results_project_dir(tmp_path / name) for name in ("First", "Second")]
        try:
            settle()
            assert actions["New"].shortcuts() == gui.QtGui.QKeySequence.keyBindings(standard.New)
            assert actions["Open"].shortcuts() == gui.QtGui.QKeySequence.keyBindings(standard.Open)
            press("New", [("project", ("First", tmp_path))])
            assert not picker_calls  # New must not begin with a Select Folder dialog.
            assert (first / "config.toml").is_file()
            assert window.windowTitle() == "CryoIDv2 — First"
            density = window.findChild(gui.QtWidgets.QLineEdit, "density_map_edit")
            pool = window.findChild(gui.QtWidgets.QLineEdit, "candidate_pool_edit")
            density.setText(str(tmp_path / "first.mrc"))
            pool.setText(str(tmp_path / "first.fasta"))
            press("New", [("project", ("Second", tmp_path))])
            assert not picker_calls
            assert density.text() == pool.text() == ""
            assert window.windowTitle() == "CryoIDv2 — Second"
            assert load_config(first / "config.toml")["last_density_map"] == str(tmp_path / "first.mrc")
            press("Open", [("folder", first.parent)])
            assert density.text() == str(tmp_path / "first.mrc")
            assert pool.text() == str(tmp_path / "first.fasta")
            assert window.windowTitle() == "CryoIDv2 — First"
            press("Open", [("folder_cancel", None)])
            press("New", [("project", None)])
            assert window.windowTitle() == "CryoIDv2 — First"
            assert load_config()["project_dir"] == str(first)
            assert (second / "config.toml").is_file()
        finally:
            timer.stop()
            # Always clear this test's windows, including on a failed dialog
            # assertion; otherwise a modal can pollute later GUI tests.
            try:
                for widget in set(app.topLevelWidgets()) - existing_windows - {window}:
                    if isinstance(widget, gui.QtWidgets.QDialog):
                        widget.reject()
                        widget.deleteLater()
                limit = time.monotonic() + 5
                while time.monotonic() < limit and any(
                    thread.isRunning() for thread in window.findChildren(gui.DiagnosticsWorker)
                ):
                    app.processEvents()
                    time.sleep(.001)
            finally:
                window.close()
                window.deleteLater()
                app.sendPostedEvents(None, gui.QtCore.QEvent.Type.DeferredDelete)
        return 0

    monkeypatch.setattr(gui.QtWidgets.QApplication, "exec", execute)
    with pytest.raises(SystemExit) as exited:
        gui.main()
    assert exited.value.code == 0
