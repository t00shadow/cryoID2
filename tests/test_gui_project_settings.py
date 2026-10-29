"""Project changes keep recent inputs and preferences available in the GUI."""
import time
import sys
import json
from pathlib import Path

import pytest

from G_GUI import main_gui as gui
from G_GUI.diagnostics import Check, Diagnostics
from G_GUI.qt_compat import QtWidgets
from cryoid_pipeline.user_config import load_config, save_config, results_project_dir


@pytest.mark.parametrize("action_name,with_history", [
    ("New", False), ("Open", False), ("Open", True), ("Open run", True),
])
def test_project_without_config_opens_blank(monkeypatch, tmp_path, action_name, with_history):
    from cryoid_pipeline.user_config import DEFAULT_CONFIG
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda kind, error, tb: errors.append(error))
    previous = results_project_dir(tmp_path / "previous")
    previous_values = dict(DEFAULT_CONFIG, project_dir=str(previous), rcl="0.75",
                           weights_dir=str(tmp_path / "previous-weights"),
                           chimerax_path=str(tmp_path / "previous-ChimeraX.exe"),
                           last_density_map=str(tmp_path / "previous.mrc"),
                           last_candidate_pool=str(tmp_path / "previous.fa"),
                           sequence_mode="rna", structure_selection="3", keep_tmp="true")
    save_config(previous_values, previous / "config.toml")
    save_config(previous_values)
    project = results_project_dir(tmp_path / "blank")
    selected = project.parent
    if with_history:
        run = project / "runs" / "run_20261001_120000"
        run.mkdir(parents=True)
        (run / "run_config.json").write_text("broken JSON should never be loaded")
        if action_name == "Open run":
            selected = run
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *args: str(selected))
    ready = Check("OK", "Available", True)
    monkeypatch.setattr(gui, "collect_diagnostics", lambda *args: Diagnostics(ready, ready, ready, "cpu"))
    inspected = []

    def inspect(dialog):
        inspected.append(type(dialog))
        if isinstance(dialog, gui.ConfigurationsWindow):
            assert dialog.weights_path() == previous_values["weights_dir"]
            assert dialog.chimerax_path() == previous_values["chimerax_path"]

        else:
            assert isinstance(dialog, gui.OptionsWindow)
            assert dialog.rcl() == 0 and dialog.protein_min_trace_length() == 10
            assert dialog.sequence_mode() == "auto"
            assert dialog.structure_selection() == "" and not dialog.keep_tmp()
            assert dialog.device() == dialog.cpu_threads() == "auto"
        return 0
    monkeypatch.setattr(gui, "_exec_dialog", inspect)

    def launch(restarted=False):
        existing = set(app.topLevelWidgets())
        def execute(app):
            window = next(w for w in app.topLevelWidgets()
                          if isinstance(w, gui.CryoIDMainWindow) and w not in existing)
            button = window.findChild(QtWidgets.QPushButton, "run_button")
            def settle():
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    app.processEvents()
                    if window.isVisible() and not button.property("checking"):
                        assert not errors, errors
                        return
                    time.sleep(.001)
                pytest.fail("GUI checks did not finish")
            try:
                settle()
                if not restarted:
                    assert window.findChild(QtWidgets.QLineEdit, "density_map_edit").text() == previous_values["last_density_map"]
                    for key in ("CRYOID_RCL", "CRYOID_UNET_DEVICE", "CRYOID_CPU_THREADS", "CRYOID_KEEP_TMP"):
                        monkeypatch.setenv(key, "invalid")
                    monkeypatch.setenv("CRYOID_UNET_WEIGHTS_DIR", "unrelated-weights")
                    monkeypatch.setenv("CRYOID_CHIMERAX_PATH", "unrelated-ChimeraX")
                    action = "New" if action_name == "New" else "Open"
                    next(a for a in window.findChildren(gui.QtGui.QAction) if a.text() == action).trigger()
                    settle()
                assert window.findChild(QtWidgets.QLineEdit, "density_map_edit").text() == ""
                assert window.findChild(QtWidgets.QLineEdit, "candidate_pool_edit").text() == ""
                next(a for a in window.findChildren(gui.QtGui.QAction) if a.text() == "Configurations").trigger()
                window.findChild(QtWidgets.QPushButton, "options_button").click()
                assert not errors, errors
                assert button.property("blocked")
                saved = load_config(project / "config.toml")
                for key in ("last_density_map", "last_candidate_pool", "last_run_dir", "sequence_ranking_run_dir", "weights_dir", "chimerax_path"):
                    assert saved[key] == ""
                assert float(saved["rcl"]) == 0
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
    launch()
    launch(restarted=True)
    assert inspected == [gui.ConfigurationsWindow, gui.OptionsWindow] * 2


@pytest.mark.parametrize("selection,missing_inputs", [
    ("parent", False), ("results", False), ("run", False), ("parent", True),
])
def test_open_existing_project_restores_its_run_inputs_and_options(monkeypatch, tmp_path, selection, missing_inputs):
    from cryoid_pipeline.user_config import DEFAULT_CONFIG
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    existing_windows = set(app.topLevelWidgets())
    slot_errors = []
    monkeypatch.setattr(sys, "excepthook", lambda kind, error, tb: slot_errors.append(error))
    project = results_project_dir(tmp_path / "old-project")
    runs = [project / "runs" / name for name in ("run_20261001_120000", "run_20261002_120000")]
    inputs = []
    for index, run in enumerate(runs):
        run.mkdir(parents=True)
        density, pool = tmp_path / f"map-{index}.mrc", tmp_path / f"pool-{index}.fa"
        density.touch()
        pool.write_text(">candidate\nMAGST\n")
        inputs.append((density, pool))
        (run / "run_config.json").write_text(json.dumps({
            "density_map": str(density), "candidate_pool": str(pool),
            "weights_path": str(tmp_path / f"weights-{index}"),
            "chimerax_path": str(tmp_path / "ChimeraX.exe"),
            "configured_rcl": 0.125 + index, "rcl": 9,
            "protein_min_trace_length": 15 + index, "requested_mode": "auto",
            "effective_mode": "protein", "configured_structure_selection": "",
            "structure_selection": {"input": "3"}, "keep_tmp": True,
            "cpu_threads": 1, "unet_device": "cpu",
        }), encoding="utf-8")
        (run / "sequence_ranking.tsv").write_text("rank\n")
    for directory in ("best_overall", "unetpredictions"):
        (project / directory).mkdir()
    # Start with no input history, as on a new installation.
    save_config({"project_dir": str(tmp_path / "unrelated"), "rcl": "5",
                 "weights_dir": str(tmp_path / "global-weights"), "chimerax_path": str(tmp_path / "ChimeraX.exe")})
    selected = {"parent": project.parent, "results": project, "run": runs[0]}[selection]
    index = 1  # Opening a run folder still opens its project's editable settings.
    density, pool = inputs[index]
    save_config(dict(DEFAULT_CONFIG, last_density_map=str(density), last_candidate_pool=str(pool), weights_dir=str(tmp_path / f"weights-{index}"), chimerax_path=str(tmp_path / "ChimeraX.exe"), rcl=str(0.125 + index), protein_min_trace_length=str(15 + index), sequence_mode="auto", structure_selection="", keep_tmp="true", cpu_threads="1", device="cpu", project_dir=str(project),
                     last_run_dir=str(runs[index]), sequence_ranking_run_dir=str(runs[index])), project / "config.toml")
    if missing_inputs:
        density.unlink()
        pool.unlink()
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *args: str(selected))
    ready = Check("OK", "Available", True)
    monkeypatch.setattr(gui, "collect_diagnostics", lambda *args: Diagnostics(ready, ready, ready, "cpu"))
    inspected = []

    def inspect(dialog):
        inspected.append(type(dialog))
        if isinstance(dialog, gui.ConfigurationsWindow):
            assert dialog.weights_path() == str(tmp_path / "global-weights")
            assert dialog.chimerax_path() == str(tmp_path / "ChimeraX.exe")

        else:
            assert isinstance(dialog, gui.OptionsWindow)
            assert dialog.rcl() == 0.125 + index
            assert dialog.protein_min_trace_length() == 15 + index
            assert dialog.sequence_mode() == "auto"
            assert dialog.structure_selection() == ""
            assert dialog.keep_tmp()
            assert dialog.cpu_threads() == "1"
            assert dialog.device() == "cpu"
        return 0

    monkeypatch.setattr(gui, "_exec_dialog", inspect)

    def execute(app):
        window = next(w for w in app.topLevelWidgets()
                      if isinstance(w, gui.CryoIDMainWindow) and w not in existing_windows)
        run_button = window.findChild(QtWidgets.QPushButton, "run_button")
        density_edit = window.findChild(QtWidgets.QLineEdit, "density_map_edit")
        pool_edit = window.findChild(QtWidgets.QLineEdit, "candidate_pool_edit")
        actions = {a.text(): a for a in window.findChildren(gui.QtGui.QAction)}

        def settle():
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                app.processEvents()
                if window.isVisible() and not run_button.property("checking"):
                    return
                time.sleep(.001)
            pytest.fail("GUI checks did not finish")

        try:
            settle()
            if not restarted:
                assert density_edit.text() == pool_edit.text() == ""
                actions["Open"].trigger()
                settle()
            assert not slot_errors, slot_errors
            assert density_edit.text() == str(density)
            assert pool_edit.text() == str(pool)
            actions["Configurations"].trigger()
            window.findChild(QtWidgets.QPushButton, "options_button").click()
            assert not slot_errors, slot_errors
            saved = load_config(project / "config.toml")
            assert saved["last_run_dir"] == str(runs[index])
            assert saved["sequence_ranking_run_dir"] == str(runs[index])
            assert saved["last_density_map"] == str(density)
            assert saved["last_candidate_pool"] == str(pool)
            assert bool(run_button.property("blocked")) == missing_inputs
        finally:
            settle()
            window.close()
            window.deleteLater()
            app.sendPostedEvents(None, gui.QtCore.QEvent.Type.DeferredDelete)
        return 0

    monkeypatch.setattr(QtWidgets.QApplication, "exec", execute)
    for restarted in (False, True):
        with pytest.raises(SystemExit) as exited:
            gui.main()
        assert exited.value.code == 0
    assert inspected == [gui.ConfigurationsWindow, gui.OptionsWindow] * 2


@pytest.mark.parametrize("with_history", [False, True])
def test_switching_projects_restores_saved_config_edits_and_reloads_on_restart(monkeypatch, tmp_path, with_history):
    from cryoid_pipeline import cli
    from cryoid_pipeline.models import PipelineResult, StageResult
    from cryoid_pipeline.user_config import DEFAULT_CONFIG

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    slot_errors, warnings = [], []
    monkeypatch.setattr(sys, "excepthook", lambda kind, error, tb: slot_errors.append(error))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *args: warnings.append(args[2]))
    ready = Check("OK", "Available", True)
    monkeypatch.setattr(gui, "collect_diagnostics", lambda *args: Diagnostics(ready, ready, ready, "cpu"))
    cli_settings = []
    cli_inputs = []

    class Pipeline:
        def __init__(self, **kwargs):
            cli_settings.append(kwargs)

        def run(self, *args, **kwargs):
            cli_inputs.append(args)
            return PipelineResult(tmp_path, [StageResult("Run", "ok", "Test pipeline")])

    monkeypatch.setattr("cryoid_pipeline.orchestrator.CryoIDPipeline", Pipeline)
    projects, expected, history = {}, {}, {}
    for index, name in enumerate(("A", "B")):
        project = results_project_dir(tmp_path / name)
        project.mkdir(parents=True)
        density, pool = tmp_path / f"{name}.mrc", tmp_path / f"{name}.fa"
        density.touch()
        pool.touch()
        projects[name] = project
        expected[name] = dict(DEFAULT_CONFIG, project_dir=str(project),
                              weights_dir=str(tmp_path / f"weights-{name}"),
                              chimerax_path=str(tmp_path / "ChimeraX.exe") if index == 0 else "",
                              rcl=str(0.125 + index), sequence_mode="protein" if index == 0 else "rna",
                              protein_min_trace_length=str(15 + index), structure_selection=str(3 + index),
                              keep_tmp="false" if index == 0 else "true",
                              device="cpu" if index == 0 else "auto",
                              cpu_threads="1" if index == 0 else "auto",
                              last_density_map=str(density), last_candidate_pool=str(pool))
        save_config(expected[name], project / "config.toml")
        if with_history:
            run = project / "runs" / "run_20261001_120000"
            run.mkdir(parents=True)
            path = run / "run_config.json"
            path.write_text(json.dumps({"rcl": 9, "weights_path": "historical-weights",
                                        "density_map": "historical.mrc", "candidate_pool": "historical.fa"}))
            history[path] = path.read_bytes()
    global_paths = {"weights_dir": str(tmp_path / "shared-weights"), "chimerax_path": str(tmp_path / "Shared-ChimeraX.exe")}
    save_config(dict(DEFAULT_CONFIG, project_dir=str(projects["A"]), **global_paths))
    selection, current, edits = [""], ["A"], {}
    monkeypatch.setattr(QtWidgets.QFileDialog, "getExistingDirectory", lambda *args: selection[0])

    def inspect(dialog):
        saved = expected[current[0]]
        if isinstance(dialog, gui.ConfigurationsWindow):
            assert dialog.weights_path() == global_paths["weights_dir"]
            assert dialog.chimerax_path() == global_paths["chimerax_path"]

            if "weights_dir" in edits:
                dialog.weights_path_edit.setText(edits["weights_dir"])
                dialog.chimerax_path_edit.setText(edits["chimerax_path"])
                return gui._dialog_accepted_code()
        else:
            assert isinstance(dialog, gui.OptionsWindow)
            assert dialog.rcl() == float(saved["rcl"])
            assert dialog.sequence_mode() == saved["sequence_mode"]
            assert dialog.protein_min_trace_length() == int(saved["protein_min_trace_length"])
            assert dialog.structure_selection() == saved["structure_selection"]
            assert dialog.keep_tmp() == (saved["keep_tmp"] == "true")
            assert dialog.device() == saved["device"]
            assert dialog.cpu_threads() == saved["cpu_threads"]
            if "rcl" in edits:
                dialog.rcl_edit.setText(edits["rcl"])
                return gui._dialog_accepted_code()
        return 0

    monkeypatch.setattr(gui, "_exec_dialog", inspect)

    def launch(restarted=False):
        existing_windows = set(app.topLevelWidgets())

        def execute(app):
            window = next(w for w in app.topLevelWidgets()
                          if isinstance(w, gui.CryoIDMainWindow) and w not in existing_windows)
            run = window.findChild(QtWidgets.QPushButton, "run_button")
            density = window.findChild(QtWidgets.QLineEdit, "density_map_edit")
            pool = window.findChild(QtWidgets.QLineEdit, "candidate_pool_edit")
            options = window.findChild(QtWidgets.QPushButton, "options_button")
            actions = {a.text(): a for a in window.findChildren(gui.QtGui.QAction)}

            def settle():
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    app.processEvents()
                    if window.isVisible() and not run.property("checking"):
                        assert not slot_errors, slot_errors
                        return
                    time.sleep(.001)
                pytest.fail("GUI checks did not finish")

            def check():
                settle()
                saved = expected[current[0]]
                assert density.text() == saved["last_density_map"]
                assert pool.text() == saved["last_candidate_pool"]
                actions["Configurations"].trigger()
                options.click()
                assert not slot_errors, slot_errors
                if not saved["last_density_map"] or not saved["last_candidate_pool"]:
                    return
                assert cli.main(["run", "--project", str(projects[current[0]])]) == 0
                assert cli_inputs[-1] == (Path(saved["last_density_map"]), Path(saved["last_candidate_pool"]))
                actual = cli_settings[-1]
                assert actual["unet_rcl"] == float(saved["rcl"])
                assert actual["unet_weights_dir"] == Path(global_paths["weights_dir"])
                assert actual["sequence_mode"] == saved["sequence_mode"]
                assert actual["protein_min_trace_length"] == int(saved["protein_min_trace_length"])
                assert actual["structure_selection"] == (saved["structure_selection"] or None)
                assert actual["keep_tmp"] == (saved["keep_tmp"] == "true")
                assert actual["unet_device"] == saved["device"]
                assert str(actual["cpu_threads"]) == saved["cpu_threads"]

            def open_project(name):
                selection[0] = str(projects[name].parent)
                actions["Open"].trigger()
                current[0] = name
                check()

            try:
                settle()
                if restarted:
                    check()  # The project file takes priority over stale global defaults.
                    return 0
                open_project("A")
                edits.update(weights_dir=str(tmp_path / "new-weights-A"), chimerax_path="")
                actions["Configurations"].trigger()
                global_paths.update(edits)
                edits.clear()
                edits["rcl"] = "0.5"
                options.click()
                expected["A"].update(edits)
                edits.clear()
                snapshot = load_config(projects["A"] / "config.toml")
                assert snapshot["rcl"] == "0.5"
                assert snapshot["weights_dir"] == ""
                assert load_config()["weights_dir"] == global_paths["weights_dir"]
                # Saving typed inputs must not require starting another run.
                density.setText(str(tmp_path / "edited-A.mrc"))
                expected["A"]["last_density_map"] = density.text()
                density.editingFinished.emit()
                assert load_config(projects["A"] / "config.toml")["last_density_map"] == density.text()
                for name in ("B", "A", "B", "A"):
                    open_project(name)
                # Opening a new project inherits current values, then gets its own file.
                projects["C"] = results_project_dir(tmp_path / "C")
                expected["C"] = dict(DEFAULT_CONFIG, project_dir=str(projects["C"]))
                selection[0] = str(projects["C"].parent)
                actions["New"].trigger()
                current[0] = "C"
                check()
                assert (projects["C"] / "config.toml").is_file()
                # Add inputs through the project editor before using the short CLI command.
                expected["C"].update(last_density_map=str(tmp_path / "C.mrc"), last_candidate_pool=str(tmp_path / "C.fa"),
                                     weights_dir=str(tmp_path / "C-weights"))
                save_config(expected["C"], projects["C"] / "config.toml")
                open_project("C")
                open_project("B")
                open_project("C")
                # Reopen the same project to read an external config-file edit.
                expected["C"]["rcl"] = "0.75"
                save_config(expected["C"], projects["C"] / "config.toml")
                open_project("C")
                # An outgoing project that cannot be saved must stay open.
                def cannot_save(*args, **kwargs):
                    raise PermissionError("Read-only project configuration")
                with monkeypatch.context() as context:
                    context.setattr(gui, "save_project_config", cannot_save)
                    selection[0] = str(projects["B"].parent)
                    actions["Open"].trigger()
                    check()
                    assert warnings and "config.toml" in warnings[-1]
                # Invalid project files must leave the current project and file intact.
                corrupt = results_project_dir(tmp_path / "corrupt")
                corrupt.mkdir(parents=True)
                (corrupt / "config.toml").write_text("broken = [")
                selection[0] = str(corrupt.parent)
                actions["Open"].trigger()
                check()
                assert warnings and "config.toml" in warnings[-1]
                assert (corrupt / "config.toml").read_text() == "broken = ["
                # Closing also remembers a typed path without a focus change.
                pool.setText(str(tmp_path / "edited-C.fa"))
                expected["C"]["last_candidate_pool"] = pool.text()
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

    launch()
    assert load_config(projects["C"] / "config.toml")["last_candidate_pool"] == expected["C"]["last_candidate_pool"]
    save_config(dict(DEFAULT_CONFIG, project_dir=str(projects["C"]), rcl="8", **global_paths))
    launch(restarted=True)
    assert all(path.read_bytes() == original for path, original in history.items())
