"""CLI modes and GUI forwarding share text; JSON remains private to widgets."""
from types import SimpleNamespace

import pytest

from cryoid_pipeline import cli, orchestrator, run_log
from cryoid_pipeline.models import PipelineResult, StageResult
from cryoid_pipeline.user_config import DEFAULT_CONFIG, project_config_path, results_project_dir, save_config


@pytest.mark.parametrize("outcome,exit_code", [("ok", 0), ("error", 1), ("blocked_missing_input", 2), ("skipped", 130)])
def test_cli_and_gui_receive_identical_readable_logs(monkeypatch, tmp_path, capsys, outcome, exit_code):
    now = [0.0]
    original_log = run_log.RunLog
    monkeypatch.setattr(run_log, "RunLog", lambda write: original_log(write, clock=lambda: now[0]))
    project = results_project_dir(tmp_path)
    save_config(dict(DEFAULT_CONFIG, project_dir=str(project)), project_config_path(project), project=True)

    class Pipeline:
        def __init__(self, **kwargs):
            pass

        def run(self, density, pool, *, progress, progress_percent, status, **kwargs):
            status(dict(name="U-Net prediction", device="cpu", stage_index=2, stage_total=6))
            progress("Stage: U-Net prediction; device: cpu")
            status(dict(kind="inference_device", device="cpu"))
            status(dict(kind="model_loading", model="key_atoms", model_index=2, model_total=3))
            now[0] += 10
            status(dict(kind="model_start", model="key_atoms"))
            for count in range(101):
                now[0] += .1
                progress(f"UNet key_atoms: predicted cube {count}/100")
                progress_percent(24 + count // 6)
                status(dict(name="U-Net prediction", device="cpu", kind="model_progress",
                            model="key_atoms", completed=count, total=100))
            progress("Warning: example diagnostic retained.")
            progress_percent(100)
            return PipelineResult(tmp_path / "run", stages=[StageResult("Run", outcome, "Example outcome")])

    monkeypatch.setattr(orchestrator, "CryoIDPipeline", Pipeline)
    args = ["run", "--map", str(tmp_path / "map.mrc"), "--pool", str(tmp_path / "pool.fasta"),
            "--project", str(tmp_path), "--no-config"]
    assert cli.main(args) == exit_code
    normal = capsys.readouterr().out.splitlines()
    now[0] = 0
    assert cli.main([*args, "--event-stream"]) == exit_code
    streamed = capsys.readouterr().out.splitlines()
    readable = [line for line in streamed if cli.decode_event(line) is None]
    assert normal == readable
    assert len([line for line in readable if line.startswith("Cubes")]) <= 4
    assert "Warning: example diagnostic retained." in readable
    assert not any("predicted cube" in line or line.startswith("Progress:") for line in readable)
    assert ("Overall pipeline: 100%" in readable) == (outcome == "ok")

    from G_GUI.main_gui import PipelineWorker
    monkeypatch.setattr("G_GUI.main_gui.subprocess.Popen", lambda *args, **kwargs: SimpleNamespace(
        stdout=[line + "\n" for line in streamed], wait=lambda: exit_code))
    worker = PipelineWorker("map.mrc", "pool.fasta", tmp_path, "", 0, 10, "auto", "", False)
    logs, events, results = [], [], []
    worker.log.connect(logs.append)
    worker.status.connect(events.append)
    worker.finished.connect(results.append)
    worker.run()
    assert logs == readable
    assert capsys.readouterr().out.splitlines() == readable  # GUI's parent console, too.
    assert any(event.get("kind") == "model_progress" for event in events)
    assert results[0].stages[0].status == outcome

    if outcome == "ok":
        _check_actual_gui_log(monkeypatch, tmp_path, project, readable, capsys)


def _check_actual_gui_log(monkeypatch, tmp_path, project, readable, capsys):
    """Also exercise the real GUI log slot and finish handler, not only its worker."""
    import sys
    import time
    from G_GUI import main_gui as gui
    from G_GUI.diagnostics import Check, Diagnostics

    app = gui.QtWidgets.QApplication.instance() or gui.QtWidgets.QApplication([])
    existing_windows = set(app.topLevelWidgets())
    density, pool = tmp_path / "map.mrc", tmp_path / "pool.fasta"
    density.write_bytes(b"input")
    pool.write_text(">candidate\nMAGST\n")
    save_config(dict(DEFAULT_CONFIG, project_dir=str(project), last_density_map=str(density),
                     last_candidate_pool=str(pool)), project_config_path(project), project=True)
    save_config(dict(DEFAULT_CONFIG, project_dir=str(project)))
    ready = Check("OK", "Available", True)
    monkeypatch.setattr(gui, "collect_diagnostics", lambda *args: Diagnostics(ready, ready, ready, "cpu"))
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda kind, error, tb: errors.append(error))

    def execute(app):
        window = next(w for w in app.topLevelWidgets()
                      if isinstance(w, gui.CryoIDMainWindow) and w not in existing_windows)
        button = window.findChild(gui.QtWidgets.QPushButton, "run_button")
        log = window.findChild(gui.QtWidgets.QPlainTextEdit, "log_edit")

        def settle():
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                app.processEvents()
                assert not errors, errors
                if not button.property("checking") and not any(
                    thread.isRunning() for thread in window.findChildren(gui.QtCore.QThread)
                ):
                    return
                time.sleep(.001)
            pytest.fail("GUI run did not finish")

        try:
            settle()
            button.click()
            settle()
            assert log.toPlainText().splitlines() == ["CryoID2 version 1.0.0", *readable]
            assert capsys.readouterr().out.splitlines() == log.toPlainText().splitlines()
        finally:
            window.close()
        return 0

    monkeypatch.setattr(gui.QtWidgets.QApplication, "exec", execute)
    with pytest.raises(SystemExit) as exited:
        gui.main()
    assert exited.value.code == 0
