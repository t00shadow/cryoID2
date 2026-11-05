import logging
import os
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from cryoid_pipeline.compute import cpu_thread_limit
from cryoid_pipeline.user_config import resolve_cpu_threads, resolve_device, validate_cpu_threads, validate_device


def test_auto_preserves_every_existing_numerical_limit_and_environment(monkeypatch):
    import torch
    import scipy.linalg
    import sklearn
    from threadpoolctl import threadpool_info
    snapshot = {p["filepath"]: p["num_threads"] for p in threadpool_info()}
    environment = dict(os.environ)
    monkeypatch.setattr(torch, "set_num_threads", lambda value: pytest.fail("Auto changed Torch threads"))
    with cpu_thread_limit(None):
        assert {p["filepath"]: p["num_threads"] for p in threadpool_info()} == snapshot
        assert dict(os.environ) == environment


def test_manual_limit_applies_and_restores_on_failure():
    import torch
    import scipy.linalg
    import sklearn
    from threadpoolctl import threadpool_info
    snapshot = {p["filepath"]: p["num_threads"] for p in threadpool_info()}
    original_torch = torch.get_num_threads()
    environment = dict(os.environ)
    with pytest.raises(RuntimeError, match="simulated"):
        with cpu_thread_limit(2):
            assert torch.get_num_threads() == 2
            assert all(p["num_threads"] == 2 for p in threadpool_info() if p["user_api"] == "blas")
            assert os.environ["OMP_NUM_THREADS"] == "2"
            raise RuntimeError("simulated run failure")
    assert torch.get_num_threads() == original_torch
    assert {p["filepath"]: p["num_threads"] for p in threadpool_info()} == snapshot
    assert dict(os.environ) == environment


def test_explicit_auto_overrides_saved_and_environment_preferences(monkeypatch):
    monkeypatch.setenv("CRYOID_CPU_THREADS", "4")
    monkeypatch.setenv("CRYOID_UNET_DEVICE", "cuda:1")
    assert resolve_cpu_threads("auto", {"cpu_threads": "8"}) is None
    assert resolve_device("auto", {"device": "cpu"}) is None
    assert resolve_cpu_threads(None, {"cpu_threads": "8"}) == 4
    assert resolve_device(None, {"device": "cpu"}) == "cuda:1"


@pytest.mark.parametrize("value", [0, -1, 1.5, True, "nan", "bad", 9])
def test_invalid_thread_counts_are_rejected(monkeypatch, value):
    monkeypatch.setattr("cryoid_pipeline.compute.available_cpu_threads", lambda: 8)
    with pytest.raises(ValueError, match="CPU threads"):
        validate_cpu_threads(value)


@pytest.mark.parametrize("value", ["gpu", "cuda:bad", "cuda:-1", "invalid"])
def test_invalid_devices_are_rejected(value):
    with pytest.raises(ValueError, match="Device"):
        validate_device(value)


@pytest.mark.parametrize("threads", [None, 4])
def test_blast_only_gets_thread_flag_for_manual_choice(monkeypatch, threads):
    from D_BLAST_searching.reduced_alphabet import search_support
    commands = []
    monkeypatch.setattr(search_support.subprocess, "Popen", lambda args, **kwargs: (
        commands.append(args) or SimpleNamespace(communicate=lambda: ("", ""), returncode=0)))
    params = SimpleNamespace(query="query.fa", database="database", outfile="hits.tsv", dbsize=100,
                             qsize=10, evalue=1.0, gapscore=(15, 3), max_target_seqs=20,
                             logger=logging.getLogger("test"), cpu_threads=threads)
    search_support.SeqBlast(params)
    if threads is None:
        assert "-num_threads" not in commands[0]
    else:
        assert commands[0][-2:] == ["-num_threads", "4"]


@pytest.fixture(scope="session")
def qt_app():
    from G_GUI.qt_compat import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_gui_compute_defaults_and_gpu_names(qt_app):
    from G_GUI.settings_dialogs import OptionsWindow
    default = OptionsWindow()
    selected = OptionsWindow(device="cuda:1", cpu_threads="4",
                             devices=[("cuda:0", "Test GPU A"), ("cuda:1", "Test GPU B")])
    try:
        assert default.device() == default.cpu_threads() == "auto"
        assert default.cpu_threads_edit.text() == ""
        assert default.cpu_threads_edit.placeholderText() == "Default: Auto (library defaults)"
        for text in ("Auto", "AUTO", "aUtO", ""):
            default.cpu_threads_edit.setText(text)
            assert default.cpu_threads() == "auto"
        default.cpu_threads_edit.setText("4")
        assert default.cpu_threads() == "4"
        assert selected.device() == "cuda:1" and selected.cpu_threads() == "4"
        assert "Test GPU B" in selected.device_combo.currentText()
        assert selected.device_combo.findData("cpu") >= 0
    finally:
        default.close()
        selected.close()
