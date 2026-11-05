"""Hardware selection is deterministic across old/new hosts, including failures."""
import json
from types import SimpleNamespace

import pytest

from cryoid_pipeline import install_hardware as hardware


@pytest.mark.parametrize("target,driver,capabilities,libc,expected", [
    ("linux-64", 11020, [[6, 1]], ("glibc", "2.17"), "cu118"),
    ("linux-64", 13000, [[6, 1]] * 4, ("glibc", "2.40"), "cu118"),
    ("win-64", 12080, [[8, 9]], None, "cu118"),
    ("linux-64", 12080, [[10, 0], [12, 0]], ("glibc", "2.28"), "cu128"),
    ("win-64", 13000, [[12, 0]], None, "cu128"),
    ("linux-64", 12080, [[12, 0]], ("glibc", "2.17"), "cpu"),
    ("linux-64", 12080, [[12, 0]], ("musl", "1.2"), "cpu"),
    ("linux-64", 12080, [[12, 0]], ("glibc", "unknown"), "cpu"),
    ("win-64", 12060, [[12, 0]], None, "cpu"),
    ("linux-64", 10020, [[6, 1]], ("glibc", "2.17"), "cpu"),
    ("linux-64", 13000, [[6, 1], [12, 0]], ("glibc", "2.40"), "cpu"),
    ("linux-64", 13000, [[99, 0]], ("glibc", "2.40"), "cpu"),
    ("linux-64", 13000, [[3, 5]], ("glibc", "2.40"), "cpu"),
    ("linux-64", 0, [], ("glibc", "2.17"), "cpu"),
    ("osx-arm64", 0, [], None, "conda-cpu"),
    ("osx-64", 0, [], None, "conda-cpu"),
])
def test_platform_driver_and_visible_gpu_matrix(target, driver, capabilities, libc, expected):
    probe = {"driver_api": driver, "devices": [{"capability": capability} for capability in capabilities]}
    profile = hardware.select_torch_profile(target, probe, libc)
    assert profile.name == expected
    assert profile.cuda == expected.startswith("cu")
    assert profile.reason


@pytest.mark.parametrize("visible", ["", "-1"])
def test_hidden_devices_are_not_used(monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(hardware.subprocess, "run", lambda *args, **kwargs: pytest.fail("Hidden GPUs were probed"))
    assert hardware.probe_cuda()["devices"] == []


@pytest.mark.parametrize("failure", ["timeout", "missing", "crash", "malformed", "invalid-device", "invalid-driver"])
def test_driver_failure_or_invalid_probe_response_selects_cpu(monkeypatch, failure):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    def run(command, **kwargs):
        if failure == "timeout":
            raise hardware.subprocess.TimeoutExpired(command, 20)
        if failure == "missing":
            raise OSError("No driver")
        if failure == "crash":
            return SimpleNamespace(returncode=-11)
        if failure == "malformed":
            return SimpleNamespace(returncode=0, stdout="not JSON")
        payload = {"devices": [{"capability": [6, 1]}], "driver_api": 11020}
        if failure == "invalid-device":
            payload["devices"][0]["capability"] = ["unknown", 0]
        else:
            payload["driver_api"] = "unknown"
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload))
    monkeypatch.setattr(hardware.subprocess, "run", run)
    probe = hardware.probe_cuda()
    assert hardware.select_torch_profile("linux-64", probe).name == "cpu"
    assert probe["reason"]


def test_driver_probe_uses_same_visibility_and_no_foreign_python_path(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-selected-by-scheduler")
    monkeypatch.setenv("PYTHONPATH", "foreign-scientific-software")
    payload = {"driver_api": 11020, "devices": [{"capability": [6, 1]}]}
    def run(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-selected-by-scheduler"
        assert "PYTHONPATH" not in kwargs["env"]
        assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
        assert kwargs["timeout"] == 20
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload))
    monkeypatch.setattr(hardware.subprocess, "run", run)
    assert hardware.probe_cuda() == payload
