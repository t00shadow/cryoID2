import os
from pathlib import Path
import subprocess
import sys

import pytest

from cryoid_pipeline.user_config import load_config, save_config
from G_GUI.local_config import gui_config_view
from cryoid_pipeline.user_config import ensure_config, resolve_run_settings


@pytest.mark.parametrize("key,value", [
    ("mode", "invalid-mode"), ("rcl", "not-a-number"), ("rcl", "nan"),
    ("rcl", "inf"), ("rcl", "-0.5"), ("protein-min-trace-length", "nan"),
    ("protein-min-trace-length", "1.5"), ("structure-selection", "2.5"),
    ("keep-tmp", "invalid-boolean"),
])
def test_invalid_cli_settings_leave_config_unchanged(tmp_path, key, value):
    config = tmp_path / "config.toml"
    save_config({"weights_dir": "keep-my-weights", "rcl": "0.25", "sequence_mode": "protein"}, config)
    before = config.read_bytes()
    result = subprocess.run([sys.executable, "-m", "cryoid_pipeline", "config", "--config", str(config),
                             "set", key, value], capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert f"Could not set {key}:" in result.stderr
    assert "Traceback" not in result.stderr
    assert config.read_bytes() == before


def test_valid_cli_values_are_normalized_without_environment_interference(tmp_path):
    config = tmp_path / "config.toml"
    env = dict(os.environ, CRYOID_RCL="invalid-environment-value")
    for key, value in (("mode", " RNA "), ("rcl", "2.5e-1")):
        result = subprocess.run([sys.executable, "-m", "cryoid_pipeline", "config", "--config", str(config),
                                 "set", key, value], env=env, capture_output=True, text=True, timeout=20)
        assert result.returncode == 0, result.stderr
    assert load_config(config)["sequence_mode"] == "rna"
    assert load_config(config)["rcl"] == "0.25"


@pytest.mark.parametrize("key,value,message", [("sequence_mode", "invalid-mode", "Sequence mode"),
                                               ("rcl", "not-a-number", "RCL")])
def test_cli_reports_invalid_saved_values_without_traceback(tmp_path, key, value, message, monkeypatch):
    monkeypatch.delenv("CRYOID_RCL", raising=False)
    config = tmp_path / "config.toml"
    save_config({key: value}, config)  # Represents a manually edited config file.
    result = subprocess.run([sys.executable, "-m", "cryoid_pipeline", "run", "--map", "missing.mrc", "--pool", "missing.fa",
                             "--config", str(config), "--project", str(tmp_path / "project")],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 2
    assert message in result.stderr
    assert "Traceback" not in result.stderr


def test_gui_loading_recovers_all_invalid_run_values_without_rewriting_file(tmp_path, monkeypatch):
    for name in ("CRYOID_RCL", "CRYOID_PROTEIN_MIN_TRACE_LENGTH", "CRYOID_STRUCTURE_SELECTION", "CRYOID_KEEP_TMP"):
        monkeypatch.delenv(name, raising=False)
    config = tmp_path / "config.toml"
    save_config({"sequence_mode": "invalid-mode", "rcl": "bad", "protein_min_trace_length": "1.5",
                 "structure_selection": "2.5", "keep_tmp": "bad", "weights_dir": "keep-my-weights"}, config)
    before = config.read_bytes()
    local = gui_config_view(ensure_config(config))
    resolved, issues = resolve_run_settings(load_config(config))
    assert local["weights_path"] == str(tmp_path / "keep-my-weights")
    assert resolved == {"device": None, "cpu_threads": None, "rcl": 0.0, "sequence_mode": "auto", "protein_min_trace_length": 10,
                        "structure_selection": None, "keep_tmp": False}
    assert set(issues) == set(resolved) - {"device", "cpu_threads"}
    assert all("Options" in issue and "invalid" in issue for issue in issues.values())
    assert config.read_bytes() == before


def test_gui_loading_reports_invalid_environment_overrides(monkeypatch):
    monkeypatch.setenv("CRYOID_RCL", "nan")
    monkeypatch.setenv("CRYOID_KEEP_TMP", "invalid-boolean")
    resolved, issues = resolve_run_settings({"rcl": "0.25", "keep_tmp": "true"})
    assert resolved["rcl"] == 0 and resolved["keep_tmp"] is False
    assert "rcl" in issues and "keep_tmp" in issues
