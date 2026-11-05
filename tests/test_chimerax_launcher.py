import os
from unittest.mock import patch

import pytest

from G_GUI.chimerax_launcher import find_chimerax


def test_chimerax_uses_current_override_and_explicit_path_priority(tmp_path):
    configured, explicit = tmp_path / "configured.exe", tmp_path / "explicit.exe"
    configured.write_bytes(b"program")
    explicit.write_bytes(b"program")
    with patch.dict(os.environ, {"CRYOID_CHIMERAX_PATH": str(configured)}, clear=True):
        assert find_chimerax() == configured
        assert find_chimerax(explicit) == explicit


def test_retired_chimerax_override_is_not_an_alias(tmp_path):
    retired = tmp_path / "retired.exe"
    retired.write_bytes(b"program")
    with (
        patch.dict(os.environ, {"CHIMERAX_PATH": str(retired)}, clear=True),
        patch("cryoid_pipeline.startup_config.platform.system", return_value="Linux"),
        patch("cryoid_pipeline.startup_config.shutil.which", return_value=None),
    ):
        with pytest.raises(FileNotFoundError, match="CRYOID_CHIMERAX_PATH"):
            find_chimerax()


def test_macos_app_bundle_can_be_configured_and_launched(tmp_path):
    from G_GUI.chimerax_launcher import launch_chimerax
    bundle = tmp_path / "ChimeraX 1.10.app"
    executable = bundle / "Contents" / "MacOS" / "ChimeraX"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"program")
    model = tmp_path / "model with spaces.cif"
    with patch("G_GUI.chimerax_launcher.subprocess.Popen") as popen:
        command = launch_chimerax([model], chimerax_path=bundle)
    assert command == [str(executable), str(model)]
    popen.assert_called_once_with(command)


def test_macos_discovers_user_applications(tmp_path, monkeypatch):
    import cryoid_pipeline.startup_config as launcher
    from G_GUI.chimerax_launcher import find_chimerax as original_find
    executable = tmp_path / "Applications" / "ChimeraX.app" / "Contents" / "MacOS" / "ChimeraX"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"program")
    monkeypatch.delenv("CRYOID_CHIMERAX_PATH", raising=False)
    monkeypatch.setattr(launcher.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(launcher.shutil, "which", lambda _: None)
    monkeypatch.setattr(launcher.Path, "home", lambda: tmp_path)

    original_glob = launcher.Path.glob

    def isolated_glob(path, pattern):
        if str(path) == "/Applications":
            return iter(())
        return original_glob(path, pattern)

    monkeypatch.setattr(launcher.Path, "glob", isolated_glob)

    with patch.dict(os.environ, {}, clear=True):
        assert original_find() == executable
