"""Keep setup/regression tests independent of the developer's saved settings."""
import os
import sys

import pytest


@pytest.fixture(scope="session")
def qt_app():
    from G_GUI.qt_compat import QtWidgets
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(autouse=True)
def isolated_user_settings(monkeypatch, tmp_path):
    # Tests use small fake bundles; never fetch multi-GB production checkpoints.
    import huggingface_hub
    def unexpected_download(*args, **kwargs):
        pytest.fail("Unexpected live weights download; supply a fake Hub backend in this test.")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", unexpected_download)
    for name in list(os.environ):
        if name.startswith("CRYOID_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("CRYOID_CONFIG", str(tmp_path / "user-settings" / "config.toml"))
    from cryoid_pipeline import startup_config
    monkeypatch.setattr(startup_config, "find_weights_dir", lambda: None)
    monkeypatch.setattr(startup_config, "find_chimerax", lambda: None)
    gui = sys.modules.get("G_GUI.main_gui")
    if gui is not None:
        monkeypatch.setattr(gui, "REPO_ROOT", tmp_path)
