from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

from G_GUI.gui_skin import CryoIDUI
from G_GUI.main_gui import apply_theme, set_run_button_state
from G_GUI.qt_compat import QtGui, QtWidgets
from G_GUI.settings_dialogs import ConfigurationsWindow, OptionsWindow


@pytest.mark.parametrize("font_size", [10, 12])
@pytest.mark.parametrize("initial_theme", ["light", "dark"])
def test_theme_changes_preserve_control_fonts_and_sizes(qt_app, font_size, initial_theme):
    app = qt_app
    old_font, old_palette, old_css = app.font(), app.palette(), app.styleSheet()
    font = QtGui.QFont(old_font)
    font.setPointSize(font_size)
    app.setFont(font)
    window = QtWidgets.QMainWindow()
    ui = CryoIDUI()
    ui.setup(window)
    assert not ui.results_section.findChildren(QtWidgets.QListWidget)
    assert [button.text() for button in ui.results_section.findChildren(QtWidgets.QPushButton)] == [
        "Sequence ranking", "Model"
    ]
    dialogs = [ConfigurationsWindow(window), OptionsWindow(window)]

    def settle():
        for _ in range(4):
            app.processEvents()

    def control_sizes():
        return [(widget.font().pointSizeF(), widget.size().width(), widget.size().height())
                for parent in [window, *dialogs]
                for widget in parent.findChildren(QtWidgets.QPushButton)
                if widget.isVisible()]

    try:
        apply_theme(app, window, ui, initial_theme)
        set_run_button_state(ui, running=False)
        window.show()
        for dialog in dialogs:
            dialog.show()
        settle()
        expected = control_sizes()
        expected_window_size = window.size()
        assert ui.options_button.font().pointSizeF() == font_size
        assert ui.density_map_edit.font().pointSizeF() == 10
        assert ui.run_button.font().pointSizeF() == 16
        for theme in ["dark", "light"] * 10:
            apply_theme(app, window, ui, theme)
            settle()
            assert control_sizes() == expected
            assert window.size() == expected_window_size
        set_run_button_state(ui, running=True)
        for theme in ("dark", "light"):
            apply_theme(app, window, ui, theme)
            settle()
            assert control_sizes() == expected
        set_run_button_state(ui, running=True, cancelling=True)
        assert ui.run_button.text() == "Abort"
        assert not ui.run_button.isEnabled()
        for theme in ("dark", "light"):
            apply_theme(app, window, ui, theme)
            settle()
            assert control_sizes() == expected
    finally:
        for dialog in dialogs:
            dialog.close()
        window.close()
        app.setStyleSheet(old_css)
        app.setPalette(old_palette)
        app.setFont(old_font)


@pytest.mark.parametrize("initial_theme", ["light", "dark"])
def test_theme_menu_label_updates_and_persists_after_toggle(qt_app, monkeypatch, initial_theme):
    import sys
    import time
    from types import SimpleNamespace
    from G_GUI import main_gui as gui
    from G_GUI.diagnostics import Check, Diagnostics

    preferences = {"theme": initial_theme}
    monkeypatch.setattr(gui.QtCore, "QSettings", lambda *args: SimpleNamespace(
        value=lambda key, default: preferences.get(key, default),
        setValue=lambda key, value: preferences.update({key: value}),
    ))
    ready = Check("OK", "Available", True)
    monkeypatch.setattr(gui, "collect_diagnostics", lambda *args: Diagnostics(ready, ready, ready, "cpu"))
    errors = []
    monkeypatch.setattr(sys, "excepthook", lambda kind, error, tb: errors.append(error))
    old_font, old_palette, old_css = qt_app.font(), qt_app.palette(), qt_app.styleSheet()

    def launch(toggle):
        existing = set(qt_app.topLevelWidgets())

        def execute(app):
            window = next(w for w in app.topLevelWidgets()
                          if isinstance(w, gui.CryoIDMainWindow) and w not in existing)
            try:
                assert window.windowTitle() == app.applicationDisplayName() == "cryoidv2"
                action = next(a for a in window.findChildren(QtGui.QAction)
                              if a.text() in {"Light mode", "Dark mode"})

                def check():
                    expected = "Light mode" if preferences["theme"] == "dark" else "Dark mode"
                    assert action.text() == action.toolTip() == expected
                    assert action.isChecked() == (preferences["theme"] == "dark")

                check()
                if toggle:
                    action.trigger()
                    assert preferences["theme"] != initial_theme
                    check()
                # Finish startup callbacks and worker cleanup before closing.
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    app.processEvents()
                    if window.isVisible() and not any(
                        t.isRunning() for t in window.findChildren(gui.DiagnosticsWorker)
                    ):
                        break
                    time.sleep(.001)
                else:
                    pytest.fail("GUI theme test did not finish startup")
                assert not errors, errors
            finally:
                window.close()
                window.deleteLater()
                app.sendPostedEvents(None, gui.QtCore.QEvent.Type.DeferredDelete)
            return 0

        monkeypatch.setattr(QtWidgets.QApplication, "exec", execute)
        with pytest.raises(SystemExit) as exited:
            gui.main()
        assert exited.value.code == 0

    try:
        launch(toggle=True)
        launch(toggle=False)
    finally:
        qt_app.setStyleSheet(old_css)
        qt_app.setPalette(old_palette)
        qt_app.setFont(old_font)
