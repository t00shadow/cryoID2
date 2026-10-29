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
def test_theme_changes_preserve_control_fonts_and_sizes(font_size, initial_theme):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
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
