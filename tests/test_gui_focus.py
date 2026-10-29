from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from G_GUI.main_gui import _clear_line_edit_highlights
from G_GUI.qt_compat import QtWidgets


def test_clear_line_edit_highlights_moves_focus_to_neutral_widget() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    window = QtWidgets.QWidget()
    layout = QtWidgets.QVBoxLayout(window)
    density_map = QtWidgets.QLineEdit("density.map")
    sequence_pool = QtWidgets.QLineEdit("pool.fasta")
    output_path = QtWidgets.QLineEdit("output")
    neutral = QtWidgets.QWidget()
    for widget in (density_map, sequence_pool, output_path, neutral):
        layout.addWidget(widget)
    window.show()

    density_map.setFocus()
    density_map.selectAll()
    app.processEvents()

    _clear_line_edit_highlights(
        (density_map, sequence_pool, output_path),
        neutral,
    )
    app.processEvents()

    assert density_map.selectedText() == ""
    assert sequence_pool.selectedText() == ""
    assert output_path.selectedText() == ""
    assert app.focusWidget() is neutral

    window.close()
