import sys
from PyQt6 import QtWidgets, QtGui
from gui_skin import Ui_cryoID


def apply_dark_palette(app: QtWidgets.QApplication) -> None:
    """Force a dark Fusion palette so the UI stays dark regardless of OS theme."""
    palette = QtGui.QPalette()
    # Main background: tweak Window/Base colors to change the overall background shade.
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#1e1e1e"))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#e2e8f0"))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#21252b"))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#282c34"))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor("#1e2127"))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor("#e2e8f0"))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#e2e8f0"))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#2f3540"))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#e2e8f0"))
    palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor("#ff7676"))
    palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor("#5d7bff"))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#347aff"))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#f4f6fb"))
    app.setPalette(palette)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")  # or "Windows"
    apply_dark_palette(app)  # ensures Fusion stays dark even if OS theme is light
    window = QtWidgets.QMainWindow()
    ui = Ui_cryoID()
    ui.setupUi(window)
    # Allow the Results dock to be closed and re-shown via View -> View Results.
    ui.actionResults_window.setCheckable(True)
    ui.actionResults_window.setChecked(True)
    ui.actionResults_window.triggered.connect(ui.dockWidget.setVisible)
    ui.dockWidget.visibilityChanged.connect(ui.actionResults_window.setChecked)
    ui.dockWidget.setWindowTitle("")
    # Results dock panel colors (overall box and inner background)
    ui.dockWidget.setStyleSheet("QDockWidget#dockWidget { background-color: #22272f; border: none; }")
    ui.dockWidgetContents.setStyleSheet(
        """
        QWidget#dockWidgetContents {
            background-color: #1f242b;
        }
        QWidget#dockWidgetContents QLabel {
            color: #e2e8f0;
            font-weight: 600;
        }
        QLabel#label_2 {
            font-weight: 700;
        }
        QWidget#dockWidgetContents QLineEdit,
        QWidget#dockWidgetContents QListWidget {
            background-color: #262b33;
            color: #e2e8f0;
            border: 1px solid #333944;
        }
        QWidget#dockWidgetContents QPushButton {
            background-color: #2f3540;
            color: #e2e8f0;
            border: 1px solid #3a404c;
            padding: 4px 8px;
        }
        QWidget#dockWidgetContents QPushButton:disabled {
            background-color: #242933;
            color: #7a818c;
        }
        """
    )
    # Menu bar colors: adjust the QMenuBar block below (background/color/border-bottom).
    # Results header row (path box + "Change folder"): tweak QLineEdit / QPushButton rules inside the dock CSS above.
    # Output box under Status: style QPlainTextEdit (or QWidget#widget) in the window stylesheet below to change its background/border.
    window.setStyleSheet(
        """
        QMainWindow, QWidget#centralwidgebel#protein_pool_file_label {
            font-weight: 600;
        }
        QMenuBar {
            background-color: #1f242b;
            color: #e2e8f0;
            border: none;
            border-bottom: 1px solid #12161d;
        }
        QMenuBar::item:selected {
            background-color: #2a303a;
        }
        QLabel#density_map_label,
        QLabel#protein_pool_file_label,
        QLabel#status_label {
            font-weight: 700;
        }
        QLineEdit {
            background-color: #262b33;
            color: #e2e8f0;
            border: 1px solid #333944;
        }
        QPlainTextEdit {
            background-color: #262b33;
            color: #e2e8f0;
            border: 1px solid #333944;
            selection-background-color: #347aff;
            selection-color: #f4f6fb;
        }
        QProgressBar {
            background-color: #262b33;
            color: #e2e8f0;
            border: 1px solid #333944;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #347aff;
        }
        QPushButton#run_button,
        QPushButton#abort_button {
            background-color: #2f6db5 !important;
            color: #f4f6fb;
            border: 1px solid #255a92;
            padding: 6px 14px;
        }
        QPushButton#run_button:hover,
        QPushButton#abort_button:hover {
            background-color: #377dd2 !important;
        }
        QPushButton#run_button:pressed,
        QPushButton#abort_button:pressed {
            background-color: #2b5c9c !important;
        }
        """
    )
    window.show()
    sys.exit(app.exec())
