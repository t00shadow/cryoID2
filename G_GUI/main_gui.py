import sys
from PyQt6 import QtWidgets, QtGui
from gui_skin import Ui_cryoID

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")  # or "Windows"
    window = QtWidgets.QMainWindow()
    ui = Ui_cryoID()
    ui.setupUi(window)
    # Darken the Results dock area while keeping black text inside
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
    window.setStyleSheet(
        """
        QMainWindow, QWidget#centralwidget {
            background-color: #252a31;
            color: #e2e8f0;
        }
        QMenuBar {
            background-color: #1d2229;
            border: none;
            border-bottom: 1px solid #121418;
        }
        QMenuBar::item {
            padding: 4px 8px;
        }
        QMenuBar::item:selected {
            background: #2a3038;
        }
        QLabel#density_map_label,
        QLabel#protein_pool_file_label {
            font-weight: 600;
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
