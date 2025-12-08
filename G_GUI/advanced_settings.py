from PyQt6 import QtWidgets, QtGui


class AdvancedSettingsWindow(QtWidgets.QWidget):
    """Lightweight advanced settings window (spawned from More options -> Advanced Settings)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Settings")
        self.setMinimumWidth(360)
        layout = QtWidgets.QFormLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        self.map_resolution = QtWidgets.QLineEdit()
        self.raster_radius = QtWidgets.QLineEdit()
        self.chimerax_path = QtWidgets.QLineEdit()
        layout.addRow("Map Resolution", self.map_resolution)
        layout.addRow("Rasterization Radius", self.raster_radius)
        layout.addRow("ChimeraX Path", self.chimerax_path)

        # Match the dark theme used in main_gui.
        self.setStyleSheet(
            """
            QWidget {
                background-color: #1f242b;
                color: #e2e8f0;
                font-weight: 600;
            }
            QLineEdit {
                background-color: #262b33;
                color: #e2e8f0;
                border: 1px solid #333944;
            }
            QLabel {
                color: #e2e8f0;
                font-weight: 700;
            }
            """
        )
