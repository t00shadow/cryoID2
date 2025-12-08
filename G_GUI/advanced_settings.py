from PyQt6 import QtGui, QtWidgets


class AdvancedSettingsWindow(QtWidgets.QDialog):
    """Standalone window for advanced cryoID settings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Advanced Settings")
        self.setModal(False)
        self.setMinimumWidth(340)

        main_layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()

        self.resolution_edit = QtWidgets.QLineEdit(self)
        self.resolution_edit.setObjectName("resolution_edit")
        self.resolution_edit.setPlaceholderText("e.g., 3.5 (Å)")
        self.resolution_edit.setValidator(QtGui.QDoubleValidator(0.0, 999.0, 3))
        form_layout.addRow("Resolution", self.resolution_edit)

        self.rasterization_radius_edit = QtWidgets.QLineEdit(self)
        self.rasterization_radius_edit.setObjectName("rasterization_radius_edit")
        self.rasterization_radius_edit.setPlaceholderText("e.g., 3.0 (Å)")
        self.rasterization_radius_edit.setValidator(QtGui.QDoubleValidator(0.0, 999.0, 3))
        form_layout.addRow("Rasterization radius", self.rasterization_radius_edit)

        main_layout.addLayout(form_layout)
