from __future__ import annotations

from pathlib import Path

from G_GUI.qt_compat import QtCore, QtGui, QtWidgets

from G_GUI.diagnostics import check_weights
from G_GUI.setup_dialogs import show_validation_issues
from cryoid_pipeline.structure_selection import default_structure_selection
from cryoid_pipeline.user_config import (
    validate_config_value,
    validate_protein_min_trace_length,
    validate_structure_selection,
)


class _SettingsDialog(QtWidgets.QDialog):
    def _add_buttons(self, layout) -> None:
        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        cancel_button = QtWidgets.QPushButton("Cancel", self)
        save_button = QtWidgets.QPushButton("Save", self)
        cancel_button.clicked.connect(self.reject)
        save_button.clicked.connect(self.accept)
        row.addWidget(cancel_button)
        row.addWidget(save_button)
        layout.addLayout(row)


class ConfigConflictDialog(QtWidgets.QDialog):
    def __init__(self, conflict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Saved settings changed")
        self.action = "cancel"
        layout = QtWidgets.QVBoxLayout(self)
        text = f"These settings changed outside this window:\n{conflict.path}\n\n" + "\n".join(
            f"{key.replace('_', ' ')}: saved {conflict.saved[key]!r}; your value {conflict.changes[key]!r}"
            for key in conflict.keys)
        label = QtWidgets.QLabel(text, self)
        label.setWordWrap(True)
        layout.addWidget(label)
        buttons = QtWidgets.QHBoxLayout()
        for title, action in (("Reload saved settings", "reload"), ("Keep my edits", "keep"), ("Cancel", "cancel")):
            button = QtWidgets.QPushButton(title, self)
            button.clicked.connect(lambda checked=False, action=action: self.choose(action))
            buttons.addWidget(button)
        layout.addLayout(buttons)

    def choose(self, action):
        self.action = action
        self.accept()


class ConfigurationsWindow(_SettingsDialog):
    """Installation paths shared by all projects."""

    def __init__(self, parent=None, *, weights_path="", chimerax_path=""):
        super().__init__(parent)
        self.setWindowTitle("Configurations")
        self.setModal(True)
        self.setMinimumWidth(520)
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.addWidget(QtWidgets.QLabel("These paths apply to all projects.", self))
        form_layout = QtWidgets.QFormLayout()

        self.weights_path_edit = QtWidgets.QLineEdit(self)
        self.weights_path_edit.setObjectName("weights_path_edit")
        self.weights_path_edit.setPlaceholderText("Default: Autodownload published weights.")
        self.weights_path_edit.setToolTip("Default: Autodownload published weights.")
        self.weights_path_edit.setText(weights_path)

        weights_row = QtWidgets.QWidget(self)
        weights_layout = QtWidgets.QHBoxLayout(weights_row)
        weights_layout.setContentsMargins(0, 0, 0, 0)
        weights_layout.addWidget(self.weights_path_edit)
        self.browse_weights_button = QtWidgets.QPushButton("Browse", weights_row)
        self.browse_weights_button.clicked.connect(self._browse_weights)
        weights_layout.addWidget(self.browse_weights_button)
        form_layout.addRow("Custom weights path", weights_row)

        self.chimerax_path_edit = QtWidgets.QLineEdit(self)
        self.chimerax_path_edit.setObjectName("chimerax_path_edit")
        self.chimerax_path_edit.setPlaceholderText("Path to ChimeraX executable")
        self.chimerax_path_edit.setText(chimerax_path)

        chimerax_row = QtWidgets.QWidget(self)
        chimerax_layout = QtWidgets.QHBoxLayout(chimerax_row)
        chimerax_layout.setContentsMargins(0, 0, 0, 0)
        chimerax_layout.addWidget(self.chimerax_path_edit)
        self.browse_chimerax_button = QtWidgets.QPushButton("Browse", chimerax_row)
        self.browse_chimerax_button.clicked.connect(self._browse_chimerax)
        chimerax_layout.addWidget(self.browse_chimerax_button)
        form_layout.addRow("ChimeraX executable", chimerax_row)

        browse_icon = QtGui.QIcon(str(Path(__file__).parent / "svg_assets" / "upload-sign-svgrepo-com.svg"))
        if browse_icon.isNull():
            standard = getattr(QtWidgets.QStyle, "StandardPixmap", QtWidgets.QStyle)
            browse_icon = self.style().standardIcon(standard.SP_DialogOpenButton)
        button_size = max(28, QtGui.QFontMetrics(self.font()).height() + 8)
        for button, tooltip in (
            (self.browse_weights_button, "Browse for weights directory"),
            (self.browse_chimerax_button, "Browse for ChimeraX executable"),
        ):
            button.setText("")
            button.setIcon(browse_icon)
            button.setIconSize(QtCore.QSize(16, 16))
            button.setFixedSize(button_size, button_size)
            button.setToolTip(tooltip)
            button.setAccessibleName(tooltip)
            button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        main_layout.addLayout(form_layout)
        self._add_buttons(main_layout)

    def weights_path(self) -> str:
        return self.weights_path_edit.text().strip()

    def chimerax_path(self) -> str:
        return self.chimerax_path_edit.text().strip()

    def _browse_weights(self) -> None:
        start_dir = str(Path(self.weights_path()).expanduser()) if self.weights_path() else ""
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select UNet weights directory",
            start_dir,
        )
        if directory:
            self.weights_path_edit.setText(directory)

    def _browse_chimerax(self) -> None:
        start_dir = str(Path(self.chimerax_path()).expanduser().parent) if self.chimerax_path() else ""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select ChimeraX executable",
            start_dir,
            "Executables (*.exe *);;All files (*)",
        )
        if filename:
            self.chimerax_path_edit.setText(filename)

    def accept(self) -> None:
        issues = []
        invalid_fields = []
        for title, field, check in (
            ("U-Net weights", self.weights_path_edit, check_weights(self.weights_path())),
        ):
            if not check.ok:
                issues.append(f"{title}: {check.detail}")
                invalid_fields.append(field)
        from G_GUI.chimerax_launcher import chimerax_executable
        if self.chimerax_path() and not chimerax_executable(self.chimerax_path()).is_file():
            issues.append("ChimeraX: The executable was not found. Choose the ChimeraX program file, "
                          "or leave this optional field empty if you do not use the model viewer.")
            invalid_fields.append(self.chimerax_path_edit)
        if issues:
            show_validation_issues(issues, self)
            invalid_fields[0].setFocus()
            return
        if self.weights_path():
            self.weights_path_edit.setText(str(Path(self.weights_path()).expanduser().resolve()))
        super().accept()


class OptionsWindow(_SettingsDialog):
    """Run settings, independent of filesystem configuration."""

    def __init__(self, parent=None, *, rcl=0.0, sequence_mode="auto",
                 protein_min_trace_length=10, structure_selection="", keep_tmp=False,
                 device="auto", cpu_threads="auto", devices=()):
        super().__init__(parent)
        self.setWindowTitle("Options")
        self.setModal(True)
        self.setMinimumWidth(520)
        main_layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()

        self.rcl_edit = QtWidgets.QLineEdit(self)
        self.rcl_edit.setObjectName("rcl_edit")
        self.rcl_edit.setPlaceholderText("Default: 0.000")
        self.rcl_edit.setValidator(QtGui.QDoubleValidator(0.0, 1_000_000.0, 3, self))
        rcl_value = float(rcl)
        self.rcl_edit.setText("" if rcl_value == 0.0 else f"{rcl_value:.3f}")
        self.rcl_edit.setToolTip("Contour for U-Net predictions")

        self.sequence_mode_combo = QtWidgets.QComboBox(self)
        self.sequence_mode_combo.setObjectName("sequence_mode_combo")
        self.sequence_mode_combo.addItem("Auto", "auto")
        self.sequence_mode_combo.addItem("Protein", "protein")
        self.sequence_mode_combo.addItem("RNA", "rna")
        mode_index = self.sequence_mode_combo.findData(str(sequence_mode or "auto").lower())
        self.sequence_mode_combo.setCurrentIndex(max(0, mode_index))
        self.sequence_mode_combo.setToolTip("Choose protein, RNA, or auto")

        self.protein_min_trace_length_edit = QtWidgets.QLineEdit(self)
        self.protein_min_trace_length_edit.setObjectName("protein_min_trace_length_edit")
        self.protein_min_trace_length_edit.setPlaceholderText("Default: 10")
        self.protein_min_trace_length_edit.setValidator(
            QtGui.QIntValidator(1, 100_000, self)
        )
        trace_length_value = int(protein_min_trace_length)
        self.protein_min_trace_length_edit.setText(
            "" if trace_length_value == 10 else str(trace_length_value)
        )
        self.protein_min_trace_length_edit.setToolTip("Minimum protein query length")

        self.structure_selection_edit = QtWidgets.QLineEdit(self)
        self.structure_selection_edit.setObjectName("structure_selection_edit")
        self.structure_selection_edit.setText(str(structure_selection or ""))
        protein_default = default_structure_selection("protein")
        rna_default = default_structure_selection("rna")
        self.structure_selection_edit.setPlaceholderText(
            f"Default: protein {protein_default.raw}; RNA {rna_default.raw} ({rna_default.numeric:.0%})"
        )
        self.structure_selection_edit.setToolTip("Candidate count or fraction")

        self.keep_tmp_checkbox = QtWidgets.QCheckBox("Keep files and tested models", self)
        self.keep_tmp_checkbox.setObjectName("keep_tmp_checkbox")
        self.keep_tmp_checkbox.setChecked(bool(keep_tmp))
        self.keep_tmp_checkbox.setToolTip("Keep temporary processing files")
        form_layout.addRow("Sequence mode", self.sequence_mode_combo)
        form_layout.addRow(
            "Protein minimum trace length", self.protein_min_trace_length_edit
        )
        form_layout.addRow("Structure candidates", self.structure_selection_edit)
        form_layout.addRow("Recommended contour level (RCL)", self.rcl_edit)
        form_layout.addRow("Temporary outputs", self.keep_tmp_checkbox)
        self.sequence_mode_combo.currentIndexChanged.connect(self._update_mode_controls)
        self._update_mode_controls()

        main_layout.addLayout(form_layout)
        compute_box = QtWidgets.QGroupBox("Compute", self)
        compute_form = QtWidgets.QFormLayout(compute_box)
        self.device_combo = QtWidgets.QComboBox(compute_box)
        self.device_combo.setObjectName("device_combo")
        self.device_combo.addItem("Auto", "auto")
        self.device_combo.addItem("CPU", "cpu")
        for identifier, name in devices:
            label = name if identifier == "mps" else f"GPU {identifier.split(':')[-1]}: {name}"
            self.device_combo.addItem(label, identifier)
        selected_device = validate_config_value("device", device)
        if selected_device == "cuda":
            cuda_devices = [
                identifier for identifier, _ in devices
                if identifier.startswith("cuda:")
            ]
            if cuda_devices:
                selected_device = cuda_devices[0]
        if self.device_combo.findData(selected_device) < 0:
            index = selected_device.split(":")[-1] if ":" in selected_device else "0"
            label = (
                "Apple GPU (unavailable)"
                if selected_device == "mps"
                else f"GPU {index} (unavailable)"
            )
            self.device_combo.addItem(label, selected_device)
        self.device_combo.setCurrentIndex(self.device_combo.findData(selected_device))
        self.device_combo.setToolTip("Choose inference compute device")
        self.cpu_threads_edit = QtWidgets.QLineEdit(compute_box)
        self.cpu_threads_edit.setObjectName("cpu_threads_edit")
        self.cpu_threads_edit.setPlaceholderText("Default: Auto (library defaults)")
        threads = validate_config_value("cpu_threads", cpu_threads)
        self.cpu_threads_edit.setText("" if threads == "auto" else threads)
        self.cpu_threads_edit.setToolTip("Enter Auto or thread count")
        compute_form.addRow("Device", self.device_combo)
        compute_form.addRow("CPU threads", self.cpu_threads_edit)
        main_layout.addWidget(compute_box)
        self._add_buttons(main_layout)

    def rcl(self) -> float:
        return float(validate_config_value("rcl", self.rcl_edit.text().strip()))

    def sequence_mode(self) -> str:
        return str(self.sequence_mode_combo.currentData() or "auto")

    def structure_selection(self) -> str:
        return self.structure_selection_edit.text().strip()

    def protein_min_trace_length(self) -> int:
        return validate_protein_min_trace_length(
            self.protein_min_trace_length_edit.text().strip() or None
        )

    def keep_tmp(self) -> bool:
        return bool(self.keep_tmp_checkbox.isChecked())

    def device(self) -> str:
        return str(self.device_combo.currentData() or "auto")

    def cpu_threads(self) -> str:
        return validate_config_value("cpu_threads", self.cpu_threads_edit.text().strip())

    def _update_mode_controls(self) -> None:
        self.protein_min_trace_length_edit.setEnabled(self.sequence_mode() != "rna")

    def accept(self) -> None:
        try:
            validate_config_value("device", self.device())
            threads = self.cpu_threads()
        except ValueError as exc:
            show_validation_issues([str(exc)], self)
            return
        self.cpu_threads_edit.setText("" if threads == "auto" else threads)
        try:
            validated_rcl = self.rcl()
        except ValueError:
            QtWidgets.QMessageBox.warning(self, "Invalid RCL", "Enter a contour level of 0 or greater, such as 0.05. Leave the field empty to use 0.")
            return
        self.rcl_edit.setText("" if validated_rcl == 0.0 else f"{validated_rcl:.3f}")
        try:
            validated_trace_length = self.protein_min_trace_length()
        except ValueError:
            QtWidgets.QMessageBox.warning(
                self, "Invalid protein minimum trace length", "Enter a whole number of 1 or greater. Leave the field empty to use 10."
            )
            return
        self.protein_min_trace_length_edit.setText(
            "" if validated_trace_length == 10 else str(validated_trace_length)
        )
        if self.structure_selection():
            try:
                validated_selection = validate_structure_selection(
                    self.structure_selection()
                )
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Invalid structure selection",
                                              "Enter a positive whole number of candidates (for example, 5), "
                                              "or a fraction greater than 0 and at most 1.0 (for example, 0.5 for 50%).")
                return
            self.structure_selection_edit.setText(validated_selection)
        super().accept()
