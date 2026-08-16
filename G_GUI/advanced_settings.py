from __future__ import annotations

from pathlib import Path

try:
    from qt_compat import QtGui, QtWidgets
except ImportError:
    from G_GUI.qt_compat import QtGui, QtWidgets

from D_BLAST_searching.stage import validate_blast_bin_dir
from cryoid_pipeline.user_config import (
    resolve_rcl,
    validate_protein_min_trace_length,
    validate_structure_selection,
)


class AdvancedSettingsWindow(QtWidgets.QDialog):
    """Standalone window for advanced CryoID settings."""

    def __init__(
        self,
        parent=None,
        *,
        weights_path: str = "",
        rcl: float = 0.0,
        chimerax_path: str = "",
        detected_chimerax_path: str = "",
        blast_bin_dir: str = "",
        detected_blast_bin_dir: str = "",
        sequence_mode: str = "auto",
        protein_min_trace_length: int = 10,
        structure_selection: str = "",
        keep_tmp: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced Settings")
        self.setModal(True)
        self.setMinimumWidth(520)

        main_layout = QtWidgets.QVBoxLayout(self)
        form_layout = QtWidgets.QFormLayout()

        self.weights_path_edit = QtWidgets.QLineEdit(self)
        self.weights_path_edit.setObjectName("weights_path_edit")
        self.weights_path_edit.setPlaceholderText("Path to UNet weights directory")
        self.weights_path_edit.setText(weights_path)

        weights_row = QtWidgets.QWidget(self)
        weights_layout = QtWidgets.QHBoxLayout(weights_row)
        weights_layout.setContentsMargins(0, 0, 0, 0)
        weights_layout.addWidget(self.weights_path_edit)
        self.browse_weights_button = QtWidgets.QPushButton("Browse", weights_row)
        self.browse_weights_button.clicked.connect(self._browse_weights)
        weights_layout.addWidget(self.browse_weights_button)
        form_layout.addRow("Weights path", weights_row)

        self.rcl_edit = QtWidgets.QLineEdit(self)
        self.rcl_edit.setObjectName("rcl_edit")
        self.rcl_edit.setPlaceholderText("Default: 0.000")
        self.rcl_edit.setValidator(QtGui.QDoubleValidator(0.0, 1_000_000.0, 3, self))
        rcl_value = float(rcl)
        self.rcl_edit.setText("" if rcl_value == 0.0 else f"{rcl_value:.3f}")
        self.rcl_edit.setToolTip(
            "Recommended contour level used only for fresh UNet predictions. "
            "Zero is used literally; the archived EMD-47655 RNA test used 0.05."
        )

        self.chimerax_path_edit = QtWidgets.QLineEdit(self)
        self.chimerax_path_edit.setObjectName("chimerax_path_edit")
        self.chimerax_path_edit.setPlaceholderText("Path to ChimeraX executable")
        self.chimerax_path_edit.setText(chimerax_path or detected_chimerax_path)

        chimerax_row = QtWidgets.QWidget(self)
        chimerax_layout = QtWidgets.QHBoxLayout(chimerax_row)
        chimerax_layout.setContentsMargins(0, 0, 0, 0)
        chimerax_layout.addWidget(self.chimerax_path_edit)
        self.browse_chimerax_button = QtWidgets.QPushButton("Browse", chimerax_row)
        self.browse_chimerax_button.clicked.connect(self._browse_chimerax)
        chimerax_layout.addWidget(self.browse_chimerax_button)
        form_layout.addRow("ChimeraX executable", chimerax_row)

        self.sequence_mode_combo = QtWidgets.QComboBox(self)
        self.sequence_mode_combo.setObjectName("sequence_mode_combo")
        self.sequence_mode_combo.addItem("Auto", "auto")
        self.sequence_mode_combo.addItem("Protein", "protein")
        self.sequence_mode_combo.addItem("RNA", "rna")
        mode_index = self.sequence_mode_combo.findData(str(sequence_mode or "auto").lower())
        self.sequence_mode_combo.setCurrentIndex(max(0, mode_index))
        self.sequence_mode_combo.setToolTip(
            "Auto inspects every candidate FASTA record before UNet inference."
        )

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
        self.protein_min_trace_length_edit.setToolTip(
            "Minimum number of connected predicted C-alpha nodes retained as a protein "
            "sequence query. The default is 10."
        )

        self.blast_bin_dir_edit = QtWidgets.QLineEdit(self)
        self.blast_bin_dir_edit.setObjectName("blast_bin_dir_edit")
        self.blast_bin_dir_edit.setPlaceholderText("Folder containing blastp, blastn, and makeblastdb")
        self.blast_bin_dir_edit.setText(blast_bin_dir or detected_blast_bin_dir)
        self.blast_bin_dir_edit.setToolTip(
            "Select the BLAST+ bin folder. CryoID also checks CRYOID_BLAST_BIN_DIR, "
            "the active Conda/Python environment, PATH, and common installation folders."
        )
        blast_row = QtWidgets.QWidget(self)
        blast_layout = QtWidgets.QHBoxLayout(blast_row)
        blast_layout.setContentsMargins(0, 0, 0, 0)
        blast_layout.addWidget(self.blast_bin_dir_edit)
        self.browse_blast_button = QtWidgets.QPushButton("Browse", blast_row)
        self.browse_blast_button.clicked.connect(self._browse_blast)
        blast_layout.addWidget(self.browse_blast_button)

        self.structure_selection_edit = QtWidgets.QLineEdit(self)
        self.structure_selection_edit.setObjectName("structure_selection_edit")
        self.structure_selection_edit.setText(str(structure_selection or ""))
        self.structure_selection_edit.setPlaceholderText(
            "Default: protein 1; RNA 0.25"
        )
        self.structure_selection_edit.setToolTip(
            "Select from the top of sequence_ranking.tsv before structure lookup. "
            "Use a positive integer for a count (1 means one sequence), or a decimal "
            "from 0 to 1 for a fraction (0.2 means 20%; 1.0 means 100%)."
        )

        self.keep_tmp_checkbox = QtWidgets.QCheckBox("Keep temporary files and tested models", self)
        self.keep_tmp_checkbox.setObjectName("keep_tmp_checkbox")
        self.keep_tmp_checkbox.setChecked(bool(keep_tmp))
        self.keep_tmp_checkbox.setToolTip(
            "Keep tracing, search databases, public sequence lists, downloaded candidates, "
            "and fitting scratch data. Disabled by default."
        )
        # Keep all filesystem/executable locations together at the top, then
        # show the remaining settings in one flat form. There is intentionally
        # no separate "Sequence Pipeline" box.
        form_layout.addRow("BLAST+ bin folder", blast_row)
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

        button_row = QtWidgets.QHBoxLayout()
        button_row.addStretch(1)
        cancel_button = QtWidgets.QPushButton("Cancel", self)
        save_button = QtWidgets.QPushButton("Save", self)
        cancel_button.clicked.connect(self.reject)
        save_button.clicked.connect(self.accept)
        button_row.addWidget(cancel_button)
        button_row.addWidget(save_button)
        main_layout.addLayout(button_row)

    def weights_path(self) -> str:
        return self.weights_path_edit.text().strip()

    def chimerax_path(self) -> str:
        return self.chimerax_path_edit.text().strip()

    def blast_bin_dir(self) -> str:
        return self.blast_bin_dir_edit.text().strip()

    def rcl(self) -> float:
        return resolve_rcl(self.rcl_edit.text().strip() or None, {})

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

    def _update_mode_controls(self) -> None:
        self.protein_min_trace_length_edit.setEnabled(self.sequence_mode() != "rna")

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

    def _browse_blast(self) -> None:
        start_dir = self.blast_bin_dir()
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select BLAST+ bin folder",
            start_dir,
        )
        if directory:
            self.blast_bin_dir_edit.setText(directory)

    def accept(self) -> None:
        try:
            validated_rcl = self.rcl()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid RCL", str(exc))
            return
        self.rcl_edit.setText("" if validated_rcl == 0.0 else f"{validated_rcl:.3f}")
        try:
            validated_trace_length = self.protein_min_trace_length()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(
                self, "Invalid protein minimum trace length", str(exc)
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
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "Invalid structure selection", str(exc))
                return
            self.structure_selection_edit.setText(validated_selection)
        if self.blast_bin_dir():
            try:
                validated = validate_blast_bin_dir(self.blast_bin_dir())
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "Invalid BLAST+ folder", str(exc))
                return
            self.blast_bin_dir_edit.setText(str(validated))
        super().accept()
