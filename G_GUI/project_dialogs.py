"""Project creation with an explicit name, parent location and Create action."""
from pathlib import Path

from G_GUI.qt_compat import QtWidgets


class NewProjectDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, *, location: Path):
        super().__init__(parent)
        self.setWindowTitle("New Project")
        self.setMinimumWidth(540)
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Create a new project folder in the location below.", self))
        form = QtWidgets.QFormLayout()
        self.name_edit = QtWidgets.QLineEdit(self)
        self.name_edit.setObjectName("project_name_edit")
        self.name_edit.setPlaceholderText("New project folder name")
        form.addRow("Name", self.name_edit)
        self.location_edit = QtWidgets.QLineEdit(str(location), self)
        self.location_edit.setObjectName("project_location_edit")
        row = QtWidgets.QWidget(self)
        location_layout = QtWidgets.QHBoxLayout(row)
        location_layout.setContentsMargins(0, 0, 0, 0)
        location_layout.addWidget(self.location_edit)
        self.browse_button = QtWidgets.QPushButton("Browse…", row)
        self.browse_button.setAutoDefault(False)
        self.browse_button.clicked.connect(self.browse_location)
        location_layout.addWidget(self.browse_button)
        form.addRow("Location", row)
        layout.addLayout(form)
        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch()
        self.cancel_button = QtWidgets.QPushButton("Cancel", self)
        self.cancel_button.setAutoDefault(False)
        self.cancel_button.clicked.connect(self.reject)
        self.create_button = QtWidgets.QPushButton("Create", self)
        self.create_button.setObjectName("create_project_button")
        self.create_button.setDefault(True)
        self.create_button.setEnabled(False)
        self.create_button.clicked.connect(self.accept)
        self.name_edit.textChanged.connect(self.update_create_button)
        self.location_edit.textChanged.connect(self.update_create_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.create_button)
        layout.addLayout(buttons)
        self.name_edit.setFocus()

    def update_create_button(self) -> None:
        self.create_button.setEnabled(bool(self.name_edit.text().strip()) and bool(self.location_edit.text().strip()))

    def browse_location(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Choose project location (existing parent folder)", self.location_edit.text(),
        )
        if directory:
            self.location_edit.setText(directory)

    def project_root(self) -> Path:
        name = self.name_edit.text().strip()
        if (not name or name in {".", ".."}
                or any(character in name for character in '/\\:*?"<>|')
                or any(ord(character) < 32 for character in name)):
            raise ValueError("Enter a project name without path separators or these characters: : * ? \" < > |")
        location = self.location_edit.text().strip()
        if not location or not Path(location).expanduser().is_dir():
            raise ValueError("Choose an existing parent folder in Location. Create will add the new project folder inside it.")
        root = Path(location).expanduser().resolve() / name
        if root.exists() or root.is_symlink():
            raise FileExistsError(
                "Choose a different project name, or use File > Open to resume an existing project.\n\n" + str(root)
            )
        return root

    def accept(self) -> None:
        try:
            self.project_root()
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(self, "Could not create project", str(exc))
            return
        super().accept()
