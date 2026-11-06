"""Readable, shared setup and configuration validation messages."""
from html import escape

from G_GUI.qt_compat import QtGui, QtWidgets


class SetupIssuesDialog(QtWidgets.QDialog):
    def __init__(self, issues: list[str], parent=None, *, validation: bool = False):
        super().__init__(parent)
        self.validation = validation
        self._issues = None
        self.setWindowTitle("Configurations" if validation else "CryoID setup")
        self.action = ""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        self.summary = QtWidgets.QLabel(self)
        self.summary.setObjectName("setup_summary")
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.details = QtWidgets.QTextBrowser(self)
        self.details.setObjectName("setup_issues")
        self.details.setOpenExternalLinks(False)
        wrap_modes = getattr(QtGui.QTextOption, "WrapMode", QtGui.QTextOption)
        self.details.setWordWrapMode(wrap_modes.WrapAtWordBoundaryOrAnywhere)
        layout.addWidget(self.details, 1)
        buttons = QtWidgets.QHBoxLayout()
        self.action_buttons = {}
        if not validation:
            for action, title in (("new_project", "New project"), ("open_project", "Open project"),
                                  ("configure", "Configurations"), ("options", "Options")):
                button = QtWidgets.QPushButton(title, self)
                button.clicked.connect(lambda checked=False, action=action: self.choose(action))
                buttons.addWidget(button)
                self.action_buttons[action] = button
        buttons.addStretch()
        close = QtWidgets.QPushButton("Close", self)
        close.clicked.connect(lambda: self.choose("close"))
        buttons.addWidget(close)
        layout.addLayout(buttons)
        self.update_issues(issues)

    def update_issues(self, issues: list[str]) -> None:
        if issues == self._issues:
            return
        self._issues = list(issues)
        titles = {issue.partition(": ")[0] for issue in issues}
        project_issue = bool(titles & {"Project", "Project folder", "Project configuration"})
        relevant = {"new_project": project_issue, "open_project": project_issue,
                    "configure": "U-Net weights" in titles, "options": "U-Net device" in titles}
        for action, button in self.action_buttons.items():
            button.setVisible(relevant[action])
        self.summary.setText(
            "Settings were not saved. Fix the following items:" if self.validation else
            f"{len(issues)} {'item needs' if len(issues) == 1 else 'items need'} attention before you can run:"
            if issues else "Good — all requirements are available. You can run CryoID."
        )
        paragraphs = []
        for issue in issues:
            title, separator, body = issue.partition(": ")
            paragraphs.append(
                f"<p><b>{escape(title)}</b><br>{escape(body).replace(chr(10), '<br>')}</p>"
                if separator else f"<p>{escape(issue).replace(chr(10), '<br>')}</p>"
            )
        self.details.setHtml("".join(paragraphs))
        self.details.setVisible(bool(issues))
        self.resize(600, min(560, 160 + 85 * len(issues)) if issues else 130)

    def choose(self, action: str) -> None:
        self.action = action
        self.accept()


def show_validation_issues(issues: list[str], parent) -> None:
    dialog = SetupIssuesDialog(issues, parent, validation=True)
    dialog.setWindowTitle(parent.windowTitle())
    execute = getattr(dialog, "exec", None) or dialog.exec_
    execute()
