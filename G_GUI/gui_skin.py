"""Code-native layout for the current CryoID desktop GUI."""
from pathlib import Path

from G_GUI.qt_compat import QtCore, QtGui, QtWidgets


def _font_width(widget: QtWidgets.QWidget, text: str) -> int:
    metrics = QtGui.QFontMetrics(widget.font())
    if hasattr(metrics, "horizontalAdvance"):
        return metrics.horizontalAdvance(text)
    return metrics.width(text)


def _font_height(widget: QtWidgets.QWidget) -> int:
    return QtGui.QFontMetrics(widget.font()).height()


def _expanding_fixed_policy() -> QtWidgets.QSizePolicy:
    return QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)


def _browse_icon(window: QtWidgets.QMainWindow) -> QtGui.QIcon:
    icon = QtGui.QIcon(str(Path(__file__).parent / "svg_assets" / "upload-sign-svgrepo-com.svg"))
    if not icon.isNull():
        return icon
    try:
        return window.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton)
    except AttributeError:
        return window.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)


class CryoIDUI:
    def setup(self, window: QtWidgets.QMainWindow) -> None:
        ui = self
        window.setObjectName("cryoID")
        ui.centralwidget = QtWidgets.QWidget(window)
        ui.centralwidget.setObjectName("centralwidget")
        window.setCentralWidget(ui.centralwidget)
        ui.main_layout = QtWidgets.QGridLayout(ui.centralwidget)
        ui.input_rows = QtWidgets.QWidget(ui.centralwidget)
        ui.input_layout = QtWidgets.QVBoxLayout(ui.input_rows)
        ui.input_layout.setSpacing(6)
        ui.density_map_label = QtWidgets.QLabel(ui.input_rows)
        ui.candidate_pool_label = QtWidgets.QLabel(ui.input_rows)
        ui.density_map_edit = QtWidgets.QLineEdit(ui.input_rows)
        ui.density_map_edit.setObjectName("density_map_edit")
        ui.density_map_edit.setPlaceholderText("your_cryoEM_map.mrc")
        ui.candidate_pool_edit = QtWidgets.QLineEdit(ui.input_rows)
        ui.candidate_pool_edit.setObjectName("candidate_pool_edit")
        ui.candidate_pool_edit.setPlaceholderText("your_pool_file.fasta")
        ui.browse_density_button = QtWidgets.QPushButton(ui.input_rows)
        ui.browse_density_button.setObjectName("browse_density_button")
        ui.browse_pool_button = QtWidgets.QPushButton(ui.input_rows)
        ui.browse_pool_button.setObjectName("browse_pool_button")
        for label, field, button in (
            (ui.density_map_label, ui.density_map_edit, ui.browse_density_button),
            (ui.candidate_pool_label, ui.candidate_pool_edit, ui.browse_pool_button),
        ):
            row = QtWidgets.QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(8)
            row.addWidget(label)
            row.addWidget(field, 1)
            row.addWidget(button)
            ui.input_layout.addLayout(row)
        ui.run_button = QtWidgets.QPushButton("Run", ui.centralwidget)
        ui.run_button.setObjectName("run_button")
        run_font = QtGui.QFont(window.font())
        run_font.setPointSize(16)
        ui.run_button.setFont(run_font)
        ui.progress_bar = QtWidgets.QProgressBar(ui.centralwidget)
        ui.progress_bar.setObjectName("progress_bar")
        ui.model_button = QtWidgets.QPushButton("Model", ui.centralwidget)
        ui.model_button.setObjectName("model_button")
        ui.log_edit = QtWidgets.QPlainTextEdit(ui.centralwidget)
        ui.log_edit.setObjectName("log_edit")
        ui.menubar = QtWidgets.QMenuBar(window)
        window.setMenuBar(ui.menubar)
        ui.menuFile = QtWidgets.QMenu("File", ui.menubar)
        ui.menuView = QtWidgets.QMenu("View", ui.menubar)
        ui.menuSettings = QtWidgets.QMenu("Settings", ui.menubar)
        ui.menuHelp = QtWidgets.QMenu("Help", ui.menubar)
        for name in ("menuFile", "menuView", "menuSettings", "menuHelp"):
            getattr(ui, name).setObjectName(name)
        for name, text in (
            ("actionNew", "New"), ("actionOpen", "Open"),
            ("actionConfigurations", "Configurations"),
            ("actionModel", "Model"), ("actionGitHub", "GitHub"),
        ):
            action = QtGui.QAction(text, window)
            action.setObjectName(name)
            setattr(ui, name, action)

        ui.centralwidget.setMinimumWidth(540)
        ui.main_layout.setContentsMargins(16, 12, 16, 14)
        ui.main_layout.setVerticalSpacing(12)

        def section(title, row):
            container = QtWidgets.QGroupBox(title, ui.centralwidget)
            layout = QtWidgets.QVBoxLayout(container)
            layout.setContentsMargins(14, 22 if title else 12, 14, 12)
            layout.setSpacing(10)
            ui.main_layout.addWidget(container, row, 0)
            return container, layout

        ui.inputs_section, inputs = section("", 0)
        ui.input_rows.setMinimumWidth(0)
        ui.input_rows.setSizePolicy(_expanding_fixed_policy())
        ui.input_layout.setContentsMargins(0, 0, 0, 0)
        input_font = QtGui.QFont(ui.density_map_edit.font())
        input_font.setPointSize(10)
        for widget in (ui.density_map_label, ui.candidate_pool_label,
                       ui.density_map_edit, ui.candidate_pool_edit):
            widget.setFont(input_font)
        field_height = max(28, _font_height(ui.density_map_edit) + 8)
        label_width = max(_font_width(ui.density_map_label, "Density Map"),
                          _font_width(ui.candidate_pool_label, "Candidate Pool")) + 22
        for label, text in ((ui.density_map_label, "Density Map"),
                            (ui.candidate_pool_label, "Candidate Pool")):
            label.setText(text)
            label.setFixedWidth(label_width)
            label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        for field in (ui.density_map_edit, ui.candidate_pool_edit):
            field.setMinimumWidth(160)
            field.setMinimumHeight(field_height)
            field.setSizePolicy(_expanding_fixed_policy())
        icon = _browse_icon(window)
        for button, tip in ((ui.browse_density_button, "Browse for density map"),
                            (ui.browse_pool_button, "Browse for candidate pool")):
            button.setIcon(icon)
            button.setIconSize(QtCore.QSize(16, 16))
            button.setText("")
            button.setFixedSize(field_height, field_height)
            button.setToolTip(tip)
            button.setAccessibleName(tip)
            button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        inputs.addWidget(ui.input_rows)
        config_row = QtWidgets.QHBoxLayout()
        config_row.addStretch()
        ui.options_button = QtWidgets.QPushButton("Options", ui.inputs_section)
        ui.options_button.setObjectName("options_button")
        config_row.addWidget(ui.options_button)
        inputs.addLayout(config_row)

        ui.run_section, run_layout = section("", 1)
        run_row = QtWidgets.QHBoxLayout()
        run_row.setSpacing(12)
        ui.run_button.setFixedSize(100, 40)
        ui.progress_bar.setFixedHeight(6)
        ui.progress_bar.setMinimumWidth(180)
        ui.progress_bar.setTextVisible(False)
        ui.progress_percent = QtWidgets.QLabel("0%", ui.run_section)
        ui.progress_percent.setObjectName("progress_percent")
        ui.progress_percent.setFixedWidth(_font_width(ui.progress_percent, "100%") + 8)
        ui.progress_bar.valueChanged.connect(lambda value: ui.progress_percent.setText(f"{value}%"))
        run_row.addWidget(ui.run_button)
        run_row.addWidget(ui.progress_bar, 1)
        run_row.addWidget(ui.progress_percent)
        run_layout.addLayout(run_row)

        ui.results_section, results = section("Results", 2)
        links = QtWidgets.QHBoxLayout()
        ui.ranking_button = QtWidgets.QPushButton("Sequence ranking", ui.results_section)
        ui.ranking_button.setObjectName("ranking_button")
        ui.ranking_button.setEnabled(False)
        ui.model_button.setEnabled(False)
        links.addWidget(ui.ranking_button)
        links.addWidget(ui.model_button)
        links.addStretch()
        results.addLayout(links)

        ui.log_section, log_layout = section("Logs", 3)
        ui.log_edit.setReadOnly(True)
        ui.log_edit.setMinimumHeight(70)
        ui.log_edit.setMaximumHeight(110)
        log_layout.addWidget(ui.log_edit)
        ui.main_layout.setRowStretch(3, 1)
        ui.log_section.setCheckable(True)
        ui.log_section.setChecked(True)
        ui.log_section.setToolTip("Click Logs to collapse or expand the log")
        log_state = {"expanded_height": 520}

        def toggle_logs(expanded: bool) -> None:
            if not expanded:
                log_state["expanded_height"] = window.height()
            collapsed_height = max(32, _font_height(ui.log_section) + 16)
            previous_height = ui.log_section.height()
            ui.log_edit.setVisible(expanded)
            ui.log_section.setMaximumHeight(16777215 if expanded else collapsed_height)
            ui.main_layout.setRowStretch(3, 1 if expanded else 0)

            def resize_log_area() -> None:
                window.layout().activate()
                height = (log_state["expanded_height"] if expanded else
                          max(1, log_state["expanded_height"] - previous_height + collapsed_height))
                window.resize(window.width(), height)

            QtCore.QTimer.singleShot(0, resize_log_area)

        ui.log_section.toggled.connect(toggle_logs)
        window.resize(620, 520)
