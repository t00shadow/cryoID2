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


class CopyLogsButton(QtWidgets.QPushButton):
    """Crossfade the copy glyph into a brief success check, then restore it."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._confirmation = 0.0
        self.set_theme(False)
        self._fade = QtCore.QPropertyAnimation(self, b"confirmation", self)
        easing = getattr(QtCore.QEasingCurve, "Type", QtCore.QEasingCurve)
        self._fade.setEasingCurve(easing.InOutCubic)
        self._fade.finished.connect(self._fade_finished)
        self._hold = QtCore.QTimer(self)
        self._hold.setSingleShot(True)
        self._hold.setInterval(750)
        self._hold.timeout.connect(lambda: self._animate_to(0.0, 180))

    def set_theme(self, dark: bool) -> None:
        assets = Path(__file__).parent / "svg_assets"
        self._check_icon = QtGui.QIcon(str(assets / ("check-dark.svg" if dark else "check.svg")))
        self.setIcon(QtGui.QIcon(str(assets / ("copy-dark.svg" if dark else "copy.svg"))))

    @QtCore.pyqtProperty(float)
    def confirmation(self) -> float:
        return self._confirmation

    @confirmation.setter
    def confirmation(self, value: float) -> None:
        self._confirmation = value
        self.update()

    def show_copied(self) -> None:
        self._hold.stop()
        self.setToolTip("Logs copied")
        self.setAccessibleName("Logs copied")
        self._animate_to(1.0, 120)

    def _animate_to(self, value: float, duration: int) -> None:
        self._fade.stop()
        self._fade.setDuration(duration)
        self._fade.setStartValue(self._confirmation)
        self._fade.setEndValue(value)
        self._fade.start()

    def _fade_finished(self) -> None:
        if self._confirmation >= 1.0:
            self._hold.start()
        else:
            self.setToolTip("Copy logs")
            self.setAccessibleName("Copy logs")

    def paintEvent(self, event) -> None:
        option = QtWidgets.QStyleOptionButton()
        self.initStyleOption(option)
        option.icon = QtGui.QIcon()
        painter = QtGui.QPainter(self)
        elements = getattr(QtWidgets.QStyle, "ControlElement", QtWidgets.QStyle)
        self.style().drawControl(elements.CE_PushButton, option, painter, self)
        size = self.iconSize()
        x, y = (self.width() - size.width()) // 2, (self.height() - size.height()) // 2
        painter.setOpacity(1.0 - self._confirmation)
        self.icon().paint(painter, x, y, size.width(), size.height())
        painter.setOpacity(self._confirmation)
        self._check_icon.paint(painter, x, y, size.width(), size.height())
        painter.end()


class LogSection(QtWidgets.QGroupBox):
    """Limit the expand/collapse tooltip to the checkbox indicator."""

    def event(self, event) -> bool:
        event_types = getattr(QtCore.QEvent, "Type", QtCore.QEvent)
        if event.type() == event_types.ToolTip:
            option = QtWidgets.QStyleOptionGroupBox()
            self.initStyleOption(option)
            controls = getattr(QtWidgets.QStyle, "ComplexControl", QtWidgets.QStyle)
            subcontrols = getattr(QtWidgets.QStyle, "SubControl", QtWidgets.QStyle)
            indicator = self.style().subControlRect(
                controls.CC_GroupBox, option, subcontrols.SC_GroupBoxCheckBox, self,
            )
            if indicator.contains(event.pos()):
                QtWidgets.QToolTip.showText(event.globalPos(), "Expand or collapse logs", self, indicator)
            else:
                QtWidgets.QToolTip.hideText()
                event.ignore()
            return True
        return super().event(event)


class LogOutput(QtWidgets.QPlainTextEdit):
    """Log pane with a copy control inset into its bottom-right corner."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setViewportMargins(0, 0, 0, 34)
        self.copy_button = CopyLogsButton(self)
        self.copy_button.setObjectName("copy_logs_button")
        self.copy_button.setFixedSize(28, 28)
        self.copy_button.setIconSize(QtCore.QSize(18, 18))
        self.copy_button.setToolTip("Copy logs")
        self.copy_button.setAccessibleName("Copy logs")
        self.copy_button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.copy_button.clicked.connect(self._copy_logs)
        self.viewport().installEventFilter(self)

    def _copy_logs(self) -> None:
        QtWidgets.QApplication.clipboard().setText(self.toPlainText())
        self.copy_button.show_copied()

    def _position_copy_button(self) -> None:
        self.copy_button.move(self.viewport().geometry().right() - self.copy_button.width() - 3,
                              self.height() - self.frameWidth() - self.copy_button.height() - 3)
        self.copy_button.raise_()

    def eventFilter(self, watched, event) -> bool:
        event_types = getattr(QtCore.QEvent, "Type", QtCore.QEvent)
        if watched is self.viewport() and event.type() == event_types.Resize:
            self._position_copy_button()
        return super().eventFilter(watched, event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_copy_button()


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
        ui.log_edit = LogOutput(ui.centralwidget)
        ui.log_edit.setObjectName("log_edit")
        ui.copy_logs_button = ui.log_edit.copy_button
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
            ("actionHuggingFace", "Hugging Face"),
            ("actionResultsFolder", "Results folder"),
        ):
            action = QtGui.QAction(text, window)
            action.setObjectName(name)
            setattr(ui, name, action)

        ui.centralwidget.setMinimumWidth(540)
        ui.main_layout.setContentsMargins(16, 12, 16, 14)
        ui.main_layout.setVerticalSpacing(12)

        def section(title, row, box_class=QtWidgets.QGroupBox):
            container = box_class(title, ui.centralwidget)
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
        ui.options_button.setToolTip("Adjust pipeline run options")
        config_row.addWidget(ui.options_button)
        inputs.addLayout(config_row)

        ui.run_section, run_layout = section("", 1)
        run_row = QtWidgets.QHBoxLayout()
        run_row.setSpacing(12)
        ui.run_button.setFixedSize(100, 40)
        ui.progress_bar.setFixedHeight(6)
        ui.progress_bar.setMinimumWidth(180)
        ui.progress_bar.setTextVisible(False)
        ui.progress_bar.setToolTip("Overall pipeline progress")
        ui.progress_percent = QtWidgets.QLabel("0%", ui.run_section)
        ui.progress_percent.setObjectName("progress_percent")
        ui.progress_percent.setFixedWidth(_font_width(ui.progress_percent, "100%") + 8)
        ui.progress_bar.valueChanged.connect(lambda value: ui.progress_percent.setText(f"{value}%"))
        run_row.addWidget(ui.run_button)
        run_row.addWidget(ui.progress_bar, 1)
        run_row.addWidget(ui.progress_percent)
        run_layout.addLayout(run_row)

        status_grid = QtWidgets.QGridLayout()
        status_grid.setHorizontalSpacing(12)
        ui.stage_label = QtWidgets.QLabel("Ready", ui.run_section)
        ui.device_label = QtWidgets.QLabel("—", ui.run_section)
        ui.elapsed_label = QtWidgets.QLabel("00:00:00", ui.run_section)
        ui.eta_label = QtWidgets.QLabel("00:00:00", ui.run_section)
        for name in ("stage_label", "device_label", "elapsed_label", "eta_label"):
            getattr(ui, name).setObjectName(name)
        ui.eta_label.setToolTip("Estimated remaining pipeline time")
        for row, column, title, value in (
            (0, 0, "Stage", ui.stage_label),
            (0, 2, "Device", ui.device_label),
            (1, 0, "Elapsed", ui.elapsed_label),
            (1, 2, "Est. remaining", ui.eta_label),
        ):
            caption = QtWidgets.QLabel(title, ui.run_section)
            caption.setProperty("muted", True)
            status_grid.addWidget(caption, row, column)
            value.setWordWrap(True)
            value.setMinimumWidth(0)
            status_grid.addWidget(value, row, column + 1)
        status_grid.setColumnStretch(1, 2)
        status_grid.setColumnStretch(3, 1)
        run_layout.addLayout(status_grid)

        ui.results_section, results = section("Results", 2)
        links = QtWidgets.QHBoxLayout()
        ui.ranking_button = QtWidgets.QPushButton("Sequence ranking", ui.results_section)
        ui.ranking_button.setObjectName("ranking_button")
        ui.ranking_button.setToolTip("No sequence rankings available")
        ui.model_button.setToolTip("No fitted model available")
        ui.ranking_button.setEnabled(False)
        ui.model_button.setEnabled(False)
        links.addWidget(ui.ranking_button)
        links.addWidget(ui.model_button)
        links.addStretch()
        results.addLayout(links)

        ui.log_section, log_layout = section("Logs", 3, LogSection)
        ui.log_edit.setReadOnly(True)
        ui.log_edit.setMinimumHeight(90)
        ui.log_edit.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        log_layout.addWidget(ui.log_edit, 1)
        ui.main_layout.setRowStretch(3, 1)
        ui.log_section.setCheckable(True)
        ui.log_section.setChecked(True)
        log_state = {"expanded_height": 600}

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
        window.resize(660, 600)
