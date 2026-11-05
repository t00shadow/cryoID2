import csv
import os
import subprocess
import sys
import uuid
from pathlib import Path

try:
    from G_GUI.qt_compat import QtCore, QtGui, QtWidgets
except (ImportError, OSError) as exc:
    raise SystemExit(
        "CryoID GUI cannot load a working Qt binding. "
        "Install or repair one GUI extra with the same Python, for example "
        "`python -m pip install --only-binary=:all: -e .[gui]` or `python -m pip install -e .[gui-pyqt5]`. "
        f"Qt error: {exc}"
    ) from exc

from G_GUI.gui_skin import CryoIDUI

from cryoid_pipeline import __version__  # noqa: E402
from cryoid_pipeline.cli import decode_event, format_command  # noqa: E402
from cryoid_pipeline.models import PipelineResult, StageResult, pipeline_result_from_dict  # noqa: E402
from cryoid_pipeline.runtime_config import (  # noqa: E402
    default_repo_root,
    display_problem,
)
from cryoid_pipeline.user_config import (  # noqa: E402
    default_config_path,
    ensure_startup_config, save_config_changes, STARTUP_KEYS, update_config, normalize_path, ConfigConflictError, resolve_run_settings, project_config_path,
    DEFAULT_CONFIG,
    load_effective_config,
    load_config as load_user_config,
    resolve_chimerax_path,
    resolve_device,
    resolve_weights_dir,
    results_project_dir,
)

from G_GUI.session_files import (
    collect_chimerax_project_files,
    ensure_project_dirs,
    is_project_directory,
    discard_uninitialized_project,
    find_project_dir_from_path,
    find_run_dir_from_path,
    flatten_chimerax_session_files,
    selected_sequence_ranking,
    sequence_ranking_files,
)
from G_GUI.settings_dialogs import ConfigurationsWindow, OptionsWindow, ConfigConflictDialog
from G_GUI.chimerax_launcher import launch_chimerax
from G_GUI.local_config import gui_config_view, save_project_config

from G_GUI.diagnostics import collect_diagnostics, check_weights, check_blast_tools, folder_write_issue
from G_GUI.run_status import RunStatus
from G_GUI.setup_dialogs import SetupIssuesDialog
from G_GUI.project_dialogs import NewProjectDialog
from cryoid_pipeline.weights import PUBLISHED_REPO


CRYOID_GITHUB_URL = "https://github.com/t00shadow/cryoID_v2"
CRYOID_HUGGINGFACE_URL = f"https://huggingface.co/{PUBLISHED_REPO}"
REPO_ROOT = default_repo_root().resolve()


class PipelineWorker(QtCore.QObject):
    log = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)
    status = QtCore.pyqtSignal(object)
    finished = QtCore.pyqtSignal(object)

    def __init__(
        self,
        density_map: str,
        candidate_pool: str,
        project_dir: Path,
        weights_path: str,
        rcl: float,
        protein_min_trace_length: int,
        sequence_mode: str,
        structure_selection: str,
        keep_tmp: bool,
        device: str | None = None,
        cpu_threads: str = "auto",
        chimerax_path: str = "",
    ) -> None:
        super().__init__()
        self.density_map = normalize_path(density_map)
        self.candidate_pool = normalize_path(candidate_pool)
        self.project_dir = Path(normalize_path(project_dir))
        self.weights_path = normalize_path(weights_path)
        self.rcl = float(rcl)
        self.protein_min_trace_length = int(protein_min_trace_length)
        self.sequence_mode = str(sequence_mode or "auto")
        self.structure_selection = str(structure_selection).strip()
        self.keep_tmp = bool(keep_tmp)
        self.device = device or "auto"
        self.cpu_threads = str(cpu_threads or "auto")
        self.chimerax_path = normalize_path(chimerax_path)
        self._cancel_requested = False
        self._cancel_file = self.project_dir / f".cryoid_cancel_{uuid.uuid4().hex}.signal"
        self._process: subprocess.Popen[str] | None = None

    def cancel(self) -> None:
        self._cancel_requested = True
        try:
            self._cancel_file.parent.mkdir(parents=True, exist_ok=True)
            self._cancel_file.touch(exist_ok=True)
        except OSError as exc:
            message = f"Could not write the CLI cancellation signal: {exc}"
            print(message, flush=True)
            self.log.emit(message)

    def _publish(self, message: str) -> None:
        print(message, flush=True)
        self.log.emit(message)

    def _failure_result(self, message: str) -> PipelineResult:
        return PipelineResult(
            run_dir=self.project_dir,
            stages=[StageResult("CLI command", "error", message)],
        )

    @QtCore.pyqtSlot()
    def run(self) -> None:
        command = [
            sys.executable,
            "-m",
            "cryoid_pipeline",
            "run",
            "--map", self.density_map,
            "--pool", self.candidate_pool,
            "--project",
            str(self.project_dir),
            "--event-stream",
            "--no-config",
            "--cancel-file",
            str(self._cancel_file),
        ]
        weights = str(Path(self.weights_path).expanduser().resolve()) if self.weights_path else ""
        command.extend(["--weights-dir", weights, "--device", self.device,
                        "--chimerax-path", self.chimerax_path])
        command.extend(["--cpu-threads", self.cpu_threads])
        command.extend(
            [
                "--rcl",
                str(self.rcl),
                "--mode",
                self.sequence_mode,
                "--protein-min-trace-length",
                str(self.protein_min_trace_length),
            ]
        )
        command.extend(["--structure-selection", self.structure_selection or "auto"])
        command.append("--keep-tmp" if self.keep_tmp else "--no-keep-tmp")

        command_text = format_command(command)
        self._publish(f"Command: {command_text}")
        env = os.environ.copy()
        # The GUI forwards its reviewed options explicitly; an empty structure
        # selector means mode defaults, not an inherited environment override.
        env.pop("CRYOID_STRUCTURE_SELECTION", None)
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHONUTF8": "1",
            }
        )
        result: PipelineResult | None = None
        return_code: int | None = None
        try:
            self._process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert self._process.stdout is not None
            for raw_line in self._process.stdout:
                line = raw_line.rstrip("\r\n")
                event = decode_event(line)
                if event is None:
                    if line:
                        self._publish(line)
                    continue
                event_type = event.get("type")
                if event_type == "progress":
                    self.progress.emit(int(event.get("value", 0)))
                elif event_type == "status":
                    self.status.emit(event)
                elif event_type == "result" and isinstance(event.get("result"), dict):
                    result = pipeline_result_from_dict(event["result"])
            return_code = self._process.wait()
        except Exception as exc:
            result = self._failure_result(f"Could not run CryoID CLI: {exc!r}")
        finally:
            self._process = None
            self._cancel_file.unlink(missing_ok=True)

        if result is None:
            result = self._failure_result(
                f"CryoID CLI exited with status {return_code} without returning a pipeline result."
            )
        elif return_code not in (None, 0):
            self._publish(f"Command exited with status {return_code}.")
        self.finished.emit(result)


class DiagnosticsWorker(QtCore.QThread):
    report = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, weights_path: str, device: str | None, parent=None):
        super().__init__(parent)
        self.arguments = (weights_path, device)

    def run(self) -> None:
        try:
            self.report.emit(collect_diagnostics(*self.arguments))
        except Exception:
            self.failed.emit("Setup checks could not finish. Try Settings > Check setup again. "
                             "If the problem continues, restart CryoID.")


def _clear_line_edit_highlights(
    line_edits: tuple[QtWidgets.QLineEdit, ...],
    focus_target: QtWidgets.QWidget,
) -> None:
    """Clear input selections and leave focus on a neutral widget."""

    for line_edit in line_edits:
        line_edit.deselect()
        line_edit.clearFocus()
        line_edit.setCursorPosition(len(line_edit.text()))
    focus_target.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)


def startup_status_message() -> str:
    return f"CryoID2 version {__version__}"


def _dialog_accepted_code() -> int:
    dialog_code = getattr(QtWidgets.QDialog, "DialogCode", None)
    if dialog_code is not None:
        accepted = dialog_code.Accepted
        return int(getattr(accepted, "value", accepted))
    return int(QtWidgets.QDialog.Accepted)


def _exec_dialog(dialog: QtWidgets.QDialog) -> int:
    exec_method = getattr(dialog, "exec", None) or getattr(dialog, "exec_", None)
    result = exec_method()
    return int(getattr(result, "value", result))


class CryoIDMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.close_request_handler = None

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self.close_request_handler is not None and self.close_request_handler(event):
            return
        super().closeEvent(event)


class SequenceRankingModel(QtCore.QAbstractTableModel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.headers: list[str] = []
        self.rows: list[list[str]] = []

    def set_table(self, headers: list[str], rows: list[list[str]]) -> None:
        self.beginResetModel()
        # Keep model installation O(1) with respect to cell inspection. Scanning
        # every cell to hide empty mode-specific columns made large rankings
        # pause the GUI after the background loader completed.
        self.headers = list(headers)
        self.rows = rows
        self.endResetModel()

    def rowCount(self, parent=QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QtCore.QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.headers)

    def data(self, index: QtCore.QModelIndex, role: int = QtCore.Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        return self.rows[index.row()][index.column()]

    def headerData(self, section: int, orientation, role: int = QtCore.Qt.ItemDataRole.DisplayRole):
        if role != QtCore.Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == QtCore.Qt.Orientation.Horizontal:
            return self.headers[section]
        return section + 1

    def sort(self, column: int, order=QtCore.Qt.SortOrder.AscendingOrder) -> None:
        if not 0 <= column < len(self.headers):
            return

        def key(row: list[str]):
            value = row[column]
            try:
                return (0, float(value))
            except (TypeError, ValueError):
                return (1, str(value).lower())

        self.layoutAboutToBeChanged.emit()
        self.rows.sort(key=key, reverse=order == QtCore.Qt.SortOrder.DescendingOrder)
        self.layoutChanged.emit()


RANKING_DISPLAY_COLUMNS = (
    ("candidate_id", "Candidate ID"),
    ("length", "Sequence Length"),
    ("primary_score", "Score"),
)


def _load_sequence_ranking_display(
    path: Path,
    *,
    limit: int = 20,
) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        source_headers = set(reader.fieldnames or [])
        missing = [key for key, _label in RANKING_DISPLAY_COLUMNS if key not in source_headers]
        if missing:
            raise ValueError(
                "Sequence ranking is missing required display column(s): "
                + ", ".join(missing)
            )
        rows = [
            [str(row.get(key) or "") for key, _label in RANKING_DISPLAY_COLUMNS]
            for _, row in zip(range(limit), reader)
        ]
    return [label for _key, label in RANKING_DISPLAY_COLUMNS], rows


class SequenceRankingLoader(QtCore.QObject):
    loaded = QtCore.pyqtSignal(object, object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            headers, rows = _load_sequence_ranking_display(self.path)
            self.loaded.emit(headers, rows)
        except FileNotFoundError:
            self.failed.emit("The ranking file was moved or deleted. Open an existing run from File > Open.")
        except PermissionError:
            self.failed.emit("The ranking file cannot be read. Check its read permissions and try again.")
        except Exception:
            self.failed.emit("The ranking file could not be read or has an invalid format. Select another run or generate a new ranking.")


def _ranking_run_label(path: Path) -> str:
    run_dir = path.parent
    details: list[str] = []
    try:
        import json

        payload = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        mode = str(payload.get("effective_mode") or "").upper()
        status = str(payload.get("status") or "").lower()
        if mode:
            details.append(mode)
        if status:
            details.append(status)
    except (OSError, ValueError):
        pass
    suffix = f" | {' | '.join(details)}" if details else ""
    return f"{run_dir.name}{suffix}"


class SequenceRankingDialog(QtWidgets.QDialog):
    run_selected = QtCore.pyqtSignal(str)

    def __init__(
        self,
        path: Path | None,
        parent=None,
        *,
        ranking_paths: list[Path] | None = None,
    ) -> None:
        super().__init__(parent)
        initial_path = path.expanduser().resolve() if path is not None else None
        candidates = [candidate.expanduser().resolve() for candidate in (ranking_paths or [])]
        if initial_path is not None and initial_path.is_file() and initial_path not in candidates:
            candidates.append(initial_path)
        self.ranking_paths = sorted(
            {candidate for candidate in candidates if candidate.is_file()},
            key=lambda candidate: candidate.parent.name,
            reverse=True,
        )
        self.path = (
            initial_path
            if initial_path is not None and initial_path in self.ranking_paths
            else (self.ranking_paths[0] if self.ranking_paths else None)
        )
        self.setWindowTitle("Sequence Ranking")
        self.setModal(True)
        self.resize(700, 650)
        layout = QtWidgets.QVBoxLayout(self)
        run_layout = QtWidgets.QFormLayout()
        self.run_combo = QtWidgets.QComboBox(self)
        self.run_combo.setObjectName("sequence_ranking_run_combo")
        self.run_combo.setToolTip("Choose a saved run")
        if self.ranking_paths:
            self.run_combo.blockSignals(True)
            for ranking_path in self.ranking_paths:
                self.run_combo.addItem(_ranking_run_label(ranking_path), str(ranking_path))
            selected_index = self.ranking_paths.index(self.path) if self.path is not None else 0
            self.run_combo.setCurrentIndex(selected_index)
            self.run_combo.blockSignals(False)
            self.run_combo.setEnabled(len(self.ranking_paths) > 1)
        else:
            self.run_combo.addItem("No runs with sequence rankings")
            self.run_combo.setEnabled(False)
        self.run_combo.currentIndexChanged.connect(self._ranking_changed)
        run_layout.addRow("Run", self.run_combo)
        layout.addLayout(run_layout)
        self.status_label = QtWidgets.QLabel("", self)
        layout.addWidget(self.status_label)
        self.table = QtWidgets.QTableView(self)
        self.model = SequenceRankingModel(self)
        self.table.setModel(self.model)
        text_elide_mode = getattr(QtCore.Qt, "TextElideMode", QtCore.Qt)
        self.table.setTextElideMode(text_elide_mode.ElideNone)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        layout.addWidget(self.table)
        self._thread: QtCore.QThread | None = None
        self._loader: SequenceRankingLoader | None = None
        if self.path is None or not self.path.is_file():
            self.path = None
            self.status_label.setText(
                "No sequence ranking is available for the current project. "
                "Run identification first; CryoID will detect sequence_ranking.tsv "
                "automatically when it exists in the run directory."
            )
        else:
            self._start_loading()

    def _ranking_changed(self, index: int) -> None:
        if self._thread is not None or index < 0:
            return
        raw_path = self.run_combo.itemData(index)
        if not raw_path:
            return
        selected = Path(str(raw_path)).expanduser().resolve()
        if not selected.is_file() or selected == self.path:
            return
        self.path = selected
        self.model.set_table([], [])
        self.run_selected.emit(str(selected.parent))
        self._start_loading()

    def _start_loading(self) -> None:
        if self.path is None:
            return
        self.run_combo.setEnabled(False)
        self.setWindowTitle(f"Sequence Ranking - {self.path.parent.name}")
        self.status_label.setText(f"Loading {self.path}...")
        self._thread = QtCore.QThread(self)
        self._loader = SequenceRankingLoader(self.path)
        self._loader.moveToThread(self._thread)
        self._thread.started.connect(self._loader.run)
        self._loader.loaded.connect(self._loaded)
        self._loader.loaded.connect(self._thread.quit)
        self._loader.failed.connect(self._failed)
        self._loader.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._loader.deleteLater)
        self._thread.finished.connect(self._loading_finished)
        self._thread.start()

    def _loading_finished(self) -> None:
        self._loader = None
        self._thread = None
        self.run_combo.setEnabled(len(self.ranking_paths) > 1)

    def _loaded(self, headers: list[str], rows: list[list[str]]) -> None:
        self.model.set_table(headers, rows)
        sequence_label = "sequence" if len(rows) == 1 else "sequences"
        self.status_label.setText(f"{len(rows)} ranked {sequence_label}")
        self.table.resizeColumnToContents(0)
        for index in range(1, len(headers)):
            self.table.setColumnWidth(index, 160)
        QtCore.QTimer.singleShot(0, self._fit_width_to_table)

    def _fit_width_to_table(self) -> None:
        column_width = sum(
            self.table.columnWidth(index) for index in range(self.model.columnCount())
        )
        margins = self.layout().contentsMargins()
        target_width = (
            column_width
            + self.table.verticalHeader().sizeHint().width()
            + self.table.verticalScrollBar().sizeHint().width()
            + (2 * self.table.frameWidth())
            + margins.left()
            + margins.right()
        )
        self.setFixedWidth(target_width)

    def _failed(self, message: str) -> None:
        self.status_label.setText(f"Could not load ranking: {message}")


def apply_theme(app, window, ui, theme: str = "light") -> None:
    """Apply one consistent palette to the main window and its dialogs."""
    dark = theme == "dark"
    bg, panel, field = ("#1e1e1e", "#1f242b", "#262b33") if dark else ("#f3f6fa", "#eaf0f7", "#ffffff")
    text, muted, border = ("#e2e8f0", "#7a818c", "#333944") if dark else ("#213247", "#64748b", "#c8d3e0")
    accent = "#347aff" if dark else "#2563eb"
    button = "#2f3540" if dark else panel
    button_border = "#3a404c" if dark else border
    menu_hover = "#2a303a" if dark else accent
    palette = QtGui.QPalette()
    for role, color in {
        "Window": bg, "WindowText": text, "Base": field,
        "AlternateBase": panel, "Text": text, "Button": panel,
        "ButtonText": text, "ToolTipBase": field, "ToolTipText": text,
        "Highlight": accent, "HighlightedText": "#ffffff", "Link": accent,
        "BrightText": "#ef4444",
    }.items():
        palette.setColor(getattr(QtGui.QPalette.ColorRole, role), QtGui.QColor(color))
    for role in ("Text", "WindowText", "ButtonText"):
        palette.setColor(getattr(QtGui.QPalette, "ColorGroup", QtGui.QPalette).Disabled,
                         getattr(QtGui.QPalette.ColorRole, role), QtGui.QColor(muted))
    if dark:
        # Preserve the palette used by the original GUI, including native controls.
        for role, color in {
            "Base": "#21252b", "AlternateBase": "#282c34",
            "ToolTipBase": "#1e2127", "Button": "#2f3540",
            "BrightText": "#ff7676", "Link": "#5d7bff",
            "HighlightedText": "#f4f6fb",
        }.items():
            palette.setColor(getattr(QtGui.QPalette.ColorRole, role), QtGui.QColor(color))
    app.setPalette(palette)
    app.setStyleSheet(f"""
        QMainWindow, QDialog {{ background: {bg}; color: {text}; }}
        QMenuBar, QMenu {{ background: {panel}; color: {text}; }}
        QMenuBar::item:selected, QMenu::item:selected {{ background: {menu_hover}; color: {text if dark else "white"}; }}
        QLabel {{ color: {text}; }}
        QLabel#density_map_label, QLabel#candidate_pool_label {{ font-weight: 600; }}
        QLineEdit, QPlainTextEdit, QTextBrowser, QListWidget, QTableView, QComboBox {{
            background: {field}; color: {text}; border: 1px solid {border};
            border-radius: 4px; selection-background-color: {accent};
            selection-color: white;
        }}
        QLineEdit, QComboBox {{ padding: 4px; }}
        QPushButton {{ background: {button}; color: {text}; border: 1px solid {button_border};
                       border-radius: 5px; padding: 5px 10px; }}
        QPushButton:hover {{ border-color: {accent}; }}
        QPushButton:disabled {{ color: {muted}; border-color: {border}; }}
        QPushButton#copy_logs_button {{ background: transparent; border: 1px solid transparent; padding: 4px; }}
        QPushButton#copy_logs_button:hover {{ background: {button}; border-color: {border}; }}
        QPushButton#copy_logs_button:focus {{ border-color: {accent}; }}
        QPushButton#run_button {{ background: #2563eb; color: white; border: none; }}
        QPushButton#run_button:hover {{ background: #1d4ed8; }}
        QPushButton#run_button[blocked="true"] {{ background: {"#9a651d" if dark else "#996117"}; color: white; }}
        QPushButton#run_button[blocked="true"]:hover {{ background: #805012; }}
        QPushButton#run_button[checking="true"] {{ background: {button}; color: {muted}; border: 1px solid {border}; }}
        QPushButton#run_button[running="true"] {{ background: #b91c1c; }}
        QPushButton#run_button[running="true"]:hover {{ background: #991b1b; }}
        QPushButton#run_button:disabled {{ background: {border}; color: {muted}; }}
        QProgressBar {{ background: {field if dark else border}; border: none; border-radius: 3px; }}
        QProgressBar::chunk {{ background: {accent}; border-radius: 3px; }}
        QLabel#progress_percent {{ color: {muted}; }}
        QLabel[muted="true"] {{ color: {muted}; font-size: 10pt; }}
        QGroupBox {{ border: 1px solid {border}; border-radius: 6px;
                     margin-top: 10px; font-weight: 600; color: {text}; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 5px; }}
    """)
    if ui is not None:
        ui.copy_logs_button.set_theme(dark)


def set_run_button_state(ui, *, running: bool, cancelling: bool = False,
                         blocked: bool = False, checking: bool = False) -> None:
    ui.run_button.setText("Abort" if running else "Run")
    blocked, checking = blocked and not running, checking and not running
    ui.run_button.setToolTip(
        "Abort the current job" if running else
        "Checking setup requirements" if checking else
        "Show missing setup requirements" if blocked else "Run the pipeline"
    )
    icon = QtGui.QIcon()
    if blocked and not checking:
        pixmaps = getattr(QtWidgets.QStyle, "StandardPixmap", QtWidgets.QStyle)
        icon = ui.run_button.style().standardIcon(pixmaps.SP_MessageBoxWarning)
    ui.run_button.setIcon(icon)
    ui.run_button.setIconSize(QtCore.QSize(16, 16))
    ui.run_button.setEnabled(not cancelling)
    ui.run_button.setProperty("running", running)
    ui.run_button.setProperty("blocked", blocked)
    ui.run_button.setProperty("checking", checking)
    ui.run_button.setAccessibleName("Run unavailable: setup required" if blocked else ui.run_button.text())
    ui.run_button.style().unpolish(ui.run_button)
    ui.run_button.style().polish(ui.run_button)



def main() -> None:
    display_issue = display_problem()
    if display_issue:
        raise SystemExit(f"CryoID GUI cannot start: {display_issue}")

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)
    app.setStyle("Fusion")
    # Keep inherited widget fonts consistent when Qt repolishes a new theme.
    font = app.font()
    font.setPointSize(12)
    app.setFont(font)
    window = CryoIDMainWindow()
    ui = CryoIDUI()
    ui.setup(window)
    window.setWindowTitle("CryoIDv2")
    theme_preferences = QtCore.QSettings("CryoID", "GUI")
    current_theme = str(theme_preferences.value("theme", "light"))
    startup_pending = True
    startup_load_issue = ""
    try:
        global_config = ensure_startup_config()
    except (OSError, ValueError) as exc:
        global_config = load_user_config()
        startup_load_issue = f"Startup configuration: {exc}"
    startup_state = {key: global_config[key] for key in STARTUP_KEYS}
    startup_baseline = dict(startup_state)
    local_config_state = gui_config_view(global_config)
    shared_config_state = global_config
    resolved_project_dir = results_project_dir(shared_config_state.get("project_dir") or REPO_ROOT)
    project_state = {"active": bool(shared_config_state.get("project_dir")) and resolved_project_dir.is_dir()}
    project_load_issue = ""
    try:
        loaded = load_effective_config(resolved_project_dir, user_config=dict(DEFAULT_CONFIG, **startup_state))
        shared_config_state = loaded.values
        local_config_state.update(gui_config_view(loaded.values))
        if loaded.source_path is not None:
            project_state["active"] = True
    except ValueError as exc:
        project_load_issue = f"Project configuration: {exc}"
        shared_config_state = dict(DEFAULT_CONFIG, **startup_state, project_dir=str(resolved_project_dir))
        local_config_state.update(gui_config_view(shared_config_state))
    project_baseline = dict(shared_config_state)
    stored_overrides = dict(shared_config_state, structure_selection=shared_config_state.get("structure_selection") or "auto")
    run_settings, startup_issues = resolve_run_settings(shared_config_state, overrides=stored_overrides)
    if project_load_issue:
        startup_issues["project_config"] = project_load_issue
    if startup_load_issue:
        startup_issues["startup_config"] = startup_load_issue
    local_config_state["project_dir"] = str(resolved_project_dir)
    ui.progress_bar.setRange(0, 100)
    ui.progress_bar.setValue(0)

    state: dict[str, object] = {
        "thread": None,
        "worker": None,
    }
    sequence_ranking_action = QtGui.QAction("Sequence Ranking", window)
    sequence_ranking_action.setToolTip("View ranked sequence matches")
    configured_weights = resolve_weights_dir(shared_config_state.get("weights_dir", ""), shared_config_state)
    configured_chimerax = resolve_chimerax_path(shared_config_state, argument=shared_config_state.get("chimerax_path", ""))
    configured_device = run_settings["device"]
    configured_rcl = run_settings["rcl"]
    configured_protein_min_trace_length = run_settings["protein_min_trace_length"]
    configured_sequence_mode = run_settings["sequence_mode"]
    configured_structure_selection = run_settings["structure_selection"]
    configured_keep_tmp = run_settings["keep_tmp"]
    settings_state: dict[str, str] = {
        "weights_path": str(configured_weights or ""),
        "device": configured_device or "auto",
        "cpu_threads": str(run_settings["cpu_threads"] or "auto"),
        "rcl": str(configured_rcl),
        "protein_min_trace_length": str(configured_protein_min_trace_length),
        "chimerax_path": configured_chimerax,
        "sequence_mode": configured_sequence_mode,
        "structure_selection": str(configured_structure_selection or ""),
        "keep_tmp": "true" if configured_keep_tmp else "false",
    }
    close_state = {"pending": False}
    run_status = RunStatus(history_path=default_config_path().with_name("timing_history.json"))
    diagnostics_state = {"thread": None, "pending": False, "report": None, "error": "", "run_requested": False}

    def finish_diagnostics_on_exit() -> None:
        worker = diagnostics_state["thread"]
        if worker is not None:
            worker.wait()

    app.aboutToQuit.connect(finish_diagnostics_on_exit)

    def refresh_run_status() -> None:
        ui.stage_label.setText(run_status.description)
        device = run_status.device
        ui.device_label.setText("CPU" if device == "cpu" else f"GPU ({device})" if device.startswith("cuda") else device)
        ui.elapsed_label.setText(run_status.elapsed)
        ui.eta_label.setText(run_status.eta)
        ui.progress_bar.setValue(run_status.percent)

    run_clock = QtCore.QTimer(window)
    run_clock.setInterval(1000)
    run_clock.timeout.connect(refresh_run_status)

    def update_stage(event: dict) -> None:
        if event.get("weights_path") and event.get("automatic"):
            path = event["weights_path"]
            # Avoid replacing a custom path selected while the run downloaded.
            worker = state.get("worker")
            if worker is not None and settings_state.get("weights_path", "") == worker.weights_path:
                if persist_startup_settings({"weights_dir": path}):
                    append_log(f"Weights: saved downloaded weights location: {path}")
                else:
                    append_log(f"Weights: using {path} for this run; the location could not be saved.")
                refresh_diagnostics()
        worker = state.get("worker")
        if worker is not None and getattr(worker, "_cancel_requested", False):
            if event.get("name") != "Cancelling / cleanup":
                return
        run_status.set_stage(event)
        refresh_run_status()

    def append_log(message: str) -> None:
        ui.log_edit.appendPlainText(str(message))
        if state.get("worker") is not None:
            run_status.consume_log(str(message))
            refresh_run_status()

    def readiness_issues() -> list[str]:
        issues = list(startup_issues.values())
        if diagnostics_state["thread"] is not None:
            issues.append("Dependency and configuration checks are still running. Please wait.")
        elif diagnostics_state["error"]:
            issues.append(str(diagnostics_state["error"]))
        elif diagnostics_state["report"] is not None:
            issues.extend(diagnostics_state["report"].issues)
        else:
            issues.append("Setup has not been checked yet.")
        for label, field in (("Density map", ui.density_map_edit), ("Candidate pool", ui.candidate_pool_edit)):
            value = field.text().strip()
            if not value:
                issues.append(f"{label}: select an input file.")
            elif not Path(value).expanduser().is_file():
                issues.append(f"{label}: This file was moved or could not be found. Select an existing input file.\nFile: {value}")
        project = current_project_dir() if project_state["active"] else None
        if project is None:
            issues.append("Project: select a project with File > New or Open.")
        location_issue = folder_write_issue(str(project) if project is not None else "")
        if project is not None and location_issue:
            issues.append(f"Project folder: {location_issue}")
        return issues

    def refresh_run_availability() -> None:
        if state.get("worker") is None:
            set_run_button_state(ui, running=False, blocked=bool(readiness_issues()),
                                 checking=diagnostics_state["thread"] is not None)

    def show_setup_issues() -> None:
        dialog = SetupIssuesDialog(readiness_issues(), window)
        refresh = QtCore.QTimer(dialog)
        refresh.setInterval(200)
        refresh.timeout.connect(lambda: dialog.update_issues(readiness_issues()))
        refresh.start()
        _exec_dialog(dialog)
        refresh.stop()
        if dialog.action == "configure":
            open_configurations()

    def check_setup() -> None:
        refresh_diagnostics()
        show_setup_issues()

    def refresh_diagnostics(*, run_after_checks: bool = False) -> None:
        diagnostics_state["run_requested"] = run_after_checks
        if diagnostics_state["thread"] is not None:
            diagnostics_state["pending"] = True
            return
        diagnostics_state["report"] = None
        diagnostics_state["error"] = ""
        worker = DiagnosticsWorker(
            settings_state.get("weights_path", ""),
            resolve_device(settings_state["device"], {}), window,
        )

        def checked(report) -> None:
            if diagnostics_state["pending"]:
                return
            diagnostics_state["report"] = report
            if state.get("worker") is None:
                run_status.device = report.device if report.compute.ok else "Unavailable"
                refresh_run_status()

        def check_failed(message: str) -> None:
            diagnostics_state["error"] = message
            run_status.device = "Unavailable"
            refresh_run_status()

        def checks_finished() -> None:
            nonlocal startup_pending
            diagnostics_state["thread"] = None
            if diagnostics_state["pending"]:
                diagnostics_state["pending"] = False
                refresh_diagnostics()
                return
            refresh_run_availability()
            if startup_pending:
                startup_pending = False
                screen = app.primaryScreen()
                if screen is not None:
                    window.move(screen.availableGeometry().center() - window.rect().center())
                app.setQuitOnLastWindowClosed(True)
                window.show()
            if diagnostics_state["run_requested"]:
                diagnostics_state["run_requested"] = False
                start_pipeline(setup_checked=True)

        worker.report.connect(checked)
        worker.failed.connect(check_failed)
        worker.finished.connect(checks_finished)
        worker.finished.connect(worker.deleteLater)
        diagnostics_state["thread"] = worker
        refresh_run_availability()
        worker.start()

    def update_run_progress(value: int) -> None:
        worker = state.get("worker")
        if worker is not None and getattr(worker, "_cancel_requested", False):
            return
        run_status.update_progress(value)
        refresh_run_status()

    def apply_saved_settings(saved) -> None:
        saved = dict(saved, **startup_state)
        shared_config_state.clear()
        shared_config_state.update(saved)
        local_config_state.update(gui_config_view(saved))
        overrides = dict(saved, structure_selection=saved.get("structure_selection") or "auto")
        resolved, issues = resolve_run_settings(saved, overrides=overrides)
        startup_issues.clear()
        startup_issues.update(issues)
        for key, value in resolved.items():
            if key in {"device", "cpu_threads"}:
                value = value or "auto"
            elif key == "keep_tmp":
                value = "true" if value else "false"
            elif key == "structure_selection":
                value = value or ""
            settings_state[key] = str(value)
        settings_state["weights_path"] = saved.get("weights_dir", "")
        settings_state["chimerax_path"] = saved.get("chimerax_path", "")
        ui.density_map_edit.setText(saved["last_density_map"])
        ui.candidate_pool_edit.setText(saved["last_candidate_pool"])
        project_baseline.clear()
        project_baseline.update(saved)

    def apply_startup_settings(saved) -> None:
        startup_state.update({key: saved[key] for key in STARTUP_KEYS})
        startup_baseline.update(startup_state)
        settings_state.update(weights_path=startup_state["weights_dir"], chimerax_path=startup_state["chimerax_path"])
        shared_config_state.update(startup_state)
        local_config_state.update(gui_config_view(dict(local_config_state, **startup_state)))

    def persist_startup_settings(updates=None, *, snapshot=False) -> bool:
        desired = dict(startup_state)
        desired.update({key: normalize_path(value) for key, value in (updates or {}).items()})
        changes = {key: value for key, value in desired.items()
                   if snapshot or value != startup_baseline[key]}
        try:
            try:
                saved = save_config_changes(changes, baseline=startup_baseline)
            except ConfigConflictError as exc:
                dialog = ConfigConflictDialog(exc, window)
                _exec_dialog(dialog)
                if dialog.action == "reload":
                    apply_startup_settings(exc.saved)
                    refresh_diagnostics()
                    return False
                if dialog.action != "keep":
                    return False
                saved = save_config_changes(changes, baseline=startup_baseline, force=True)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(window, "Could not save startup settings",
                                          f"The shared installation paths could not be saved.\n\n{default_config_path()}\n\n{exc}")
            return False
        apply_startup_settings(saved)
        startup_issues.pop("startup_config", None)
        return True

    def persist_config(*, reviewed_options: bool = False, snapshot: bool = False,
                       initialize: bool = False, **updates: object) -> bool:
        to_save = dict(local_config_state, **settings_state)
        to_save.update({key: str(value) if value is not None else "" for key, value in updates.items()})
        to_save["last_density_map"] = normalize_path(ui.density_map_edit.text().strip())
        to_save["last_candidate_pool"] = normalize_path(ui.candidate_pool_edit.text().strip())
        for key in startup_issues:
            if not reviewed_options:
                to_save[key] = shared_config_state.get(key, "")
        active = project_state["active"]
        project = current_project_dir() if active else None
        if project is None:
            settings_state.update({key: str(value) for key, value in updates.items() if key in settings_state})
            return True
        previous_settings = dict(settings_state)
        try:
            try:
                saved = save_project_config(to_save, project, baseline=project_baseline,
                                            initialize=initialize or "project_dir" in updates, snapshot=snapshot)
            except ConfigConflictError as exc:
                dialog = ConfigConflictDialog(exc, window)
                _exec_dialog(dialog)
                if dialog.action == "reload":
                    apply_saved_settings(dict(exc.saved, project_dir=str(project)))
                    refresh_diagnostics()
                    return False
                if dialog.action != "keep":
                    return False
                saved = save_project_config(to_save, project, baseline=project_baseline,
                                            initialize=True, snapshot=snapshot, force=True)
            update_config(project_dir=str(project))
        except (OSError, ValueError, TimeoutError) as exc:
            QtWidgets.QMessageBox.warning(window, "Could not save configuration",
                                          f"The configuration could not be saved. Check this file and its permissions.\n\n{project_config_path(project)}\n\n{exc}")
            return False
        apply_saved_settings(saved)
        if previous_settings != settings_state and not snapshot:
            refresh_diagnostics()
        project_state["active"] = active
        return True

    def remember_edited_inputs() -> bool:
        if (ui.density_map_edit.text().strip() != local_config_state.get("last_density_map", "")
                or ui.candidate_pool_edit.text().strip() != local_config_state.get("last_candidate_pool", "")):
            return persist_config()
        return True

    def current_project_dir() -> Path | None:
        if not project_state["active"]:
            return None
        selected = str(local_config_state.get("project_dir") or "").strip()
        return results_project_dir(selected) if selected else None

    def refresh_project_title() -> None:
        project = current_project_dir()
        window.setWindowTitle(f"CryoIDv2 — {project.parent.name}" if project else "CryoIDv2")
        window.setWindowFilePath(str(project_config_path(project)) if project else "")

    def clear_input_highlight() -> None:
        _clear_line_edit_highlights(
            (ui.density_map_edit, ui.candidate_pool_edit),
            ui.centralwidget,
        )

    def current_sequence_ranking() -> Path | None:
        project = current_project_dir()
        if project is None:
            return None
        return selected_sequence_ranking(
            project,
            local_config_state.get("sequence_ranking_run_dir"),
        )

    def refresh_result_actions() -> None:
        project = current_project_dir()
        has_ranking = bool(project and sequence_ranking_files(project))
        has_model = bool(project and collect_chimerax_project_files(project, None)["aligned_models"])
        ui.ranking_button.setEnabled(has_ranking)
        ui.ranking_button.setToolTip(
            "View ranked sequence matches" if has_ranking else
            "No sequence rankings available"
        )
        sequence_ranking_action.setEnabled(True)
        ui.model_button.setEnabled(has_model)
        ui.model_button.setToolTip(
            "Open model in ChimeraX" if has_model else
            "No fitted model available"
        )
        ui.actionModel.setEnabled(True)

    def open_results_folder() -> None:
        project = current_project_dir()
        if project is None:
            QtWidgets.QMessageBox.warning(window, "Results folder", "Select a project with File > New or Open first.")
            return
        try:
            project.mkdir(parents=True, exist_ok=True)
        except OSError:
            QtWidgets.QMessageBox.warning(window, "Results folder",
                                          f"The results folder could not be opened. Check that its location is accessible.\n\n{project}")
            return
        if not QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(project))):
            QtWidgets.QMessageBox.warning(window, "Results folder",
                                          f"The file manager could not open this folder. You can open it manually:\n\n{project}")

    def open_sequence_ranking(path: Path | None = None) -> None:
        project = current_project_dir()
        rankings = sequence_ranking_files(project) if project is not None else []
        ranking = path.expanduser().resolve() if path is not None else current_sequence_ranking()
        if ranking is None or not ranking.is_file():
            ranking = None
        elif path is not None:
            persist_config(sequence_ranking_run_dir=ranking.parent)
        dialog = SequenceRankingDialog(ranking, window, ranking_paths=rankings)

        def remember_selected_run(run_dir: str) -> None:
            persist_config(sequence_ranking_run_dir=run_dir)

        dialog.run_selected.connect(remember_selected_run)
        _exec_dialog(dialog)

    def browse_density_map() -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            window,
            "Select density map",
            str(REPO_ROOT),
            "Density maps (*.mrc *.map *.ccp4);;All files (*)",
        )
        if filename:
            ui.density_map_edit.setText(filename)
            persist_config(last_density_map=Path(filename).expanduser().resolve())

    def browse_candidate_pool() -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            window,
            "Select candidate sequence pool",
            str(REPO_ROOT),
            "FASTA sequence pools (*.fasta *.fa *.faa *.fna);;All files (*)",
        )
        if filename:
            ui.candidate_pool_edit.setText(filename)
            persist_config(last_candidate_pool=Path(filename).expanduser().resolve())

    def start_pipeline(*_args, setup_checked: bool = False) -> None:
        if readiness_issues():
            show_setup_issues()
            return
        if not persist_startup_settings(snapshot=True):
            return
        if not setup_checked:
            # Recheck in the background so a repaired/changed environment or
            # bundle cannot be accepted using a stale startup report.
            refresh_diagnostics(run_after_checks=True)
            return
        density_map = ui.density_map_edit.text().strip()
        candidate_pool = ui.candidate_pool_edit.text().strip()
        project_dir = current_project_dir()
        if not density_map or not candidate_pool:
            append_log("Run blocked: provide both a density map and a candidate sequence pool.")
            return
        if project_dir is None:
            append_log("Run blocked: select a project with File > New or Open.")
            return
        weights_path = settings_state.get("weights_path", "").strip()
        weights_dir = Path(weights_path).expanduser().resolve() if weights_path else ""
        weights_check = check_weights(weights_path)
        report = diagnostics_state["report"]
        report.weights = weights_check
        if not weights_check.ok:
            refresh_run_availability()
            show_setup_issues()
            return
        if state.get("thread") is not None:
            append_log("A run is already active.")
            return

        try:
            project_dir.mkdir(parents=True, exist_ok=True)
            ensure_project_dirs(project_dir)
        except OSError as exc:
            append_log(f"Run blocked: could not create project folder {project_dir}: {exc}")
            return
        local_config_state["project_dir"] = str(project_dir)
        if not persist_config(
            snapshot=True,
            project_dir=project_dir,
            weights_path=weights_dir,
            device=settings_state["device"],
            cpu_threads=settings_state["cpu_threads"],
            rcl=settings_state.get("rcl", "0"),
            protein_min_trace_length=settings_state.get(
                "protein_min_trace_length", "10"
            ),
            chimerax_path=settings_state.get("chimerax_path", ""),
            last_density_map=Path(density_map).expanduser().resolve(),
            last_candidate_pool=Path(candidate_pool).expanduser().resolve(),
            sequence_mode=settings_state.get("sequence_mode", "auto"),
            structure_selection=settings_state.get("structure_selection", ""),
            keep_tmp=settings_state.get("keep_tmp", "false") == "true",
        ):
            return
        density_map = ui.density_map_edit.text().strip()
        candidate_pool = ui.candidate_pool_edit.text().strip()
        ui.progress_bar.setRange(0, 100)
        ui.progress_bar.setValue(0)
        set_run_button_state(ui, running=True)
        run_status.start(
            report.device if report is not None else configured_device or "cpu",
            workload={
                "map_bytes": Path(density_map).stat().st_size,
                "pool_bytes": Path(candidate_pool).stat().st_size,
                "mode": settings_state.get("sequence_mode", "auto"),
                "structure_selection": settings_state.get("structure_selection", ""),
                "trace_length": settings_state.get("protein_min_trace_length", "10"),
                "rcl": settings_state.get("rcl", "0"),
                "hardware": report.compute.detail,
                "cpu_threads": settings_state["cpu_threads"],
                "cpu": next((line for line in report.lines if line.startswith("CPU:")), ""),
            },
        )
        refresh_run_status()
        run_clock.start()
        clear_input_highlight()
        append_log("Starting cryoID pipeline...")

        thread = QtCore.QThread(window)
        worker = PipelineWorker(
            density_map,
            candidate_pool,
            project_dir,
            str(weights_dir),
            float(settings_state.get("rcl", "0")),
            int(settings_state.get("protein_min_trace_length", "10")),
            settings_state.get("sequence_mode", "auto"),
            settings_state.get("structure_selection", ""),
            settings_state.get("keep_tmp", "false") == "true",
            device=report.device if report is not None else configured_device,
            cpu_threads=settings_state["cpu_threads"],
            chimerax_path=settings_state.get("chimerax_path", ""),
        )
        worker.moveToThread(thread)

        def on_finished(result: object) -> None:
            close_after_abort = close_state["pending"]
            ui.progress_bar.setRange(0, 100)
            result_statuses = {str(stage.status) for stage in getattr(result, "stages", [])}
            cancelled = "skipped" in result_statuses or getattr(worker, "_cancel_requested", False)
            label = (
                "Failed" if "error" in result_statuses else
                "Cancelled" if cancelled else
                "Complete" if result_statuses == {"ok"} else "Blocked"
            )
            run_status.finish(label, reset_progress=cancelled)
            run_clock.stop()
            refresh_run_status()
            set_run_button_state(ui, running=False)
            state["thread"] = None
            state["worker"] = None
            refresh_run_availability()
            if hasattr(result, "stages"):
                for stage in result.stages:
                    append_log(f"{stage.name}: {stage.status} - {stage.message}")
                ambiguous_mode = next(
                    (
                        stage
                        for stage in result.stages
                        if stage.name == "Sequence mode"
                        and stage.status != "ok"
                        and ("mixed" in stage.message.lower() or "ambiguous" in stage.message.lower())
                    ),
                    None,
                )
                if ambiguous_mode is not None:
                    QtWidgets.QMessageBox.warning(
                        window,
                        "Sequence mode could not be detected",
                        ambiguous_mode.message
                        + "\n\nChoose Protein or RNA using Options.",
                    )
                config_updates: dict[str, object] = {
                    "project_dir": project_dir,
                    "last_density_map": Path(density_map).expanduser().resolve(),
                    "last_candidate_pool": Path(candidate_pool).expanduser().resolve(),
                    "weights_path": settings_state.get("weights_path", ""),
                    "rcl": settings_state.get("rcl", "0"),
                    "chimerax_path": settings_state.get("chimerax_path", ""),
                    "sequence_mode": settings_state.get("sequence_mode", "auto"),
                    "structure_selection": settings_state.get("structure_selection", ""),
                    "keep_tmp": settings_state.get("keep_tmp", "false"),
                }
                result_run_dir = Path(result.run_dir)
                if (result_run_dir / "run_config.json").is_file():
                    config_updates["last_run_dir"] = result_run_dir
                persist_config(
                    **config_updates,
                )
                refresh_result_actions()
                log_path = Path(result.run_dir) / "pipeline.log"
                if log_path.is_file():
                    append_log(f"Run log: {log_path}")
                if result.selected_hit is not None:
                    append_log(
                        f"Top sequence: {result.selected_hit.record_id} "
                        f"(score {result.selected_hit.score:.3f})"
                    )
                statuses = {str(stage.status) for stage in result.stages}
                if "error" in statuses:
                    append_log("Run finished with errors.")
                elif "skipped" in statuses:
                    append_log("Run cancelled.")
                elif any(status.startswith("blocked") for status in statuses):
                    append_log("Run blocked before completion.")
                else:
                    append_log("Run finished.")
            else:
                append_log("Run finished.")
            if close_after_abort:
                append_log("Abort finished. Closing window.")
                QtCore.QTimer.singleShot(0, window.close)

        thread.started.connect(worker.run)
        worker.log.connect(append_log)
        worker.progress.connect(update_run_progress)
        worker.status.connect(update_stage)
        worker.finished.connect(on_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        state["thread"] = thread
        state["worker"] = worker
        thread.start()

    def abort_pipeline(checked: bool = False, *, close_requested: bool = False) -> None:
        worker = state.get("worker")
        if worker is None:
            append_log("No active run to abort.")
            return
        already_requested = bool(getattr(worker, "_cancel_requested", False))
        worker.cancel()
        set_run_button_state(ui, running=True, cancelling=True)
        run_status.set_stage({"name": "Cancelling / waiting for cleanup", "device": run_status.device})
        refresh_run_status()
        clear_input_highlight()
        QtCore.QTimer.singleShot(0, clear_input_highlight)
        if close_requested:
            if already_requested:
                append_log("Close requested. Waiting for the active abort to finish before closing.")
            else:
                append_log("Close requested. Aborting active run before closing.")
        elif already_requested:
            append_log("Abort is already pending. Waiting for unfinished outputs to be cleared.")
        else:
            append_log("Abort requested. Current work will stop and unfinished outputs will be cleared.")

    def handle_close_request(event: QtGui.QCloseEvent) -> bool:
        if state.get("worker") is None:
            if not remember_edited_inputs():
                event.ignore()
                return True
            return False
        event.ignore()
        close_state["pending"] = True
        abort_pipeline(close_requested=True)
        return True

    window.close_request_handler = handle_close_request

    def activate_project(project: Path, *, create_root: Path | None = None) -> None:
        try:
            latest_global = load_user_config(strict=True)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(window, "Could not load startup settings", str(exc))
            return
        apply_startup_settings(latest_global)
        config_path = project_config_path(project)
        try:
            loaded = load_effective_config(
                project, user_config=dict(DEFAULT_CONFIG, **startup_state),
                use_project_config=create_root is None,
            )
            saved_settings = loaded.values
            # Read the destination before saving the outgoing project. When
            # reopening this project, merge any typed inputs before reloading
            # so neither local inputs nor external settings edits are lost.
            same_project = project == current_project_dir()
            inputs_changed = (
                ui.density_map_edit.text().strip() != local_config_state.get("last_density_map", "")
                or ui.candidate_pool_edit.text().strip() != local_config_state.get("last_candidate_pool", "")
            )
            if project_state["active"] and (not same_project or inputs_changed):
                if not persist_config():
                    return
                if same_project:
                    saved_settings = load_effective_config(
                        project, user_config=dict(DEFAULT_CONFIG, **startup_state)
                    ).values
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(
                window, "Could not open project",
                f"The saved project settings could not be read.\n\n{config_path}\n\n{exc}",
            )
            return
        created_root = False
        try:
            if create_root is not None:
                # Reserve a new folder atomically; never reuse an existing
                # project or read its settings through File > New.
                try:
                    create_root.mkdir()
                    created_root = True
                except FileExistsError:
                    QtWidgets.QMessageBox.warning(
                        window, "Project folder already exists",
                        "Choose a different project name, or use File > Open to resume an existing project.\n\n" + str(create_root),
                    )
                    return
            ensure_project_dirs(project)
        except OSError as exc:
            if created_root:
                discard_uninitialized_project(create_root, project)
            QtWidgets.QMessageBox.warning(window, "Could not open project",
                                          f"The results folders could not be created.\n\n{project}\n\n{exc}")
            return
        saved_settings["project_dir"] = str(project)
        try:
            saved_settings = save_project_config(gui_config_view(saved_settings), project,
                                                 baseline=saved_settings, initialize=True)
            update_config(project_dir=str(project))
        except (OSError, ValueError, TimeoutError) as exc:
            if created_root:
                discard_uninitialized_project(create_root, project)
            QtWidgets.QMessageBox.warning(window, "Could not open project",
                                          f"The project configuration could not be saved.\n\n{config_path}\n\n{exc}")
            return
        apply_saved_settings(saved_settings)
        project_state["active"] = True
        refresh_project_title()
        append_log(f"{'Created' if create_root is not None else 'Opened'} project: {project}")
        run_status.reset()
        refresh_run_status()
        refresh_result_actions()
        refresh_diagnostics()

    def new_project() -> None:
        if state.get("worker") is not None:
            QtWidgets.QMessageBox.warning(window, "Run active", "Abort or wait for completion before creating a new project.")
            return
        dialog = NewProjectDialog(
            window, location=current_project_dir().parent.parent if current_project_dir() else REPO_ROOT,
        )
        if _exec_dialog(dialog) != _dialog_accepted_code():
            return
        try:
            root = dialog.project_root()
            project = results_project_dir(root)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.warning(window, "Invalid project location", str(exc))
            return
        activate_project(project, create_root=root)

    def open_project() -> None:
        if state.get("worker") is not None:
            QtWidgets.QMessageBox.warning(window, "Run active", "Abort or wait for completion before opening another project.")
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            window,
            "Open results parent or cryoid_results directory",
            str(current_project_dir() or REPO_ROOT),
        )
        if directory:
            selected = Path(directory).expanduser().resolve()
            project = results_project_dir(selected)
            # An explicitly selected project takes priority over an ancestor
            # project when the user creates projects inside another folder.
            if not project.is_dir():
                selected_run = find_run_dir_from_path(selected)
                detected_project = find_project_dir_from_path(selected)
                if detected_project is None and selected_run is not None and selected_run.parent.name == "runs":
                    detected_project = selected_run.parent.parent
                project = results_project_dir(detected_project or selected)
            if not is_project_directory(project):
                QtWidgets.QMessageBox.warning(
                    window, "No CryoID project found",
                    "Select an existing project, its cryoid_results folder, or a run folder. "
                    "Use File > New to create a project.\n\n" + str(selected),
                )
                return
            activate_project(project)

    def launch_files(files: list[Path]) -> bool:
        configured_chimerax = settings_state.get("chimerax_path") or None
        try:
            launch_chimerax(files, chimerax_path=configured_chimerax)
        except FileNotFoundError:
            message = "ChimeraX could not be found. Choose its executable in Settings > Configurations."
            QtWidgets.QMessageBox.critical(window, "ChimeraX not found", message)
            return False
        except Exception:
            QtWidgets.QMessageBox.critical(window, "ChimeraX launch failed",
                                          "ChimeraX could not start. Check that the selected program works "
                                          "on this computer, then try again.")
            return False
        return True

    def open_chimerax_session() -> None:
        project_dir = current_project_dir()
        density_map_text = ui.density_map_edit.text().strip()
        density_map = Path(density_map_text).expanduser().resolve() if density_map_text else None

        if project_dir is None:
            QtWidgets.QMessageBox.warning(window, "ChimeraX", "Select a CryoID results directory before opening ChimeraX.")
            return
        groups = collect_chimerax_project_files(project_dir, density_map)
        files = flatten_chimerax_session_files(groups)
        if not groups["density_maps"]:
            QtWidgets.QMessageBox.warning(window, "ChimeraX", "Provide an existing input density map.")
            return
        if not groups["aligned_models"]:
            QtWidgets.QMessageBox.warning(window, "ChimeraX", "No fitted model is available yet. Complete a run with structure fitting before opening a model.")
            return
        launch_files(files)

    def open_configurations() -> None:
        dialog = ConfigurationsWindow(
            window,
            weights_path=settings_state.get("weights_path", ""),
            chimerax_path=settings_state.get("chimerax_path", ""),
        )
        if _exec_dialog(dialog) != _dialog_accepted_code():
            return
        updates = {"weights_dir": dialog.weights_path(), "chimerax_path": dialog.chimerax_path()}
        if not persist_startup_settings(updates):
            return
        refresh_result_actions()
        refresh_diagnostics()

    def open_options() -> None:
        report = diagnostics_state["report"]
        dialog = OptionsWindow(
            window,
            rcl=float(settings_state.get("rcl", "0")),
            protein_min_trace_length=int(settings_state.get("protein_min_trace_length", "10")),
            sequence_mode=settings_state.get("sequence_mode", "auto"),
            structure_selection=settings_state.get("structure_selection", ""),
            keep_tmp=settings_state.get("keep_tmp", "false").lower() == "true",
            device=settings_state["device"],
            cpu_threads=settings_state["cpu_threads"],
            devices=report.devices if report is not None else (),
        )
        if _exec_dialog(dialog) != _dialog_accepted_code():
            return
        updates = {
            "rcl": str(dialog.rcl()),
            "protein_min_trace_length": str(dialog.protein_min_trace_length()),
            "sequence_mode": dialog.sequence_mode(),
            "structure_selection": dialog.structure_selection(),
            "keep_tmp": "true" if dialog.keep_tmp() else "false",
            "device": dialog.device(),
            "cpu_threads": dialog.cpu_threads(),
        }
        if not persist_config(reviewed_options=True, initialize=True, **updates):
            return
        startup_issues.clear()
        refresh_diagnostics()

    def open_repository(name: str, url: str) -> None:
        opened = QtGui.QDesktopServices.openUrl(QtCore.QUrl(url))
        if opened:
            return
        message = f"Could not open the default browser.\n\n{url}"
        QtWidgets.QMessageBox.warning(window, f"Could not open {name}", message)

    def update_theme_action() -> None:
        target = "Light" if current_theme == "dark" else "Dark"
        theme_action.setText(f"Switch to {target} Mode")
        theme_action.setToolTip(f"Switch to {target.lower()} mode")

    def toggle_theme() -> None:
        nonlocal current_theme
        current_theme = "light" if current_theme == "dark" else "dark"
        theme_preferences.setValue("theme", current_theme)
        theme_action.setChecked(current_theme == "dark")
        update_theme_action()
        apply_theme(app, window, ui, current_theme)

    theme_action = QtGui.QAction("Theme", window)
    theme_action.setCheckable(True)
    theme_action.setChecked(current_theme == "dark")
    update_theme_action()
    theme_action.triggered.connect(toggle_theme)

    def configure_menus() -> None:
        ui.menuFile.clear()
        ui.menuFile.addAction(ui.actionNew)
        ui.menuFile.addAction(ui.actionOpen)
        ui.actionNew.setToolTip("Create a new project folder with blank inputs")
        ui.actionOpen.setToolTip("Resume an existing CryoID project or run")
        standard_keys = getattr(QtGui.QKeySequence, "StandardKey", QtGui.QKeySequence)
        ui.actionNew.setShortcuts(QtGui.QKeySequence.keyBindings(standard_keys.New))
        ui.actionOpen.setShortcuts(QtGui.QKeySequence.keyBindings(standard_keys.Open))
        ui.menuView.clear()
        ui.menuView.addAction(sequence_ranking_action)
        ui.menuView.addAction(ui.actionModel)
        ui.menuView.addAction(ui.actionResultsFolder)
        ui.actionResultsFolder.setToolTip("Open project results folder")
        ui.menuSettings.clear()
        ui.menuSettings.addAction(ui.actionConfigurations)
        check_setup_action = ui.menuSettings.addAction("Check setup")
        check_setup_action.triggered.connect(check_setup)
        ui.menuSettings.addAction(theme_action)
        ui.menuHelp.clear()
        ui.menuHelp.addAction(ui.actionGitHub)
        ui.actionGitHub.setToolTip("Open CryoID GitHub repository")
        ui.menuHelp.addAction(ui.actionHuggingFace)
        ui.actionHuggingFace.setToolTip("Open the published CryoID weights on Hugging Face")
        ui.menubar.clear()
        ui.menubar.addAction(ui.menuFile.menuAction())
        ui.menubar.addAction(ui.menuView.menuAction())
        ui.menubar.addAction(ui.menuSettings.menuAction())
        ui.menubar.addAction(ui.menuHelp.menuAction())

    configure_menus()

    def restore_startup_state() -> None:
        project = Path(local_config_state.get("project_dir") or results_project_dir(REPO_ROOT)).expanduser().resolve()
        local_config_state["project_dir"] = str(project)
        if project.is_dir():
            ensure_project_dirs(project)
            refresh_result_actions()

        density_text = local_config_state.get("last_density_map", "")
        if density_text:
            density_path = Path(density_text).expanduser()
            ui.density_map_edit.setText(str(density_path))

        pool_text = local_config_state.get("last_candidate_pool", "")
        if pool_text:
            pool_path = Path(pool_text).expanduser()
            ui.candidate_pool_edit.setText(str(pool_path))

    restore_startup_state()
    refresh_project_title()
    refresh_result_actions()
    ui.log_edit.clear()
    append_log(startup_status_message())

    ui.browse_density_button.clicked.connect(browse_density_map)
    ui.browse_pool_button.clicked.connect(browse_candidate_pool)
    ui.run_button.clicked.connect(
        lambda: abort_pipeline() if state.get("worker") is not None else start_pipeline()
    )
    ui.model_button.clicked.connect(open_chimerax_session)

    ui.actionNew.triggered.connect(new_project)
    ui.actionOpen.triggered.connect(open_project)
    ui.actionConfigurations.triggered.connect(open_configurations)
    ui.actionGitHub.triggered.connect(lambda: open_repository("GitHub", CRYOID_GITHUB_URL))
    ui.actionHuggingFace.triggered.connect(lambda: open_repository("Hugging Face", CRYOID_HUGGINGFACE_URL))
    ui.actionModel.triggered.connect(lambda: launch_files([]))
    ui.actionResultsFolder.triggered.connect(open_results_folder)
    sequence_ranking_action.triggered.connect(lambda: open_sequence_ranking())
    ui.options_button.clicked.connect(open_options)
    ui.density_map_edit.textChanged.connect(refresh_run_availability)
    ui.candidate_pool_edit.textChanged.connect(refresh_run_availability)
    ui.density_map_edit.editingFinished.connect(remember_edited_inputs)
    ui.candidate_pool_edit.editingFinished.connect(remember_edited_inputs)
    ui.ranking_button.clicked.connect(lambda: open_sequence_ranking())
    refresh_run_availability()
    apply_theme(app, window, ui, current_theme)
    QtCore.QTimer.singleShot(0, refresh_diagnostics)
    smoke_test_ms = os.environ.get("CRYOID_GUI_SMOKE_TEST_MS", "").strip()
    if smoke_test_ms:
        QtCore.QTimer.singleShot(max(1, int(smoke_test_ms)), app.quit)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
