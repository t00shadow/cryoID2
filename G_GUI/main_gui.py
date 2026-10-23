import csv
import os
import subprocess
import sys
import uuid
from pathlib import Path

try:
    from G_GUI.qt_compat import QtCore, QtGui, QtWidgets
except ImportError as exc:
    raise SystemExit(
        "CryoID GUI cannot start because no Qt binding is installed. "
        "Install one GUI extra, for example `pip install .[gui-pyqt6]` or `pip install .[gui-pyqt5]`."
    ) from exc

from G_GUI.gui_skin import CryoIDUI

from cryoid_pipeline import __version__  # noqa: E402
from cryoid_pipeline.cli import decode_event, format_command  # noqa: E402
from D_BLAST_searching.stage import (  # noqa: E402
    autodetect_blast_bin_dir,
    validate_blast_bin_dir,
)
from cryoid_pipeline.models import PipelineResult, StageResult, pipeline_result_from_dict  # noqa: E402
from cryoid_pipeline.runtime_config import (  # noqa: E402
    RuntimeConfig,
    display_problem,
)
from cryoid_pipeline.user_config import (  # noqa: E402
    load_config as load_user_config,
    resolve_chimerax_path,
    resolve_device,
    resolve_keep_tmp,
    resolve_project_dir,
    resolve_protein_min_trace_length,
    resolve_rcl,
    resolve_sequence_mode,
    resolve_structure_selection,
    resolve_weights_dir,
    results_project_dir,
)

from G_GUI.session_files import (
    collect_chimerax_project_files,
    ensure_project_dirs,
    find_project_dir_from_path,
    find_run_dir_from_path,
    flatten_chimerax_session_files,
    selected_sequence_ranking,
    sequence_ranking_files,
)
from G_GUI.settings_dialogs import ConfigurationsWindow, OptionsWindow
from G_GUI.chimerax_launcher import find_chimerax, launch_chimerax
from G_GUI.local_config import ensure_local_config, save_local_config


CRYOID_GITHUB_URL = "https://github.com/t00shadow/cryoID_v2"
CONFIG = RuntimeConfig.from_env()
REPO_ROOT = CONFIG.repo_root


class PipelineWorker(QtCore.QObject):
    log = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(int)
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
        blast_bin_dir: str,
    ) -> None:
        super().__init__()
        self.density_map = density_map
        self.candidate_pool = candidate_pool
        self.project_dir = project_dir
        self.weights_path = weights_path
        self.rcl = float(rcl)
        self.protein_min_trace_length = int(protein_min_trace_length)
        self.sequence_mode = str(sequence_mode)
        self.structure_selection = str(structure_selection).strip()
        self.keep_tmp = bool(keep_tmp)
        self.blast_bin_dir = str(blast_bin_dir)
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
            self.density_map,
            self.candidate_pool,
            "--project-dir",
            str(self.project_dir),
            "--repo-root",
            str(REPO_ROOT),
            "--event-stream",
            "--cancel-file",
            str(self._cancel_file),
        ]
        if self.weights_path:
            command.extend(["--weights-dir", str(Path(self.weights_path).expanduser().resolve())])
        if self.blast_bin_dir:
            command.extend(["--blast-bin-dir", str(Path(self.blast_bin_dir).expanduser().resolve())])
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
        if self.structure_selection:
            command.extend(["--structure-selection", self.structure_selection])
        if self.keep_tmp:
            command.append("--keep-tmp")

        command_text = format_command(command)
        self._publish(f"Command: {command_text}")
        env = os.environ.copy()
        env.update(
            {
                "PYTHONUNBUFFERED": "1",
                "PYTHONUTF8": "1",
                "CRYOID_REPO_ROOT": str(REPO_ROOT),
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


def _detect_chimerax_path() -> str:
    try:
        return str(find_chimerax())
    except FileNotFoundError:
        return ""


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
        except Exception as exc:
            self.failed.emit(str(exc))


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
        self.run_combo.setToolTip(
            "Choose which run's sequence_ranking.tsv to view. The saved choice is reused "
            "until another run is selected."
        )
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
        QLineEdit, QPlainTextEdit, QListWidget, QTableView, QComboBox {{
            background: {field}; color: {text}; border: 1px solid {border};
            border-radius: 4px; selection-background-color: {accent};
            selection-color: white;
        }}
        QLineEdit, QComboBox {{ padding: 4px; }}
        QPushButton {{ background: {button}; color: {text}; border: 1px solid {button_border};
                       border-radius: 5px; padding: 5px 10px; }}
        QPushButton:hover {{ border-color: {accent}; }}
        QPushButton:disabled {{ color: {muted}; border-color: {border}; }}
        QPushButton#run_button {{ background: #2563eb; color: white; border: none; }}
        QPushButton#run_button:hover {{ background: #1d4ed8; }}
        QPushButton#run_button[running="true"] {{ background: #b91c1c; }}
        QPushButton#run_button[running="true"]:hover {{ background: #991b1b; }}
        QPushButton#run_button:disabled {{ background: {border}; color: {muted}; }}
        QProgressBar {{ background: {field if dark else border}; border: none; border-radius: 3px; }}
        QProgressBar::chunk {{ background: {accent}; border-radius: 3px; }}
        QLabel#progress_percent {{ color: {muted}; }}
        QGroupBox {{ border: 1px solid {border}; border-radius: 6px;
                     margin-top: 10px; font-weight: 600; color: {text}; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 5px; }}
    """)


def set_run_button_state(ui, *, running: bool, cancelling: bool = False) -> None:
    ui.run_button.setText("Aborting…" if cancelling else "Abort" if running else "Run")
    ui.run_button.setToolTip("Abort the current job" if running else "Run the pipeline")
    ui.run_button.setEnabled(not cancelling)
    ui.run_button.setProperty("running", running)
    ui.run_button.style().unpolish(ui.run_button)
    ui.run_button.style().polish(ui.run_button)



def main() -> None:
    display_issue = display_problem()
    if display_issue:
        raise SystemExit(f"CryoID GUI cannot start: {display_issue}")

    app = QtWidgets.QApplication(sys.argv)
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
    local_config_state = ensure_local_config()
    shared_config_state = load_user_config()
    resolved_project_dir = resolve_project_dir(None, shared_config_state, default=REPO_ROOT)
    local_config_state["project_dir"] = str(resolved_project_dir)
    ui.progress_bar.setRange(0, 100)
    ui.progress_bar.setValue(0)

    state: dict[str, object] = {
        "thread": None,
        "worker": None,
        "last_running_progress": 0,
    }
    sequence_ranking_action = QtGui.QAction("Sequence Ranking", window)
    sequence_ranking_action.setToolTip(
        "Open a run's sequence ranking and choose which run to keep viewing"
    )
    detected_chimerax = _detect_chimerax_path()
    configured_weights = resolve_weights_dir(None, shared_config_state)
    configured_chimerax = resolve_chimerax_path(shared_config_state)
    configured_device = resolve_device(None, shared_config_state)
    configured_rcl = resolve_rcl(None, shared_config_state)
    configured_protein_min_trace_length = resolve_protein_min_trace_length(
        None, shared_config_state
    )
    configured_sequence_mode = resolve_sequence_mode(None, shared_config_state)
    configured_structure_selection = resolve_structure_selection(None, shared_config_state)
    configured_keep_tmp = resolve_keep_tmp(None, shared_config_state)
    configured_blast = str(shared_config_state.get("blast_bin_dir") or "").strip()
    detected_blast = autodetect_blast_bin_dir()
    settings_state: dict[str, str] = {
        "weights_path": str(configured_weights or ""),
        "rcl": str(configured_rcl),
        "protein_min_trace_length": str(configured_protein_min_trace_length),
        "chimerax_path": configured_chimerax or detected_chimerax,
        "sequence_mode": configured_sequence_mode,
        "structure_selection": str(configured_structure_selection or ""),
        "keep_tmp": "true" if configured_keep_tmp else "false",
        "blast_bin_dir": configured_blast or str(detected_blast or ""),
    }
    close_state = {"pending": False}

    def append_log(message: str) -> None:
        ui.log_edit.appendPlainText(str(message))

    def update_run_progress(value: int) -> None:
        if value < 100:
            state["last_running_progress"] = value
        ui.progress_bar.setValue(value)

    def persist_config(**updates: object) -> None:
        for key, value in updates.items():
            local_config_state[key] = str(value) if value is not None else ""
        local_config_state.update(save_local_config(local_config_state))

    def current_project_dir() -> Path | None:
        selected = str(local_config_state.get("project_dir") or "").strip()
        return results_project_dir(selected) if selected else None

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
        sequence_ranking_action.setEnabled(True)
        ui.model_button.setEnabled(has_model)
        ui.actionModel.setEnabled(True)

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

    def start_pipeline() -> None:
        density_map = ui.density_map_edit.text().strip()
        candidate_pool = ui.candidate_pool_edit.text().strip()
        project_dir = current_project_dir()
        if not density_map or not candidate_pool:
            append_log("Run blocked: provide both a density map and a candidate sequence pool.")
            return
        if project_dir is None:
            append_log("Run blocked: select an output parent directory.")
            return
        weights_path = settings_state.get("weights_path", "").strip()
        if not weights_path:
            append_log("Run blocked: set the weights path in Settings > Configurations.")
            return
        weights_dir = Path(weights_path).expanduser().resolve()
        if not weights_dir.is_dir():
            append_log(f"Run blocked: weights path does not exist: {weights_dir}")
            return
        blast_bin_dir = settings_state.get("blast_bin_dir", "").strip()
        if blast_bin_dir:
            try:
                blast_bin_dir = str(validate_blast_bin_dir(blast_bin_dir))
            except ValueError as exc:
                append_log(f"Run blocked: {exc}")
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
        persist_config(
            project_dir=project_dir,
            weights_path=weights_dir,
            rcl=settings_state.get("rcl", "0"),
            protein_min_trace_length=settings_state.get(
                "protein_min_trace_length", "10"
            ),
            chimerax_path=settings_state.get("chimerax_path", ""),
            blast_bin_dir=blast_bin_dir,
            last_density_map=Path(density_map).expanduser().resolve(),
            last_candidate_pool=Path(candidate_pool).expanduser().resolve(),
            sequence_mode=settings_state.get("sequence_mode", "auto"),
            structure_selection=settings_state.get("structure_selection", ""),
            keep_tmp=settings_state.get("keep_tmp", "false") == "true",
        )
        ui.progress_bar.setRange(0, 100)
        ui.progress_bar.setValue(0)
        state["last_running_progress"] = 0
        set_run_button_state(ui, running=True)
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
            blast_bin_dir,
        )
        worker.moveToThread(thread)

        def on_finished(result: object) -> None:
            close_after_abort = close_state["pending"]
            ui.progress_bar.setRange(0, 100)
            result_statuses = {str(stage.status) for stage in getattr(result, "stages", [])}
            if result_statuses and result_statuses == {"ok"}:
                ui.progress_bar.setValue(100)
            else:
                ui.progress_bar.setValue(int(state.get("last_running_progress", 0)))
            set_run_button_state(ui, running=False)
            state["thread"] = None
            state["worker"] = None
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
                    "weights_path": weights_dir,
                    "rcl": settings_state.get("rcl", "0"),
                    "chimerax_path": settings_state.get("chimerax_path", ""),
                    "blast_bin_dir": settings_state.get("blast_bin_dir", ""),
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
            return False
        event.ignore()
        close_state["pending"] = True
        abort_pipeline(close_requested=True)
        return True

    window.close_request_handler = handle_close_request

    def new_project() -> None:
        if state.get("worker") is not None:
            QtWidgets.QMessageBox.warning(window, "Run active", "Abort or wait for completion before creating a new project.")
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            window,
            "Select or create results parent directory",
            str(REPO_ROOT),
        )
        if not directory:
            return
        project = results_project_dir(directory)
        ensure_project_dirs(project)
        ui.density_map_edit.clear()
        ui.candidate_pool_edit.clear()
        local_config_state["project_dir"] = str(project)
        ui.progress_bar.setRange(0, 100)
        ui.progress_bar.setValue(0)
        persist_config(
            project_dir=project,
            last_run_dir="",
            sequence_ranking_run_dir="",
            last_density_map="",
            last_candidate_pool="",
            weights_path=settings_state.get("weights_path", ""),
            rcl=settings_state.get("rcl", "0"),
            protein_min_trace_length=settings_state.get(
                "protein_min_trace_length", "10"
            ),
            chimerax_path=settings_state.get("chimerax_path", ""),
            blast_bin_dir=settings_state.get("blast_bin_dir", ""),
            sequence_mode=settings_state.get("sequence_mode", "auto"),
            structure_selection=settings_state.get("structure_selection", ""),
            keep_tmp=settings_state.get("keep_tmp", "false") == "true",
        )
        refresh_result_actions()

    def open_project() -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            window,
            "Open results parent or cryoid_results directory",
            str(current_project_dir() or REPO_ROOT),
        )
        if directory:
            selected = Path(directory).expanduser().resolve()
            selected_run = find_run_dir_from_path(selected)
            detected_project = find_project_dir_from_path(selected)
            project = results_project_dir(detected_project or selected)
            ensure_project_dirs(project)
            local_config_state["project_dir"] = str(project)
            if selected_run is not None:
                config_updates: dict[str, object] = {
                    "project_dir": project,
                    "last_run_dir": selected_run,
                }
                if (selected_run / "sequence_ranking.tsv").is_file():
                    config_updates["sequence_ranking_run_dir"] = selected_run
                persist_config(**config_updates)
            else:
                persist_config(project_dir=project, last_run_dir="")
            refresh_result_actions()

    def launch_files(files: list[Path]) -> bool:
        configured_chimerax = settings_state.get("chimerax_path") or None
        try:
            launch_chimerax(files, chimerax_path=configured_chimerax)
        except FileNotFoundError as exc:
            message = (
                f"{exc}\n\nSet the ChimeraX executable path in Settings > Configurations."
            )
            QtWidgets.QMessageBox.critical(window, "ChimeraX not found", message)
            return False
        except Exception as exc:
            QtWidgets.QMessageBox.critical(window, "ChimeraX launch failed", str(exc))
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
            QtWidgets.QMessageBox.warning(window, "ChimeraX", "No best aligned model is recorded in best_overall.")
            return
        launch_files(files)

    def open_configurations() -> None:
        dialog = ConfigurationsWindow(
            window,
            weights_path=settings_state.get("weights_path", ""),
            chimerax_path=settings_state.get("chimerax_path", ""),
            detected_chimerax_path=_detect_chimerax_path(),
            blast_bin_dir=settings_state.get("blast_bin_dir", ""),
            detected_blast_bin_dir=str(autodetect_blast_bin_dir() or ""),
            results_directory=str(current_project_dir() or results_project_dir(REPO_ROOT)),
        )
        active_run = state.get("worker") is not None
        dialog.results_directory_edit.setEnabled(not active_run)
        dialog.browse_results_button.setEnabled(not active_run)
        if _exec_dialog(dialog) != _dialog_accepted_code():
            return
        project = results_project_dir(dialog.results_directory())
        try:
            ensure_project_dirs(project)
        except OSError as exc:
            QtWidgets.QMessageBox.warning(window, "Invalid results location", str(exc))
            return
        updates = {
            "weights_path": dialog.weights_path(),
            "chimerax_path": dialog.chimerax_path(),
            "blast_bin_dir": dialog.blast_bin_dir(),
        }
        settings_state.update(updates)
        project_updates = {"project_dir": project}
        if project != current_project_dir():
            project_updates.update(last_run_dir="", sequence_ranking_run_dir="")
        persist_config(**updates, **project_updates)
        refresh_result_actions()

    def open_options() -> None:
        dialog = OptionsWindow(
            window,
            rcl=float(settings_state.get("rcl", "0")),
            protein_min_trace_length=int(settings_state.get("protein_min_trace_length", "10")),
            sequence_mode=settings_state.get("sequence_mode", "auto"),
            structure_selection=settings_state.get("structure_selection", ""),
            keep_tmp=settings_state.get("keep_tmp", "false").lower() == "true",
        )
        if _exec_dialog(dialog) != _dialog_accepted_code():
            return
        updates = {
            "rcl": str(dialog.rcl()),
            "protein_min_trace_length": str(dialog.protein_min_trace_length()),
            "sequence_mode": dialog.sequence_mode(),
            "structure_selection": dialog.structure_selection(),
            "keep_tmp": "true" if dialog.keep_tmp() else "false",
        }
        settings_state.update(updates)
        persist_config(**updates)

    def open_github() -> None:
        opened = QtGui.QDesktopServices.openUrl(QtCore.QUrl(CRYOID_GITHUB_URL))
        if opened:
            return
        message = f"Could not open the default browser.\n\n{CRYOID_GITHUB_URL}"
        QtWidgets.QMessageBox.warning(window, "Could not open GitHub", message)

    def toggle_theme() -> None:
        nonlocal current_theme
        current_theme = "light" if current_theme == "dark" else "dark"
        theme_preferences.setValue("theme", current_theme)
        theme_action.setChecked(current_theme == "dark")
        apply_theme(app, window, ui, current_theme)

    theme_action = QtGui.QAction("Theme", window)
    theme_action.setCheckable(True)
    theme_action.setChecked(current_theme == "dark")
    theme_action.setToolTip("Switch between light and dark mode (checked means dark)")
    theme_action.triggered.connect(toggle_theme)

    def configure_menus() -> None:
        ui.menuFile.clear()
        ui.menuFile.addAction(ui.actionNew)
        ui.menuFile.addAction(ui.actionOpen)
        ui.menuView.clear()
        ui.menuView.addAction(sequence_ranking_action)
        ui.menuView.addAction(ui.actionModel)
        ui.menuSettings.clear()
        ui.menuSettings.addAction(ui.actionConfigurations)
        ui.menuSettings.addAction(theme_action)
        ui.menuHelp.clear()
        ui.menuHelp.addAction(ui.actionGitHub)
        ui.actionGitHub.setToolTip("Open the CryoIDv2 GitHub repository")
        ui.menubar.clear()
        ui.menubar.addAction(ui.menuFile.menuAction())
        ui.menubar.addAction(ui.menuView.menuAction())
        ui.menubar.addAction(ui.menuSettings.menuAction())
        ui.menubar.addAction(ui.menuHelp.menuAction())

    configure_menus()

    def restore_startup_state() -> None:
        project_text = local_config_state.get("project_dir", "")
        if project_text:
            project = Path(project_text).expanduser()
            if project.is_dir():
                project = project.resolve()
                ensure_project_dirs(project)
                local_config_state["project_dir"] = str(project)
                refresh_result_actions()
            else:
                local_config_state["project_dir"] = str(REPO_ROOT.resolve())
        else:
            local_config_state["project_dir"] = str(REPO_ROOT.resolve())

        density_text = local_config_state.get("last_density_map", "")
        if density_text:
            density_path = Path(density_text).expanduser()
            if density_path.is_file():
                ui.density_map_edit.setText(str(density_path.resolve()))
            else:
                ui.density_map_edit.clear()

        pool_text = local_config_state.get("last_candidate_pool", "")
        if pool_text:
            pool_path = Path(pool_text).expanduser()
            if pool_path.is_file():
                ui.candidate_pool_edit.setText(str(pool_path.resolve()))
            else:
                ui.candidate_pool_edit.clear()

    restore_startup_state()
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
    ui.actionGitHub.triggered.connect(open_github)
    ui.actionModel.triggered.connect(lambda: launch_files([]))
    sequence_ranking_action.triggered.connect(lambda: open_sequence_ranking())
    ui.options_button.clicked.connect(open_options)
    ui.ranking_button.clicked.connect(lambda: open_sequence_ranking())
    set_run_button_state(ui, running=False)
    apply_theme(app, window, ui, current_theme)
    window.show()
    smoke_test_ms = os.environ.get("CRYOID_GUI_SMOKE_TEST_MS", "").strip()
    if smoke_test_ms:
        QtCore.QTimer.singleShot(max(1, int(smoke_test_ms)), app.quit)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
