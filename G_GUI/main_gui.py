import csv
import os
import shutil
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

from G_GUI.gui_skin import Ui_cryoID

from cryoid_pipeline import __version__  # noqa: E402
from cryoid_pipeline.cli import decode_event, format_command  # noqa: E402
from D_BLAST_searching.stage import (  # noqa: E402
    autodetect_blast_bin_dir,
    validate_blast_bin_dir,
)
from cryoid_pipeline.models import PipelineResult, StageResult, pipeline_result_from_dict  # noqa: E402
from cryoid_pipeline.runtime_config import (  # noqa: E402
    RuntimeConfig,
    cuda_status_message,
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

try:
    from session_files import (
        collect_chimerax_project_files,
        collect_project_display_files,
        collect_result_display_files,
        ensure_project_dirs,
        find_project_dir_from_path,
        find_run_dir_from_path,
        flatten_chimerax_session_files,
        latest_sequence_ranking,
    )
except ImportError:  # Allows `python -m G_GUI.main_gui` from the repo root.
    from G_GUI.session_files import (
        collect_chimerax_project_files,
        collect_project_display_files,
        collect_result_display_files,
        ensure_project_dirs,
        find_project_dir_from_path,
        find_run_dir_from_path,
        flatten_chimerax_session_files,
        latest_sequence_ranking,
    )

try:
    from advanced_settings import AdvancedSettingsWindow
except ImportError:  # Allows `python -m G_GUI.main_gui` from the repo root.
    from G_GUI.advanced_settings import AdvancedSettingsWindow

try:
    from chimerax_launcher import find_chimerax, launch_chimerax
except ImportError:  # Allows `python -m G_GUI.main_gui` from the repo root.
    from G_GUI.chimerax_launcher import find_chimerax, launch_chimerax

try:
    from local_config import ensure_local_config, save_local_config
except ImportError:  # Allows `python -m G_GUI.main_gui` from the repo root.
    from G_GUI.local_config import ensure_local_config, save_local_config


PLACEHOLDERS = {"your_cryoEM_map.mrc", "your_pool_file", "your_pool_file.fasta", "output_file_path"}
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


def _usable_text(value: str) -> str:
    text = value.strip()
    return "" if text in PLACEHOLDERS else text


def _project_dir_from_text(value: str) -> Path | None:
    text = _usable_text(value)
    if not text:
        return None
    path = Path(text).expanduser()
    return path.resolve()


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
    return f"{cuda_status_message()}\nCryoID version {__version__}"


def _item_path(item: QtWidgets.QListWidgetItem | None) -> Path | None:
    if item is None:
        return None
    raw = item.data(QtCore.Qt.ItemDataRole.UserRole)
    return Path(str(raw)) if raw else None


def _result_label(path: Path, run_dir: Path | None) -> str:
    if path.name == "sequence_ranking.tsv":
        detected_run = path.parent
        mode = ""
        try:
            import json

            payload = json.loads((detected_run / "run_config.json").read_text(encoding="utf-8"))
            mode = str(payload.get("effective_mode") or "").upper()
        except (OSError, ValueError):
            pass
        suffix = f" | {mode}" if mode else ""
        return f"[{detected_run.name}{suffix}] sequence_ranking.tsv"
    if run_dir is None:
        return path.name
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except ValueError:
        return path.name


def _style_link_item(item: QtWidgets.QListWidgetItem) -> None:
    font = item.font()
    font.setUnderline(True)
    item.setFont(font)
    item.setForeground(QtGui.QBrush(QtGui.QColor("#7db1ff")))


def _open_file_location(path: Path) -> None:
    target = path.expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Result path no longer exists: {target}")
    if sys.platform.startswith("win"):
        if target.is_file():
            subprocess.Popen(["explorer", f"/select,{target}"])
        else:
            subprocess.Popen(["explorer", str(target)])
        return
    if sys.platform == "darwin":
        if target.is_file():
            subprocess.Popen(["open", "-R", str(target)])
        else:
            subprocess.Popen(["open", str(target)])
        return

    directory = target if target.is_dir() else target.parent
    opener = shutil.which("xdg-open")
    if opener:
        subprocess.Popen([opener, str(directory)])
        return
    opened = QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(directory)))
    if not opened:
        raise RuntimeError(f"No desktop file opener is available for {directory}")


def _open_file(path: Path) -> None:
    target = path.expanduser().resolve()
    if not target.is_file():
        raise FileNotFoundError(f"Result file no longer exists: {target}")
    opened = QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(target)))
    if not opened:
        raise RuntimeError(f"No desktop file opener is available for {target}")


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


class SequenceRankingDialog(QtWidgets.QDialog):
    def __init__(self, path: Path | None, parent=None) -> None:
        super().__init__(parent)
        self.path = path.expanduser().resolve() if path is not None else None
        self.setWindowTitle("Sequence Ranking")
        self.setModal(True)
        self.resize(700, 650)
        layout = QtWidgets.QVBoxLayout(self)
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

    def _start_loading(self) -> None:
        if self.path is None:
            return
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


def apply_dark_palette(app: QtWidgets.QApplication) -> None:
    """Force a dark Fusion palette so the UI stays dark regardless of OS theme."""
    palette = QtGui.QPalette()
    # Main background: tweak Window/Base colors to change the overall background shade.
    palette.setColor(QtGui.QPalette.ColorRole.Window, QtGui.QColor("#1e1e1e"))
    palette.setColor(QtGui.QPalette.ColorRole.WindowText, QtGui.QColor("#e2e8f0"))
    palette.setColor(QtGui.QPalette.ColorRole.Base, QtGui.QColor("#21252b"))
    palette.setColor(QtGui.QPalette.ColorRole.AlternateBase, QtGui.QColor("#282c34"))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipBase, QtGui.QColor("#1e2127"))
    palette.setColor(QtGui.QPalette.ColorRole.ToolTipText, QtGui.QColor("#e2e8f0"))
    palette.setColor(QtGui.QPalette.ColorRole.Text, QtGui.QColor("#e2e8f0"))
    palette.setColor(QtGui.QPalette.ColorRole.Button, QtGui.QColor("#2f3540"))
    palette.setColor(QtGui.QPalette.ColorRole.ButtonText, QtGui.QColor("#e2e8f0"))
    palette.setColor(QtGui.QPalette.ColorRole.BrightText, QtGui.QColor("#ff7676"))
    palette.setColor(QtGui.QPalette.ColorRole.Link, QtGui.QColor("#5d7bff"))
    palette.setColor(QtGui.QPalette.ColorRole.Highlight, QtGui.QColor("#347aff"))
    palette.setColor(QtGui.QPalette.ColorRole.HighlightedText, QtGui.QColor("#f4f6fb"))
    app.setPalette(palette)


def _font_width(widget: QtWidgets.QWidget, text: str) -> int:
    metrics = QtGui.QFontMetrics(widget.font())
    if hasattr(metrics, "horizontalAdvance"):
        return metrics.horizontalAdvance(text)
    return metrics.width(text)


def _font_height(widget: QtWidgets.QWidget) -> int:
    return QtGui.QFontMetrics(widget.font()).height()


def _fixed_policy() -> QtWidgets.QSizePolicy:
    return QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Fixed)


def _expanding_fixed_policy() -> QtWidgets.QSizePolicy:
    return QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)


def _browse_icon(window: QtWidgets.QMainWindow) -> QtGui.QIcon:
    icon = QtGui.QIcon(str(REPO_ROOT / "G_GUI" / "svg_assets" / "upload-sign-svgrepo-com.svg"))
    if not icon.isNull():
        return icon
    try:
        return window.style().standardIcon(QtWidgets.QStyle.StandardPixmap.SP_DialogOpenButton)
    except AttributeError:
        return window.style().standardIcon(QtWidgets.QStyle.SP_DialogOpenButton)


def _remove_spacers(layout: QtWidgets.QLayout) -> None:
    for index in range(layout.count() - 1, -1, -1):
        item = layout.itemAt(index)
        if item is not None and item.widget() is None:
            layout.takeAt(index)


def _set_layout_stretches(layout: QtWidgets.QBoxLayout, stretches: list[int]) -> None:
    for index, stretch in enumerate(stretches):
        if index < layout.count():
            layout.setStretch(index, stretch)


def _clear_widget_alignment(layout: QtWidgets.QLayout, widget: QtWidgets.QWidget) -> None:
    try:
        layout.setAlignment(widget, QtCore.Qt.AlignmentFlag(0))
    except TypeError:
        layout.setAlignment(widget, QtCore.Qt.Alignment(0))


def apply_runtime_layout_fixes(window: QtWidgets.QMainWindow, ui: Ui_cryoID) -> None:
    field_button_gap = 10
    row_margin = 9
    change_folder_font = QtGui.QFont(ui.mrc_file_editbox.font())
    if change_folder_font.pointSize() > 0:
        change_folder_font.setPointSize(max(11, change_folder_font.pointSize() - 1))
    ui.change_folder_button_results.setFont(change_folder_font)

    field_height = max(30, _font_height(ui.mrc_file_editbox) + 12)
    browse_size = QtCore.QSize(field_height + 14, field_height)
    action_width = max(104, _font_width(ui.run_button, "Abort") + 42)
    action_height = max(48, field_height + 14)
    label_width = max(
        _font_width(ui.density_map_label, ui.density_map_label.text()),
        _font_width(ui.protein_pool_file_label, ui.protein_pool_file_label.text()),
    ) + 16
    input_width = max(300, _font_width(ui.mrc_file_editbox, "your_cryoEM_map.mrc") + 90)
    dock_width = max(350, _font_width(ui.change_folder_button_results, "Change folder") + 220)

    ui.top_widget.setSizePolicy(_expanding_fixed_policy())
    ui.bot_widget.setSizePolicy(_expanding_fixed_policy())
    ui.top_widget_right.setSizePolicy(_expanding_fixed_policy())
    ui.top_widget_right.setMinimumWidth(label_width + input_width + browse_size.width() + (field_button_gap * 2) + 36)
    ui.top_widget.setMinimumWidth(action_width + ui.top_widget_right.minimumWidth() + 95)

    for layout in (ui.horizontalLayout_6, ui.horizontalLayout_16):
        _remove_spacers(layout)
        layout.setContentsMargins(row_margin, row_margin, row_margin, row_margin)
        layout.setSpacing(field_button_gap)
        layout.invalidate()
    _clear_widget_alignment(ui.horizontalLayout_6, ui.top_widget_right)
    _set_layout_stretches(ui.horizontalLayout_6, [0, 1])
    _set_layout_stretches(ui.horizontalLayout_16, [0, 0, 1])

    for layout in (ui.horizontalLayout, ui.horizontalLayout_14):
        _remove_spacers(layout)
        layout.setSpacing(field_button_gap)
        _set_layout_stretches(layout, [0, 1, 0])
        layout.invalidate()

    for label in (ui.density_map_label, ui.protein_pool_file_label):
        label.setFixedWidth(label_width)
        label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

    for line_edit in (ui.mrc_file_editbox, ui.pool_file_editbox):
        line_edit.setMinimumWidth(input_width)
        line_edit.setMinimumHeight(field_height)
        line_edit.setSizePolicy(_expanding_fixed_policy())

    ui.lineEdit_2.setFont(ui.mrc_file_editbox.font())

    icon = _browse_icon(window)
    for button in (ui.browse_mrc_file_button, ui.browse_pool_fil_button):
        button.setIcon(icon)
        button.setIconSize(QtCore.QSize(20, 20))
        button.setText("")
        button.setFixedSize(browse_size)
        button.setSizePolicy(_fixed_policy())
        button.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)

    for button in (ui.run_button, ui.abort_button):
        button.setFixedSize(action_width, action_height)
        button.setSizePolicy(_fixed_policy())

    ui.progressBar.setMinimumHeight(field_height)
    ui.progressBar.setMinimumWidth(max(340, input_width + browse_size.width()))
    ui.progressBar.setSizePolicy(_expanding_fixed_policy())
    ui.status_label.setFixedWidth(label_width)
    ui.status_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)

    ui.dockWidget.setMinimumWidth(dock_width)
    ui.dockWidgetContents.setMinimumWidth(dock_width - 20)
    ui.results_label_widget.setMinimumWidth(dock_width - 24)
    ui.horizontalLayout_7.setSpacing(field_button_gap)
    _set_layout_stretches(ui.horizontalLayout_7, [1, 0])
    ui.lineEdit_2.setMinimumWidth(max(130, dock_width - _font_width(ui.change_folder_button_results, "Change folder") - 92))
    ui.lineEdit_2.setMinimumHeight(field_height)
    ui.lineEdit_2.setSizePolicy(_expanding_fixed_policy())
    ui.change_folder_button_results.setFixedSize(
        _font_width(ui.change_folder_button_results, "Change folder") + 30,
        field_height,
    )
    ui.view_in_chimera_button.setMinimumHeight(field_height)
    try:
        window.resizeDocks([ui.dockWidget], [dock_width], QtCore.Qt.Orientation.Horizontal)
    except Exception:
        pass


def main() -> None:
    display_issue = display_problem()
    if display_issue:
        raise SystemExit(f"CryoID GUI cannot start: {display_issue}")

    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")  # or "Windows"
    apply_dark_palette(app)  # ensures Fusion stays dark even if OS theme is light
    window = CryoIDMainWindow()
    ui = Ui_cryoID()
    ui.setupUi(window)
    window.setWindowTitle("CryoIDv2")
    ui.lineEdit_2.setPlaceholderText("Output directory (.../cryoid_results)")
    ui.change_folder_button_results.setText("Browse")
    local_config_state = ensure_local_config()
    shared_config_state = load_user_config()
    resolved_project_dir = resolve_project_dir(None, shared_config_state, default=REPO_ROOT)
    local_config_state["project_dir"] = str(resolved_project_dir)
    ui.lineEdit_2.setText(str(resolved_project_dir))
    ui.progressBar.setRange(0, 100)
    ui.progressBar.setValue(0)
    apply_runtime_layout_fixes(window, ui)

    state: dict[str, object] = {
        "thread": None,
        "worker": None,
        "last_result": None,
        "loaded_project_dir": None,
    }
    sequence_ranking_action = QtGui.QAction("Sequence Ranking", window)
    sequence_ranking_action.setToolTip("Open the latest run's sequence ranking in a separate window")
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
        ui.plainTextEdit.appendPlainText(str(message))

    def persist_config(**updates: object) -> None:
        for key, value in updates.items():
            local_config_state[key] = str(value) if value is not None else ""
        local_config_state.update(save_local_config(local_config_state))

    def current_project_dir() -> Path | None:
        selected = _project_dir_from_text(ui.lineEdit_2.text())
        return results_project_dir(selected) if selected is not None else None

    def clear_input_highlight() -> None:
        _clear_line_edit_highlights(
            (ui.mrc_file_editbox, ui.pool_file_editbox, ui.lineEdit_2),
            ui.centralwidget,
        )

    def current_sequence_ranking() -> Path | None:
        project = current_project_dir()
        result = state.get("last_result")
        result_ranking = getattr(result, "ranking_path", None)
        if result_ranking is not None and Path(result_ranking).is_file():
            result_project = find_project_dir_from_path(Path(result_ranking))
            if project is not None and result_project == project.resolve():
                return Path(result_ranking).resolve()
        last_run = str(local_config_state.get("last_run_dir") or "").strip()
        if last_run:
            ranking = Path(last_run).expanduser() / "sequence_ranking.tsv"
            if ranking.is_file() and project is not None and find_project_dir_from_path(ranking) == project.resolve():
                return ranking.resolve()
        return latest_sequence_ranking(project) if project is not None else None

    def refresh_ranking_action() -> None:
        # Keep this command usable like Advanced Settings. The dialog reports
        # an empty state when the expected run-level ranking is absent.
        sequence_ranking_action.setEnabled(True)

    def open_sequence_ranking(path: Path | None = None) -> None:
        ranking = path or current_sequence_ranking()
        if ranking is None or not ranking.is_file():
            append_log("No run-specific sequence ranking is available.")
            ranking = None
        dialog = SequenceRankingDialog(ranking, window)
        _exec_dialog(dialog)

    def populate_results(paths: list[Path], project_dir: Path | None = None) -> None:
        ui.file_list_widget_results.clear()
        if project_dir is not None:
            state["loaded_project_dir"] = project_dir
        for path in paths:
            item = QtWidgets.QListWidgetItem(_result_label(path, project_dir))
            item.setToolTip(str(path))
            item.setData(QtCore.Qt.ItemDataRole.UserRole, str(path))
            _style_link_item(item)
            ui.file_list_widget_results.addItem(item)
        refresh_ranking_action()

    def populate_project_results(
        project_dir: Path | None = None,
        output_files: list[Path] | None = None,
    ) -> None:
        project = project_dir or current_project_dir()
        if project is None:
            ui.file_list_widget_results.clear()
            return
        if not project.is_dir():
            append_log(f"Project folder not found: {project}")
            return
        ensure_project_dirs(project)
        files = collect_result_display_files(project, output_files)
        populate_results(files, project_dir=project)
        append_log(f"Loaded {len(files)} project result file(s) from {project}")

    def browse_density_map() -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            window,
            "Select density map",
            str(REPO_ROOT),
            "Density maps (*.mrc *.map *.ccp4);;All files (*)",
        )
        if filename:
            ui.mrc_file_editbox.setText(filename)
            persist_config(last_density_map=Path(filename).expanduser().resolve())

    def browse_candidate_pool() -> None:
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            window,
            "Select candidate sequence pool",
            str(REPO_ROOT),
            "FASTA sequence pools (*.fasta *.fa *.faa *.fna);;All files (*)",
        )
        if filename:
            ui.pool_file_editbox.setText(filename)
            persist_config(last_candidate_pool=Path(filename).expanduser().resolve())

    def choose_results_folder() -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            window,
            "Select results parent directory",
            str(current_project_dir() or REPO_ROOT),
        )
        if directory:
            project = results_project_dir(directory)
            ensure_project_dirs(project)
            ui.lineEdit_2.setText(str(project))
            persist_config(project_dir=project)
            populate_project_results(project)

    def start_pipeline() -> None:
        density_map = _usable_text(ui.mrc_file_editbox.text())
        candidate_pool = _usable_text(ui.pool_file_editbox.text())
        project_dir = current_project_dir()
        if not density_map or not candidate_pool:
            append_log("Run blocked: provide both a density map and a candidate sequence pool.")
            return
        if project_dir is None:
            append_log("Run blocked: select an output parent directory.")
            return
        weights_path = settings_state.get("weights_path", "").strip()
        if not weights_path:
            append_log("Run blocked: set the weights path in More Options > Advanced Settings.")
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
        ui.lineEdit_2.setText(str(project_dir))
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
        ui.progressBar.setRange(0, 100)
        ui.progressBar.setValue(0)
        ui.run_button.setEnabled(False)
        ui.abort_button.setEnabled(True)
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
            state["last_result"] = result
            ui.progressBar.setRange(0, 100)
            ui.progressBar.setValue(100)
            ui.run_button.setEnabled(True)
            ui.abort_button.setEnabled(False)
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
                        + "\n\nChoose Protein or RNA in More Options > Advanced Settings.",
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
                populate_project_results(project_dir, list(getattr(result, "output_files", []) or []))
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
        worker.progress.connect(ui.progressBar.setValue)
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
        ui.abort_button.setEnabled(False)
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
            append_log("Cannot create a new project while a run is active. Abort or wait for completion first.")
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
        ui.mrc_file_editbox.clear()
        ui.pool_file_editbox.clear()
        ui.lineEdit_2.setText(str(project))
        ui.progressBar.setRange(0, 100)
        ui.progressBar.setValue(0)
        persist_config(
            project_dir=project,
            last_run_dir="",
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
        populate_project_results(project)
        append_log(f"Project opened: {project}")

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
            ui.lineEdit_2.setText(str(project))
            state["last_result"] = None
            selected_outputs: list[Path] = []
            if selected_run is not None:
                persist_config(project_dir=project, last_run_dir=selected_run)
            else:
                persist_config(project_dir=project, last_run_dir="")
            populate_project_results(project, selected_outputs)
            append_log(f"Project opened: {project}")

    def save_log() -> None:
        append_log("GUI log export is not part of the CryoIDv2 project workflow.")

    def open_chimerax_session() -> None:
        project_dir = current_project_dir()
        density_map_text = _usable_text(ui.mrc_file_editbox.text())
        density_map = Path(density_map_text).expanduser().resolve() if density_map_text else None
        configured_chimerax = settings_state.get("chimerax_path") or None

        def launch_files(files: list[Path]) -> bool:
            try:
                launch_chimerax(files, chimerax_path=configured_chimerax)
            except FileNotFoundError as exc:
                message = (
                    f"{exc}\n\nSet the ChimeraX executable path in More options > Advanced Settings."
                )
                QtWidgets.QMessageBox.critical(window, "ChimeraX not found", message)
                append_log(f"ChimeraX launch failed: {message}")
                return False
            except Exception as exc:
                QtWidgets.QMessageBox.critical(window, "ChimeraX launch failed", str(exc))
                append_log(f"ChimeraX launch failed: {exc}")
                return False
            append_log(f"Opened in ChimeraX: {', '.join(str(path) for path in files)}")
            return True

        if project_dir is None:
            append_log("Select a CryoID results directory before opening ChimeraX.")
            return
        groups = collect_chimerax_project_files(project_dir, density_map)
        files = flatten_chimerax_session_files(groups)
        if not groups["density_maps"]:
            append_log("ChimeraX launch blocked: provide an existing input density map.")
            return
        if not groups["aligned_models"]:
            append_log("ChimeraX launch blocked: no best aligned model is recorded in best_overall.")
            return
        if launch_files(files):
            append_log(
                "Opened in ChimeraX: "
                f"{len(groups['density_maps'])} map(s), "
                f"{len(groups['aligned_models'])} aligned model(s)."
            )

    def open_advanced_settings() -> None:
        dialog = AdvancedSettingsWindow(
            window,
            weights_path=settings_state.get("weights_path", ""),
            rcl=float(settings_state.get("rcl", "0")),
            protein_min_trace_length=int(
                settings_state.get("protein_min_trace_length", "10")
            ),
            chimerax_path=settings_state.get("chimerax_path", ""),
            detected_chimerax_path=_detect_chimerax_path(),
            blast_bin_dir=settings_state.get("blast_bin_dir", ""),
            detected_blast_bin_dir=str(autodetect_blast_bin_dir() or ""),
            sequence_mode=settings_state.get("sequence_mode", "auto"),
            structure_selection=settings_state.get("structure_selection", ""),
            keep_tmp=settings_state.get("keep_tmp", "false").lower() == "true",
        )
        if _exec_dialog(dialog) != _dialog_accepted_code():
            return
        settings_state["weights_path"] = dialog.weights_path()
        settings_state["rcl"] = str(dialog.rcl())
        settings_state["protein_min_trace_length"] = str(
            dialog.protein_min_trace_length()
        )
        settings_state["chimerax_path"] = dialog.chimerax_path()
        settings_state["blast_bin_dir"] = dialog.blast_bin_dir()
        settings_state["sequence_mode"] = dialog.sequence_mode()
        settings_state["structure_selection"] = dialog.structure_selection()
        settings_state["keep_tmp"] = "true" if dialog.keep_tmp() else "false"
        persist_config(
            weights_path=settings_state["weights_path"],
            rcl=settings_state["rcl"],
            protein_min_trace_length=settings_state["protein_min_trace_length"],
            chimerax_path=settings_state["chimerax_path"],
            blast_bin_dir=settings_state["blast_bin_dir"],
            sequence_mode=settings_state["sequence_mode"],
            structure_selection=settings_state["structure_selection"],
            keep_tmp=settings_state["keep_tmp"],
        )
        if settings_state["weights_path"]:
            append_log(f"Weights path set to: {settings_state['weights_path']}")
        else:
            append_log("Weights path cleared. Runs require a weights path.")
        append_log(f"Recommended contour level (RCL) set to: {float(settings_state['rcl']):g}")
        append_log(
            "Protein minimum trace length set to: "
            f"{int(settings_state['protein_min_trace_length'])}."
        )
        if settings_state["chimerax_path"]:
            append_log(f"ChimeraX path set to: {settings_state['chimerax_path']}")
        else:
            append_log("ChimeraX path cleared. View in ChimeraX will try autodetection.")
        if settings_state["blast_bin_dir"]:
            append_log(f"BLAST+ bin folder set to: {settings_state['blast_bin_dir']}")
        else:
            append_log("BLAST+ path cleared. CryoID will try automatic detection.")
        append_log(f"Sequence mode set to: {settings_state['sequence_mode']}.")
        if settings_state["structure_selection"]:
            append_log(
                "Structure candidate selection set to: "
                f"{settings_state['structure_selection']}."
            )
        else:
            append_log("Structure candidate selection uses mode defaults: protein 1; RNA 0.25.")

    def open_selected_result_location(item: QtWidgets.QListWidgetItem) -> None:
        path = _item_path(item)
        if path is None:
            return
        if path.name == "sequence_ranking.tsv":
            open_sequence_ranking(path)
            return
        try:
            _open_file_location(path)
        except Exception as exc:
            append_log(f"Could not open result location: {exc}")

    def log_not_ready(feature: str) -> None:
        append_log(f"{feature}: no separate dialog is wired yet. Use Run for the current default workflow.")

    def open_github() -> None:
        opened = QtGui.QDesktopServices.openUrl(QtCore.QUrl(CRYOID_GITHUB_URL))
        if opened:
            append_log(f"Opened CryoIDv2 GitHub: {CRYOID_GITHUB_URL}")
            return
        message = f"Could not open the default browser.\n\n{CRYOID_GITHUB_URL}"
        QtWidgets.QMessageBox.warning(window, "Could not open GitHub", message)
        append_log(f"Could not open CryoIDv2 GitHub: {CRYOID_GITHUB_URL}")

    def configure_menus() -> None:
        ui.menucryoID.setTitle("File")
        ui.menuView.setTitle("View")
        ui.menuMore_Options.setTitle("More Options")
        ui.menuHelp.setTitle("Help")
        ui.menucryoID.clear()
        ui.menucryoID.addAction(ui.actionNew)
        ui.menucryoID.addAction(ui.actionOpen)
        ui.menuView.clear()
        ui.menuView.addAction(ui.actionResults_window)
        ui.menuView.addAction(sequence_ranking_action)
        ui.menuView.addAction(ui.actionChimeraX)
        ui.menuMore_Options.clear()
        ui.menuMore_Options.addAction(ui.actionAdvanced_Settings)
        ui.menuHelp.clear()
        ui.menuHelp.addAction(ui.actionNonesofar)
        ui.actionNonesofar.setText("GitHub")
        ui.actionNonesofar.setToolTip("Open the CryoIDv2 GitHub repository")
        for action in (
            ui.actionRecent,
            ui.action_L_oad_save,
            ui.actionTBD,
            ui.actionPhenix,
            ui.actionCOOT,
            ui.actionMRCs,
            ui.actionProgress_bar,
            ui.actionPreviewer,
        ):
            action.setVisible(False)
        ui.menubar.clear()
        ui.menubar.addAction(ui.menucryoID.menuAction())
        ui.menubar.addAction(ui.menuView.menuAction())
        ui.menubar.addAction(ui.menuMore_Options.menuAction())
        ui.menubar.addAction(ui.menuHelp.menuAction())

    configure_menus()

    def restore_startup_state() -> None:
        project_text = local_config_state.get("project_dir", "")
        if project_text:
            project = Path(project_text).expanduser()
            if project.is_dir():
                project = project.resolve()
                ensure_project_dirs(project)
                ui.lineEdit_2.setText(str(project))
                populate_project_results(project)
            else:
                ui.lineEdit_2.setText(str(REPO_ROOT.resolve()))
                append_log(f"Saved project path no longer exists: {project}")
        else:
            ui.lineEdit_2.setText(str(REPO_ROOT.resolve()))

        density_text = local_config_state.get("last_density_map", "")
        if density_text:
            density_path = Path(density_text).expanduser()
            if density_path.is_file():
                ui.mrc_file_editbox.setText(str(density_path.resolve()))
            else:
                ui.mrc_file_editbox.clear()
                append_log(f"Saved density map no longer exists: {density_path}")

        pool_text = local_config_state.get("last_candidate_pool", "")
        if pool_text:
            pool_path = Path(pool_text).expanduser()
            if pool_path.is_file():
                ui.pool_file_editbox.setText(str(pool_path.resolve()))
            else:
                ui.pool_file_editbox.clear()
                append_log(f"Saved candidate pool no longer exists: {pool_path}")

    restore_startup_state()
    ui.plainTextEdit.clear()
    append_log(startup_status_message())

    ui.browse_mrc_file_button.clicked.connect(browse_density_map)
    ui.browse_pool_fil_button.clicked.connect(browse_candidate_pool)
    ui.change_folder_button_results.clicked.connect(choose_results_folder)
    ui.run_button.clicked.connect(start_pipeline)
    ui.abort_button.clicked.connect(abort_pipeline)
    ui.view_in_chimera_button.clicked.connect(open_chimerax_session)
    ui.file_list_widget_results.itemClicked.connect(open_selected_result_location)
    ui.abort_button.setEnabled(False)

    ui.actionNew.triggered.connect(new_project)
    ui.actionOpen.triggered.connect(open_project)
    ui.actionAdvanced_Settings.triggered.connect(open_advanced_settings)
    ui.actionNonesofar.triggered.connect(open_github)
    ui.actionChimeraX.triggered.connect(open_chimerax_session)
    sequence_ranking_action.triggered.connect(lambda: open_sequence_ranking())
    # Allow the Results dock to be closed and re-shown via View -> View Results.
    ui.actionResults_window.setCheckable(True)
    ui.actionResults_window.setChecked(True)
    ui.actionResults_window.triggered.connect(ui.dockWidget.setVisible)
    ui.dockWidget.visibilityChanged.connect(ui.actionResults_window.setChecked)
    ui.dockWidget.setWindowTitle("")
    # Results dock panel colors (overall box and inner background)
    ui.dockWidget.setStyleSheet("QDockWidget#dockWidget { background-color: #22272f; border: none; }")
    ui.dockWidgetContents.setStyleSheet(
        """
        QWidget#dockWidgetContents {
            background-color: #1f242b;
        }
        QWidget#dockWidgetContents QLabel {
            color: #e2e8f0;
            font-weight: 600;
        }
        QLabel#label_2 {
            font-weight: 700;
        }
        QWidget#dockWidgetContents QLineEdit,
        QWidget#dockWidgetContents QListWidget {
            background-color: #262b33;
            color: #e2e8f0;
            border: 1px solid #333944;
        }
        QWidget#dockWidgetContents QPushButton {
            background-color: #2f3540;
            color: #e2e8f0;
            border: 1px solid #3a404c;
            padding: 4px 8px;
        }
        QWidget#dockWidgetContents QPushButton:disabled {
            background-color: #242933;
            color: #7a818c;
        }
        """
    )
    # Menu bar colors: adjust the QMenuBar block below (background/color/border-bottom).
    # Results header row (path box + "Change folder"): tweak QLineEdit / QPushButton rules inside the dock CSS above.
    # Output box under Status: style QPlainTextEdit (or QWidget#widget) in the window stylesheet below to change its background/border.
    window.setStyleSheet(
        """
        QMainWindow, QWidget#centralwidgebel#protein_pool_file_label {
            font-weight: 600;
        }
        QMenuBar {
            background-color: #1f242b;
            color: #e2e8f0;
            border: none;
            border-bottom: 1px solid #12161d;
        }
        QMenuBar::item:selected {
            background-color: #2a303a;
        }
        QLabel#density_map_label,
        QLabel#protein_pool_file_label,
        QLabel#status_label {
            font-weight: 700;
        }
        QLineEdit {
            background-color: #262b33;
            color: #e2e8f0;
            border: 1px solid #333944;
        }
        QPlainTextEdit {
            background-color: #262b33;
            color: #e2e8f0;
            border: 1px solid #333944;
            selection-background-color: #347aff;
            selection-color: #f4f6fb;
        }
        QProgressBar {
            background-color: #262b33;
            color: #e2e8f0;
            border: 1px solid #333944;
            text-align: center;
        }
        QProgressBar::chunk {
            background-color: #347aff;
        }
        QPushButton#run_button,
        QPushButton#abort_button {
            background-color: #2f6db5 !important;
            color: #f4f6fb;
            border: 1px solid #255a92;
            padding: 6px 14px;
        }
        QPushButton#run_button:hover,
        QPushButton#abort_button:hover {
            background-color: #377dd2 !important;
        }
        QPushButton#run_button:pressed,
        QPushButton#abort_button:pressed {
            background-color: #2b5c9c !important;
        }
        """
    )
    window.show()
    smoke_test_ms = os.environ.get("CRYOID_GUI_SMOKE_TEST_MS", "").strip()
    if smoke_test_ms:
        QtCore.QTimer.singleShot(max(1, int(smoke_test_ms)), app.quit)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
