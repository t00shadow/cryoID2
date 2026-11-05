from __future__ import annotations

import json
import os
import platform
import re
import stat
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping

from .structure_selection import parse_structure_selection

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]


ENV_CONFIG = "CRYOID_CONFIG"
ENV_PROJECT_DIR = "CRYOID_PROJECT_DIR"
ENV_WEIGHTS_DIR = "CRYOID_UNET_WEIGHTS_DIR"
ENV_DEVICE = "CRYOID_UNET_DEVICE"
ENV_CPU_THREADS = "CRYOID_CPU_THREADS"
ENV_RCL = "CRYOID_RCL"
ENV_PROTEIN_MIN_TRACE_LENGTH = "CRYOID_PROTEIN_MIN_TRACE_LENGTH"
ENV_STRUCTURE_SELECTION = "CRYOID_STRUCTURE_SELECTION"
ENV_KEEP_TMP = "CRYOID_KEEP_TMP"
ENV_CHIMERAX_PATH = "CRYOID_CHIMERAX_PATH"
RESULTS_FOLDER_NAME = "cryoid_results"
STARTUP_KEYS = ("weights_dir", "chimerax_path")
PROJECT_STATE_KEYS = ("last_run_dir", "sequence_ranking_run_dir", "last_density_map", "last_candidate_pool")
PATH_KEYS = ("project_dir", "weights_dir", "chimerax_path", *PROJECT_STATE_KEYS)

DEFAULT_CONFIG: dict[str, str] = {
    "project_dir": "",
    "weights_dir": "",
    "device": "auto",
    "cpu_threads": "auto",
    "rcl": "0",
    "protein_min_trace_length": "10",
    "structure_selection": "",
    "keep_tmp": "false",
    "chimerax_path": "",
    "last_run_dir": "",
    "sequence_ranking_run_dir": "",
    "last_density_map": "",
    "last_candidate_pool": "",
    "sequence_mode": "auto",
}

DEFAULT_KEYS = (
    "project_dir",
    "weights_dir",
    "device",
    "cpu_threads",
    "rcl",
    "protein_min_trace_length",
    "structure_selection",
    "keep_tmp",
    "sequence_mode",
)
GUI_KEYS = (
    "chimerax_path",
    "last_run_dir",
    "sequence_ranking_run_dir",
    "last_density_map",
    "last_candidate_pool",
)


def default_config_path() -> Path:
    override = os.environ.get(ENV_CONFIG)
    if override:
        return Path(override).expanduser().resolve()

    system = platform.system().lower()
    if system == "windows":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
    elif system == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return (base / "cryoid" / "config.toml").expanduser().resolve()


def _normalized(raw: Mapping[str, Any] | None) -> dict[str, str]:
    config = dict(DEFAULT_CONFIG)
    if raw is None:
        return config
    for key in config:
        value = raw.get(key, config[key])
        config[key] = str(value) if value is not None else ""
    return config


def load_config(
    path: Path | str | None = None, *, strict: bool = False,
    defaults: Mapping[str, Any] | None = None, project: bool = False,
) -> dict[str, str]:
    config_path = Path(path).expanduser().resolve() if path is not None else default_config_path()
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (FileNotFoundError, OSError, tomllib.TOMLDecodeError):
        if strict:
            raise
        return dict(DEFAULT_CONFIG)
    if not isinstance(raw, dict):
        if strict:
            raise ValueError("The configuration must contain a TOML table.")
        return dict(DEFAULT_CONFIG)

    combined: dict[str, Any] = dict(defaults or {})
    defaults = raw.get("defaults", {})
    gui = raw.get("gui", {})
    startup = raw.get("startup", {})
    if strict and any(not isinstance(section, dict) for section in (defaults, gui, startup)):
        raise ValueError("The startup, defaults and gui sections must be TOML tables.")
    stored = {}
    if isinstance(defaults, dict):
        stored.update({key: defaults[key] for key in DEFAULT_KEYS if key in defaults})
    if isinstance(gui, dict):
        stored.update({key: gui[key] for key in GUI_KEYS if key in gui})
    if project:
        for key in STARTUP_KEYS:
            stored.pop(key, None)
    elif isinstance(startup, dict):
        stored.update({key: startup[key] for key in STARTUP_KEYS if key in startup})
    for key in PATH_KEYS:
        if key in stored:
            stored[key] = normalize_path(stored[key], base_dir=config_path.parent)
    combined.update(stored)
    return _normalized(combined)


def _toml_string(value: object) -> str:
    return json.dumps(str(value) if value is not None else "", ensure_ascii=False)


def normalize_path(value: Any, *, base_dir: Path | None = None) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    candidate = Path(text).expanduser()
    if PureWindowsPath(text).is_absolute() and os.name != "nt":
        return text
    if not candidate.is_absolute():
        candidate = (base_dir or Path.cwd()) / candidate
    return str(candidate.resolve())


@contextmanager
def _config_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_name(f".{path.name}.lock")
    with lock_path.open("a+b") as handle:
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        deadline = time.monotonic() + 5
        while True:
            try:
                handle.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Configuration is busy; try again: {path}")
                time.sleep(.025)
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _write_config(config: Mapping[str, Any], path: Path, *, project: bool = False) -> dict[str, str]:
    normalized = _normalized(config)
    lines = []
    if project:
        normalized.update({key: "" for key in STARTUP_KEYS})
    else:
        lines.append("[startup]")
        lines.extend(f"{key} = {_toml_string(normalized[key])}" for key in STARTUP_KEYS)
        lines.append("")
    lines.append("[defaults]")
    lines.extend(f"{key} = {_toml_string(normalized[key])}" for key in DEFAULT_KEYS if key not in STARTUP_KEYS)
    lines.extend(["", "[gui]"])
    lines.extend(f"{key} = {_toml_string(normalized[key])}" for key in GUI_KEYS if key not in STARTUP_KEYS)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write("\n".join(lines) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists():
            temporary.chmod(stat.S_IMODE(path.stat().st_mode))
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.chmod(stat.S_IREAD | stat.S_IWRITE)
            temporary.unlink(missing_ok=True)
    return normalized


def save_config(config: Mapping[str, Any], path: Path | str | None = None, *, overwrite: bool = True,
                project: bool = False) -> dict[str, str]:
    config_path = Path(path).expanduser().resolve() if path is not None else default_config_path()
    with _config_lock(config_path):
        if not overwrite and config_path.exists():
            raise FileExistsError(f"Configuration already exists: {config_path}")
        return _write_config(config, config_path, project=project)


def config_value(key: str, value: Any) -> str:
    if key in PATH_KEYS:
        return os.path.normcase(normalize_path(value))
    if key == "structure_selection" and not str(value or "").strip():
        return ""
    try:
        return validate_config_value(key, value)
    except (ValueError, TypeError, OverflowError):
        return str(value or "")


class ConfigConflictError(ValueError):
    def __init__(self, path: Path, keys: list[str], saved: Mapping[str, Any], changes: Mapping[str, Any]):
        super().__init__(f"Settings changed outside this window: {', '.join(keys)}\n{path}")
        self.path, self.keys = path, keys
        self.saved, self.changes = dict(saved), dict(changes)


def save_config_changes(
    changes: Mapping[str, Any], path: Path | str | None = None, *,
    baseline: Mapping[str, Any] | None = None, initial: Mapping[str, Any] | None = None,
    force: bool = False, project: bool = False,
) -> dict[str, str]:
    config_path = Path(path).expanduser().resolve() if path is not None else default_config_path()
    changes = {key: value for key, value in changes.items() if key in DEFAULT_CONFIG}
    if project:
        changes = {key: value for key, value in changes.items() if key not in STARTUP_KEYS}
    with _config_lock(config_path):
        exists = config_path.exists()
        saved = load_config(config_path, strict=True, project=project) if exists else _normalized(initial)
        remove_startup = False
        if project and exists:
            with config_path.open("rb") as handle:
                raw = tomllib.load(handle)
            remove_startup = "startup" in raw or any(
                key in raw.get(section, {}) for section in ("defaults", "gui") for key in STARTUP_KEYS)
        if exists and baseline is not None and not force:
            conflicts = [key for key, value in changes.items() if key != "project_dir"
                         if config_value(key, saved[key]) != config_value(key, baseline.get(key, DEFAULT_CONFIG[key]))
                         and config_value(key, saved[key]) != config_value(key, value)]
            if conflicts:
                raise ConfigConflictError(config_path, conflicts, saved, changes)
        changes = {key: value for key, value in changes.items() if config_value(key, saved[key]) != config_value(key, value)}
        if not changes and not remove_startup and (exists or initial is None):
            return saved
        saved.update(changes)
        return _write_config(saved, config_path, project=project)


def ensure_config(path: Path | str | None = None) -> dict[str, str]:
    config_path = Path(path).expanduser().resolve() if path is not None else default_config_path()
    if config_path.exists():
        return load_config(config_path)
    try:
        return save_config(DEFAULT_CONFIG, config_path, overwrite=False)
    except FileExistsError:
        return load_config(config_path, strict=True)


def ensure_startup_config(path: Path | str | None = None) -> dict[str, str]:
    """Initialize machine paths once, migrating existing values before discovery."""
    from .startup_config import find_weights_dir, find_chimerax
    config_path = Path(path).expanduser().resolve() if path is not None else default_config_path()
    with _config_lock(config_path):
        values = load_config(config_path, strict=True) if config_path.exists() else dict(DEFAULT_CONFIG)
        if config_path.exists():
            with config_path.open("rb") as handle:
                raw = tomllib.load(handle)
            if isinstance(raw.get("startup"), dict):
                return values
        # Only this one-time migration may import paths from the last project.
        selected = values.get("project_dir")
        previous = {}
        if selected:
            old_path = project_config_path(results_project_dir(selected))
            if old_path.is_file():
                previous = load_config(old_path)
        for key in STARTUP_KEYS:
            values[key] = values[key] or previous.get(key, "")
        if not values["weights_dir"]:
            detected = find_weights_dir()
            values["weights_dir"] = normalize_path(detected) if detected else ""
        if not values["chimerax_path"]:
            try:
                values["chimerax_path"] = normalize_path(find_chimerax())
            except FileNotFoundError:
                pass
        return _write_config(values, config_path)


def update_config(path: Path | str | None = None, **updates: Any) -> dict[str, str]:
    return save_config_changes(updates, path)


def _value_from_env(name: str) -> str:
    return str(os.environ.get(name) or "").strip()


def _resolved_path(value: Path | str) -> Path:
    return Path(value).expanduser().resolve()


def results_project_dir(value: Path | str) -> Path:
    """Return the dedicated CryoID output folder below a selected parent path."""

    selected = _resolved_path(value)
    if selected.name.casefold() == RESULTS_FOLDER_NAME.casefold():
        return selected
    return (selected / RESULTS_FOLDER_NAME).resolve()


def project_config_path(project_dir: Path) -> Path:
    """Configuration inside an already resolved CryoID results directory."""
    return Path(project_dir) / "config.toml"


def load_project_config(
    project_dir: Path, *, defaults: Mapping[str, Any] | None = None,
) -> dict[str, str] | None:
    path = project_config_path(project_dir)
    return load_config(path, strict=True, defaults=defaults, project=True) if path.exists() else None


@dataclass(frozen=True)
class LoadedConfig:
    project_dir: Path
    values: dict[str, str]
    source_path: Path | None


def load_effective_config(
    project_dir: Path | str | None = None, *, config_path: Path | str | None = None,
    user_config: Mapping[str, Any] | None = None,
    default_project_dir: Path | str | None = None, use_project_config: bool = True,
) -> LoadedConfig:
    """Load user defaults, then project settings, without writing either file.

    Run snapshots never supply editable project settings. Callers that already
    have explicit settings can disable reading the project file altogether.
    Flag/environment overrides are applied by the shared resolvers after loading.
    """
    defaults = load_config(config_path) if user_config is None else _normalized(user_config)
    project = resolve_project_dir(project_dir, defaults, default=default_project_dir)
    project_defaults = dict(defaults)
    for key in PROJECT_STATE_KEYS:
        project_defaults[key] = ""
    source = project_config_path(project)
    try:
        values = load_project_config(project, defaults=project_defaults) if use_project_config else None
        if values is None:
            values, source = project_defaults, None
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read configuration {source}: {exc}") from exc
    values = _normalized(values)
    values.update({key: defaults[key] for key in STARTUP_KEYS})
    # A stored project_dir must not redirect an explicitly selected project.
    values["project_dir"] = str(project)
    return LoadedConfig(project, values, source)


def resolve_project_dir(
    argument: Path | str | None,
    config: Mapping[str, Any] | None = None,
    *,
    default: Path | str | None = None,
) -> Path:
    if argument is not None:
        return results_project_dir(argument)
    env_value = _value_from_env(ENV_PROJECT_DIR)
    if env_value:
        return results_project_dir(env_value)
    config_value = str((config or {}).get("project_dir") or "").strip()
    if config_value:
        return results_project_dir(config_value)
    return results_project_dir(default if default is not None else Path.cwd())


def resolve_weights_dir(
    argument: Path | str | None,
    config: Mapping[str, Any] | None = None,
) -> Path | None:
    if argument is not None:
        return _resolved_path(argument) if str(argument).strip() else None
    env_value = _value_from_env(ENV_WEIGHTS_DIR)
    if env_value:
        return _resolved_path(env_value)
    config_value = str((config or {}).get("weights_dir") or "").strip()
    return _resolved_path(config_value) if config_value else None


def resolve_device(argument: str | None, config: Mapping[str, Any] | None = None) -> str | None:
    value = argument if argument is not None else _value_from_env(ENV_DEVICE) or (config or {}).get("device")
    device = validate_device(value)
    return None if device == "auto" else device


def validate_device(value: str | None) -> str:
    device = str(value or "auto").strip().lower()
    if device not in {"auto", "cpu", "cuda", "mps"} and not re.fullmatch(r"cuda:\d+", device):
        raise ValueError("Device must be auto, cpu, mps, or a CUDA GPU such as cuda:0.")
    return device


def validate_cpu_threads(value) -> int | None:
    from .compute import available_cpu_threads

    text = str(value if value is not None else "auto").strip().lower()
    if text in {"", "auto"}:
        return None
    available = available_cpu_threads()
    if not re.fullmatch(r"[1-9]\d*", text) or int(text) > available:
        raise ValueError(f"CPU threads must be auto or a whole number from 1 to {available}.")
    return int(text)


def resolve_cpu_threads(argument=None, config: Mapping[str, Any] | None = None) -> int | None:
    value = argument if argument is not None else _value_from_env(ENV_CPU_THREADS) or (config or {}).get("cpu_threads", "auto")
    return validate_cpu_threads(value)


def resolve_rcl(argument: float | str | None, config: Mapping[str, Any] | None = None) -> float:
    from C_Unet_model_predicting.prediction_provenance import validate_rcl

    if argument is not None:
        return validate_rcl(argument)
    env_value = _value_from_env(ENV_RCL)
    if env_value:
        return validate_rcl(env_value)
    config_value = str((config or {}).get("rcl") or "").strip()
    return validate_rcl(config_value)


def validate_protein_min_trace_length(value: int | str | None) -> int:
    if value in (None, ""):
        return 10
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Protein minimum trace length must be a whole number.") from exc
    if not numeric.is_integer():
        raise ValueError("Protein minimum trace length must be a whole number.")
    length = int(numeric)
    if length < 1:
        raise ValueError("Protein minimum trace length must be at least 1.")
    return length


def resolve_protein_min_trace_length(
    argument: int | str | None,
    config: Mapping[str, Any] | None = None,
) -> int:
    if argument is not None:
        return validate_protein_min_trace_length(argument)
    env_value = _value_from_env(ENV_PROTEIN_MIN_TRACE_LENGTH)
    if env_value:
        return validate_protein_min_trace_length(env_value)
    return validate_protein_min_trace_length(
        (config or {}).get("protein_min_trace_length", "10")
    )


def resolve_chimerax_path(
    config: Mapping[str, Any] | None = None, *, argument: Path | str | None = None,
) -> str:
    if argument is not None:
        return str(_resolved_path(argument)) if str(argument).strip() else ""
    env_value = _value_from_env(ENV_CHIMERAX_PATH)
    if env_value:
        return str(_resolved_path(env_value))
    config_value = str((config or {}).get("chimerax_path") or "").strip()
    return str(_resolved_path(config_value)) if config_value else ""


def resolve_sequence_mode(argument: str | None, config: Mapping[str, Any] | None = None) -> str:
    value = argument if argument is not None else str((config or {}).get("sequence_mode") or "auto")
    return validate_sequence_mode(value)


def validate_sequence_mode(value: str | None) -> str:
    mode = str(value or "auto").strip().lower()
    if mode not in {"auto", "protein", "rna"}:
        raise ValueError("Sequence mode must be auto, protein, or rna.")
    return mode


def validate_structure_selection(value: str | int | float) -> str:
    if str(value).strip().lower() == "auto":
        return ""
    return parse_structure_selection(value).raw


def resolve_structure_selection(
    argument: str | int | float | None,
    config: Mapping[str, Any] | None = None,
) -> str | None:
    if argument is not None:
        return validate_structure_selection(argument) or None
    env_value = _value_from_env(ENV_STRUCTURE_SELECTION)
    if env_value:
        return validate_structure_selection(env_value) or None
    config_value = str((config or {}).get("structure_selection") or "").strip()
    if config_value:
        return validate_structure_selection(config_value) or None
    return None


def validate_keep_tmp(value: bool | str | int | None) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    if text in {"", "0", "false", "no", "off"}:
        return False
    if text in {"1", "true", "yes", "on"}:
        return True
    raise ValueError("Keep tmp must be true or false.")


def resolve_keep_tmp(
    argument: bool | str | int | None,
    config: Mapping[str, Any] | None = None,
) -> bool:
    if argument is not None:
        return validate_keep_tmp(argument)
    env_value = _value_from_env(ENV_KEEP_TMP)
    if env_value:
        return validate_keep_tmp(env_value)
    return validate_keep_tmp((config or {}).get("keep_tmp", "false"))


def validate_config_value(key: str, value: Any) -> str:
    """Validate a changed setting before writing it, without environment overrides."""
    if key == "device":
        return validate_device(value)
    if key == "cpu_threads":
        count = validate_cpu_threads(value)
        return "auto" if count is None else str(count)
    if key == "sequence_mode":
        return validate_sequence_mode(value)
    if key == "rcl":
        from C_Unet_model_predicting.prediction_provenance import validate_rcl

        return str(validate_rcl(value))
    if key == "protein_min_trace_length":
        return str(validate_protein_min_trace_length(value))
    if key == "structure_selection":
        return validate_structure_selection(value)
    if key == "keep_tmp":
        return "true" if validate_keep_tmp(value) else "false"
    return str(value)


def resolve_run_settings(
    config: Mapping[str, Any], *, overrides: Mapping[str, Any] | None = None, strict: bool = False,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve the same run options for CLI and GUI; GUI can request review on errors."""
    resolved, issues = {}, {}
    for key, resolver, fallback, label, guidance in (
        ("device", resolve_device, None, "Device", "Choose Auto, CPU, or an available GPU"),
        ("cpu_threads", resolve_cpu_threads, None, "CPU threads", "Choose Auto or an available thread count"),
        ("rcl", resolve_rcl, 0.0, "RCL", "Enter a finite number of 0 or greater"),
        ("sequence_mode", resolve_sequence_mode, "auto", "Sequence mode", "Choose Auto, Protein, or RNA"),
        ("protein_min_trace_length", resolve_protein_min_trace_length, 10, "Protein trace length", "Enter a positive whole number"),
        ("structure_selection", resolve_structure_selection, None, "Structure candidates", "Choose a positive count or a fraction greater than 0 and at most 1"),
        ("keep_tmp", resolve_keep_tmp, False, "Temporary files", "Choose whether to keep temporary files"),
    ):
        try:
            resolved[key] = resolver((overrides or {}).get(key), config)
        except (ValueError, TypeError, OverflowError):
            if strict:
                raise
            resolved[key] = fallback
            issues[key] = f"{label}: A saved setting or environment override is invalid. {guidance} in Options and save."
    return resolved, issues
