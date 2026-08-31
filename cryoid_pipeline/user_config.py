from __future__ import annotations

import json
import os
import platform
from pathlib import Path
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
ENV_RCL = "CRYOID_RCL"
ENV_PROTEIN_MIN_TRACE_LENGTH = "CRYOID_PROTEIN_MIN_TRACE_LENGTH"
ENV_STRUCTURE_SELECTION = "CRYOID_STRUCTURE_SELECTION"
ENV_KEEP_TMP = "CRYOID_KEEP_TMP"
ENV_CHIMERAX_PATH = "CRYOID_CHIMERAX_PATH"
RESULTS_FOLDER_NAME = "cryoid_results"

DEFAULT_CONFIG: dict[str, str] = {
    "project_dir": "",
    "weights_dir": "",
    "device": "",
    "rcl": "0",
    "protein_min_trace_length": "10",
    "structure_selection": "",
    "keep_tmp": "false",
    "chimerax_path": "",
    "blast_bin_dir": "",
    "last_run_dir": "",
    "sequence_ranking_run_dir": "",
    "last_density_map": "",
    "last_candidate_pool": "",
    "sequence_mode": "auto",
    # Loaded only so an old file can be rewritten without a parse failure. It
    # is intentionally ignored: the new shared selector has different units
    # and mode-specific defaults, so silently applying an old RNA-only value
    # to protein runs would be unsafe.
    "rna_structure_cutoff_percent": "",
}

DEFAULT_KEYS = (
    "project_dir",
    "weights_dir",
    "device",
    "rcl",
    "protein_min_trace_length",
    "structure_selection",
    "keep_tmp",
    "sequence_mode",
    "blast_bin_dir",
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


def load_config(path: Path | str | None = None) -> dict[str, str]:
    config_path = Path(path).expanduser().resolve() if path is not None else default_config_path()
    try:
        with config_path.open("rb") as handle:
            raw = tomllib.load(handle)
    except (FileNotFoundError, OSError, tomllib.TOMLDecodeError):
        return dict(DEFAULT_CONFIG)
    if not isinstance(raw, dict):
        return dict(DEFAULT_CONFIG)

    combined: dict[str, Any] = {}
    defaults = raw.get("defaults", {})
    gui = raw.get("gui", {})
    if isinstance(defaults, dict):
        combined.update(defaults)
    if isinstance(gui, dict):
        combined.update(gui)
    # Accept a flat file as a convenience, while writing the documented sections.
    combined.update({key: raw[key] for key in DEFAULT_CONFIG if key in raw})
    return _normalized(combined)


def _toml_string(value: object) -> str:
    return json.dumps(str(value) if value is not None else "", ensure_ascii=False)


def save_config(config: Mapping[str, Any], path: Path | str | None = None) -> dict[str, str]:
    config_path = Path(path).expanduser().resolve() if path is not None else default_config_path()
    normalized = _normalized(config)
    lines = ["[defaults]"]
    lines.extend(f"{key} = {_toml_string(normalized[key])}" for key in DEFAULT_KEYS)
    lines.extend(["", "[gui]"])
    lines.extend(f"{key} = {_toml_string(normalized[key])}" for key in GUI_KEYS)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return normalized


def ensure_config(path: Path | str | None = None) -> dict[str, str]:
    config_path = Path(path).expanduser().resolve() if path is not None else default_config_path()
    if config_path.exists():
        return load_config(config_path)
    return save_config(DEFAULT_CONFIG, config_path)


def update_config(path: Path | str | None = None, **updates: Any) -> dict[str, str]:
    config = load_config(path)
    for key, value in updates.items():
        if key in DEFAULT_CONFIG:
            config[key] = str(value) if value is not None else ""
    return save_config(config, path)


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
        return _resolved_path(argument)
    env_value = _value_from_env(ENV_WEIGHTS_DIR)
    if env_value:
        return _resolved_path(env_value)
    config_value = str((config or {}).get("weights_dir") or "").strip()
    return _resolved_path(config_value) if config_value else None


def resolve_device(argument: str | None, config: Mapping[str, Any] | None = None) -> str | None:
    if argument:
        return str(argument)
    env_value = _value_from_env(ENV_DEVICE)
    if env_value:
        return env_value
    config_value = str((config or {}).get("device") or "").strip()
    return config_value or None


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


def resolve_chimerax_path(config: Mapping[str, Any] | None = None) -> str:
    env_value = _value_from_env(ENV_CHIMERAX_PATH)
    if env_value:
        return str(_resolved_path(env_value))
    config_value = str((config or {}).get("chimerax_path") or "").strip()
    return str(_resolved_path(config_value)) if config_value else ""


def resolve_blast_bin_dir(
    argument: Path | str | None,
    config: Mapping[str, Any] | None = None,
) -> Path | None:
    from D_BLAST_searching.blast_tools import resolve_blast_bin_dir as resolve

    return resolve(argument, config)


def resolve_sequence_mode(argument: str | None, config: Mapping[str, Any] | None = None) -> str:
    from D_BLAST_searching.sequence_mode import validate_requested_mode

    value = argument if argument is not None else str((config or {}).get("sequence_mode") or "auto")
    return validate_requested_mode(value)


def validate_structure_selection(value: str | int | float) -> str:
    return parse_structure_selection(value).raw


def resolve_structure_selection(
    argument: str | int | float | None,
    config: Mapping[str, Any] | None = None,
) -> str | None:
    if argument is not None:
        return validate_structure_selection(argument)
    env_value = _value_from_env(ENV_STRUCTURE_SELECTION)
    if env_value:
        return validate_structure_selection(env_value)
    config_value = str((config or {}).get("structure_selection") or "").strip()
    if config_value:
        return validate_structure_selection(config_value)
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
