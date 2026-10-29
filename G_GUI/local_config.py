from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from cryoid_pipeline.user_config import (
    DEFAULT_CONFIG, PATH_KEYS, PROJECT_STATE_KEYS, STARTUP_KEYS, config_value, normalize_path,
    project_config_path, save_config_changes,
)


def gui_config_view(config: Mapping[str, Any]) -> dict[str, str]:
    values = {key: str(config.get(key, default)) for key, default in DEFAULT_CONFIG.items()}
    values["weights_path"] = values.pop("weights_dir")
    return values


def shared_config_view(config: Mapping[str, Any]) -> dict[str, str]:
    values = {key: str(config.get(key, default)) for key, default in DEFAULT_CONFIG.items()}
    values["weights_dir"] = str(config.get("weights_path", config.get("weights_dir", "")))
    for key in PATH_KEYS:
        values[key] = normalize_path(values[key])
    return values


def save_project_config(
    config: Mapping[str, Any], project_dir: Path, *, baseline: Mapping[str, Any],
    initialize: bool = False, snapshot: bool = False, force: bool = False,
) -> dict[str, str]:
    values = shared_config_view(config)
    values["project_dir"] = str(project_dir)
    changes = {key: value for key, value in values.items() if key not in STARTUP_KEYS
               if config_value(key, value) != config_value(key, baseline.get(key, DEFAULT_CONFIG[key]))}
    if snapshot:
        changes.update({key: value for key, value in values.items()
                        if key not in STARTUP_KEYS and (key not in PROJECT_STATE_KEYS or key in {"last_density_map", "last_candidate_pool"})})
    initial = values if initialize or snapshot or changes else None
    if initial is not None:
        changes["project_dir"] = str(project_dir)
    saved = save_config_changes(changes, project_config_path(project_dir), baseline=baseline, initial=initial, force=force, project=True)
    saved["project_dir"] = str(project_dir)
    return saved
