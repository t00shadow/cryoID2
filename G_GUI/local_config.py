from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from cryoid_pipeline.user_config import (
    ensure_config,
    load_config,
    resolve_structure_selection,
    resolve_keep_tmp,
    results_project_dir,
    save_config,
)


DEFAULT_LOCAL_CONFIG: dict[str, str] = {
    "project_dir": "",
    "last_run_dir": "",
    "sequence_ranking_run_dir": "",
    "weights_path": "",
    "rcl": "0",
    "protein_min_trace_length": "10",
    "structure_selection": "",
    "keep_tmp": "false",
    "chimerax_path": "",
    "blast_bin_dir": "",
    "last_density_map": "",
    "last_candidate_pool": "",
    "sequence_mode": "auto",
}


def _gui_view(config: Mapping[str, Any]) -> dict[str, str]:
    return {
        "project_dir": str(config.get("project_dir") or ""),
        "last_run_dir": str(config.get("last_run_dir") or ""),
        "sequence_ranking_run_dir": str(config.get("sequence_ranking_run_dir") or ""),
        "weights_path": str(config.get("weights_dir") or ""),
        "rcl": str(config.get("rcl") or "0"),
        "protein_min_trace_length": str(config.get("protein_min_trace_length") or "10"),
        "chimerax_path": str(config.get("chimerax_path") or ""),
        "blast_bin_dir": str(config.get("blast_bin_dir") or ""),
        "last_density_map": str(config.get("last_density_map") or ""),
        "last_candidate_pool": str(config.get("last_candidate_pool") or ""),
        "sequence_mode": str(config.get("sequence_mode") or "auto"),
        "structure_selection": str(resolve_structure_selection(None, config) or ""),
        "keep_tmp": "true" if resolve_keep_tmp(None, config) else "false",
    }


def load_local_config(path: Path | None = None) -> dict[str, str]:
    return _gui_view(load_config(path))


def save_local_config(config: Mapping[str, Any], path: Path | None = None) -> dict[str, str]:
    shared = load_config(path)
    shared.update(
        {
            "project_dir": config.get("project_dir", ""),
            "last_run_dir": config.get("last_run_dir", ""),
            "sequence_ranking_run_dir": config.get("sequence_ranking_run_dir", ""),
            "weights_dir": config.get("weights_path", config.get("weights_dir", "")),
            "rcl": config.get("rcl", "0"),
            "protein_min_trace_length": config.get("protein_min_trace_length", "10"),
            "chimerax_path": config.get("chimerax_path", ""),
            "blast_bin_dir": config.get("blast_bin_dir", ""),
            "last_density_map": config.get("last_density_map", ""),
            "last_candidate_pool": config.get("last_candidate_pool", ""),
            "sequence_mode": config.get("sequence_mode", "auto"),
            "structure_selection": config.get("structure_selection", ""),
            "keep_tmp": config.get("keep_tmp", "false"),
        }
    )
    return _gui_view(save_config(shared, path))


def ensure_local_config(path: Path | None = None) -> dict[str, str]:
    return _gui_view(ensure_config(path))


def project_dir_or_default(config: dict[str, Any], default_project_dir: Path | str) -> Path:
    configured = str(config.get("project_dir") or "").strip()
    if configured:
        path = Path(configured).expanduser()
        if path.is_dir():
            return results_project_dir(path)
    return results_project_dir(default_project_dir)


def update_local_config(path: Path | None = None, **updates: Any) -> dict[str, str]:
    config = load_local_config(path)
    for key, value in updates.items():
        if key in DEFAULT_LOCAL_CONFIG:
            config[key] = str(value) if value is not None else ""
    return save_local_config(config, path)
