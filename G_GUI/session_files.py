from __future__ import annotations

import csv
from pathlib import Path


DENSITY_SUFFIXES = {".ccp4", ".map", ".mrc"}
MODEL_SUFFIXES = {".cif", ".mmcif", ".pdb"}
DISPLAY_SUFFIXES = {
    ".ccp4",
    ".csv",
    ".fasta",
    ".fa",
    ".map",
    ".mrc",
    ".pdb",
    ".cif",
    ".mmcif",
    ".tsv",
}


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for path in paths:
        key = str(path.resolve()).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path.resolve())
    return unique


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def ensure_project_dirs(project_dir: Path) -> None:
    project_dir = Path(project_dir).expanduser().resolve()
    for directory in (
        project_dir / "unetpredictions",
        project_dir / "runs",
        project_dir / "best_overall",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def find_project_dir_from_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    candidate = path if path.is_dir() else path.parent
    try:
        candidate = candidate.resolve()
    except OSError:
        pass
    for current in (candidate, *candidate.parents):
        if (
            (current / "runs").is_dir()
            and (current / "best_overall").is_dir()
            and (current / "unetpredictions").is_dir()
        ):
            return current
    return None


def find_run_dir_from_path(path: Path | None) -> Path | None:
    if path is None:
        return None
    candidate = path if path.is_dir() else path.parent
    try:
        candidate = candidate.resolve()
    except OSError:
        pass
    for current in (candidate, *candidate.parents):
        if (current / "run_config.json").is_file() and (current / "pipeline.log").is_file():
            return current
    return None


def _display_files_under(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [
        path.resolve()
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in DISPLAY_SUFFIXES
    ]


def collect_project_display_files(project_dir: Path) -> list[Path]:
    project_dir = Path(project_dir).expanduser().resolve()
    paths: list[Path] = []
    paths.extend(_display_files_under(project_dir / "best_overall"))
    paths.extend(_display_files_under(project_dir / "unetpredictions"))
    return _dedupe_paths(paths)


def collect_result_display_files(project_dir: Path, output_files: list[Path] | None = None) -> list[Path]:
    # The ranking has a dedicated dialog and must not appear in the default
    # results-file list. Keep the optional argument for the stable GUI API.
    _ = output_files
    return collect_project_display_files(project_dir)


def latest_run_dir(project_dir: Path) -> Path | None:
    runs = Path(project_dir).expanduser().resolve() / "runs"
    if not runs.is_dir():
        return None
    candidates = sorted(
        (
            path.resolve()
            for path in runs.iterdir()
            if path.is_dir() and (path / "run_config.json").is_file()
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def latest_sequence_ranking(project_dir: Path) -> Path | None:
    run_dir = latest_run_dir(project_dir)
    if run_dir is None:
        return None
    ranking = run_dir / "sequence_ranking.tsv"
    return ranking.resolve() if ranking.is_file() else None


def _float_or_none(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def best_overall_model(project_dir: Path) -> Path | None:
    project = Path(project_dir).expanduser().resolve()
    rows: list[dict[str, str]] = []
    for csv_path in sorted((project / "runs").glob("run_*/alignment_summary.csv")):
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as handle:
                rows.extend(dict(row) for row in csv.DictReader(handle))
        except OSError:
            continue
    candidates: list[tuple[float, int, str, Path]] = []
    for row in rows:
        if row.get("is_best_for_type") != "true":
            continue
        score = _float_or_none(row.get("score"))
        model_path = Path(
            str(row.get("aligned_model") or row.get("model_path") or "")
        ).expanduser()
        if score is None or model_path.suffix.lower() not in MODEL_SUFFIXES or not model_path.is_file():
            continue
        source_type = str(row.get("source_type") or "").lower()
        tie_priority = 1 if source_type == "experimental" else 0
        candidates.append((score, tie_priority, source_type, model_path.resolve()))
    if not candidates:
        return None
    _score, _priority, source_type, run_model = max(
        candidates, key=lambda item: (item[0], item[1])
    )
    project_models = sorted((project / "best_overall").glob(f"best_{source_type}_model.*"))
    return project_models[0].resolve() if project_models else run_model


def collect_chimerax_project_files(project_dir: Path, input_map: Path | None) -> dict[str, list[Path]]:
    density_maps: list[Path] = []
    aligned_models: list[Path] = []
    if input_map is not None:
        input_map = Path(input_map).expanduser()
        if input_map.suffix.lower() in DENSITY_SUFFIXES and input_map.is_file():
            density_maps.append(input_map.resolve())
    model = best_overall_model(project_dir)
    if model is not None:
        aligned_models.append(model)
    return {
        "density_maps": _dedupe_paths(density_maps),
        "source_models": [],
        "aligned_models": _dedupe_paths(aligned_models),
    }


def collect_chimerax_session_files(run_or_project_dir: Path) -> dict[str, list[Path]]:
    path = Path(run_or_project_dir).expanduser().resolve()
    project_dir = find_project_dir_from_path(path)
    if project_dir is None:
        run_dir = find_run_dir_from_path(path)
        if run_dir is not None and run_dir.parent.name == "runs":
            project_dir = run_dir.parent.parent
    if project_dir is None:
        project_dir = path
    return collect_chimerax_project_files(project_dir, None)


def flatten_chimerax_session_files(groups: dict[str, list[Path]]) -> list[Path]:
    return [
        *groups.get("density_maps", []),
        *groups.get("aligned_models", []),
    ]
