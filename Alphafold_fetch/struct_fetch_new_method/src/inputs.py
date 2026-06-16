"""
Input contract for both pipeline and benchmark workflows.

Each input CSV row must provide a protein ``sequence`` and a ``density_dir``
path. Every other column is optional metadata used for labelling or for the
GT-aware code paths in benchmarks. The functions in this module resolve those
inputs into concrete on-disk paths and validate that all required prediction
maps exist.

A ``density_dir`` is expected to be a directory containing:

  - ``ground_truth/<one .map / .mrc / .ccp4 file>``
  - ``predictions/prediction_density_ca_key_atoms.mrc``
  - ``predictions/prediction_density_secondary_structure_{helix,sheet,loop}.mrc``

For benchmark rows that name a ``ground_truth_model`` or ``fitted_pdbs`` PDB
identifier, the deposited GT model is located in a sibling ``models/``
directory (see :func:`ground_truth_model_path_for_row`).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cases import infer_emd_id, normalize_pdb
from paths import METHOD_ROOT


PIPELINE_REQUIRED_CSV_COLUMNS = (
    "sequence",
    "density_dir",
)

MAP_EXTENSIONS = {".map", ".mrc", ".ccp4"}
MODEL_EXTENSIONS = {".cif", ".mmcif", ".pdb"}


@dataclass(frozen=True)
class DensityInputs:
    """Resolved paths and EMDB identifier for one CSV row's density inputs."""

    density_dir: Path
    ground_truth_map_path: Path
    prediction_dir: Path
    raw_key_atoms_1_path: Path
    emdb_id: str


def resolve_csv_path(row: dict[str, Any], column: str) -> Path:
    """Resolve a path-valued CSV column, raising on missing column or missing file."""
    raw = str(row.get(column) or "").strip()
    if not raw:
        raise ValueError(f"Input CSV is missing required path value: {column}")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = (METHOD_ROOT / path).resolve()
    else:
        path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input CSV path does not exist: {column}={path}")
    return path


def resolve_csv_dir(row: dict[str, Any], column: str) -> Path:
    """Like :func:`resolve_csv_path` but additionally asserts the path is a directory."""
    path = resolve_csv_path(row, column)
    if not path.is_dir():
        raise FileNotFoundError(f"Input CSV path is not a directory: {column}={path}")
    return path


def ground_truth_map_from_density_dir(density_dir: Path) -> Path:
    """Return the single map file inside ``<density_dir>/ground_truth/``.

    Raises if the subdirectory is missing, empty, or contains more than one
    map file (the expected layout is exactly one ground-truth map per case).
    """
    gt_map_dir = Path(density_dir) / "ground_truth"
    if not gt_map_dir.is_dir():
        raise FileNotFoundError(f"Input density_dir is missing ground_truth subdirectory: {gt_map_dir}")
    candidates = sorted(
        path for path in gt_map_dir.iterdir()
        if path.is_file() and path.suffix.lower() in MAP_EXTENSIONS
    )
    if not candidates:
        raise FileNotFoundError(f"Input density_dir ground_truth subdirectory contains no map file: {gt_map_dir}")
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one ground-truth map in {gt_map_dir}; found {len(candidates)}")
    return candidates[0]


def resolve_density_inputs(row: dict[str, Any]) -> DensityInputs:
    """Resolve ``density_dir`` and all required prediction maps for a CSV row.

    Validates that every expected prediction map file exists. Raises with a
    clear message naming the missing file(s) if any are absent.
    """
    density_dir = resolve_csv_dir(row, "density_dir")
    prediction_dir = density_dir / "predictions"
    if not prediction_dir.is_dir():
        raise FileNotFoundError(f"Input density_dir is missing predictions subdirectory: {prediction_dir}")
    ground_truth_map_path = ground_truth_map_from_density_dir(density_dir)
    raw_key_atoms_1_path = prediction_dir / "prediction_density_ca_key_atoms.mrc"
    required_prediction_paths = [
        raw_key_atoms_1_path,
        prediction_dir / "prediction_density_secondary_structure_helix.mrc",
        prediction_dir / "prediction_density_secondary_structure_sheet.mrc",
        prediction_dir / "prediction_density_secondary_structure_loop.mrc",
    ]
    missing = [path for path in required_prediction_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Input density_dir is missing required map file(s): " + ", ".join(str(path) for path in missing))
    return DensityInputs(
        density_dir=density_dir,
        ground_truth_map_path=ground_truth_map_path,
        prediction_dir=prediction_dir,
        raw_key_atoms_1_path=raw_key_atoms_1_path,
        emdb_id=infer_emd_id(ground_truth_map_path.name),
    )


def ground_truth_model_id(csv_row: dict[str, Any], gt_model_path: Path | None = None) -> str:
    """Return the GT PDB identifier from CSV columns, or infer it from the model filename."""
    pdb_id = normalize_pdb(csv_row.get("ground_truth_model") or csv_row.get("fitted_pdbs") or "")
    if pdb_id:
        return pdb_id
    if gt_model_path is not None:
        stem = gt_model_path.stem
        prefix = "ground_truth_model_"
        if stem.lower().startswith(prefix):
            return normalize_pdb(stem[len(prefix):])
        return normalize_pdb(stem)
    return ""


def ground_truth_model_path_for_row(csv_row: dict[str, Any]) -> Path:
    """Locate the deposited GT model on disk for a benchmark row.

    Expects a sibling ``models/`` directory next to ``density_dir``. When the
    row names a PDB identifier, looks for ``ground_truth_model_<PDB>.{cif,mmcif,pdb}``.
    Otherwise it requires exactly one model file in ``models/`` to remove
    ambiguity.
    """
    density_dir = resolve_csv_dir(csv_row, "density_dir")
    models_dir = density_dir.parent / "models"
    if not models_dir.is_dir():
        raise FileNotFoundError(f"Input density_dir has no sibling models directory: {models_dir}")
    pdb_id = ground_truth_model_id(csv_row)
    if pdb_id:
        candidates = [
            models_dir / f"ground_truth_model_{pdb_id}.cif",
            models_dir / f"ground_truth_model_{pdb_id}.mmcif",
            models_dir / f"ground_truth_model_{pdb_id}.pdb",
        ]
        for path in candidates:
            if path.is_file():
                return path
        raise FileNotFoundError(f"Ground-truth model is missing from models directory for {pdb_id}: {models_dir}")
    candidates = sorted(
        path for path in models_dir.iterdir()
        if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS
    )
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected exactly one ground-truth model in {models_dir}; found {len(candidates)}")
    return candidates[0]
