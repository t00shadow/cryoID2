"""
Case identifiers, ID normalization, and GT pose-metric helpers.

A :class:`CaseSpec` is the immutable description of one benchmark/pipeline
case passed into :func:`rawfield.run_raw_field`. The other helpers normalize
gene names, PDB identifiers, and EMDB identifiers into canonical forms, and
compute symmetry-aware pose comparison metrics against deposited GT models.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.spatial import cKDTree

import map_utils
from paths import METHOD_ROOT, resolve_path


@dataclass(frozen=True)
class CaseSpec:
    """Immutable bundle of identifiers and paths describing one case.

    ``key`` is the human-readable case identifier (``gene_pdb`` form). ``seed``
    derives a deterministic RNG seed from the EMDB number so randomized starts
    are reproducible across reruns.
    """

    key: str
    gene: str
    pdb_id: str
    emdb_id: str
    resolution: Optional[float]
    recommended_contour_level: Optional[float]
    gt_model_path: Path
    map_path: Path
    raw_prediction_dir: Path
    raw_key_atoms_1_path: Path

    @property
    def emd_num(self) -> str:
        """Numeric tail of the EMDB identifier (e.g. ``"22610"``)."""
        return normalize_emd(self.emdb_id).split("-")[-1]

    @property
    def seed(self) -> int:
        """Deterministic RNG seed derived from the EMDB number (falls back to a stable hash)."""
        try:
            return int(self.emd_num)
        except ValueError:
            return abs(hash(self.emdb_id)) % (2**31)


def normalize_pdb(value: Any) -> str:
    """Upper-case + strip PDB identifier and drop any model-file extension."""
    text = str(value or "").strip().upper()
    for suffix in (".MMCIF", ".CIF", ".PDB", ".ENT"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text


def normalize_emd(value: Any) -> str:
    """Return ``value`` in the canonical ``EMD-<number>`` form (empty string if blank)."""
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if text.startswith("EMD-"):
        return text
    if text.startswith("EMD"):
        return "EMD-" + text[3:].lstrip("-_")
    return text


def infer_emd_id(value: Any) -> str:
    """Extract the canonical EMDB identifier from a filename like ``emd_22610.map``."""
    text = str(value or "")
    match = re.search(r"(?:^|[^A-Za-z0-9])EMD[-_]?(\d+)(?:$|[^A-Za-z0-9])", text, flags=re.IGNORECASE)
    if not match:
        return ""
    return normalize_emd(f"EMD-{match.group(1)}")


def float_or_none(value: Any) -> Optional[float]:
    """Coerce ``value`` to ``float`` or return ``None`` for blanks/non-finite/non-numeric input."""
    try:
        out = float(value)
        return out if math.isfinite(out) else None
    except Exception:
        return None


def case_key_from_parts(gene: str, pdb_id: str, emdb_id: str) -> str:
    """Compose the canonical ``gene_pdb`` case key, falling back gracefully if either part is empty."""
    gene_part = str(gene).strip()
    pdb_part = normalize_pdb(pdb_id)
    return f"{gene_part}_{pdb_part}" if gene_part and pdb_part else gene_part or pdb_part


def resolve_existing_path(value: Any) -> Optional[Path]:
    """Resolve ``value`` to a Path if it exists on disk, else ``None``."""
    text = str(value or "").strip()
    if not text:
        return None
    path = resolve_path(text, base=METHOD_ROOT, must_exist=False)
    return path if path.exists() else None


def raw_prediction_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Return the resolved raw-field directory + key-atoms map paths from ``meta``.

    Raises if either path is missing from ``meta`` or doesn't exist on disk.
    The pipeline populates ``meta`` from :func:`inputs.resolve_density_inputs`
    so existence is normally guaranteed; the explicit check guards against
    direct callers like ``benchmark_gt_only`` that build their own meta dicts.
    """
    direct_path_fields = {
        "raw_key_atoms_1_path": meta.get("raw_key_atoms_1_path"),
        "raw_prediction_dir": meta.get("raw_prediction_dir"),
    }
    for field, value in direct_path_fields.items():
        if str(value or "").strip() and resolve_existing_path(value) is None:
            raise FileNotFoundError(f"Raw-field path from CSV does not exist: {field}={value}")
    direct_key = resolve_existing_path(direct_path_fields["raw_key_atoms_1_path"])
    direct_raw_dir = resolve_existing_path(direct_path_fields["raw_prediction_dir"])
    if direct_key is not None and direct_raw_dir is not None:
        return {
            "raw_prediction_dir": direct_raw_dir,
            "raw_key_atoms_1_path": direct_key,
        }
    raise FileNotFoundError(
        "Raw-field metadata is missing. The pipeline must resolve raw_prediction_dir "
        "and raw_key_atoms_1_path from density_dir, and all referenced paths must exist."
    )


def pose_metrics_vs_gt_symmetry_aware(gt_model_path: Path, fitted_model_path: Path, *, success_rmsd_A: float = 2.0) -> dict[str, Any]:
    """Compute pose comparison metrics against the GT model with symmetry awareness.

    Augments the direct atom-by-atom RMSD from :func:`map_utils.pose_metrics_vs_gt`
    with a bidirectional nearest-neighbour RMSD that succeeds on symmetric
    assemblies whose chain labelling differs from the GT. Reports both
    independently and a combined ``success_mode`` summary.
    """
    metrics = dict(map_utils.pose_metrics_vs_gt(gt_model_path, fitted_model_path))
    try:
        gt_st, gt_rep = map_utils.read_model_coords(gt_model_path)
        fit_st, fit_rep = map_utils.read_model_coords(fitted_model_path)
        gt_atoms = map_utils.collect_non_h_coords(gt_st)
        fit_atoms = map_utils.collect_non_h_coords(fit_st)
        if gt_atoms.size == 0:
            gt_atoms = np.asarray(gt_rep, dtype=np.float64)
        if fit_atoms.size == 0:
            fit_atoms = np.asarray(fit_rep, dtype=np.float64)
        if gt_atoms.shape[0] > 0 and fit_atoms.shape[0] > 0:
            d_fit_to_gt, _ = cKDTree(gt_atoms).query(fit_atoms, k=1)
            d_gt_to_fit, _ = cKDTree(fit_atoms).query(gt_atoms, k=1)
            symmetry_nn_rmsd = float(np.sqrt(0.5 * (np.mean(np.square(d_fit_to_gt)) + np.mean(np.square(d_gt_to_fit)))))
        else:
            symmetry_nn_rmsd = float("inf")
    except Exception as exc:
        metrics["symmetry_metric_error"] = repr(exc)
        symmetry_nn_rmsd = float("inf")
    direct_ok = bool(metrics.get("close_to_gt_direct_rmsd_le_2A"))
    symmetry_ok = bool(np.isfinite(symmetry_nn_rmsd) and symmetry_nn_rmsd <= float(success_rmsd_A))
    metrics["symmetry_metric_basis"] = "all_non_h_atoms_bidirectional_nn_rmsd"
    metrics["symmetry_nn_rmsd_A"] = symmetry_nn_rmsd
    metrics["close_to_gt_symmetry_nn_le_2A"] = bool(symmetry_ok)
    metrics["close_to_gt_symmetry_or_direct_le_2A"] = bool(direct_ok or symmetry_ok)
    if direct_ok and symmetry_ok:
        success_mode = "direct_and_symmetry"
    elif direct_ok:
        success_mode = "direct_only"
    elif symmetry_ok:
        success_mode = "symmetry_only"
    else:
        success_mode = "none"
    metrics["success_mode"] = success_mode
    return metrics
