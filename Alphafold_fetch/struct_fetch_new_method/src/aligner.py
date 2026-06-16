"""
Thin wrapper around the rawfield_ss_dominant alignment runner.

:func:`align_model_to_map` is the single public entry point used by
``pipeline.py`` and ``benchmark_gt_only.py``. It:

1. Builds a :class:`cases.CaseSpec` from the input metadata.
2. Optionally writes a randomized starting pose (used by benchmark runs to
   prevent benchmark success from being trivially explained by deposited
   coordinates already being well-placed).
3. Delegates to :func:`rawfield.run_raw_field`.
4. Packages the resulting payload into a stable dict shape that includes the
   alignment score, the score metric name, the fitted output path, and rich
   diagnostic metadata.

Only ``rawfield_ss_dominant`` is supported as an alignment method.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.spatial.transform import Rotation as SciRot

from cases import (
    CaseSpec,
    case_key_from_parts,
    float_or_none,
    infer_emd_id,
    normalize_emd,
    normalize_pdb,
    raw_prediction_meta,
    resolve_existing_path,
)
from io_utils import to_jsonable, write_json
import map_utils
from rawfield import RAWFIELD_SS_DOMINANT_CONFIG, SCORE_METRIC, run_raw_field


@dataclass(frozen=True)
class AlignmentConfig:
    """Tunable knobs for one alignment invocation.

    Attributes
    ----------
    method_id
        Hard-coded to ``"rawfield_ss_dominant"``. Reserved for future methods.
    atom_mode
        Which atoms drive the secondary-structure term: ``"backbone"`` (N/CA/C/O)
        or ``"heavy"`` (all non-hydrogen).
    score_metric
        Name of the SW-against-rawfields score, propagated into output for
        downstream consumers (set to :data:`rawfield.SCORE_METRIC`).
    compute_gt_metrics
        When True, compute symmetry-aware pose metrics against the deposited
        GT model. Useful in benchmark runs; turned off for production runs.
    randomize_start
        When True, write a random-rotation-and-translation starting pose for
        the candidate (used by benchmarks).
    random_start_margin_A
        Requested in-bounds margin (Angstroms) for the random start; falls
        back to 0 when the rotated model doesn't fit at the requested margin.
    rawfield_config_overrides
        Per-call overrides merged into :data:`rawfield.RAWFIELD_SS_DOMINANT_CONFIG`
        for advanced users; usually ``None``.
    """

    method_id: str = "rawfield_ss_dominant"
    atom_mode: str = "backbone"
    score_metric: str = SCORE_METRIC
    compute_gt_metrics: bool = True
    randomize_start: bool = False
    random_start_margin_A: float = 4.0
    rawfield_config_overrides: Optional[dict[str, Any]] = None


def _output_suffix_for_model(model_path: Path) -> str:
    """Return ``.cif`` / ``.mmcif`` from input, else ``.pdb`` (default for unknown formats)."""
    suffix = model_path.suffix.lower()
    return suffix if suffix in (".cif", ".mmcif") else ".pdb"


def fitted_output_path(fitted_dir: Path, label: str, model_path: Path) -> Path:
    """Compose the per-candidate fitted-model output path."""
    return Path(fitted_dir) / f"{label}_rawfield{_output_suffix_for_model(Path(model_path))}"


def _write_randomized_start_model(
    *,
    case: CaseSpec,
    source_model_path: Path,
    output_model_path: Path,
    method_run_dir: Path,
    seed: int,
    margin_A: float,
) -> tuple[Path, dict[str, Any]]:
    """Write a deterministic random-rotated, random-translated copy of the input model.

    The translation is drawn so the rotated model's bounding box fits within
    the map's bounding box minus ``margin_A``. If that's infeasible the margin
    is silently dropped to 0; if still infeasible we raise.

    Returns ``(randomized_model_path, meta_dict)`` describing the placement.
    The meta dict is also written next to the model as a JSON sidecar.
    """
    mapinfo = map_utils.get_mapinfo(case.map_path, denoise_sigma=0.0)
    _, rep_coords = map_utils.read_model_coords(source_model_path)
    rep_coords = np.asarray(rep_coords, dtype=np.float64)
    if rep_coords.size == 0:
        raise RuntimeError(f"No representative atoms found for random start: {source_model_path}")

    box_min = np.array([mapinfo.eff_x0, mapinfo.eff_y0, mapinfo.eff_z0], dtype=np.float64)
    box_max = np.array(
        [
            mapinfo.eff_x0 + mapinfo.extent_x,
            mapinfo.eff_y0 + mapinfo.extent_y,
            mapinfo.eff_z0 + mapinfo.extent_z,
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(int(seed))
    rotation = SciRot.random(random_state=rng).as_matrix()
    rep_rot = (rotation @ rep_coords.T).T
    margin = float(max(0.0, margin_A))
    min_bounds = box_min + margin
    max_bounds = box_max - margin
    low = min_bounds - rep_rot.min(axis=0)
    high = max_bounds - rep_rot.max(axis=0)
    used_margin = margin
    if np.any(low > high):
        min_bounds = box_min
        max_bounds = box_max
        low = min_bounds - rep_rot.min(axis=0)
        high = max_bounds - rep_rot.max(axis=0)
        used_margin = 0.0
    if np.any(low > high):
        raise RuntimeError(f"Could not place randomized start within map bounds for {case.key}")

    translation = rng.uniform(low=low, high=high)
    random_dir = method_run_dir / "randomized_input"
    random_dir.mkdir(parents=True, exist_ok=True)
    random_model_path = random_dir / f"{output_model_path.stem}_randomized_start{_output_suffix_for_model(source_model_path)}"
    map_utils.write_transformed_structure(source_model_path, random_model_path, rotation, translation)
    meta = {
        "enabled": True,
        "seed": int(seed),
        "seed_source": "emdb_number",
        "source_model_path": str(source_model_path),
        "randomized_model_path": str(random_model_path),
        "attempts_used": 1,
        "max_attempts": 1,
        "rotation_matrix": rotation,
        "translation_vector": translation,
        "in_bounds_margin_A": float(used_margin),
        "requested_margin_A": float(margin_A),
        "placement_stage_name": "simple_in_bounds",
        "placement_pad_frac": 0.0,
        "stage_specs": [],
        "placement_box_min_xyz": min_bounds,
        "placement_box_max_xyz": max_bounds,
        "placement_method": "single_random_rotation_in_map_bounds",
        "validation": {"representative_atoms_in_bounds": True},
        "attempt_preview": [
            {
                "attempt": 1,
                "stage": 1,
                "stage_name": "simple_in_bounds",
                "margin_A": float(used_margin),
                "pad_frac": 0.0,
                "status": "chosen",
            }
        ],
    }
    write_json(random_dir / f"{output_model_path.stem}_randomized_start.json", meta)
    return random_model_path, meta


def _case_key_from_meta(meta: dict[str, Any], gene_id: str, emdb_id: str, gt_model_path: Optional[Path], model_path: Path) -> str:
    """Compose the case key, preferring an explicit override in ``meta``."""
    explicit = str(meta.get("case_key") or "").strip()
    if explicit:
        return explicit
    pdb_id = normalize_pdb(meta.get("ground_truth_model") or "")
    if not pdb_id and gt_model_path is not None:
        pdb_id = normalize_pdb(gt_model_path.name)
    if not pdb_id:
        pdb_id = normalize_pdb(model_path.name)
    return case_key_from_parts(gene_id, pdb_id, emdb_id)


def _build_case_spec(
    *,
    map_path: Path,
    model_path: Path,
    recommended_contour_level: Optional[float],
    meta: dict[str, Any],
) -> tuple[CaseSpec, dict[str, Any]]:
    """Build the ``CaseSpec`` describing one input row for the rawfield runner.

    Pipeline always populates ``meta["gene_id"]`` (from CSV or from the top-hit
    UniProt accession) before reaching this code path; the assertion guards
    against accidental future regressions.
    """
    gene_id = str(meta.get("gene_id") or meta.get("target") or "").strip()
    if not gene_id:
        raise ValueError("internal: gene_id is missing from input_meta; pipeline._candidate_selection should always set it")
    emdb_id = normalize_emd(meta.get("emdb_id") or "") or infer_emd_id(map_path.name)
    gt_model_path = resolve_existing_path(meta.get("gt_model_path"))
    pdb_id = normalize_pdb(meta.get("ground_truth_model") or "")
    if not pdb_id and gt_model_path is not None:
        pdb_id = normalize_pdb(gt_model_path.name)
    raw_meta = raw_prediction_meta(meta)
    case = CaseSpec(
        key=_case_key_from_meta(meta, gene_id, emdb_id, gt_model_path, model_path),
        gene=gene_id,
        pdb_id=pdb_id or normalize_pdb(model_path.name),
        emdb_id=emdb_id,
        resolution=float_or_none(meta.get("resolution")),
        recommended_contour_level=recommended_contour_level,
        gt_model_path=gt_model_path or model_path,
        map_path=map_path,
        raw_prediction_dir=Path(raw_meta["raw_prediction_dir"]),
        raw_key_atoms_1_path=Path(raw_meta["raw_key_atoms_1_path"]),
    )
    return case, raw_meta


def align_model_to_map(
    map_path: Path | str,
    model_path: Path | str,
    output_model_path: Path | str,
    *,
    recommended_contour_level: Optional[float] = None,
    config: Optional[AlignmentConfig] = None,
    input_meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Align one candidate model to a cryo-EM map with the rawfield_ss_dominant method.

    Writes the fitted model to ``output_model_path`` and returns a dict with
    the alignment score, score metric, fitted output path, optional GT
    pose-metric comparison, and a rich ``alignment_meta`` block including the
    selected method, runner, search trace, and randomized-start record.
    """
    started = time.perf_counter()
    config = config or AlignmentConfig()
    if config.method_id != "rawfield_ss_dominant":
        raise ValueError("Only rawfield_ss_dominant is supported")
    map_path = Path(map_path)
    model_path = Path(model_path)
    output_model_path = Path(output_model_path)
    if not map_path.exists():
        raise FileNotFoundError(f"Map not found: {map_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    meta = dict(input_meta or {})
    case, raw_meta = _build_case_spec(map_path=map_path, model_path=model_path, recommended_contour_level=recommended_contour_level, meta=meta)

    method_run_dir = output_model_path.parent.parent / "rawfield_case_runs" / output_model_path.stem
    alignment_input_model_path = model_path
    randomized_start_meta: dict[str, Any] = {"enabled": False}
    if bool(config.randomize_start):
        seed = int(case.seed)
        alignment_input_model_path, randomized_start_meta = _write_randomized_start_model(
            case=case,
            source_model_path=model_path,
            output_model_path=output_model_path,
            method_run_dir=method_run_dir,
            seed=seed,
            margin_A=float(config.random_start_margin_A),
        )

    method_cfg = dict(RAWFIELD_SS_DOMINANT_CONFIG)
    method_cfg["atom_mode"] = str(config.atom_mode)
    if config.rawfield_config_overrides:
        method_cfg.update(dict(config.rawfield_config_overrides))
    payload = run_raw_field(
        case,
        {
            "case_dir": method_run_dir.parent,
            "source_model_path": alignment_input_model_path,
            "original_model_path": model_path,
            "output_model_path": output_model_path,
            "compute_gt_metrics": bool(config.compute_gt_metrics),
            "randomized_start_meta": randomized_start_meta,
        },
        method_run_dir,
        method_cfg,
    )
    score = float(payload["score"])
    if not math.isfinite(score):
        raise RuntimeError(f"Non-finite alignment score for {case.key}: {score}")
    return to_jsonable(
        {
            "score": score,
            "score_metric": config.score_metric,
            "optimization_score": score,
            "selected_method_id": config.method_id,
            "alignment_runner": "local_rawfield.run_raw_field",
            "method_runner": "raw_field",
            "mean_z": payload.get("mean_z"),
            "final_map_meanz": payload.get("final_map_meanz"),
            "raw_field_score": payload.get("raw_field_score"),
            "raw_field_metrics": payload.get("raw_field_metrics") or {},
            "map_cc_score": payload.get("map_cc_score"),
            "fitted_path": output_model_path,
            "source_model": model_path,
            "alignment_input_model": alignment_input_model_path,
            "alignment_seconds": float(time.perf_counter() - started),
            "gt_metrics": payload.get("gt_metrics") or {},
            "alignment_meta": {
                "case_key": case.key,
                "method_run_dir": method_run_dir,
                "raw_prediction_dir": case.raw_prediction_dir,
                "raw_key_atoms_1_path": case.raw_key_atoms_1_path,
                "selected_method_id": config.method_id,
                "runner": "raw_field",
                "randomized_start": randomized_start_meta,
                "search_meta": payload.get("search_meta") or {},
                "raw_prediction_meta": raw_meta,
            },
        }
    )
