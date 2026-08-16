"""
Raw-field SS-dominant rigid alignment runner.

The runner takes a candidate model and a target cryo-EM map and finds a rigid
transform that maximizes a weighted sum of three signals:

  - CA atoms landing on predicted CA-key-atom field peaks (``key1``)
  - backbone atoms landing on the per-voxel max secondary-structure field (``ssmax``)
  - backbone atoms landing in strong experimental-map density (z-score)

Search proceeds in three stages:

  1. Seed enumeration: high-percentile field voxels yield weighted principal
     frames; the 24 chiral axis-aligned rotations between candidate-model and
     field frames produce an initial bank of seeds.
  2. Pose refinement: deterministic point-cloud ICP + axis spin/shift +
     bounded Powell on the weighted score.
  3. Final map-aware refinement: a second pair of Powell branches incorporates
     the smoothed map term, then the best pose across branches is selected.

Q-score is not used inside the search and is computed separately by ``qscore.py``
as an independent validation metric.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import mrcfile
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as SciRot

from . import map_utils
from .cases import CaseSpec, pose_metrics_vs_gt_symmetry_aware


SCORE_METRIC = "rawfield_ss_dominant_score"

RAWFIELD_SS_DOMINANT_CONFIG: dict[str, Any] = {
    "id": "rawfield_ss_dominant",
    "runner": "raw_field",
    "atom_mode": "backbone",
    "field_specs": [
        {"field": "key1", "model": "ca", "percentile": 99.95, "max_points": 12000},
        {"field": "key1", "model": "ca", "percentile": 99.50, "max_points": 12000},
        {"field": "ssmax", "model": "backbone", "percentile": 99.95, "max_points": 16000},
        {"field": "ssmax", "model": "backbone", "percentile": 99.50, "max_points": 16000},
    ],
    "coarse_ca_coef": 0.8,
    "coarse_ss_coef": 1.2,
    "coarse_map_coef": 0.04,
    "refine_ca_coef": 0.8,
    "refine_ss_coef": 1.2,
    "refine_map_coef": 0.08,
    "top_k_refine": 1,
    "top_k_final_map_refine": 4,
    "icp_iterations": 8,
    "icp_percentile": 99.95,
    "icp_key_max_points": 600,
    "icp_ss_max_points": 900,
    "icp_keep_fraction": 0.70,
    "raw_refine_maxiter": 96,
    "final_map_maxiter": 128,
    "short_final_map_maxiter": 128,
    "refine_rot_bound_deg": 45.0,
    "final_map_rot_bound_deg": 35.0,
    "short_final_map_rot_bound_deg": 35.0,
    "refine_trans_bound_factor": 0.35,
    "final_map_trans_bound_factor": 0.20,
    "short_final_map_trans_bound_factor": 0.20,
    "fallback_probe_cc_radius_A": 6.0,
    "primary_seed_pose_dedupe_rmsd_A": 2.0,
    "primary_seed_pose_dedupe_min_keep": 8,
    "primary_seed_pose_backfill_rmsd_A": 1.0,
    "coarse_only": False,
    "skip_final_map_refine": False,
}


class _RawFieldMRC:
    """Lightweight stand-in for ``mrcfile.MrcFile`` that wraps a precomputed array + header."""

    def __init__(self, data: np.ndarray, header: Any, voxel_size: Any):
        self.data = data
        self.header = header
        self.voxel_size = voxel_size


_RAW_FIELD_CACHE: dict[tuple[str, ...], map_utils.MapInfo] = {}
_AXIS_ALIGNMENT_BANK: list[np.ndarray] = []
def _default_local_cache_dir() -> Path:
    env_tmp = os.environ.get("CRYOID_STRUCT_FETCH_TMP_DIR")
    if env_tmp:
        return Path(env_tmp).expanduser() / "raw_mrc_cache"
    return Path.home() / ".cryoid" / "cache" / "structure_fitting" / "raw_mrc"


_RAW_FIELD_LOCAL_CACHE = _default_local_cache_dir()


def configure_local_cache_dir(cache_dir: Path | str) -> None:
    """Route staged raw MRC copies into a run-local temp dir."""
    global _RAW_FIELD_LOCAL_CACHE
    _RAW_FIELD_LOCAL_CACHE = Path(cache_dir).expanduser().resolve()


def clear_per_case_caches() -> None:
    """Drop caches that hold case-specific data; called between cases.

    Clears :data:`_RAW_FIELD_CACHE` (the prediction-map MapInfos for the
    just-finished case have no reuse value once the next case starts, since
    each case has its own ``key1`` / ``ssmax`` paths) and wipes the on-disk
    :data:`_RAW_FIELD_LOCAL_CACHE` directory of staged MRC copies for the
    same reason. Leaves :data:`_AXIS_ALIGNMENT_BANK` intact since the 24
    chiral rotation matrices are identical for every case.
    """
    _RAW_FIELD_CACHE.clear()
    if _RAW_FIELD_LOCAL_CACHE.exists():
        try:
            shutil.rmtree(_RAW_FIELD_LOCAL_CACHE, ignore_errors=True)
        except OSError as exc:
            print(f"[rawfield] could not remove staged MRC cache {_RAW_FIELD_LOCAL_CACHE}: {exc}", flush=True)


def clear_caches() -> None:
    """Drop every module-level cache. Called by pipeline.run_csv at end of run.

    The end-of-run wipe is broader than :func:`clear_per_case_caches`: it also
    clears the axis-alignment bank, so a fresh Python process starts with
    absolutely no residual state.
    """
    clear_per_case_caches()
    _AXIS_ALIGNMENT_BANK.clear()


def model_search_data(model_path: Path, atom_mode: str = "backbone") -> dict[str, Any]:
    """Read a candidate model and pre-extract the atom subsets used during search.

    Returns a dict with CA coords, scoring coords (backbone or all heavy),
    representative coords, and the centroid of representative coords.
    """
    st, rep_coords = map_utils.read_model_coords(model_path)
    score_coords = map_utils.collect_non_h_coords(st) if atom_mode == "heavy" else map_utils.collect_backbone_coords(st)
    if score_coords.size == 0:
        score_coords = rep_coords
    ca_coords = map_utils.collect_ca_coords(st)
    if ca_coords.size == 0:
        ca_coords = np.asarray(rep_coords, dtype=np.float64)
    rep_coords = np.asarray(rep_coords, dtype=np.float64)
    return {
        "rep_coords": rep_coords,
        "score_coords": np.asarray(score_coords, dtype=np.float64),
        "ca_coords": np.asarray(ca_coords, dtype=np.float64),
        "model_com": rep_coords.mean(axis=0),
    }


def _squeeze_mrc_volume(data: np.ndarray) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D raw prediction volume after squeeze, got {arr.shape}")
    return np.asarray(arr, dtype=np.float32)


def _local_stage_path_for_raw_mrc(path: Path) -> Path:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    stem = path.stem.replace(".", "_")
    return _RAW_FIELD_LOCAL_CACHE / f"{stem}_{digest}{path.suffix}"


def _read_raw_mrc_volume_with_retry(path: Path) -> tuple[np.ndarray, Any, Any]:
    last_exc: Exception | None = None
    staged = _local_stage_path_for_raw_mrc(path)
    for use_staged in (False, True):
        current = staged if use_staged else Path(path)
        try:
            if use_staged:
                _RAW_FIELD_LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
                if (not staged.exists()) or staged.stat().st_size <= 0:
                    shutil.copyfile(str(path), str(staged))
            with mrcfile.open(str(current), permissive=True) as mrc:
                arr = _squeeze_mrc_volume(mrc.data)
                header = mrc.header.copy()
                voxel_size = mrc.voxel_size.copy()
            return arr, header, voxel_size
        except Exception as exc:
            last_exc = exc
            if use_staged and staged.exists():
                staged.unlink(missing_ok=True)
    if last_exc is None:
        raise RuntimeError(f"Failed to read raw prediction volume: {path}")
    raise last_exc


def _mapinfo_from_prediction_paths(paths: Iterable[Path]) -> map_utils.MapInfo:
    key = tuple(str(Path(p)) for p in paths)
    cached = _RAW_FIELD_CACHE.get(key)
    if cached is not None:
        return cached
    concrete_paths = [Path(p) for p in key]
    arr, header, voxel_size = _read_raw_mrc_volume_with_retry(concrete_paths[0])
    mapinfo = map_utils.MapInfo(_RawFieldMRC(arr, header, voxel_size))
    for path in concrete_paths[1:]:
        arr, _, _ = _read_raw_mrc_volume_with_retry(path)
        mapinfo._set_fit_map(np.maximum(mapinfo.fit_map, arr))
    _RAW_FIELD_CACHE[key] = mapinfo
    return mapinfo


def _secondary_structure_paths_from_key(raw_key_atoms_1_path: Path) -> list[Path]:
    suffix = "_key_atoms_1_possibility.mrc"
    name = raw_key_atoms_1_path.name
    if not name.endswith(suffix):
        return []
    prefix = name[: -len(suffix)]
    paths = [
        raw_key_atoms_1_path.parent / f"{prefix}_secondary_strctures_{idx}_possibility.mrc"
        for idx in (1, 2, 3)
    ]
    return paths if all(path.exists() for path in paths) else []


def _secondary_structure_prediction_paths(case: CaseSpec) -> list[Path]:
    paths_from_key = _secondary_structure_paths_from_key(Path(case.raw_key_atoms_1_path))
    if paths_from_key:
        return paths_from_key

    base = Path(case.raw_prediction_dir)
    emd = case.emd_num
    for raw_dir in (base / "possibility", base):
        if emd:
            ss_paths = sorted(raw_dir.glob(f"emd_{emd}.map_secondary_strctures_[123]_possibility.mrc"))
            if len(ss_paths) == 3:
                return ss_paths
        renamed = [
            raw_dir / "prediction_density_secondary_structure_helix.mrc",
            raw_dir / "prediction_density_secondary_structure_sheet.mrc",
            raw_dir / "prediction_density_secondary_structure_loop.mrc",
        ]
        if all(path.exists() for path in renamed):
            return renamed
    raise FileNotFoundError(
        f"Expected 3 secondary-structure raw maps for {case.key} under {base} "
        "using either original emd_* names or organized prediction_density_* names"
    )


def _raw_field_mapinfos(case: CaseSpec, method_cfg: dict[str, Any]) -> dict[str, map_utils.MapInfo]:
    key1 = _mapinfo_from_prediction_paths([case.raw_key_atoms_1_path])
    ss_paths = _secondary_structure_prediction_paths(case)
    return {"key1": key1, "ssmax": _mapinfo_from_prediction_paths(ss_paths)}


def _axis_alignment_bank() -> list[np.ndarray]:
    global _AXIS_ALIGNMENT_BANK
    if _AXIS_ALIGNMENT_BANK:
        return _AXIS_ALIGNMENT_BANK
    eye = np.eye(3, dtype=np.float64)
    mats: list[np.ndarray] = []
    for perm in itertools.permutations(range(3)):
        p = eye[:, perm]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            mat = p @ np.diag(np.asarray(signs, dtype=np.float64))
            if np.linalg.det(mat) > 0:
                mats.append(mat)
    _AXIS_ALIGNMENT_BANK = mats
    return mats


def _weighted_frame(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    w = np.maximum(np.asarray(weights, dtype=np.float64).reshape(-1), 1e-8)
    if pts.shape[0] != w.shape[0] or pts.shape[0] < 3:
        raise ValueError("Need >=3 weighted points")
    center = np.average(pts, axis=0, weights=w)
    centered = pts - center
    cov = (centered * w[:, None]).T @ centered / float(np.sum(w))
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = np.asarray(evals[order], dtype=np.float64)
    evecs = np.asarray(evecs[:, order], dtype=np.float64)
    if np.linalg.det(evecs) < 0:
        evecs[:, -1] *= -1.0
    return np.asarray(center, dtype=np.float64), evecs, evals


def _unweighted_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 3:
        raise ValueError("Need >=3 points for frame")
    center = pts.mean(axis=0)
    centered = pts - center
    cov = centered.T @ centered / float(max(pts.shape[0], 1))
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = np.asarray(evals[order], dtype=np.float64)
    evecs = np.asarray(evecs[:, order], dtype=np.float64)
    if np.linalg.det(evecs) < 0:
        evecs[:, -1] *= -1.0
    return np.asarray(center, dtype=np.float64), evecs, evals


def _top_weighted_world_points(mapinfo: map_utils.MapInfo, percentile: float, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    flat = np.asarray(mapinfo.fit_map, dtype=np.float32).reshape(-1)
    threshold = float(np.percentile(flat, float(percentile)))
    idx = np.flatnonzero(flat >= threshold)
    if idx.size == 0:
        raise RuntimeError(f"No voxels found above percentile {percentile}")
    if idx.size > int(max_points):
        vals = flat[idx]
        keep = np.argpartition(vals, -int(max_points))[-int(max_points):]
        idx = idx[keep]
    vals = flat[idx]
    order = np.argsort(vals)[::-1]
    idx = idx[order]
    vals = vals[order]
    iz, iy, ix = np.unravel_index(idx, mapinfo.fit_map.shape)
    xs, ys, zs = mapinfo.grid_to_world_many(ix, iy, iz)
    return np.column_stack([xs, ys, zs]).astype(np.float64, copy=False), np.asarray(vals, dtype=np.float64)


def _raw_field_metrics(
    field_maps: dict[str, map_utils.MapInfo],
    real_mapinfo: map_utils.MapInfo,
    ca_coords: np.ndarray,
    backbone_coords: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    ca_coef: float,
    ss_coef: float,
    map_coef: float,
) -> dict[str, float]:
    ca_world = (rotation @ np.asarray(ca_coords, dtype=np.float64).T).T + translation
    bb_world = (rotation @ np.asarray(backbone_coords, dtype=np.float64).T).T + translation
    key_mean = float(np.mean(field_maps["key1"].interp_many(ca_world)))
    ss_mean = float(np.mean(field_maps["ssmax"].interp_many(bb_world)))
    map_meanz = float(map_utils.score_mean_z(real_mapinfo, bb_world))
    score = float(ca_coef * key_mean + ss_coef * ss_mean + map_coef * map_meanz)
    return {"raw_field_score": score, "raw_key1_mean": key_mean, "raw_ss_mean": ss_mean, "raw_map_meanz": map_meanz}


def _seed_candidates_from_raw_fields(search: dict[str, Any], field_maps: dict[str, map_utils.MapInfo], real_mapinfo: map_utils.MapInfo, method_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    model_sets = {"ca": np.asarray(search["ca_coords"], dtype=np.float64), "backbone": np.asarray(search["score_coords"], dtype=np.float64)}
    model_frames = {name: _unweighted_frame(coords) for name, coords in model_sets.items()}
    spin_steps = max(1, int(method_cfg.get("field_spin_steps", 1)))
    spin_angles = np.array([0.0], dtype=np.float64) if spin_steps <= 1 else np.linspace(0.0, 2.0 * math.pi, num=spin_steps, endpoint=False, dtype=np.float64)
    for spec in method_cfg.get("field_specs", []):
        field_name = str(spec["field"])
        model_name = str(spec["model"])
        field_points, field_weights = _top_weighted_world_points(field_maps[field_name], percentile=float(spec.get("percentile", 99.9)), max_points=int(spec.get("max_points", 12000)))
        field_center, field_axes, field_evals = _weighted_frame(field_points, field_weights)
        model_center, model_axes, model_evals = model_frames[model_name]
        spin_axis = np.asarray(field_axes[:, 0], dtype=np.float64)
        for orient_idx, orient in enumerate(_axis_alignment_bank()):
            base_rotation = np.asarray(field_axes @ orient @ model_axes.T, dtype=np.float64)
            for spin_idx, angle in enumerate(spin_angles):
                spin_rotation = SciRot.from_rotvec(spin_axis * float(angle)).as_matrix()
                seed_rotation = np.asarray(spin_rotation @ base_rotation, dtype=np.float64)
                seed_translation = np.asarray(field_center - (seed_rotation @ model_center), dtype=np.float64)
                metrics = _raw_field_metrics(
                    field_maps,
                    real_mapinfo,
                    search["ca_coords"],
                    search["score_coords"],
                    seed_rotation,
                    seed_translation,
                    ca_coef=float(method_cfg.get("coarse_ca_coef", 1.0)),
                    ss_coef=float(method_cfg.get("coarse_ss_coef", 1.0)),
                    map_coef=float(method_cfg.get("coarse_map_coef", 0.0)),
                )
                out.append(
                    {
                        "seed_rank": None,
                        "seed_label": f"{field_name}_p{str(spec.get('percentile')).replace('.', 'p')}",
                        "target_center_mode": f"{field_name}_weighted_frame",
                        "target_center_xyz": field_center,
                        "rotation_idx": int(orient_idx),
                        "jitter_idx": int(spin_idx),
                        "field_name": field_name,
                        "model_name": model_name,
                        "field_percentile": float(spec.get("percentile", 99.9)),
                        "field_weighted_eval": field_evals,
                        "model_eval": model_evals,
                        "R": seed_rotation,
                        "t": seed_translation,
                        **metrics,
                    }
                )
    out.sort(key=lambda row: float(row["raw_field_score"]), reverse=True)
    for idx, row in enumerate(out):
        row["seed_rank"] = int(idx)
    return out


def _spin_about_world_axis(rotation: np.ndarray, translation: np.ndarray, model_com: np.ndarray, world_axis: np.ndarray, angle_rad: float) -> tuple[np.ndarray, np.ndarray]:
    axis = np.asarray(world_axis, dtype=np.float64)
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    spin = SciRot.from_rotvec(axis * float(angle_rad)).as_matrix()
    world_com = (np.asarray(rotation, dtype=np.float64) @ np.asarray(model_com, dtype=np.float64)) + np.asarray(translation, dtype=np.float64)
    new_rotation = np.asarray(spin @ np.asarray(rotation, dtype=np.float64), dtype=np.float64)
    new_translation = np.asarray(world_com - (new_rotation @ np.asarray(model_com, dtype=np.float64)), dtype=np.float64)
    return new_rotation, new_translation


def _axis_refine_pose(search: dict[str, Any], field_maps: dict[str, map_utils.MapInfo], real_mapinfo: map_utils.MapInfo, rotation_seed: np.ndarray, translation_seed: np.ndarray, method_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    steps = max(0, int(method_cfg.get("axis_refine_steps", 0)))
    if steps <= 0:
        return np.asarray(rotation_seed, dtype=np.float64), np.asarray(translation_seed, dtype=np.float64)
    _, model_axes, _ = _unweighted_frame(np.asarray(search["score_coords"], dtype=np.float64))
    model_axis = np.asarray(model_axes[:, 0], dtype=np.float64)
    model_com = np.asarray(search["model_com"], dtype=np.float64)
    shift_bound = float(method_cfg.get("axis_refine_shift_bound_factor", 0.18)) * float(max(real_mapinfo.longest_extent, 1e-6))
    map_coef = float(method_cfg.get("axis_refine_map_coef", method_cfg.get("refine_map_coef", 0.06)))
    rotation_cur = np.asarray(rotation_seed, dtype=np.float64)
    translation_cur = np.asarray(translation_seed, dtype=np.float64)
    for _ in range(steps):
        world_axis = np.asarray(rotation_cur @ model_axis, dtype=np.float64)
        world_axis = world_axis / max(np.linalg.norm(world_axis), 1e-12)

        def _spin_score(angle: float) -> float:
            rotation_try, translation_try = _spin_about_world_axis(rotation_cur, translation_cur, model_com, world_axis, float(angle))
            return _raw_field_metrics(
                field_maps,
                real_mapinfo,
                search["ca_coords"],
                search["score_coords"],
                rotation_try,
                translation_try,
                ca_coef=float(method_cfg.get("refine_ca_coef", 1.0)),
                ss_coef=float(method_cfg.get("refine_ss_coef", 1.0)),
                map_coef=map_coef,
            )["raw_field_score"]

        spin_res = minimize_scalar(lambda a: -_spin_score(float(a)), bounds=(-math.pi, math.pi), method="bounded", options={"maxiter": 64, "xatol": 1e-3})
        rotation_cur, translation_cur = _spin_about_world_axis(rotation_cur, translation_cur, model_com, world_axis, float(spin_res.x))

        def _shift_score(shift: float) -> float:
            return _raw_field_metrics(
                field_maps,
                real_mapinfo,
                search["ca_coords"],
                search["score_coords"],
                rotation_cur,
                np.asarray(translation_cur + world_axis * float(shift), dtype=np.float64),
                ca_coef=float(method_cfg.get("refine_ca_coef", 1.0)),
                ss_coef=float(method_cfg.get("refine_ss_coef", 1.0)),
                map_coef=map_coef,
            )["raw_field_score"]

        shift_res = minimize_scalar(lambda s: -_shift_score(float(s)), bounds=(-shift_bound, shift_bound), method="bounded", options={"maxiter": 64, "xatol": 1e-2})
        translation_cur = np.asarray(translation_cur + world_axis * float(shift_res.x), dtype=np.float64)
    return np.asarray(rotation_cur, dtype=np.float64), np.asarray(translation_cur, dtype=np.float64)


def kabsch_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form rigid Kabsch transform mapping ``src`` onto ``dst``: returns ``(R, t)``."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    c_src = src.mean(axis=0)
    c_dst = dst.mean(axis=0)
    u, _, vt = np.linalg.svd((src - c_src).T @ (dst - c_dst))
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    return np.asarray(rotation, dtype=np.float64), np.asarray(c_dst - (rotation @ c_src), dtype=np.float64)


def _deterministic_raw_point_icp(search: dict[str, Any], field_maps: dict[str, map_utils.MapInfo], rotation_seed: np.ndarray, translation_seed: np.ndarray, method_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    key_max_points = int(method_cfg.get("icp_key_max_points", 600))
    ss_max_points = int(method_cfg.get("icp_ss_max_points", 900))
    key_points = np.empty((0, 3), dtype=np.float64)
    ss_points = np.empty((0, 3), dtype=np.float64)
    if key_max_points > 0:
        key_points, _ = _top_weighted_world_points(field_maps["key1"], percentile=float(method_cfg.get("icp_percentile", 99.95)), max_points=key_max_points)
    if ss_max_points > 0:
        ss_points, _ = _top_weighted_world_points(field_maps["ssmax"], percentile=float(method_cfg.get("icp_percentile", 99.95)), max_points=ss_max_points)
    model_ca = np.asarray(search["ca_coords"], dtype=np.float64)
    model_bb = np.asarray(search["score_coords"], dtype=np.float64)
    keep_frac = float(method_cfg.get("icp_keep_fraction", 0.70))
    rotation_cur = np.asarray(rotation_seed, dtype=np.float64)
    translation_cur = np.asarray(translation_seed, dtype=np.float64)
    for _ in range(max(1, int(method_cfg.get("icp_iterations", 8)))):
        src_parts: list[np.ndarray] = []
        dst_parts: list[np.ndarray] = []
        if key_points.size:
            world_ca = (rotation_cur @ model_ca.T).T + translation_cur
            key_d, key_idx = cKDTree(world_ca).query(key_points, k=1)
            keep_n = max(3, int(math.ceil(keep_frac * key_points.shape[0])))
            keep = np.argsort(np.asarray(key_d, dtype=np.float64))[:keep_n]
            src_parts.append(model_ca[np.asarray(key_idx, dtype=np.int64)[keep]])
            dst_parts.append(key_points[keep])
        if ss_points.size:
            world_bb = (rotation_cur @ model_bb.T).T + translation_cur
            ss_d, ss_idx = cKDTree(world_bb).query(ss_points, k=1)
            keep_n = max(6, int(math.ceil(keep_frac * ss_points.shape[0])))
            if key_points.size:
                keep_n = min(keep_n, max(2 * int(key_points.shape[0]), 64))
            keep = np.argsort(np.asarray(ss_d, dtype=np.float64))[:keep_n]
            src_parts.append(model_bb[np.asarray(ss_idx, dtype=np.int64)[keep]])
            dst_parts.append(ss_points[keep])
        if not src_parts:
            break
        src = np.vstack(src_parts)
        dst = np.vstack(dst_parts)
        if src.shape[0] < 3:
            break
        rotation_cur, translation_cur = kabsch_transform(src, dst)
    return np.asarray(rotation_cur, dtype=np.float64), np.asarray(translation_cur, dtype=np.float64)


def _map_cc_score(mapinfo: map_utils.MapInfo, model_pts_world: np.ndarray, *, radius_A: float, max_dim: int = 128) -> float:
    density_ds, pts_zyx_ds, scale_mean = map_utils.prepare_density_inputs(mapinfo, np.asarray(model_pts_world, dtype=np.float64), int(max_dim))
    vs_mean = float((mapinfo.sx + mapinfo.sy + mapinfo.sz) / 3.0)
    radius_ds = float(radius_A) / max(vs_mean * float(scale_mean), 1e-6)
    occ = map_utils.model_occupancy_from_points(density_ds.shape, pts_zyx_ds, radius_ds).astype(np.float32, copy=False).reshape(-1)
    x = occ - float(np.mean(occ))
    y = np.asarray(density_ds, dtype=np.float32).reshape(-1)
    y = y - float(np.mean(y))
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    return 0.0 if den <= 0.0 else float(np.dot(x, y) / den)


def _seed_pose_moment_payload(model_coords: np.ndarray) -> dict[str, np.ndarray]:
    coords = np.asarray(model_coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[0] <= 0 or coords.shape[1] != 3:
        raise ValueError("Expected non-empty Nx3 model coordinates")
    return {"mean": np.mean(coords, axis=0, dtype=np.float64), "second_moment": (coords.T @ coords) / float(coords.shape[0])}


def _seed_pose_rmsd_A(row_a: dict[str, Any], row_b: dict[str, Any], pose_payload: dict[str, np.ndarray]) -> float:
    rotation_a = np.asarray(row_a["R"], dtype=np.float64)
    translation_a = np.asarray(row_a["t"], dtype=np.float64)
    rotation_b = np.asarray(row_b["R"], dtype=np.float64)
    translation_b = np.asarray(row_b["t"], dtype=np.float64)
    mean = np.asarray(pose_payload["mean"], dtype=np.float64)
    second = np.asarray(pose_payload["second_moment"], dtype=np.float64)
    delta_rotation = rotation_a - rotation_b
    delta_translation = translation_a - translation_b
    mse = float(np.trace(delta_rotation @ second @ delta_rotation.T) + 2.0 * (delta_translation @ (delta_rotation @ mean)) + (delta_translation @ delta_translation))
    if mse < 0.0 and mse > -1e-8:
        mse = 0.0
    return float(math.sqrt(max(mse, 0.0)))


def _build_primary_seed_bank(seeds: list[dict[str, Any]], method_cfg: dict[str, Any], *, pose_payload: dict[str, np.ndarray] | None = None) -> list[dict[str, Any]]:
    top_overall = max(1, int(method_cfg.get("primary_seed_top_overall", method_cfg.get("top_k_refine", 12))))
    pose_dedupe = max(0.0, float(method_cfg.get("primary_seed_pose_dedupe_rmsd_A", 0.0)))
    pose_backfill = max(0.0, float(method_cfg.get("primary_seed_pose_backfill_rmsd_A", 0.0)))
    pose_min_keep = max(1, int(method_cfg.get("primary_seed_pose_dedupe_min_keep", top_overall)))
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, float, int, int]] = set()
    skipped_pose_dupes: list[dict[str, Any]] = []

    def _metadata_key(row: dict[str, Any]) -> tuple[str, float, int, int]:
        return str(row.get("field_name", "")), float(row.get("field_percentile", 0.0)), int(row.get("rotation_idx", -1)), int(row.get("jitter_idx", -1))

    def _pose_is_far_enough(row: dict[str, Any], min_rmsd: float) -> bool:
        if pose_payload is None or min_rmsd <= 0.0:
            return True
        return all(_seed_pose_rmsd_A(row, kept, pose_payload) > float(min_rmsd) for kept in out)

    def _add(row: dict[str, Any], *, min_rmsd: float, collect_skip: bool) -> bool:
        key = _metadata_key(row)
        if key in seen:
            return False
        if not _pose_is_far_enough(row, min_rmsd):
            if collect_skip:
                skipped_pose_dupes.append(row)
            return False
        seen.add(key)
        out.append(row)
        return True

    for row in seeds:
        if len(out) >= top_overall:
            break
        _add(row, min_rmsd=pose_dedupe, collect_skip=True)
    if len(out) < min(top_overall, pose_min_keep) and skipped_pose_dupes:
        for row in skipped_pose_dupes:
            if len(out) >= min(top_overall, pose_min_keep):
                break
            _add(row, min_rmsd=pose_backfill, collect_skip=False)
    if not out and seeds:
        _add(seeds[0], min_rmsd=0.0, collect_skip=False)
    return out


def _sort_refined_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = list(rows)
    out.sort(
        key=lambda row: (
            float(row.get("raw_field_score", -1e18)),
            -int(row.get("seed_rank", 10**9)),
        ),
        reverse=True,
    )
    return out


def _payload(
    case: CaseSpec,
    case_ctx: dict[str, Any],
    run_dir: Path,
    method_cfg: dict[str, Any],
    best: dict[str, Any],
    refined_preview: list[dict[str, Any]],
) -> dict[str, Any]:
    final_model = Path(case_ctx.get("output_model_path") or (run_dir / "final_fitted.cif"))
    final_model.parent.mkdir(parents=True, exist_ok=True)
    source_model_path = Path(case_ctx.get("source_model_path") or case_ctx["randomized_model_path"])
    map_utils.write_transformed_structure(source_model_path, final_model, best["R"], best["t"])
    score = float(best.get("raw_field_score", 0.0))
    gt_metrics: dict[str, Any] = {}
    if bool(case_ctx.get("compute_gt_metrics", True)) and Path(case.gt_model_path).exists():
        try:
            gt_metrics = pose_metrics_vs_gt_symmetry_aware(case.gt_model_path, final_model)
        except Exception as exc:
            gt_metrics = {"error": repr(exc)}
    return {
        "final_model_path": final_model,
        "score": score,
        "score_metric": SCORE_METRIC,
        "optimization_score": score,
        "mean_z": best.get("raw_map_meanz"),
        "final_map_meanz": best.get("final_map_meanz"),
        "raw_field_score": score,
        "gt_metrics": gt_metrics,
        "raw_field_metrics": {
            "raw_key1_mean": best.get("raw_key1_mean"),
            "raw_ss_mean": best.get("raw_ss_mean"),
            "raw_map_meanz": best.get("raw_map_meanz"),
        },
        "map_cc_score": best.get("map_cc_score"),
        "search_meta": {
            "selected_method_id": str(method_cfg.get("id", "")),
            "runner": "raw_field",
            "seed_label": best.get("seed_label"),
            "target_center_mode": best.get("target_center_mode"),
            "target_center_xyz": best.get("target_center_xyz"),
            "rotation_idx": best.get("rotation_idx"),
            "jitter_idx": best.get("jitter_idx"),
            "rank_score": score,
            "rank_score_metric": SCORE_METRIC,
            "final_mode": best.get("final_mode"),
            "refined_candidates": [
                {
                    "rank": idx + 1,
                    "R": row["R"],
                    "t": row["t"],
                    "seed_label": row.get("seed_label"),
                    "field_name": row.get("field_name"),
                    "field_percentile": row.get("field_percentile"),
                    "raw_field_score": row.get("raw_field_score"),
                    "raw_key1_mean": row.get("raw_key1_mean"),
                    "raw_ss_mean": row.get("raw_ss_mean"),
                    "raw_map_meanz": row.get("raw_map_meanz"),
                    "powell_meta": row.get("powell_meta"),
                    "final_map_meanz": row.get("final_map_meanz"),
                    "map_cc_score": row.get("map_cc_score"),
                    "final_mode": row.get("final_mode"),
                }
                for idx, row in enumerate(refined_preview[:8])
            ],
        },
    }


def run_raw_field(case: CaseSpec, case_ctx: dict[str, Any], run_dir: Path, method_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the full three-stage rawfield_ss_dominant alignment for one case.

    Parameters
    ----------
    case
        Identifiers and resolved paths describing the input case.
    case_ctx
        Per-call context dict with ``source_model_path`` /
        ``randomized_model_path``, ``output_model_path``, and the
        ``compute_gt_metrics`` flag.
    run_dir
        Scratch directory for this candidate. Removed by the pipeline once the
        candidate finishes.
    method_cfg
        Override knobs merged into :data:`RAWFIELD_SS_DOMINANT_CONFIG`; ``None``
        uses the defaults.

    Returns the rawfield payload consumed by :func:`aligner.align_model_to_map`:
    final score, score metric, fitted output path, search trace, and optional
    GT pose metrics.
    """
    method_cfg = dict(RAWFIELD_SS_DOMINANT_CONFIG if method_cfg is None else method_cfg)
    if str(method_cfg.get("id")) != "rawfield_ss_dominant":
        raise ValueError("struct_fetch_new_method only supports rawfield_ss_dominant")

    source_model_path = Path(case_ctx.get("source_model_path") or case_ctx["randomized_model_path"])
    search = model_search_data(source_model_path, atom_mode=str(method_cfg.get("atom_mode", "backbone")))
    field_maps = _raw_field_mapinfos(case, method_cfg)
    real_mapinfo = map_utils.get_mapinfo(case.map_path, denoise_sigma=0.0)
    seeds = _seed_candidates_from_raw_fields(search, field_maps, real_mapinfo, method_cfg)
    if not seeds:
        raise RuntimeError(f"No raw-field seeds generated for {case.key}")
    if bool(method_cfg.get("coarse_only", False)):
        best = dict(seeds[0])
        pts_world = (np.asarray(best["R"], dtype=np.float64) @ np.asarray(search["score_coords"], dtype=np.float64).T).T + np.asarray(best["t"], dtype=np.float64)
        best.update(
            {
                "final_mode": "coarse_only",
                "map_cc_score": _map_cc_score(
                    real_mapinfo,
                    pts_world,
                    radius_A=float(method_cfg.get("fallback_probe_cc_radius_A", 6.0)),
                    max_dim=128,
                ),
                "final_map_meanz": float(best.get("raw_map_meanz", float("nan"))),
                "powell_meta": {"skipped": True, "reason": "coarse_only"},
            }
        )
        return _payload(case, case_ctx, run_dir, method_cfg, best, [best])
    primary_seed_bank = _build_primary_seed_bank(seeds, method_cfg, pose_payload=_seed_pose_moment_payload(search["ca_coords"]))
    refine_model_center = np.mean(np.asarray(search["score_coords"], dtype=np.float64), axis=0)

    raw_refined: list[dict[str, Any]] = []
    trans_bound = float(method_cfg.get("refine_trans_bound_factor", 0.35)) * float(max(real_mapinfo.longest_extent, 1e-6))
    for row in primary_seed_bank:
        rotation_work, translation_work = _deterministic_raw_point_icp(search, field_maps, np.asarray(row["R"], dtype=np.float64), np.asarray(row["t"], dtype=np.float64), method_cfg)
        rotation_axis, translation_axis = _axis_refine_pose(search, field_maps, real_mapinfo, rotation_work, translation_work, method_cfg)

        def _scorer(rotation: np.ndarray, translation: np.ndarray) -> float:
            return _raw_field_metrics(
                field_maps,
                real_mapinfo,
                search["ca_coords"],
                search["score_coords"],
                rotation,
                translation,
                ca_coef=float(method_cfg.get("refine_ca_coef", 1.0)),
                ss_coef=float(method_cfg.get("refine_ss_coef", 1.0)),
                map_coef=float(method_cfg.get("refine_map_coef", 0.0)),
            )["raw_field_score"]

        rotation_ref, translation_ref, powell_meta = map_utils.bounded_powell_refine(
            rotation_axis,
            translation_axis,
            _scorer,
            model_center=refine_model_center,
            rot_bound_deg=float(method_cfg.get("refine_rot_bound_deg", 45.0)),
            trans_bound_ang=trans_bound,
            maxiter=int(method_cfg.get("raw_refine_maxiter", 96)),
        )
        metrics = _raw_field_metrics(
            field_maps,
            real_mapinfo,
            search["ca_coords"],
            search["score_coords"],
            rotation_ref,
            translation_ref,
            ca_coef=float(method_cfg.get("refine_ca_coef", 1.0)),
            ss_coef=float(method_cfg.get("refine_ss_coef", 1.0)),
            map_coef=float(method_cfg.get("refine_map_coef", 0.0)),
        )
        raw_refined.append({**row, **metrics, "R": rotation_ref, "t": translation_ref, "R_icp": rotation_work, "t_icp": translation_work, "R_axis": rotation_axis, "t_axis": translation_axis, "powell_meta": powell_meta, "seed_rank": int(row.get("seed_rank", 10**9))})

    refined: list[dict[str, Any]] = []
    final_refine_cap = max(1, int(method_cfg.get("top_k_final_map_refine", len(raw_refined))))
    final_inputs = _sort_refined_rows(raw_refined)[:final_refine_cap]
    branch_specs = [] if bool(method_cfg.get("skip_final_map_refine", False)) else [
        ("meanz_short", float(method_cfg.get("short_final_map_trans_bound_factor", 0.20)) * float(max(real_mapinfo.longest_extent, 1e-6)), int(method_cfg.get("short_final_map_maxiter", 128)), float(method_cfg.get("short_final_map_rot_bound_deg", 35.0))),
        ("meanz_long", float(method_cfg.get("final_map_trans_bound_factor", 0.20)) * float(max(real_mapinfo.longest_extent, 1e-6)), int(method_cfg.get("final_map_maxiter", 128)), float(method_cfg.get("final_map_rot_bound_deg", 35.0))),
    ]
    cc_eval_radius = float(method_cfg.get("fallback_probe_cc_radius_A", 6.0))
    for row in final_inputs:
        pts_world = (np.asarray(row["R"], dtype=np.float64) @ np.asarray(search["score_coords"], dtype=np.float64).T).T + np.asarray(row["t"], dtype=np.float64)
        refined.append(
            {
                **row,
                "final_mode": "rawfield_only",
                "map_cc_score": _map_cc_score(real_mapinfo, pts_world, radius_A=cc_eval_radius, max_dim=128),
                "final_map_meanz": float(row.get("raw_map_meanz", row.get("final_map_meanz", float("nan")))),
            }
        )
        seen_branch_keys: set[tuple[float, int, float]] = set()
        for final_mode, trans_bound_ang, maxiter, rot_bound_deg in branch_specs:
            branch_key = (round(float(trans_bound_ang), 6), int(maxiter), round(float(rot_bound_deg), 6))
            if branch_key in seen_branch_keys:
                continue
            seen_branch_keys.add(branch_key)
            rotation_final, translation_final, final_map_meanz = map_utils.joint_powell_refine(
                real_mapinfo,
                search["score_coords"],
                np.asarray(row["R"], dtype=np.float64),
                np.asarray(row["t"], dtype=np.float64),
                trans_bound_ang=float(trans_bound_ang),
                maxiter=int(maxiter),
                rot_bound_deg=float(rot_bound_deg),
            )
            final_metrics = _raw_field_metrics(
                field_maps,
                real_mapinfo,
                search["ca_coords"],
                search["score_coords"],
                rotation_final,
                translation_final,
                ca_coef=float(method_cfg.get("refine_ca_coef", 1.0)),
                ss_coef=float(method_cfg.get("refine_ss_coef", 1.0)),
                map_coef=float(method_cfg.get("refine_map_coef", 0.0)),
            )
            pts_world = (np.asarray(rotation_final, dtype=np.float64) @ np.asarray(search["score_coords"], dtype=np.float64).T).T + np.asarray(translation_final, dtype=np.float64)
            refined.append({**row, **final_metrics, "R": rotation_final, "t": translation_final, "final_mode": final_mode, "map_cc_score": _map_cc_score(real_mapinfo, pts_world, radius_A=cc_eval_radius, max_dim=128), "final_map_meanz": float(final_map_meanz)})

    refined = _sort_refined_rows(refined)
    if not refined:
        raise RuntimeError(f"No refined raw-field candidates for {case.key}")
    return _payload(case, case_ctx, run_dir, method_cfg, refined[0], refined)
