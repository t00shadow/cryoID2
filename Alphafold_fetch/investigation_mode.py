#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any, Optional

import mrcfile
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as SciRot

import fit_v5 as fv


STEP_DIR_NAMES = [
    "00_inputs",
    "00_randomized_start_pose",
    "01_map_variants",
    "02_com_translation",
    "03_fib_rotations_sample",
    "04_best_fib_rotation",
    "05_self_spin",
    "06_translation_sweep",
    "07_joint_powell_refine",
    "08_vof_scoring",
    "09_final",
]

# Randomized start pose translation span relative to map extents (xyz).
RANDOM_START_TRANSLATION_FRACTION = 0.25


def sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "item"


def _json_default(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def truthy(v: Any) -> bool:
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    return str(v).strip().lower() in {"1", "true", "t", "yes", "y"}


def _safe_float(v: Any) -> Optional[float]:
    try:
        x = float(v)
    except Exception:
        return None
    return x if np.isfinite(x) else None


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _resolve_existing_path(path_str: str, bases: list[Path]) -> Path:
    p = _resolve_path_maybe_relative(path_str, bases)
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")
    return p.resolve()


def _structure_suffix_for(model_path: Path) -> str:
    s = model_path.suffix.lower()
    return s if s in {".cif", ".mmcif"} else ".pdb"


def _write_structure_snapshot(
    source_model_path: Path,
    out_path: Path,
    Rm: Optional[np.ndarray] = None,
    t: Optional[np.ndarray] = None,
) -> Path:
    st = fv.gemmi.read_structure(str(source_model_path))
    if Rm is not None and t is not None:
        fv.apply_transform_to_structure(st, Rm, t)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in (".cif", ".mmcif"):
        st.make_mmcif_document().write_file(str(out_path))
    else:
        st.write_pdb(str(out_path))
    return out_path


def _write_mrc_snapshot(
    arr: np.ndarray,
    out_path: Path,
    mapinfo: fv.MapInfo,
    source_shape_zyx: Optional[tuple[int, int, int]] = None,
) -> Path:
    data = np.asarray(arr, dtype=np.float32)
    src_shape = np.asarray(source_shape_zyx or mapinfo.fit_map.shape, dtype=np.float64)
    dst_shape = np.asarray(data.shape, dtype=np.float64)
    scale_ax = src_shape / dst_shape
    sx_new = float(mapinfo.sx * scale_ax[mapinfo.axX])
    sy_new = float(mapinfo.sy * scale_ax[mapinfo.axY])
    sz_new = float(mapinfo.sz * scale_ax[mapinfo.axZ])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(str(out_path), overwrite=True) as mrc:
        mrc.set_data(np.ascontiguousarray(data))
        try:
            mrc.voxel_size = (sx_new, sy_new, sz_new)
        except Exception:
            pass
        try:
            mrc.header.mapc = int(mapinfo.mapc)
            mrc.header.mapr = int(mapinfo.mapr)
            mrc.header.maps = int(mapinfo.maps)
            mrc.header.xorigin = float(mapinfo.eff_x0)
            mrc.header.yorigin = float(mapinfo.eff_y0)
            mrc.header.zorigin = float(mapinfo.eff_z0)
            mrc.update_header_stats()
        except Exception:
            pass
    return out_path


def _make_randomized_start_pose(
    original_model_path: Path,
    map_path: Path,
    out_dir: Path,
    seed: int,
) -> tuple[Path, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    raw_mapinfo = fv._get_cached_mapinfo(map_path, denoise_sigma=0.0)
    _, rep_coords = fv.read_model_coords(original_model_path)
    model_com = rep_coords.mean(axis=0)

    R_rand = SciRot.random(random_state=rng).as_matrix()
    map_center = np.array(
        [
            raw_mapinfo.eff_x0 + 0.5 * raw_mapinfo.extent_x,
            raw_mapinfo.eff_y0 + 0.5 * raw_mapinfo.extent_y,
            raw_mapinfo.eff_z0 + 0.5 * raw_mapinfo.extent_z,
        ],
        dtype=np.float64,
    )
    span = np.array(
        [raw_mapinfo.extent_x, raw_mapinfo.extent_y, raw_mapinfo.extent_z],
        dtype=np.float64,
    ) * float(RANDOM_START_TRANSLATION_FRACTION)
    offset = rng.uniform(low=-span, high=span)
    target_center = map_center + offset
    t_rand = target_center - (R_rand @ model_com)

    suffix = _structure_suffix_for(original_model_path)
    out_model = out_dir / f"randomized_start_pose{suffix}"
    _write_structure_snapshot(original_model_path, out_model, Rm=R_rand, t=t_rand)

    meta = {
        "seed": int(seed),
        "rotation_matrix": R_rand,
        "translation_vector": t_rand,
        "model_com_before": model_com,
        "map_center_xyz": map_center,
        "target_center_xyz": target_center,
        "random_offset_xyz": offset,
        "translation_fraction_of_map_extent": float(RANDOM_START_TRANSLATION_FRACTION),
    }
    write_json(out_dir / "randomized_start_pose.json", meta)
    return out_model, meta


def _resolve_default_map_from_struct_fetch(repo_root: Path, alphafold_dir: Path) -> Optional[Path]:
    sf = alphafold_dir / "struct_fetch_v4.py"
    if not sf.exists():
        return None
    text = sf.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"MAP_PATH\s*=\s*pathlib\.Path\((['\"])(.+?)\1\)", text)
    if not m:
        return None
    raw = m.group(2)
    p = Path(raw)
    for cand in [p, repo_root / p, alphafold_dir / p]:
        if cand.exists():
            return cand.resolve()
    return None


def _find_log_for_gene(alphafold_dir: Path, gene_id: str) -> Optional[Path]:
    target = f"{gene_id}__fit_scores.csv".lower()
    hits = [p for p in alphafold_dir.rglob("*__fit_scores.csv") if p.name.lower() == target]
    if not hits:
        return None
    hits.sort(key=lambda p: (p.stat().st_mtime, str(p)), reverse=True)
    return hits[0]


def _resolve_path_maybe_relative(path_str: str, bases: list[Path]) -> Path:
    p = Path(str(path_str))
    if p.is_absolute() and p.exists():
        return p
    for base in bases:
        cand = (base / p).resolve()
        if cand.exists():
            return cand
    return p


def _load_and_select_candidate(
    log_csv: Path,
    repo_root: Path,
    candidate_label: Optional[str],
    candidate_id: Optional[str],
) -> tuple[dict[str, Any], pd.DataFrame]:
    df = pd.read_csv(log_csv)
    if df.empty:
        raise RuntimeError(f"Empty log: {log_csv}")

    if candidate_label:
        rows = df[df["label"].astype(str) == str(candidate_label)]
        if rows.empty:
            raise RuntimeError(f"Candidate label not found: {candidate_label}")
        row = rows.iloc[0]
    elif candidate_id:
        rows = df[df["id"].astype(str) == str(candidate_id)]
        if rows.empty:
            raise RuntimeError(f"Candidate id not found: {candidate_id}")
        scores = pd.to_numeric(rows.get("score"), errors="coerce").fillna(float("-inf"))
        row = rows.iloc[int(scores.argmax())]
    else:
        kept_mask = df["kept"].map(truthy) if "kept" in df.columns else pd.Series([False] * len(df))
        rows = df[kept_mask] if kept_mask.any() else df
        scores = pd.to_numeric(rows.get("score"), errors="coerce").fillna(float("-inf"))
        row = rows.iloc[int(scores.argmax())]

    out = {str(k): row[k] for k in row.index}
    source_model = _resolve_path_maybe_relative(str(out["source_model"]), [log_csv.parent, repo_root, repo_root / "Alphafold_fetch"])
    if not source_model.exists():
        raise FileNotFoundError(f"Source model not found: {source_model}")
    out["log_csv"] = str(log_csv)
    out["source_model_resolved"] = str(source_model)
    return out, df


def _build_map_variants(map_path: Path, recommended_contour_level: Optional[float]) -> list[tuple[str, fv.MapInfo]]:
    if fv.GLOBAL_MAP_DENOISE_ENABLED and fv.GLOBAL_MAP_DENOISE_SIGMA > 0:
        den_tag = f"gaussian_sigma{fv.GLOBAL_MAP_DENOISE_SIGMA:g}"
        if fv.GLOBAL_MAP_DENOISE_CENTER_BLOB_ONLY:
            den_tag += "_center_blob"
        den_variant = (
            den_tag,
            fv._get_cached_mapinfo(
                map_path,
                denoise_sigma=fv.GLOBAL_MAP_DENOISE_SIGMA,
                denoise_center_blob_only=fv.GLOBAL_MAP_DENOISE_CENTER_BLOB_ONLY,
                denoise_center_blob_level=recommended_contour_level,
                denoise_center_blob_z_fallback=fv.GLOBAL_MAP_DENOISE_CENTER_BLOB_Z_FALLBACK,
                denoise_center_blob_cluster_disconnected=fv.GLOBAL_MAP_DENOISE_CENTER_BLOB_CLUSTER_DISCONNECTED,
                denoise_center_blob_link_factor=fv.GLOBAL_MAP_DENOISE_CENTER_BLOB_LINK_FACTOR,
                denoise_center_blob_center_factor=fv.GLOBAL_MAP_DENOISE_CENTER_BLOB_CENTER_FACTOR,
                denoise_center_blob_min_component_vox=fv.GLOBAL_MAP_DENOISE_CENTER_BLOB_MIN_COMPONENT_VOX,
            ),
        )
        if fv.GLOBAL_MAP_USE_DENOISED_ONLY:
            return [den_variant]
        return [("raw", fv._get_cached_mapinfo(map_path, denoise_sigma=0.0)), den_variant]
    return [("raw", fv._get_cached_mapinfo(map_path, denoise_sigma=0.0))]


def _global_rotation_align_debug(
    mapinfo: fv.MapInfo,
    model_pts_world: np.ndarray,
    max_dim: int = fv.GLOBAL_SEARCH_MAX_DIM,
    z_thresh: float = fv.GLOBAL_SEARCH_Z,
    radius_ang: float = fv.GLOBAL_SEARCH_RADIUS_A,
    n_rot: int = fv.GLOBAL_SEARCH_ROTATIONS,
    seed: int = fv.GLOBAL_SEARCH_SEED,
) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    m = mapinfo.fit_map
    m_ds = fv._downsample_to_maxdim(m, max_dim)
    mZ_ds = (m_ds - mapinfo.mu) / mapinfo.sd
    map_mask = (mZ_ds >= float(z_thresh))
    map_sum = float(np.count_nonzero(map_mask))
    if map_sum <= 0:
        dbg = {
            "m_ds": m_ds,
            "mZ_ds": mZ_ds,
            "map_mask": map_mask,
            "map_com": np.zeros(3),
            "model_com": np.zeros(3),
            "R_base": np.eye(3),
            "t_base": np.zeros(3),
            "rot_scores": [],
            "R_seed": np.eye(3),
            "t0_seed": np.zeros(3),
            "best_spin_R": np.eye(3),
            "best_spin_t0": np.zeros(3),
            "best_R": np.eye(3),
            "best_t": np.zeros(3),
            "best_iou": 0.0,
            "best_score": 0.0,
            "best_spin_iou": 0.0,
            "best_spin_score": 0.0,
            "spin_steps": int(max(1, int(fv.GLOBAL_SEARCH_SELF_SPIN_STEPS))),
            "n_rot_tested": 0,
        }
        return np.eye(3), np.zeros(3), 0.0, dbg

    scale_z = m.shape[0] / m_ds.shape[0]
    scale_y = m.shape[1] / m_ds.shape[1]
    scale_x = m.shape[2] / m_ds.shape[2]
    scale_mean = float(np.mean([scale_z, scale_y, scale_x]))
    vs_mean = float((mapinfo.sx + mapinfo.sy + mapinfo.sz) / 3.0)
    r_vox = float(radius_ang) / max(vs_mean, 1e-6)
    rad_ds = r_vox / max(scale_mean, 1e-6)

    map_pts, map_wts, map_com = fv.sample_map_points(mapinfo, stride=3, target_min=2500)
    model_com = model_pts_world.mean(axis=0)
    map_frame = None
    model_frame = None
    model_pc1_local = None
    rotset_target_axis = None
    R_base = np.eye(3, dtype=np.float64)
    try:
        map_frame, map_center = fv._build_signed_pca_frame(map_pts, weights=map_wts)
        map_com = map_center
        rotset_target_axis = map_frame[:, 0].copy()
    except Exception:
        map_frame = None
    try:
        model_frame, model_center = fv._build_signed_pca_frame(model_pts_world, weights=None)
        model_com = model_center
        model_pc1_local = model_frame[:, 0].copy()
    except Exception:
        model_frame = None

    if fv.GLOBAL_SEARCH_USE_PCA_SEED and map_frame is not None and model_frame is not None:
        R_base = map_frame @ model_frame.T
        if np.linalg.det(R_base) < 0.0:
            fix = model_frame.copy()
            fix[:, 1] *= -1.0
            fix[:, 2] = fv._safe_normalize(np.cross(fix[:, 0], fix[:, 1]), fix[:, 2])
            fix[:, 1] = fv._safe_normalize(np.cross(fix[:, 2], fix[:, 0]), fix[:, 1])
            R_base = map_frame @ fix.T

    # Match fit_v5: fixed 24 proper signed-permutation rotations, PCA-axis oriented.
    # `n_rot` and `seed` are kept for signature compatibility but ignored.
    cube_axes, cube_angles = fv._cube24_rotation_axis_angles()
    if fv.GLOBAL_SEARCH_USE_PCA_SEED and rotset_target_axis is not None:
        cube_axes = fv._align_rotation_axes_to_target(cube_axes, rotset_target_axis)
    rotations_delta = np.stack(
        [SciRot.from_rotvec(axis * angle).as_matrix() for axis, angle in zip(cube_axes, cube_angles)],
        axis=0,
    )

    if model_pts_world.shape[0] > 5000:
        idx_occ = np.linspace(0, model_pts_world.shape[0] - 1, 5000, dtype=int)
        model_pts_occ = model_pts_world[idx_occ]
    else:
        model_pts_occ = model_pts_world
    if model_pts_world.shape[0] > 2000:
        idx_score = np.linspace(0, model_pts_world.shape[0] - 1, 2000, dtype=int)
        model_pts_score = model_pts_world[idx_score]
    else:
        model_pts_score = model_pts_world

    if model_pc1_local is not None:
        spin_axis_local = model_pc1_local.copy()
    else:
        spin_axis_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    spin_axis_local /= (np.linalg.norm(spin_axis_local) + 1e-12)

    model_pts_occ_ctr = model_pts_occ - model_com
    model_pts_score_ctr = model_pts_score - model_com

    base_step_ang = max(vs_mean * scale_mean, 1.0) * float(fv.GLOBAL_SEARCH_TRANS_STEP_SCALE)
    offsets_fine = [i * base_step_ang for i in range(-int(fv.GLOBAL_SEARCH_TRANS_RANGE), int(fv.GLOBAL_SEARCH_TRANS_RANGE) + 1)]
    trans_offsets_fine = np.array(list(itertools.product(offsets_fine, offsets_fine, offsets_fine)), dtype=np.float64)

    if bool(fv.GLOBAL_SEARCH_COARSE_TRANSLATION):
        step_coarse = base_step_ang * float(fv.GLOBAL_SEARCH_COARSE_MULT)
        offsets_coarse = [i * step_coarse for i in range(-int(fv.GLOBAL_SEARCH_COARSE_RANGE), int(fv.GLOBAL_SEARCH_COARSE_RANGE) + 1)]
        trans_offsets_coarse = np.array(list(itertools.product(offsets_coarse, offsets_coarse, offsets_coarse)), dtype=np.float64)
    else:
        trans_offsets_coarse = None

    def _score_pose(pts_world: np.ndarray, pts_world_score: Optional[np.ndarray]):
        fz, fy, fx = mapinfo.world_to_array_frac_many(pts_world)
        pts_zyx_ds = np.column_stack([fz / scale_z, fy / scale_y, fx / scale_x])
        model_occ = fv._model_occupancy_from_points(m_ds.shape, pts_zyx_ds, rad_ds)
        model_sum = float(np.count_nonzero(model_occ))
        if model_sum <= 0:
            return float("-inf"), 0.0
        inter = float(np.logical_and(map_mask, model_occ > 0).sum())
        union = map_sum + model_sum - inter
        iou = (inter / union) if union > 0 else 0.0
        mean_z = fv._score_meanZ(mapinfo, pts_world_score)
        return iou * max(mean_z, 0.0), iou

    rot_scores: list[dict[str, Any]] = []
    for i_rot, R_delta in enumerate(rotations_delta):
        Rm = R_delta @ R_base
        t0 = map_com - (Rm @ model_com)
        pts_world = (Rm @ model_pts_occ_ctr.T).T + map_com
        pts_world_score = (Rm @ model_pts_score_ctr.T).T + map_com if model_pts_score is not None else None
        score, iou = _score_pose(pts_world, pts_world_score)
        rot_scores.append({"idx": int(i_rot), "score": float(score), "iou": float(iou), "R": Rm, "t": t0})

    if not rot_scores:
        dbg = {
            "m_ds": m_ds,
            "mZ_ds": mZ_ds,
            "map_mask": map_mask,
            "map_com": map_com,
            "model_com": model_com,
            "R_base": R_base,
            "t_base": map_com - (R_base @ model_com),
            "rot_scores": [],
            "R_seed": np.eye(3),
            "t0_seed": np.zeros(3),
            "best_spin_R": np.eye(3),
            "best_spin_t0": np.zeros(3),
            "best_R": np.eye(3),
            "best_t": np.zeros(3),
            "best_iou": 0.0,
            "best_score": 0.0,
            "best_spin_iou": 0.0,
            "best_spin_score": 0.0,
            "spin_steps": int(max(1, int(fv.GLOBAL_SEARCH_SELF_SPIN_STEPS))),
            "n_rot_tested": 0,
        }
        return np.eye(3), np.zeros(3), 0.0, dbg

    best_seed_entry = max(rot_scores, key=lambda x: (x["score"], x["iou"]))
    R_seed = np.asarray(best_seed_entry["R"], dtype=np.float64)
    t0_seed = np.asarray(best_seed_entry["t"], dtype=np.float64)

    spin_steps = max(1, int(fv.GLOBAL_SEARCH_SELF_SPIN_STEPS))
    best_spin_score = float("-inf")
    best_spin_iou = float("-inf")
    best_spin_R = R_seed
    best_spin_t0 = t0_seed
    best_spin_pts_rot = (R_seed @ model_pts_occ_ctr.T).T + map_com
    best_spin_pts_rot_score = (R_seed @ model_pts_score_ctr.T).T + map_com if model_pts_score is not None else None

    axis_world = R_seed @ spin_axis_local
    axis_world /= (np.linalg.norm(axis_world) + 1e-12)
    for sidx in range(spin_steps):
        ang = (2.0 * math.pi * float(sidx)) / float(spin_steps)
        R_spin = SciRot.from_rotvec(axis_world * ang).as_matrix()
        Rm = R_spin @ R_seed
        t0 = map_com - (Rm @ model_com)
        pts_rot = (Rm @ model_pts_occ_ctr.T).T + map_com
        pts_rot_score = (Rm @ model_pts_score_ctr.T).T + map_com if model_pts_score is not None else None
        score, iou = _score_pose(pts_rot, pts_rot_score)
        if (score > best_spin_score) or (score == best_spin_score and iou > best_spin_iou):
            best_spin_score = score
            best_spin_iou = iou
            best_spin_R = Rm
            best_spin_t0 = t0
            best_spin_pts_rot = pts_rot
            best_spin_pts_rot_score = pts_rot_score

    best_iou = float("-inf")
    best_score = float("-inf")
    best_R = best_spin_R
    best_t = best_spin_t0
    delta_seed = np.zeros(3, dtype=np.float64)
    if trans_offsets_coarse is not None:
        local_best = (float("-inf"), 0.0, delta_seed)
        for delta in trans_offsets_coarse:
            pts_world = best_spin_pts_rot + delta
            pts_world_score = best_spin_pts_rot_score + delta if best_spin_pts_rot_score is not None else None
            score, iou = _score_pose(pts_world, pts_world_score)
            if score > local_best[0]:
                local_best = (score, iou, delta)
        delta_seed = local_best[2]
    for delta in trans_offsets_fine:
        delta_total = delta_seed + delta
        pts_world = best_spin_pts_rot + delta_total
        pts_world_score = best_spin_pts_rot_score + delta_total if best_spin_pts_rot_score is not None else None
        score, iou = _score_pose(pts_world, pts_world_score)
        if score > best_score:
            best_score = score
            best_iou = iou
            best_R = best_spin_R
            best_t = best_spin_t0 + delta_total

    dbg = {
        "m_ds": m_ds,
        "mZ_ds": mZ_ds,
        "map_mask": map_mask,
        "map_com": np.asarray(map_com, dtype=np.float64),
        "model_com": np.asarray(model_com, dtype=np.float64),
        "R_base": np.asarray(R_base, dtype=np.float64),
        "t_base": np.asarray(map_com - (R_base @ model_com), dtype=np.float64),
        "rot_scores": rot_scores,
        "R_seed": np.asarray(R_seed, dtype=np.float64),
        "t0_seed": np.asarray(t0_seed, dtype=np.float64),
        "best_spin_R": np.asarray(best_spin_R, dtype=np.float64),
        "best_spin_t0": np.asarray(best_spin_t0, dtype=np.float64),
        "best_R": np.asarray(best_R, dtype=np.float64),
        "best_t": np.asarray(best_t, dtype=np.float64),
        "best_iou": float(best_iou),
        "best_score": float(best_score),
        "best_spin_iou": float(best_spin_iou),
        "best_spin_score": float(best_spin_score),
        "spin_steps": int(spin_steps),
        "n_rot_tested": int(len(rot_scores)),
    }
    return best_R, best_t, float(best_iou), dbg


def _write_step_pose(step_dir: Path, source_model_path: Path, base_name: str, suffix: str, Rm: np.ndarray, t: np.ndarray) -> Path:
    return _write_structure_snapshot(source_model_path, step_dir / f"{sanitize_name(base_name)}{suffix}", Rm=Rm, t=t)


def _variant_prefix(variant_name: str) -> str:
    return sanitize_name(variant_name)


def _candidate_rng_seed(name_key: str, extra_key: str, override: Optional[int]) -> int:
    if override is not None:
        return int(override)
    h = hashlib.sha256(f"{name_key}|{extra_key}".encode("utf-8", errors="ignore")).hexdigest()[:8]
    return int(h, 16)


def _run_variant_investigation(
    gene_id: str,
    variant_name: str,
    mapinfo: fv.MapInfo,
    source_model_path: Path,
    score_coords_full: np.ndarray,
    align_pts: np.ndarray,
    step_dirs: dict[str, Path],
    recommended_contour_level: Optional[float],
    sample_seed: int,
    include_variant_prefix: bool,
) -> dict[str, Any]:
    vp = _variant_prefix(variant_name)
    pref = f"{vp}_" if include_variant_prefix else ""
    suffix = _structure_suffix_for(source_model_path)

    # 01_map_variants
    _write_mrc_snapshot(mapinfo.fit_map, step_dirs["01_map_variants"] / f"{pref}working_map.mrc", mapinfo)
    write_json(
        step_dirs["01_map_variants"] / f"{pref}map_variant_meta.json",
        {
            "gene_id": gene_id,
            "variant_name": variant_name,
            "shape_zyx": list(mapinfo.fit_map.shape),
            "voxel_size_xyz": [mapinfo.sx, mapinfo.sy, mapinfo.sz],
            "origin_xyz": [mapinfo.eff_x0, mapinfo.eff_y0, mapinfo.eff_z0],
            "mu": mapinfo.mu,
            "sd": mapinfo.sd,
        },
    )

    # 02-06 global alignment stages
    R0, t0, coarse_iou, dbg = _global_rotation_align_debug(
        mapinfo,
        align_pts,
        max_dim=fv.GLOBAL_SEARCH_MAX_DIM,
        z_thresh=fv.GLOBAL_SEARCH_Z,
        radius_ang=fv.GLOBAL_SEARCH_RADIUS_A,
        n_rot=fv.GLOBAL_SEARCH_ROTATIONS,
        seed=fv.GLOBAL_SEARCH_SEED,
    )

    _write_step_pose(step_dirs["02_com_translation"], source_model_path, f"{pref}com_seed_pose", suffix, np.asarray(dbg["R_base"]), np.asarray(dbg["t_base"]))
    _write_mrc_snapshot(np.asarray(dbg["m_ds"]), step_dirs["02_com_translation"] / f"{pref}coarse_map_downsampled.mrc", mapinfo, tuple(mapinfo.fit_map.shape))
    write_json(
        step_dirs["02_com_translation"] / f"{pref}com_seed_meta.json",
        {"variant_name": variant_name, "coarse_iou_best": coarse_iou, "map_com": dbg["map_com"], "model_com": dbg["model_com"], "n_rot_tested": dbg["n_rot_tested"]},
    )

    rot_entries = list(dbg["rot_scores"])
    rng = np.random.default_rng(sample_seed)
    k = min(5, len(rot_entries))
    chosen = np.arange(len(rot_entries)) if k == len(rot_entries) else (np.sort(rng.choice(np.arange(len(rot_entries)), size=k, replace=False)) if k > 0 else np.array([], dtype=int))
    samples_meta = []
    for out_i, idx_sel in enumerate(chosen.tolist()):
        ent = rot_entries[idx_sel]
        s_val = _safe_float(ent["score"])
        s_tag = "nan" if s_val is None else f"{s_val:.4f}"
        _write_step_pose(
            step_dirs["03_fib_rotations_sample"],
            source_model_path,
            f"{pref}pca_sym_rot_sample_{out_i:02d}_idx{int(ent['idx']):03d}_score{s_tag}",
            suffix,
            np.asarray(ent["R"]),
            np.asarray(ent["t"]),
        )
        samples_meta.append({"sample_order": out_i, "rotation_index": int(ent["idx"]), "score": ent["score"], "iou": ent["iou"]})
    if k > 0:
        _write_mrc_snapshot(np.asarray(dbg["m_ds"]), step_dirs["03_fib_rotations_sample"] / f"{pref}coarse_map_downsampled.mrc", mapinfo, tuple(mapinfo.fit_map.shape))
    write_json(
        step_dirs["03_fib_rotations_sample"] / f"{pref}pca_symmetry_rotation_sample_meta.json",
        {
            "variant_name": variant_name,
            "rotation_sampler": "pca_aligned_cube24",
            "sample_seed": int(sample_seed),
            "total_rotations_tested": int(len(rot_entries)),
            "sample_count": int(k),
            "samples": samples_meta,
        },
    )

    _write_step_pose(step_dirs["04_best_fib_rotation"], source_model_path, f"{pref}best_pca_symmetry_rotation_pose", suffix, np.asarray(dbg["R_seed"]), np.asarray(dbg["t0_seed"]))
    write_json(
        step_dirs["04_best_fib_rotation"] / f"{pref}best_pca_symmetry_rotation_meta.json",
        {
            "variant_name": variant_name,
            "rotation_sampler": "pca_aligned_cube24",
            "best_rotation_index": int(max(rot_entries, key=lambda x: (x['score'], x['iou']))['idx']) if rot_entries else None,
        },
    )

    _write_step_pose(step_dirs["05_self_spin"], source_model_path, f"{pref}self_spin_pose", suffix, np.asarray(dbg["best_spin_R"]), np.asarray(dbg["best_spin_t0"]))
    write_json(
        step_dirs["05_self_spin"] / f"{pref}self_spin_meta.json",
        {"variant_name": variant_name, "spin_steps": int(dbg["spin_steps"]), "best_spin_score": dbg["best_spin_score"], "best_spin_iou": dbg["best_spin_iou"]},
    )

    _write_step_pose(step_dirs["06_translation_sweep"], source_model_path, f"{pref}translation_sweep_best_pose", suffix, np.asarray(dbg["best_R"]), np.asarray(dbg["best_t"]))
    write_json(
        step_dirs["06_translation_sweep"] / f"{pref}translation_sweep_meta.json",
        {"variant_name": variant_name, "best_global_score": dbg["best_score"], "best_global_iou": dbg["best_iou"]},
    )

    # 07 joint Powell refine
    longest = max(mapinfo.longest_extent, 1e-6)
    trans_bound_ang = fv.GLOBAL_JOINT_TRANS_FACTOR * longest
    R_ref, t_ref, refine_meanz = fv.joint_powell_refine(mapinfo, score_coords_full, np.asarray(R0), np.asarray(t0), trans_bound_ang=trans_bound_ang)
    _write_step_pose(step_dirs["07_joint_powell_refine"], source_model_path, f"{pref}joint_powell_refined_pose", suffix, R_ref, t_ref)
    write_json(
        step_dirs["07_joint_powell_refine"] / f"{pref}joint_powell_meta.json",
        {"variant_name": variant_name, "refine_mean_z_objective": float(refine_meanz), "trans_bound_ang": float(trans_bound_ang)},
    )

    # 08 VOF scoring
    pts_world = (R_ref @ score_coords_full.T).T + t_ref
    vof, vof_meta = fv.compute_vof_optimized(mapinfo, pts_world, max_dim=fv.VOF_MAX_DIM, recommended_contour_level=recommended_contour_level)
    mean_z = fv._score_meanZ(mapinfo, pts_world)
    score = fv._score_from_vof_meanz(float(vof), mean_z)
    _write_step_pose(step_dirs["08_vof_scoring"], source_model_path, f"{pref}vof_scoring_pose", suffix, R_ref, t_ref)

    m_ds_vof, mZ_ds_vof, pts_zyx_ds, scale_mean_vof = fv._prepare_vof_inputs(mapinfo, pts_world, fv.VOF_MAX_DIM)
    _write_mrc_snapshot(m_ds_vof, step_dirs["08_vof_scoring"] / f"{pref}vof_map_downsampled.mrc", mapinfo, tuple(mapinfo.fit_map.shape))
    if vof_meta and vof_meta.get("threshold_z") is not None and vof_meta.get("radius_ang") is not None:
        _write_mrc_snapshot((mZ_ds_vof >= float(vof_meta["threshold_z"])).astype(np.float32), step_dirs["08_vof_scoring"] / f"{pref}vof_threshold_mask.mrc", mapinfo, tuple(mapinfo.fit_map.shape))
        vs_mean = float((mapinfo.sx + mapinfo.sy + mapinfo.sz) / 3.0)
        rad_ds = (float(vof_meta["radius_ang"]) / max(vs_mean, 1e-6)) / max(scale_mean_vof, 1e-6)
        occ = fv._model_occupancy_from_points(m_ds_vof.shape, pts_zyx_ds, rad_ds).astype(np.float32)
        _write_mrc_snapshot(occ, step_dirs["08_vof_scoring"] / f"{pref}vof_model_occupancy_mask.mrc", mapinfo, tuple(mapinfo.fit_map.shape))
    write_json(
        step_dirs["08_vof_scoring"] / f"{pref}vof_scoring_meta.json",
        {"variant_name": variant_name, "vof": float(vof), "mean_z": float(mean_z), "score": float(score), "vof_meta": vof_meta},
    )

    return {
        "variant_name": variant_name,
        "R": R_ref,
        "t": t_ref,
        "vof": float(vof),
        "mean_z": float(mean_z),
        "score": float(score),
        "vof_meta": vof_meta,
        "coarse_iou": float(coarse_iou),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dump fit_v5 intermediate outputs for one ground-truth model + map.")
    p.add_argument("model_path", help="Ground-truth structure (.pdb/.cif/.mmcif).")
    p.add_argument("map_path", help="Ground-truth density map (.mrc/.map).")
    p.add_argument("--run-name", help="Optional output folder name (default: model filename stem).")
    p.add_argument("--recommended-contour-level", type=float, default=None, help="Optional contour level passed to fit_v5 VOF scoring.")
    p.add_argument("--sample-seed", type=int, default=None, help="Optional RNG seed for the 5 coarse PCA-symmetry rotation samples.")
    p.add_argument("--random-start-seed", type=int, default=None, help="Optional RNG seed for the randomized starting pose.")
    p.add_argument("--output-root", help="Optional investigations root. Default: Alphafold_fetch/investigations")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    script_path = Path(__file__).resolve()
    alphafold_dir = script_path.parent
    repo_root = alphafold_dir.parent
    source_model_original = _resolve_existing_path(str(args.model_path), [repo_root, alphafold_dir, Path.cwd()])
    map_path = _resolve_existing_path(str(args.map_path), [repo_root, alphafold_dir, Path.cwd()])
    run_name = sanitize_name(str(args.run_name).strip()) if args.run_name else sanitize_name(source_model_original.stem)
    if not run_name:
        raise SystemExit("Could not derive run name from model file. Use --run-name.")

    out_root = Path(args.output_root).resolve() if args.output_root else (alphafold_dir / "investigations").resolve()
    gene_out_dir = out_root / run_name
    gene_out_dir.mkdir(parents=True, exist_ok=True)
    step_dirs = {name: (gene_out_dir / name) for name in STEP_DIR_NAMES}
    for d in step_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Randomized start pose (new baseline input to the rest of the investigation)
    random_start_seed = (
        int(args.random_start_seed)
        if args.random_start_seed is not None
        else int(np.random.default_rng().integers(0, 2**32 - 1))
    )
    source_model_path, random_start_meta = _make_randomized_start_pose(
        original_model_path=source_model_original,
        map_path=map_path,
        out_dir=step_dirs["00_randomized_start_pose"],
        seed=random_start_seed,
    )

    print(f"[investigation] run_name={run_name}")
    print(f"[investigation] source_model_original={source_model_original}")
    print(f"[investigation] source_model_randomized={source_model_path}")
    print(f"[investigation] map_path={map_path}")
    print(f"[investigation] output_dir={gene_out_dir}")

    # 00_inputs
    suffix = _structure_suffix_for(source_model_original)
    _copy_if_exists(source_model_original, step_dirs["00_inputs"] / f"groundtruth_input{suffix}")
    _copy_if_exists(map_path, step_dirs["00_inputs"] / map_path.name)
    write_json(
        step_dirs["00_inputs"] / "selection.json",
        {
            "run_name": run_name,
            "input_model_path": str(source_model_original),
            "map_path": str(map_path),
            "randomized_start_pose_path": str(source_model_path),
            "fit_v5": {
                "GLOBAL_SEARCH_MAX_DIM": fv.GLOBAL_SEARCH_MAX_DIM,
                "GLOBAL_SEARCH_Z": fv.GLOBAL_SEARCH_Z,
                "GLOBAL_SEARCH_ROTATIONS": fv.GLOBAL_SEARCH_ROTATIONS,
                "GLOBAL_SEARCH_SELF_SPIN_STEPS": fv.GLOBAL_SEARCH_SELF_SPIN_STEPS,
                "VOF_MAX_DIM": fv.VOF_MAX_DIM,
                "VOF_SCORE_ATOMS": fv.VOF_SCORE_ATOMS,
                "GLOBAL_MAP_DENOISE_ENABLED": fv.GLOBAL_MAP_DENOISE_ENABLED,
            },
        },
    )

    # Load model / scoring atom set same as fit_v5.fit_model_and_score
    st, model_coords_full = fv.read_model_coords(source_model_path)
    if fv.VOF_SCORE_ATOMS == "backbone":
        score_coords_full = fv._collect_backbone_coords(st)
    elif fv.VOF_SCORE_ATOMS == "heavy":
        score_coords_full = fv._collect_non_h_coords(st)
    else:
        score_coords_full = model_coords_full
    if score_coords_full.size == 0:
        score_coords_full = model_coords_full
    align_pts = score_coords_full if score_coords_full.size else model_coords_full

    map_variants = _build_map_variants(map_path, args.recommended_contour_level)
    include_variant_prefix = len(map_variants) > 1
    seed_base = _candidate_rng_seed(run_name, f"{source_model_original.name}|{map_path.name}", args.sample_seed)
    variant_results: list[dict[str, Any]] = []
    for i, (variant_name, mapinfo) in enumerate(map_variants, start=1):
        print(f"[investigation] variant {i}/{len(map_variants)} -> {variant_name}")
        variant_results.append(
            _run_variant_investigation(
                gene_id=run_name,
                variant_name=variant_name,
                mapinfo=mapinfo,
                source_model_path=source_model_path,
                score_coords_full=score_coords_full,
                align_pts=align_pts,
                step_dirs=step_dirs,
                recommended_contour_level=args.recommended_contour_level,
                sample_seed=int(seed_base + i * 7919),
                include_variant_prefix=include_variant_prefix,
            )
        )

    if not variant_results:
        raise SystemExit("No variant results produced.")
    best = max(variant_results, key=lambda x: x["score"])
    best_mapinfo = next(mi for name, mi in map_variants if name == best["variant_name"])

    final_struct = step_dirs["09_final"] / f"final_{sanitize_name(best['variant_name'])}{suffix}"
    _write_structure_snapshot(source_model_path, final_struct, Rm=np.asarray(best["R"]), t=np.asarray(best["t"]))
    _write_mrc_snapshot(best_mapinfo.fit_map, step_dirs["09_final"] / f"final_{sanitize_name(best['variant_name'])}_working_map.mrc", best_mapinfo)

    summary = {
        "run_name": run_name,
        "input_model_path": str(source_model_original),
        "randomized_start_model_path": str(source_model_path),
        "randomized_start_pose": random_start_meta,
        "map_path": str(map_path),
        "output_dir": str(gene_out_dir),
        "recommended_contour_level": args.recommended_contour_level,
        "rotation_sample_seed_base": int(seed_base),
        "random_start_seed": int(random_start_seed),
        "variants": [
            {
                "variant_name": r["variant_name"],
                "score": r["score"],
                "vof": r["vof"],
                "mean_z": r["mean_z"],
                "coarse_iou": r["coarse_iou"],
                "vof_meta": r["vof_meta"],
            }
            for r in variant_results
        ],
        "best_variant": {
            "variant_name": best["variant_name"],
            "score": best["score"],
            "vof": best["vof"],
            "mean_z": best["mean_z"],
            "vof_meta": best["vof_meta"],
            "final_structure": str(final_struct),
        },
        "steps": {k: str(v) for k, v in step_dirs.items()},
    }
    write_json(gene_out_dir / "summary.json", summary)
    write_json(step_dirs["09_final"] / "summary.json", summary)

    print(f"[investigation] done -> {gene_out_dir}")
    print(f"[investigation] final structure -> {final_struct}")


if __name__ == "__main__":
    main()
