#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fit_v5 as fv5  # noqa: E402


MAP_PATH = Path("Alphafold_fetch/qscore_groundtruth_run_2_10/001_RYR1/emd_38042.map")
MODEL_PATH = Path("Alphafold_fetch/qscore_groundtruth_run_2_10/001_RYR1/8X48.cif")
OUT_DIR = Path("Alphafold_fetch/ryr1_new_method_tests")


@dataclass
class FitResult:
    method: str
    run_type: str
    runtime_sec: float
    score_custom: float
    vof: float
    mean_z: float
    fitted_points: np.ndarray
    details: Dict[str, float]


def _rmsd_direct(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    d2 = np.sum((a - b) ** 2, axis=1)
    return float(np.sqrt(np.mean(d2)))


def _score_pose_fast(mapinfo: fv5.MapInfo, pts_world: np.ndarray) -> Tuple[float, float, float, dict]:
    # Custom fast objective for method bakeoff:
    # score = meanZ only (VOF omitted to keep these experiments fast).
    mean_z = fv5._score_meanZ(mapinfo, pts_world)
    return float(mean_z), float("nan"), float(mean_z), {}


def _build_rotations(points: np.ndarray, n_rot: int = 32, use_pca_seed: bool = True) -> np.ndarray:
    n_eff = max(1, int(n_rot))
    fib_axes, fib_angles = fv5._fib_axes_angles(n_eff)
    if use_pca_seed and n_eff > 1:
        try:
            U, _ = fv5.pca(points)
            pca_axis = np.asarray(U[:, 0], dtype=np.float64)
            fib_axes = fv5._align_fib_axes_to_target(fib_axes, pca_axis)
        except Exception:
            pass
    rotations = np.stack(
        [R.from_rotvec(axis * angle).as_matrix() for axis, angle in zip(fib_axes, fib_angles)],
        axis=0,
    )
    return rotations


def _select_topk_seeds(mapinfo: fv5.MapInfo, points: np.ndarray, n_rot: int, top_k: int) -> List[Tuple[float, np.ndarray, np.ndarray]]:
    model_com = points.mean(axis=0)
    _, _, map_com = fv5.sample_map_points(mapinfo, stride=3, target_min=2500)
    rots = _build_rotations(points, n_rot=n_rot, use_pca_seed=True)

    n = points.shape[0]
    if n > 2500:
        idx = np.linspace(0, n - 1, 2500, dtype=int)
        pts_score = points[idx]
    else:
        pts_score = points

    seeds = []
    for Rm in rots:
        t0 = map_com - (Rm @ model_com)
        pts = (Rm @ pts_score.T).T + t0
        mz = fv5._score_meanZ(mapinfo, pts)
        seeds.append((float(mz), Rm, t0))
    seeds.sort(key=lambda x: x[0], reverse=True)
    return seeds[: max(1, int(top_k))]


def _run_topk_powell(
    mapinfo: fv5.MapInfo,
    points: np.ndarray,
    n_rot: int = 32,
    top_k: int = 6,
    powell_iter: int = 12,
    trans_factor: float = 0.30,
) -> Tuple[np.ndarray, Dict[str, float]]:
    seeds = _select_topk_seeds(mapinfo, points, n_rot=n_rot, top_k=top_k)
    trans_bound = float(trans_factor) * max(mapinfo.longest_extent, 1e-6)

    best = None
    for coarse_mz, R0, t0 in seeds:
        Rb, tb, _ = fv5.joint_powell_refine(
            mapinfo,
            points,
            R0,
            t0,
            trans_bound_ang=trans_bound,
            maxiter=int(powell_iter),
            rot_bound_deg=20.0,
        )
        pts = (Rb @ points.T).T + tb
        score, vof, mean_z, _ = _score_pose_fast(mapinfo, pts)
        rec = {
            "score": score,
            "vof": vof,
            "mean_z": mean_z,
            "coarse_mz": float(coarse_mz),
        }
        if best is None or rec["score"] > best["meta"]["score"]:
            best = {"pts": pts, "meta": rec}
    if best is None:
        raise RuntimeError("No seed produced a result in topk+powell")
    return best["pts"], best["meta"]


def _run_multires_topk(
    map_path: Path,
    points: np.ndarray,
    n_rot: int = 32,
    top_k_stage1: int = 8,
    keep_stage1: int = 3,
) -> Tuple[np.ndarray, Dict[str, float]]:
    # Coarse smoother map variant
    mapinfo_smooth = fv5._get_cached_mapinfo(
        map_path,
        denoise_sigma=2.0,
        denoise_center_blob_only=True,
        denoise_center_blob_level=None,
        denoise_center_blob_z_fallback=0.50,
        denoise_center_blob_cluster_disconnected=True,
        denoise_center_blob_link_factor=0.08,
        denoise_center_blob_center_factor=0.40,
        denoise_center_blob_min_component_vox=64,
    )
    # Main map variant (same settings as current pipeline default)
    mapinfo_main = fv5._get_cached_mapinfo(
        map_path,
        denoise_sigma=fv5.GLOBAL_MAP_DENOISE_SIGMA if fv5.GLOBAL_MAP_DENOISE_ENABLED else 0.0,
        denoise_center_blob_only=fv5.GLOBAL_MAP_DENOISE_CENTER_BLOB_ONLY,
        denoise_center_blob_level=None,
        denoise_center_blob_z_fallback=fv5.GLOBAL_MAP_DENOISE_CENTER_BLOB_Z_FALLBACK,
        denoise_center_blob_cluster_disconnected=fv5.GLOBAL_MAP_DENOISE_CENTER_BLOB_CLUSTER_DISCONNECTED,
        denoise_center_blob_link_factor=fv5.GLOBAL_MAP_DENOISE_CENTER_BLOB_LINK_FACTOR,
        denoise_center_blob_center_factor=fv5.GLOBAL_MAP_DENOISE_CENTER_BLOB_CENTER_FACTOR,
        denoise_center_blob_min_component_vox=fv5.GLOBAL_MAP_DENOISE_CENTER_BLOB_MIN_COMPONENT_VOX,
    )

    # Stage 1 seeds on smoother map
    seeds = _select_topk_seeds(mapinfo_smooth, points, n_rot=n_rot, top_k=top_k_stage1)
    stage1 = []
    trans_bound1 = 0.45 * max(mapinfo_smooth.longest_extent, 1e-6)
    for coarse_mz, R0, t0 in seeds:
        R1, t1, _ = fv5.joint_powell_refine(
            mapinfo_smooth,
            points,
            R0,
            t0,
            trans_bound_ang=trans_bound1,
            maxiter=12,
            rot_bound_deg=25.0,
        )
        pts1 = (R1 @ points.T).T + t1
        mz1 = fv5._score_meanZ(mapinfo_smooth, pts1)
        stage1.append((float(mz1), R1, t1, float(coarse_mz)))
    stage1.sort(key=lambda x: x[0], reverse=True)
    stage1 = stage1[: max(1, int(keep_stage1))]

    # Stage 2 on main map
    best = None
    trans_bound2 = 0.30 * max(mapinfo_main.longest_extent, 1e-6)
    for mz1, R1, t1, coarse_mz in stage1:
        R2, t2, _ = fv5.joint_powell_refine(
            mapinfo_main,
            points,
            R1,
            t1,
            trans_bound_ang=trans_bound2,
            maxiter=14,
            rot_bound_deg=20.0,
        )
        pts2 = (R2 @ points.T).T + t2
        score, vof, mean_z, _ = _score_pose_fast(mapinfo_main, pts2)
        rec = {
            "score": score,
            "vof": vof,
            "mean_z": mean_z,
            "stage1_mz": float(mz1),
            "coarse_mz": float(coarse_mz),
        }
        if best is None or rec["score"] > best["meta"]["score"]:
            best = {"pts": pts2, "meta": rec}
    if best is None:
        raise RuntimeError("No seed produced a result in multires_topk")
    return best["pts"], best["meta"]


def _run_annealed_powell(
    mapinfo: fv5.MapInfo,
    points: np.ndarray,
    n_rot: int = 32,
    n_steps: int = 80,
) -> Tuple[np.ndarray, Dict[str, float]]:
    # Seed from best coarse rotation only.
    seeds = _select_topk_seeds(mapinfo, points, n_rot=n_rot, top_k=1)
    _, R_seed, t_seed = seeds[0]

    longest = max(mapinfo.longest_extent, 1e-6)
    rot_bound = math.radians(30.0)
    trans_bound = 0.35 * longest

    def _project(x: np.ndarray) -> np.ndarray:
        y = np.asarray(x, dtype=np.float64).copy()
        y[:3] = np.clip(y[:3], -rot_bound, rot_bound)
        y[3:] = np.clip(y[3:], -trans_bound, trans_bound)
        return y

    def _score_x(x: np.ndarray) -> float:
        x = _project(x)
        rv = x[:3]
        dt = x[3:]
        R_delta = R.from_rotvec(rv).as_matrix()
        R_try = R_delta @ R_seed
        t_try = t_seed + dt
        pts = (R_try @ points.T).T + t_try
        return float(fv5._score_meanZ(mapinfo, pts))

    x = np.zeros(6, dtype=np.float64)
    cur = _score_x(x)
    best_x = x.copy()
    best_s = cur
    rng = np.random.default_rng(20260211)

    rot_sigma0 = math.radians(8.0)
    trans_sigma0 = 0.12 * longest
    for i in range(max(1, int(n_steps))):
        frac = 1.0 - (i / max(1, n_steps - 1))
        temp = 0.05 + 0.95 * frac
        rot_sigma = rot_sigma0 * temp
        trans_sigma = trans_sigma0 * temp
        step = np.concatenate(
            [
                rng.normal(0.0, rot_sigma, size=3),
                rng.normal(0.0, trans_sigma, size=3),
            ]
        )
        xp = _project(x + step)
        sp = _score_x(xp)
        accept = False
        if sp >= cur:
            accept = True
        else:
            prob = math.exp((sp - cur) / max(1e-6, temp))
            if rng.random() < prob:
                accept = True
        if accept:
            x = xp
            cur = sp
            if cur > best_s:
                best_s = cur
                best_x = x.copy()

    # Final local polish from annealed pose.
    rv = best_x[:3]
    dt = best_x[3:]
    R0 = R.from_rotvec(rv).as_matrix() @ R_seed
    t0 = t_seed + dt
    Rb, tb, _ = fv5.joint_powell_refine(
        mapinfo,
        points,
        R0,
        t0,
        trans_bound_ang=0.30 * longest,
        maxiter=14,
        rot_bound_deg=20.0,
    )
    pts = (Rb @ points.T).T + tb
    score, vof, mean_z, _ = _score_pose_fast(mapinfo, pts)
    return pts, {
        "score": score,
        "vof": vof,
        "mean_z": mean_z,
        "anneal_best_mz": float(best_s),
    }


def _randomize_start(points: np.ndarray, mapinfo: fv5.MapInfo, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    com = points.mean(axis=0)
    axis = rng.normal(size=3)
    axis /= (np.linalg.norm(axis) + 1e-12)
    angle = rng.uniform(-math.pi, math.pi)
    R_rand = R.from_rotvec(axis * angle).as_matrix()
    t_mag = 0.35 * max(mapinfo.longest_extent, 1e-6)
    delta = rng.uniform(-t_mag, t_mag, size=3)
    pts = (R_rand @ (points - com).T).T + com + delta
    return pts


def _run_method_once(
    method_name: str,
    method_fn: Callable[[np.ndarray], Tuple[np.ndarray, Dict[str, float]]],
    points_start: np.ndarray,
    run_type: str,
    mapinfo_for_score: fv5.MapInfo,
) -> FitResult:
    t0 = time.time()
    pts_fit, meta = method_fn(points_start)
    dt = time.time() - t0
    score, vof, mean_z, _ = _score_pose_fast(mapinfo_for_score, pts_fit)
    details = dict(meta)
    details["score_recomputed"] = float(score)
    return FitResult(
        method=method_name,
        run_type=run_type,
        runtime_sec=float(dt),
        score_custom=float(score),
        vof=float(vof),
        mean_z=float(mean_z),
        fitted_points=np.asarray(pts_fit, dtype=np.float64),
        details=details,
    )


def main() -> int:
    if not MAP_PATH.exists():
        raise SystemExit(f"Map not found: {MAP_PATH}")
    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found: {MODEL_PATH}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Main map variant (align/scoring map)
    mapinfo_main = fv5._get_cached_mapinfo(
        MAP_PATH,
        denoise_sigma=fv5.GLOBAL_MAP_DENOISE_SIGMA if fv5.GLOBAL_MAP_DENOISE_ENABLED else 0.0,
        denoise_center_blob_only=fv5.GLOBAL_MAP_DENOISE_CENTER_BLOB_ONLY,
        denoise_center_blob_level=None,
        denoise_center_blob_z_fallback=fv5.GLOBAL_MAP_DENOISE_CENTER_BLOB_Z_FALLBACK,
        denoise_center_blob_cluster_disconnected=fv5.GLOBAL_MAP_DENOISE_CENTER_BLOB_CLUSTER_DISCONNECTED,
        denoise_center_blob_link_factor=fv5.GLOBAL_MAP_DENOISE_CENTER_BLOB_LINK_FACTOR,
        denoise_center_blob_center_factor=fv5.GLOBAL_MAP_DENOISE_CENTER_BLOB_CENTER_FACTOR,
        denoise_center_blob_min_component_vox=fv5.GLOBAL_MAP_DENOISE_CENTER_BLOB_MIN_COMPONENT_VOX,
    )

    # Use representative atoms for speed and consistent comparison.
    _, points_native = fv5.read_model_coords(MODEL_PATH)
    points_random = _randomize_start(points_native, mapinfo_main, seed=20260211)

    methods: Dict[str, Callable[[np.ndarray], Tuple[np.ndarray, Dict[str, float]]]] = {
        "topk_powell_meanz": lambda pts: _run_topk_powell(
            mapinfo=mapinfo_main,
            points=pts,
            n_rot=32,
            top_k=8,
            powell_iter=24,
            trans_factor=0.30,
        ),
        "multires_topk_powell": lambda pts: _run_multires_topk(
            map_path=MAP_PATH,
            points=pts,
            n_rot=32,
            top_k_stage1=12,
            keep_stage1=4,
        ),
        "annealed_powell": lambda pts: _run_annealed_powell(
            mapinfo=mapinfo_main,
            points=pts,
            n_rot=32,
            n_steps=180,
        ),
    }

    rows = []
    summary = []

    for method_name, fn in methods.items():
        print(f"\n[method] {method_name} ...")
        ref_res = _run_method_once(
            method_name=method_name,
            method_fn=fn,
            points_start=points_native,
            run_type="native_start",
            mapinfo_for_score=mapinfo_main,
        )
        rand_res = _run_method_once(
            method_name=method_name,
            method_fn=fn,
            points_start=points_random,
            run_type="randomized_start",
            mapinfo_for_score=mapinfo_main,
        )

        rmsd = _rmsd_direct(ref_res.fitted_points, rand_res.fitted_points)
        score_delta = float(rand_res.score_custom - ref_res.score_custom)
        abs_score_delta = abs(score_delta)
        success = bool((rmsd <= 2.0) and (abs_score_delta <= 0.05))

        summary.append(
            {
                "method": method_name,
                "native_time_sec": round(ref_res.runtime_sec, 3),
                "random_time_sec": round(rand_res.runtime_sec, 3),
                "native_score_custom": round(ref_res.score_custom, 6),
                "random_score_custom": round(rand_res.score_custom, 6),
                "score_delta_random_minus_native": round(score_delta, 6),
                "direct_rmsd_A": round(rmsd, 6),
                "success": success,
            }
        )

        for r in (ref_res, rand_res):
            rows.append(
                {
                    "method": r.method,
                    "run_type": r.run_type,
                    "runtime_sec": round(r.runtime_sec, 6),
                    "score_custom": round(r.score_custom, 8),
                    "vof": round(r.vof, 8),
                    "mean_z": round(r.mean_z, 8),
                    "details_json": json.dumps(r.details, ensure_ascii=True),
                }
            )

    detail_csv = OUT_DIR / "ryr1_method_runs.csv"
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)

    summary_csv = OUT_DIR / "ryr1_method_summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else [])
        w.writeheader()
        w.writerows(summary)

    summary_json = OUT_DIR / "ryr1_method_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n[done] summary:")
    for s in summary:
        print(json.dumps(s, ensure_ascii=True))
    print(f"[done] detail_csv={detail_csv}")
    print(f"[done] summary_csv={summary_csv}")
    print(f"[done] summary_json={summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
