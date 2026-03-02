# =============================
# File: fit_v5.py
# =============================
#!/usr/bin/env python3
"""
Rigid-body fit MODEL+MAP, then evaluate with:
  - VOF (Volume Overlap Fraction) from a 3D voxel IoU between a thresholded map and
    a model occupancy, optimized over rasterization radius + density cutoff.

Returns `(score, fitted_path)`, where `score = raw_vof * max(mean_z, 0)`.

Deps: gemmi, numpy, scipy, mrcfile
  pip install gemmi numpy scipy mrcfile
"""

from pathlib import Path
import math, itertools, os
from typing import Tuple, Optional
import numpy as np
import mrcfile, gemmi
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
from scipy.ndimage import gaussian_filter, label, center_of_mass

_MAPINFO_CACHE: dict[tuple[Path, float, bool, Optional[float], float], tuple[int, int, "MapInfo"]] = {}
_SPHERE_OFFSETS_CACHE: dict[tuple[int, float], np.ndarray] = {}
_FALLBACK_NOTICE_SEEN: set[str] = set()

def _fallback_notice(tag: str, message: str, once: bool = False) -> None:
    if once and tag in _FALLBACK_NOTICE_SEEN:
        return
    _FALLBACK_NOTICE_SEEN.add(tag)
    print(f"[fit_v5][fallback] {message}")

# -------------------- MRC helpers --------------------
def _safe_get_origin(mrc: mrcfile.mrcfile.MrcFile):
    h = mrc.header
    try:
        return float(h.xorigin), float(h.yorigin), float(h.zorigin)
    except Exception:
        _fallback_notice(
            "safe_get_origin_header_fields",
            "Header origin fields unavailable; trying legacy origin array.",
        )
        try:
            arr = np.asarray(h.origin, dtype=np.float64).ravel()
            if arr.size >= 3:
                return float(arr[0]), float(arr[1]), float(arr[2])
            _fallback_notice(
                "safe_get_origin_zero",
                "Legacy origin array incomplete; using (0,0,0) origin.",
            )
        except Exception:
            _fallback_notice(
                "safe_get_origin_zero",
                "Legacy origin array unavailable; using (0,0,0) origin.",
            )
    return 0.0, 0.0, 0.0

def _safe_get_step(mrc: mrcfile.mrcfile.MrcFile):
    try:
        vs = mrc.voxel_size
        return float(vs.x), float(vs.y), float(vs.z)
    except Exception:
        _fallback_notice(
            "safe_get_step_header",
            "Voxel size unavailable; deriving voxel step from header cell dimensions.",
        )
        h = mrc.header
        cella = np.asarray(getattr(h, "cella", [1.0, 1.0, 1.0]), dtype=np.float64).ravel()
        mx, my, mz = int(getattr(h, "mx", 0) or 0), int(getattr(h, "my", 0) or 0), int(getattr(h, "mz", 0) or 0)
        nx, ny, nz = int(h.nx), int(h.ny), int(h.nz)

        cellx = float(cella[0]) if cella.size > 0 else 1.0
        celly = float(cella[1]) if cella.size > 1 else 1.0
        cellz = float(cella[2]) if cella.size > 2 else 1.0

        def _axis_step(cell: float, m: int, n: int) -> float:
            if m <= 0 and n <= 0:
                _fallback_notice(
                    "safe_get_step_denom_default",
                    "Header axis sampling counts are missing/zero; using denominator fallback of 1.",
                )
            denom = m if m > 0 else (n if n > 0 else 1)
            step = float(cell) / float(denom)
            if (not np.isfinite(step)) or step == 0.0:
                _fallback_notice(
                    "safe_get_step_axis_default",
                    "Invalid axis voxel step encountered; falling back to 1.0 A step.",
                )
                return 1.0
            return abs(float(step))

        sx = _axis_step(cellx, mx, nx)
        sy = _axis_step(celly, my, ny)
        sz = _axis_step(cellz, mz, nz)
        return float(sx), float(sy), float(sz)

class MapInfo:
    """MRC mapping + vectorized interpolation on a working copy."""
    def __init__(self, mrc: mrcfile.mrcfile.MrcFile):
        raw = mrc.data.astype(np.float32, copy=False)
        if not np.isfinite(raw).all():
            _fallback_notice(
                "mapinfo_raw_nonfinite",
                "Input map has non-finite voxels; replacing with finite median values.",
            )
            finite = np.isfinite(raw)
            fill = np.nanmedian(raw[finite]) if np.any(finite) else 0.0
            raw = np.where(finite, raw, fill).astype(np.float32, copy=False)

        self.raw = raw
        self.nz, self.ny, self.nx = raw.shape
        self.sx, self.sy, self.sz = _safe_get_step(mrc)

        h = mrc.header
        self.nxstart = int(getattr(h, "nxstart", 0) or 0)
        self.nystart = int(getattr(h, "nystart", 0) or 0)
        self.nzstart = int(getattr(h, "nzstart", 0) or 0)

        self.mapc = int(getattr(h, "mapc", 1) or 1)
        self.mapr = int(getattr(h, "mapr", 2) or 2)
        self.maps = int(getattr(h, "maps", 3) or 3)
        if {self.mapc, self.mapr, self.maps} != {1, 2, 3}:
            _fallback_notice(
                "mapinfo_axis_order_default",
                f"Non-standard map axis order ({self.mapc},{self.mapr},{self.maps}); using default (1,2,3).",
            )
            self.mapc, self.mapr, self.maps = 1, 2, 3

        x0, y0, z0 = _safe_get_origin(mrc)
        self.eff_x0 = x0 + self._start_for_world_axis(1) * self.sx
        self.eff_y0 = y0 + self._start_for_world_axis(2) * self.sy
        self.eff_z0 = z0 + self._start_for_world_axis(3) * self.sz

        self.cx = self._count_for_world_axis(1)
        self.cy = self._count_for_world_axis(2)
        self.cz = self._count_for_world_axis(3)

        self.axX = self._array_axis_for_world_axis(1)
        self.axY = self._array_axis_for_world_axis(2)
        self.axZ = self._array_axis_for_world_axis(3)

        self.fit_map = self.raw
        self.mu = 0.0
        self.sd = 1.0
        self.z_map = self.raw
        self._set_fit_map(self.raw)
        self._coords_buf = np.empty((3, 0), dtype=np.float64)

        self.extent_x = self.sx * (self.cx - 1)
        self.extent_y = self.sy * (self.cy - 1)
        self.extent_z = self.sz * (self.cz - 1)
        self.longest_extent = float(max(self.extent_x, self.extent_y, self.extent_z))

    def _set_fit_map(self, fit_map: np.ndarray):
        arr = np.asarray(fit_map, dtype=np.float32, order="C")
        if arr.shape != self.raw.shape:
            raise ValueError(f"fit_map shape mismatch: got {arr.shape}, expected {self.raw.shape}")
        if not np.isfinite(arr).all():
            _fallback_notice(
                "mapinfo_fit_nonfinite",
                "Working map has non-finite voxels; replacing with finite median values.",
            )
            finite = np.isfinite(arr)
            fill = np.nanmedian(arr[finite]) if np.any(finite) else 0.0
            arr = np.where(finite, arr, fill).astype(np.float32, copy=False)
        self.fit_map = arr
        self.mu = float(np.mean(self.fit_map))
        self.sd = float(np.std(self.fit_map, ddof=0)) + 1e-12
        self.z_map = ((self.fit_map - self.mu) / self.sd).astype(np.float32, copy=False)

    def _array_axis_for_world_axis(self, axis_code: int) -> int:
        if self.maps == axis_code:
            return 0
        if self.mapr == axis_code:
            return 1
        if self.mapc == axis_code:
            return 2
        raise ValueError(f"Axis code {axis_code} not present in map axis ordering.")

    def _start_for_world_axis(self, axis_code: int) -> int:
        if self.mapc == axis_code:
            return self.nxstart
        if self.mapr == axis_code:
            return self.nystart
        if self.maps == axis_code:
            return self.nzstart
        raise ValueError(f"Axis code {axis_code} not present in map axis ordering.")

    def _count_for_world_axis(self, axis_code: int) -> int:
        if self.mapc == axis_code:
            return self.nx
        if self.mapr == axis_code:
            return self.ny
        if self.maps == axis_code:
            return self.nz
        raise ValueError(f"Axis code {axis_code} not present in map axis ordering.")

    def _grid_index_for_world_axis(self, axis_code: int, ix, iy, iz):
        if self.mapc == axis_code:
            return ix
        if self.mapr == axis_code:
            return iy
        if self.maps == axis_code:
            return iz
        raise ValueError(f"Axis code {axis_code} not present in map axis ordering.")

    def grid_to_world_many(self, ix, iy, iz):
        idx_x = self._grid_index_for_world_axis(1, ix, iy, iz)
        idx_y = self._grid_index_for_world_axis(2, ix, iy, iz)
        idx_z = self._grid_index_for_world_axis(3, ix, iy, iz)
        xs = self.eff_x0 + idx_x * self.sx
        ys = self.eff_y0 + idx_y * self.sy
        zs = self.eff_z0 + idx_z * self.sz
        return xs, ys, zs

    def world_to_array_frac_many(self, pts: np.ndarray):
        xs = pts[:, 0]; ys = pts[:, 1]; zs = pts[:, 2]
        idxX = (xs - self.eff_x0) / self.sx
        idxY = (ys - self.eff_y0) / self.sy
        idxZ = (zs - self.eff_z0) / self.sz
        f = [None, None, None]
        f[self.axX] = idxX
        f[self.axY] = idxY
        f[self.axZ] = idxZ
        return f  # [fz, fy, fx] in array-axis order (z,y,x)

    def _world_to_coords(self, pts_world: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_world, dtype=np.float64)
        n = int(pts.shape[0])
        if self._coords_buf.shape[1] < n:
            self._coords_buf = np.empty((3, n), dtype=np.float64)
        coords = self._coords_buf[:, :n]

        xs = pts[:, 0]
        ys = pts[:, 1]
        zs = pts[:, 2]
        coords[self.axX] = (xs - self.eff_x0) / self.sx
        coords[self.axY] = (ys - self.eff_y0) / self.sy
        coords[self.axZ] = (zs - self.eff_z0) / self.sz
        return coords

    def interp_many(self, pts_world: np.ndarray) -> np.ndarray:
        coords = self._world_to_coords(pts_world)
        vals = map_coordinates(self.fit_map, coords, order=1, mode='nearest', cval=0.0)
        return vals.astype(np.float64, copy=False)


def _center_blob_mask(
    arr: np.ndarray,
    threshold_value: float,
    cluster_disconnected: bool = True,
    centroid_link_vox: float = 8.0,
    center_gate_vox: float = 32.0,
    min_component_vox: int = 64,
) -> np.ndarray:
    """Return a center blob mask; optionally cluster nearby disconnected components."""
    mask = np.asarray(arr >= float(threshold_value), dtype=bool)
    if not np.any(mask):
        _fallback_notice(
            "center_blob_empty_mask",
            "Center-blob mask is empty at threshold; using full-map mask.",
        )
        return np.ones(arr.shape, dtype=bool)

    labs, nlab = label(mask)
    if int(nlab) <= 1:
        return mask

    counts = np.bincount(labs.ravel(), minlength=int(nlab) + 1)
    comp_labels = np.arange(1, int(nlab) + 1, dtype=np.int32)
    comp_counts = counts[1:].astype(np.int64, copy=False)
    valid = comp_counts >= int(max(1, min_component_vox))
    if not np.any(valid):
        valid = np.ones_like(comp_labels, dtype=bool)
    comp_labels = comp_labels[valid]
    if comp_labels.size == 0:
        return mask

    center_idx = np.array(
        [(arr.shape[0] - 1) * 0.5, (arr.shape[1] - 1) * 0.5, (arr.shape[2] - 1) * 0.5],
        dtype=np.float64,
    )
    cz = int(round(center_idx[0]))
    cy = int(round(center_idx[1]))
    cx = int(round(center_idx[2]))
    cz = int(np.clip(cz, 0, arr.shape[0] - 1))
    cy = int(np.clip(cy, 0, arr.shape[1] - 1))
    cx = int(np.clip(cx, 0, arr.shape[2] - 1))

    center_lab = int(labs[cz, cy, cx])
    centroids = []
    for lab_id in comp_labels.tolist():
        cm = center_of_mass(mask, labs, int(lab_id))
        if cm is None or len(cm) != 3:
            centroids.append((np.nan, np.nan, np.nan))
            continue
        centroids.append(cm)
    centroids = np.asarray(centroids, dtype=np.float64)
    finite = np.isfinite(centroids).all(axis=1)
    if not np.any(finite):
        _fallback_notice(
            "center_blob_centroid_invalid",
            "Component centroids invalid; falling back to first component.",
        )
        return labs == int(comp_labels[0])
    comp_labels = comp_labels[finite]
    centroids = centroids[finite]

    best_lab = int(comp_labels[0])
    best_d2 = float("inf")
    if center_lab > 0 and np.any(comp_labels == center_lab):
        best_lab = int(center_lab)
    else:
        for lab_id, c in zip(comp_labels.tolist(), centroids):
            d2 = float(np.sum((c - center_idx) ** 2))
            if d2 < best_d2:
                best_d2 = d2
                best_lab = int(lab_id)

    if not cluster_disconnected or comp_labels.size == 1:
        return labs == best_lab

    seed_idx = int(np.where(comp_labels == best_lab)[0][0])
    d_center = np.linalg.norm(centroids - center_idx.reshape(1, 3), axis=1)
    gate = float(max(4.0, center_gate_vox))
    gated = d_center <= gate
    if not np.any(gated):
        _fallback_notice(
            "center_blob_gate_empty",
            "No components within center gate; falling back to nearest center component.",
        )
        return labs == best_lab

    gated_indices = np.where(gated)[0]
    gated_labels = comp_labels[gated_indices]
    gated_centers = centroids[gated_indices]
    if gated_labels.size == 1:
        return labs == int(gated_labels[0])

    if best_lab in gated_labels:
        seed_local = int(np.where(gated_labels == best_lab)[0][0])
    else:
        _fallback_notice(
            "center_blob_seed_missing",
            "Best center component missing after gating; falling back to nearest center component.",
        )
        return labs == best_lab

    link = float(max(2.0, centroid_link_vox))
    dmat = np.linalg.norm(gated_centers[:, None, :] - gated_centers[None, :, :], axis=2)
    adjacency = dmat <= link
    visited = np.zeros(gated_labels.shape[0], dtype=bool)
    stack = [seed_local]
    visited[seed_local] = True
    while stack:
        cur = stack.pop()
        nbrs = np.where(adjacency[cur])[0]
        for nb in nbrs.tolist():
            if not visited[nb]:
                visited[nb] = True
                stack.append(nb)

    selected_labels = gated_labels[visited]
    if selected_labels.size == 0:
        _fallback_notice(
            "center_blob_cluster_empty",
            "Disconnected-component clustering selected none; falling back to nearest center component.",
        )
        selected_labels = np.array([best_lab], dtype=np.int32)
    return np.isin(labs, selected_labels)


def _get_cached_mapinfo(
    map_path: Path,
    denoise_sigma: float = 0.0,
    denoise_center_blob_only: bool = False,
    denoise_center_blob_level: Optional[float] = None,
    denoise_center_blob_z_fallback: float = 0.50,
    denoise_center_blob_cluster_disconnected: bool = True,
    denoise_center_blob_link_factor: float = 0.08,
    denoise_center_blob_center_factor: float = 0.40,
    denoise_center_blob_min_component_vox: int = 64,
) -> MapInfo:
    map_path = Path(map_path).resolve()
    st = map_path.stat()
    sig = (int(st.st_mtime_ns), int(st.st_size))
    sigma = float(max(0.0, denoise_sigma))
    blob_level = None
    if denoise_center_blob_level is not None and np.isfinite(denoise_center_blob_level):
        blob_level = float(denoise_center_blob_level)
    key = (
        map_path,
        sigma,
        bool(denoise_center_blob_only),
        (None if blob_level is None else round(blob_level, 6)),
        float(denoise_center_blob_z_fallback),
        bool(denoise_center_blob_cluster_disconnected),
        round(float(denoise_center_blob_link_factor), 6),
        round(float(denoise_center_blob_center_factor), 6),
        int(denoise_center_blob_min_component_vox),
    )
    cached = _MAPINFO_CACHE.get(key)
    if cached and (cached[0], cached[1]) == sig:
        return cached[2]

    with mrcfile.open(str(map_path), mode="r") as mrc:
        mapinfo = MapInfo(mrc)
    if sigma > 0.0:
        src = mapinfo.raw
        if denoise_center_blob_only:
            if blob_level is None:
                zf = float(denoise_center_blob_z_fallback)
                _fallback_notice(
                    "denoise_blob_level_z_fallback",
                    f"No contour level provided for center-blob denoise; using z-fallback {zf:.2f}.",
                )
                thr = float(np.mean(src) + zf * (float(np.std(src)) + 1e-12))
            else:
                thr = float(blob_level)
            min_dim = float(max(1, min(src.shape)))
            link_vox = max(2.0, float(denoise_center_blob_link_factor) * min_dim)
            center_gate_vox = max(4.0, float(denoise_center_blob_center_factor) * min_dim)
            keep = _center_blob_mask(
                src,
                threshold_value=thr,
                cluster_disconnected=bool(denoise_center_blob_cluster_disconnected),
                centroid_link_vox=float(link_vox),
                center_gate_vox=float(center_gate_vox),
                min_component_vox=int(max(1, denoise_center_blob_min_component_vox)),
            )
            work = np.where(keep, src, 0.0).astype(np.float32, copy=False)
            den = gaussian_filter(work, sigma=sigma).astype(np.float32, copy=False)
            den = np.where(keep, den, 0.0).astype(np.float32, copy=False)
        else:
            den = gaussian_filter(src, sigma=sigma).astype(np.float32, copy=False)
        mapinfo._set_fit_map(den)

    _MAPINFO_CACHE[key] = (sig[0], sig[1], mapinfo)
    if len(_MAPINFO_CACHE) > 8:
        keep = {key: _MAPINFO_CACHE[key]}
        _MAPINFO_CACHE.clear()
        _MAPINFO_CACHE.update(keep)
    return mapinfo

# -------------------- Structure I/O & transforms --------------------
def read_model_coords(st_path: Path):
    """Use representative atoms: CA, then P, else first non-H atom."""
    st = gemmi.read_structure(str(st_path))
    if len(st) == 0:
        raise RuntimeError("Empty structure (no models).")
    coords = []
    for model in st:
        for chain in model:
            for res in chain:
                atom = None
                for a in res:
                    if a.name == "CA": atom = a; break
                if atom is None:
                    for a in res:
                        if a.name == "P": atom = a; break
                if atom is None:
                    for a in res:
                        if (a.element and a.element.name != "H"): atom = a; break
                if atom is not None:
                    p = atom.pos; coords.append((p.x, p.y, p.z))
    if not coords:
        raise RuntimeError("No atoms found in model.")
    return st, np.asarray(coords, dtype=np.float64)

def _collect_non_h_coords(st: gemmi.Structure) -> np.ndarray:
    coords = []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    if atom.element and atom.element.name == "H":
                        continue
                    p = atom.pos
                    coords.append((p.x, p.y, p.z))
    return np.asarray(coords, dtype=np.float64) if coords else np.empty((0, 3), dtype=np.float64)

def _collect_backbone_coords(st: gemmi.Structure) -> np.ndarray:
    coords = []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    if atom.name in ("N", "CA", "C", "O", "P"):
                        if atom.element and atom.element.name == "H":
                            continue
                        p = atom.pos
                        coords.append((p.x, p.y, p.z))
    if coords:
        return np.asarray(coords, dtype=np.float64)
    return _collect_non_h_coords(st)

def apply_transform_to_structure(st: gemmi.Structure, Rm: np.ndarray, t: np.ndarray):
    Rm = np.asarray(Rm, dtype=np.float64); t = np.asarray(t, dtype=np.float64)
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    p = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)
                    q = Rm @ p + t
                    atom.pos = gemmi.Position(float(q[0]), float(q[1]), float(q[2]))

# -------------------- PCA & coarse alignment --------------------
def pca(points: np.ndarray, weights: np.ndarray | None = None):
    if weights is None:
        center = points.mean(axis=0)
        X = points - center
        C = (X.T @ X) / max(len(points) - 1, 1)
    else:
        w = weights.reshape(-1, 1); wsum = float(weights.sum()) or 1.0
        center = (points * w).sum(axis=0) / wsum
        X = points - center
        C = (X.T * weights) @ X / wsum
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    return vecs, center

def _safe_normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        _fallback_notice(
            "safe_normalize_vector",
            "Attempted to normalize a near-zero/invalid vector; using fallback axis.",
        )
        out = np.asarray(fallback, dtype=np.float64)
        out_n = float(np.linalg.norm(out))
        if not np.isfinite(out_n) or out_n < 1e-12:
            _fallback_notice(
                "safe_normalize_default_x",
                "Fallback axis is invalid; using hard default [1,0,0].",
            )
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return out / out_n
    return v / n

def _orthogonal_fallback_axis(axis: np.ndarray) -> np.ndarray:
    axis = _safe_normalize(axis, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    cands = (
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
        np.array([0.0, 1.0, 0.0], dtype=np.float64),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    best = cands[0]
    best_norm = -1.0
    for cand in cands:
        v = cand - axis * float(np.dot(cand, axis))
        n = float(np.linalg.norm(v))
        if n > best_norm:
            best_norm = n
            best = v
    return _safe_normalize(best, np.array([0.0, 1.0, 0.0], dtype=np.float64))

def _weighted_skewness(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0
    if weights is None:
        x0 = x - float(np.mean(x))
        m2 = float(np.mean(x0 * x0))
        if m2 < 1e-16:
            return 0.0
        m3 = float(np.mean(x0 * x0 * x0))
        return m3 / ((m2 ** 1.5) + 1e-18)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if w.size != x.size:
        _fallback_notice(
            "weighted_skew_mismatch",
            "Skewness weights length mismatch; returning 0 skewness fallback.",
        )
        return 0.0
    w = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)
    wsum = float(np.sum(w))
    if wsum <= 0.0:
        _fallback_notice(
            "weighted_skew_unweighted",
            "Skewness weights sum is non-positive; falling back to unweighted skewness.",
        )
        return _weighted_skewness(x, weights=None)
    mu = float(np.sum(w * x) / wsum)
    x0 = x - mu
    m2 = float(np.sum(w * x0 * x0) / wsum)
    if m2 < 1e-16:
        return 0.0
    m3 = float(np.sum(w * x0 * x0 * x0) / wsum)
    return m3 / ((m2 ** 1.5) + 1e-18)

def _orient_axis_by_skewness(
    axis: np.ndarray,
    points: np.ndarray,
    center: np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    axis = _safe_normalize(axis, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    pts = np.asarray(points, dtype=np.float64)
    ctr = np.asarray(center, dtype=np.float64)
    proj = (pts - ctr) @ axis
    skew = _weighted_skewness(proj, weights)
    if skew < 0.0:
        axis = -axis
    elif abs(skew) <= 1e-8:
        if weights is None:
            pos = int(np.count_nonzero(proj >= 0.0))
            neg = int(np.count_nonzero(proj < 0.0))
            if pos < neg:
                axis = -axis
        else:
            w = np.asarray(weights, dtype=np.float64).reshape(-1)
            if w.size == proj.size:
                w = np.where(np.isfinite(w), np.maximum(w, 0.0), 0.0)
                pos = float(np.sum(w[proj >= 0.0]))
                neg = float(np.sum(w[proj < 0.0]))
                if pos < neg:
                    axis = -axis
        # deterministic tie-break
        k = int(np.argmax(np.abs(axis)))
        if axis[k] < 0.0:
            axis = -axis
    return axis

def _build_signed_pca_frame(
    points: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    vecs, center = pca(points, weights=weights)
    pc1 = _orient_axis_by_skewness(vecs[:, 0], points, center, weights)

    pc2_raw = np.asarray(vecs[:, 1], dtype=np.float64)
    pc2_raw = pc2_raw - pc1 * float(np.dot(pc2_raw, pc1))
    if float(np.linalg.norm(pc2_raw)) < 1e-12:
        _fallback_notice(
            "pca_frame_pc2_fallback",
            "PC2 is degenerate after orthogonalization; using orthogonal fallback axis.",
        )
        pc2_raw = _orthogonal_fallback_axis(pc1)
    pc2 = _orient_axis_by_skewness(pc2_raw, points, center, weights)
    pc2 = _safe_normalize(pc2 - pc1 * float(np.dot(pc2, pc1)), _orthogonal_fallback_axis(pc1))

    pc3 = _safe_normalize(np.cross(pc1, pc2), _orthogonal_fallback_axis(pc1))
    pc2 = _safe_normalize(np.cross(pc3, pc1), pc2)
    frame = np.column_stack([pc1, pc2, pc3]).astype(np.float64, copy=False)

    # Numerical guard to preserve right-handedness.
    if np.linalg.det(frame) < 0.0:
        _fallback_notice(
            "pca_frame_handedness_fix",
            "PCA frame had negative determinant; flipping PC2 to enforce right-handed frame.",
        )
        pc2 = -pc2
        pc3 = _safe_normalize(np.cross(pc1, pc2), pc3)
        pc2 = _safe_normalize(np.cross(pc3, pc1), pc2)
        frame = np.column_stack([pc1, pc2, pc3]).astype(np.float64, copy=False)
    return frame, np.asarray(center, dtype=np.float64)

def sample_map_points(mapinfo: MapInfo, stride: int, target_min: int):
    A = mapinfo.fit_map
    sub = A[::stride, ::stride, ::stride]
    mu, sd = float(sub.mean()), float(sub.std(ddof=0))
    thresholds = [mu + 1.5*sd, mu + 1.0*sd, np.quantile(sub, 0.98), np.quantile(sub, 0.95), mu + 0.5*sd]
    pts = wts = None

    for th in thresholds:
        mask = sub >= th
        if int(mask.sum()) >= target_min:
            iz, iy, ix = np.nonzero(mask)
            iz *= stride; iy *= stride; ix *= stride
            vals = A[iz, iy, ix].astype(np.float64, copy=False)
            xs, ys, zs = mapinfo.grid_to_world_many(ix, iy, iz)
            pts = np.column_stack([xs, ys, zs]); wts = vals
            break

    if pts is None:
        _fallback_notice(
            "sample_map_points_full_map",
            "No threshold produced enough map points; falling back to full-map sampling for PCA/COM.",
        )
        iz, iy, ix = np.nonzero(np.ones_like(sub, dtype=bool))
        iz *= stride; iy *= stride; ix *= stride
        vals = A[iz, iy, ix].astype(np.float64, copy=False)
        xs, ys, zs = mapinfo.grid_to_world_many(ix, iy, iz)
        pts = np.column_stack([xs, ys, zs]); wts = vals

    wsum = float(wts.sum()) or 1.0
    com = (pts * (wts.reshape(-1, 1))).sum(axis=0) / wsum
    return pts, wts, com

def _score_meanZ(mapinfo: MapInfo, pts_world: np.ndarray) -> float:
    vals = mapinfo.interp_many(pts_world)
    return float(np.mean((vals - mapinfo.mu) / mapinfo.sd))

# -------------------- VOF (3D IoU) --------------------
def _downsample_to_maxdim(vol: np.ndarray, max_dim: int) -> np.ndarray:
    z, y, x = vol.shape
    scale = max(1, int(np.ceil(max([z,y,x]) / max_dim)))
    if scale == 1: return vol
    return vol[::scale, ::scale, ::scale]

# --- VOF optimization defaults ---
VOF_RADIUS_RANGE_A = (4.0, 8.0)
VOF_RADIUS_STEP_A = 0.25
VOF_Z_THRESHOLDS = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5]
VOF_Z_QUANTILES = [0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995]
VOF_MIN_MAP_FRAC = 0.001
VOF_MIN_MODEL_FRAC = 0.0005
VOF_MIN_UNION_FRAC = 0.0015
VOF_MAX_DIM = 160
VOF_SCORE_ATOMS = "backbone"  # "rep", "backbone", or "heavy"

# --- Global coarse search ---
GLOBAL_SEARCH_MAX_DIM = 64
GLOBAL_SEARCH_Z = 1.5
GLOBAL_SEARCH_RADIUS_A = 3.5
GLOBAL_SEARCH_ROTATIONS = 24  # fixed PCA-aligned cube-rotation sampler size (kept for API compatibility)
GLOBAL_SEARCH_TRANS_RANGE = 1
GLOBAL_SEARCH_TRANS_STEP_SCALE = 1.0
GLOBAL_SEARCH_USE_PCA_SEED = True
GLOBAL_SEARCH_COARSE_TRANSLATION = False
GLOBAL_SEARCH_COARSE_MULT = 4.0
GLOBAL_SEARCH_COARSE_RANGE = 2
GLOBAL_SEARCH_SEED = 0  # legacy (unused by fixed 24-rotation sampler; kept for API compatibility)
GLOBAL_SEARCH_SELF_SPIN_STEPS = 1  # 1 => disable extra self-spin sweep

# --- Joint local refinement (replaces greedy_refine in fit_model_and_score) ---
GLOBAL_JOINT_POWELL_MAXITER = 32
GLOBAL_JOINT_POWELL_XTOL = 1e-6
GLOBAL_JOINT_POWELL_FTOL = 1e-6
GLOBAL_JOINT_ROT_BOUND_DEG = 20.0
GLOBAL_JOINT_TRANS_FACTOR = 0.30
GLOBAL_MAP_DENOISE_ENABLED = False
GLOBAL_MAP_DENOISE_SIGMA = 1.0
# If True, evaluate only the denoised map variant (no raw-map fit branch).
GLOBAL_MAP_USE_DENOISED_ONLY = True
GLOBAL_MAP_DENOISE_CENTER_BLOB_ONLY = True
GLOBAL_MAP_DENOISE_CENTER_BLOB_Z_FALLBACK = 0.50
GLOBAL_MAP_DENOISE_CENTER_BLOB_CLUSTER_DISCONNECTED = True
GLOBAL_MAP_DENOISE_CENTER_BLOB_LINK_FACTOR = 0.08
GLOBAL_MAP_DENOISE_CENTER_BLOB_CENTER_FACTOR = 0.40
GLOBAL_MAP_DENOISE_CENTER_BLOB_MIN_COMPONENT_VOX = 64

def _score_from_vof_meanz(raw_vof: float, mean_z: float) -> float:
    return float(raw_vof) * max(float(mean_z), 0.0)


def joint_powell_refine(
    mapinfo: MapInfo,
    model_pts_scoring: np.ndarray,
    R_seed: np.ndarray,
    t_seed: np.ndarray,
    trans_bound_ang: float,
    maxiter: int = GLOBAL_JOINT_POWELL_MAXITER,
    xtol: float = GLOBAL_JOINT_POWELL_XTOL,
    ftol: float = GLOBAL_JOINT_POWELL_FTOL,
    rot_bound_deg: float = GLOBAL_JOINT_ROT_BOUND_DEG,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Joint 6-DoF local optimization (spin+rotation+translation) using Powell on mean Z."""
    model_pts_scoring = np.asarray(model_pts_scoring, dtype=np.float64)
    R_seed = np.asarray(R_seed, dtype=np.float64)
    t_seed = np.asarray(t_seed, dtype=np.float64)

    rot_bound = math.radians(float(rot_bound_deg))
    trans_bound = abs(float(trans_bound_ang))
    lb = np.array(
        [-rot_bound, -rot_bound, -rot_bound, -trans_bound, -trans_bound, -trans_bound],
        dtype=np.float64,
    )
    ub = np.array(
        [rot_bound, rot_bound, rot_bound, trans_bound, trans_bound, trans_bound],
        dtype=np.float64,
    )

    def _project(v: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(v, dtype=np.float64), lb, ub)

    def _score(v: np.ndarray) -> float:
        v = _project(v)
        rv = v[:3]
        dt = v[3:]
        R_delta = R.from_rotvec(rv).as_matrix()
        R_try = R_delta @ R_seed
        t_try = t_seed + dt
        pts_world = (R_try @ model_pts_scoring.T).T + t_try
        return _score_meanZ(mapinfo, pts_world)

    x0 = np.zeros(6, dtype=np.float64)
    best_x = x0.copy()
    best_s = _score(best_x)

    res = minimize(
        lambda v: -_score(v),
        x0,
        method="Powell",
        bounds=list(zip(lb, ub)),
        options={
            "maxiter": int(maxiter),
            "xtol": float(xtol),
            "ftol": float(ftol),
            "disp": False,
        },
    )
    if res.x is not None:
        x_try = _project(res.x)
        s_try = _score(x_try)
        if s_try > best_s:
            best_s = s_try
            best_x = x_try

    rv = best_x[:3]
    dt = best_x[3:]
    R_delta = R.from_rotvec(rv).as_matrix()
    R_best = R_delta @ R_seed
    t_best = t_seed + dt
    return R_best, t_best, float(best_s)

def _model_occupancy_from_points(shape: tuple[int,int,int], pts_zyx: np.ndarray, rad_vox: float) -> np.ndarray:
    occ = np.zeros(shape, dtype=np.uint8)
    if pts_zyx.size == 0:
        return occ

    zmax, ymax, xmax = shape
    rv = float(rad_vox)
    rr = int(max(1, np.ceil(rv)))
    key = (rr, rv)
    offsets = _SPHERE_OFFSETS_CACHE.get(key)
    if offsets is None:
        off = []
        r2 = rv * rv + 1e-6
        for dz in range(-rr, rr + 1):
            for dy in range(-rr, rr + 1):
                for dx in range(-rr, rr + 1):
                    if (dx * dx + dy * dy + dz * dz) <= r2:
                        off.append((dz, dy, dx))
        offsets = np.asarray(off, dtype=np.int64)
        _SPHERE_OFFSETS_CACHE[key] = offsets

    idx = np.round(pts_zyx).astype(np.int64)

    all_coords = idx[:, None, :] + offsets[None, :, :]
    all_coords = all_coords.reshape(-1, 3)
    all_coords[:, 0] = np.clip(all_coords[:, 0], 0, zmax - 1)
    all_coords[:, 1] = np.clip(all_coords[:, 1], 0, ymax - 1)
    all_coords[:, 2] = np.clip(all_coords[:, 2], 0, xmax - 1)
    occ[all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]] = 1
    return occ

def _prepare_vof_inputs(mapinfo: MapInfo, model_pts_world: np.ndarray, max_dim: int):
    m = mapinfo.fit_map
    m_ds = _downsample_to_maxdim(m, max_dim)
    mZ_ds = _downsample_to_maxdim(mapinfo.z_map, max_dim)

    fz, fy, fx = mapinfo.world_to_array_frac_many(model_pts_world)
    scale_z = m.shape[0] / m_ds.shape[0]
    scale_y = m.shape[1] / m_ds.shape[1]
    scale_x = m.shape[2] / m_ds.shape[2]
    pts_zyx_ds = np.column_stack([fz/scale_z, fy/scale_y, fx/scale_x])

    scale_mean = float(np.mean([scale_z, scale_y, scale_x]))
    return m_ds, mZ_ds, pts_zyx_ds, scale_mean

def _build_radius_candidates(rmin: float, rmax: float, step: float, extra: Optional[list[float]] = None) -> list[float]:
    base = np.arange(rmin, rmax + 1e-6, step, dtype=np.float64).tolist()
    if extra:
        base.extend(extra)
    uniq = sorted({round(float(r), 4) for r in base if rmin <= float(r) <= rmax})
    return uniq

def _build_threshold_candidates(
    mZ_ds: np.ndarray,
    mapinfo: MapInfo,
    recommended_contour_level: Optional[float] = None,
    extra: Optional[list[float]] = None,
) -> list[float]:
    candidates: list[float] = []
    candidates.extend(VOF_Z_THRESHOLDS)
    if VOF_Z_QUANTILES:
        try:
            q_vals = np.quantile(mZ_ds, VOF_Z_QUANTILES).astype(np.float64).ravel().tolist()
            candidates.extend(q_vals)
        except Exception:
            pass
    if recommended_contour_level is not None and np.isfinite(recommended_contour_level):
        z_rec = (float(recommended_contour_level) - mapinfo.mu) / mapinfo.sd
        if np.isfinite(z_rec):
            candidates.extend([z_rec * 0.8, z_rec, z_rec * 1.2, z_rec - 0.25, z_rec + 0.25])
    if extra:
        candidates.extend(extra)
    uniq = sorted({round(float(z), 5) for z in candidates if np.isfinite(z)})
    return uniq

def _compute_vof_from_masks(map_mask: np.ndarray, model_occ: np.ndarray) -> Tuple[float, int, int, int]:
    inter = int(np.count_nonzero(map_mask & (model_occ > 0)))
    map_cnt = int(np.count_nonzero(map_mask))
    model_cnt = int(np.count_nonzero(model_occ))
    union = map_cnt + model_cnt - inter
    vof = float(inter) / float(union) if union > 0 else 0.0
    return vof, inter, map_cnt, model_cnt

def compute_vof_for_radius(
    mapinfo: MapInfo,
    model_pts_world: np.ndarray,
    radius_ang: float,
    max_dim: int = 128,
    recommended_contour_level: Optional[float] = None,
) -> Tuple[float, dict]:
    """
    Compute VOF for a fixed radius, optimizing only the density cutoff (Z-score).
    Returns (best_vof, details).
    """
    m_ds, mZ_ds, pts_zyx_ds, scale_mean = _prepare_vof_inputs(mapinfo, model_pts_world, max_dim)
    total_vox = int(m_ds.size)
    min_map_vox = max(200, int(VOF_MIN_MAP_FRAC * total_vox))
    min_model_vox = max(120, int(VOF_MIN_MODEL_FRAC * total_vox))
    min_union_vox = max(min_map_vox, min_model_vox, int(VOF_MIN_UNION_FRAC * total_vox))

    vs_mean = float((mapinfo.sx + mapinfo.sy + mapinfo.sz) / 3.0)
    r_vox = float(radius_ang) / max(vs_mean, 1e-6)
    rad_ds = r_vox / max(scale_mean, 1e-6)
    model_occ = _model_occupancy_from_points(m_ds.shape, pts_zyx_ds, rad_ds)

    thresholds = _build_threshold_candidates(
        mZ_ds,
        mapinfo,
        recommended_contour_level=recommended_contour_level,
    )
    best = None
    best_any = None
    for z_th in thresholds:
        mask = mZ_ds >= z_th
        vof, inter, map_cnt, model_cnt = _compute_vof_from_masks(mask, model_occ)
        union = map_cnt + model_cnt - inter
        payload = {
            "vof": vof,
            "radius_ang": float(radius_ang),
            "threshold_z": float(z_th),
            "map_mask_vox": map_cnt,
            "model_mask_vox": model_cnt,
            "union_vox": union,
        }
        if best_any is None or vof > best_any["vof"]:
            best_any = payload
        if map_cnt >= min_map_vox and model_cnt >= min_model_vox and union >= min_union_vox:
            if best is None or vof > best["vof"]:
                best = payload

    chosen = best or best_any
    if not chosen:
        _fallback_notice(
            "vof_for_radius_no_candidate",
            "VOF(radius) found no valid threshold candidate; returning zero score.",
        )
        return 0.0, {
            "radius_ang": float(radius_ang),
            "threshold_z": None,
            "threshold_value": None,
            "map_mask_vox": 0,
            "model_mask_vox": 0,
            "union_vox": 0,
            "constrained": False,
        }

    threshold_val = mapinfo.mu + chosen["threshold_z"] * mapinfo.sd
    details = {
        "radius_ang": float(chosen["radius_ang"]),
        "threshold_z": float(chosen["threshold_z"]),
        "threshold_value": float(threshold_val),
        "map_mask_vox": int(chosen["map_mask_vox"]),
        "model_mask_vox": int(chosen["model_mask_vox"]),
        "union_vox": int(chosen["union_vox"]),
        "constrained": bool(best is not None),
    }
    if best is None and best_any is not None:
        _fallback_notice(
            "vof_for_radius_unconstrained",
            "VOF(radius) constrained mask criteria not met; using best unconstrained threshold.",
        )
    return float(chosen["vof"]), details

def compute_vof_optimized(
    mapinfo: MapInfo,
    model_pts_world: np.ndarray,
    max_dim: int = 128,
    recommended_contour_level: Optional[float] = None,
) -> Tuple[float, dict]:
    """
    Optimize VOF over (radius, density threshold) using a plateau-aware
    exponential bracketing search in radius per threshold.
    Density thresholds are applied in Z-score space; if a recommended contour
    level is provided, it is converted to Z and included in candidates.
    Returns (best_vof, details).
    """
    m_ds, mZ_ds, pts_zyx_ds, scale_mean = _prepare_vof_inputs(mapinfo, model_pts_world, max_dim)
    total_vox = int(m_ds.size)
    min_map_vox = max(200, int(VOF_MIN_MAP_FRAC * total_vox))
    min_model_vox = max(120, int(VOF_MIN_MODEL_FRAC * total_vox))
    min_union_vox = max(min_map_vox, min_model_vox, int(VOF_MIN_UNION_FRAC * total_vox))

    vs_mean = float((mapinfo.sx + mapinfo.sy + mapinfo.sz) / 3.0)
    rmin, rmax = VOF_RADIUS_RANGE_A
    radii = _build_radius_candidates(rmin, rmax, VOF_RADIUS_STEP_A)
    z_thresh = _build_threshold_candidates(mZ_ds, mapinfo, recommended_contour_level=recommended_contour_level)
    if not radii or not z_thresh:
        _fallback_notice(
            "vof_optimized_empty_grid",
            "VOF optimization has empty radius/threshold grid; returning zero score.",
        )
        return 0.0, {
            "radius_ang": None,
            "threshold_z": None,
            "threshold_value": None,
            "map_mask_vox": 0,
            "model_mask_vox": 0,
            "union_vox": 0,
            "constrained": False,
        }

    map_masks: dict[float, tuple[np.ndarray, int]] = {}
    for z_th in z_thresh:
        zf = float(z_th)
        mask = mZ_ds >= zf
        map_masks[zf] = (mask, int(np.count_nonzero(mask)))

    occ_cache: dict[float, tuple[np.ndarray, int]] = {}
    pair_cache: dict[tuple[float, float], dict] = {}

    def _get_occ(r_ang: float) -> tuple[np.ndarray, int]:
        rf = float(r_ang)
        cached = occ_cache.get(rf)
        if cached is not None:
            return cached
        r_vox = rf / max(vs_mean, 1e-6)
        rad_ds = r_vox / max(scale_mean, 1e-6)
        occ = _model_occupancy_from_points(m_ds.shape, pts_zyx_ds, rad_ds)
        payload = (occ, int(np.count_nonzero(occ)))
        occ_cache[rf] = payload
        return payload

    def _eval_pair(z_th: float, r_ang: float) -> dict:
        zf = float(z_th)
        rf = float(r_ang)
        key = (zf, rf)
        cached = pair_cache.get(key)
        if cached is not None:
            return cached

        mask, _ = map_masks[zf]
        occ, _ = _get_occ(rf)
        vof, inter, map_cnt_actual, model_cnt_actual = _compute_vof_from_masks(mask, occ)
        union = map_cnt_actual + model_cnt_actual - inter
        payload = {
            "vof": float(vof),
            "radius_ang": rf,
            "threshold_z": zf,
            "map_mask_vox": int(map_cnt_actual),
            "model_mask_vox": int(model_cnt_actual),
            "union_vox": int(union),
        }
        pair_cache[key] = payload
        return payload

    best = None
    best_any = None

    def _update_best(payload: dict) -> None:
        nonlocal best, best_any
        if best_any is None or payload["vof"] > best_any["vof"]:
            best_any = payload
        if (
            payload["map_mask_vox"] >= min_map_vox
            and payload["model_mask_vox"] >= min_model_vox
            and payload["union_vox"] >= min_union_vox
        ):
            if best is None or payload["vof"] > best["vof"]:
                best = payload

    n_r = len(radii)
    mid_idx = n_r // 2

    # For each threshold, search radius on the discrete grid:
    # 1) choose direction from center (plateau-aware),
    # 2) exponentially bracket candidate peak,
    # 3) local ternary narrowing in the bracket.
    for zf in [float(z) for z in z_thresh]:
        local: dict[int, dict] = {}

        def _get_idx(i: int) -> Optional[dict]:
            if i < 0 or i >= n_r:
                return None
            cached = local.get(i)
            if cached is not None:
                return cached
            payload = _eval_pair(zf, radii[i])
            local[i] = payload
            return payload

        center = _get_idx(mid_idx)
        if center is None:
            continue
        left = _get_idx(mid_idx - 1)
        right = _get_idx(mid_idx + 1)

        direction = 0
        if right is not None and right["vof"] >= center["vof"] and (left is None or right["vof"] >= left["vof"]):
            direction = +1
        elif left is not None and left["vof"] >= center["vof"]:
            direction = -1

        if direction == 0:
            for payload in local.values():
                _update_best(payload)
            continue

        prev = mid_idx
        cur = mid_idx + direction
        step = 2
        while True:
            p_cur = _get_idx(cur)
            if p_cur is None:
                break

            nxt = cur + direction * step
            if nxt < 0 or nxt >= n_r:
                boundary = 0 if nxt < 0 else (n_r - 1)
                _get_idx(boundary)
                lo = min(prev, boundary)
                hi = max(prev, boundary)
                break

            p_nxt = _get_idx(nxt)
            if p_nxt is not None and p_nxt["vof"] >= p_cur["vof"]:
                prev = cur
                cur = nxt
                step *= 2
            else:
                lo = min(prev, nxt)
                hi = max(prev, nxt)
                break

        while hi - lo > 3:
            m1 = lo + (hi - lo) // 3
            m2 = hi - (hi - lo) // 3
            p1 = _get_idx(m1)
            p2 = _get_idx(m2)
            if p1 is None or p2 is None:
                break
            if p1["vof"] < p2["vof"]:
                lo = m1
            elif p1["vof"] > p2["vof"]:
                hi = m2
            else:
                lo = m1
                hi = m2

        for i in range(lo, hi + 1):
            _get_idx(i)
        for payload in local.values():
            _update_best(payload)

    chosen = best or best_any
    if not chosen:
        _fallback_notice(
            "vof_optimized_no_candidate",
            "VOF optimization found no candidate; returning zero score.",
        )
        return 0.0, {
            "radius_ang": None,
            "threshold_z": None,
            "threshold_value": None,
            "map_mask_vox": 0,
            "model_mask_vox": 0,
            "union_vox": 0,
            "constrained": False,
        }

    threshold_val = mapinfo.mu + chosen["threshold_z"] * mapinfo.sd
    details = {
        "radius_ang": float(chosen["radius_ang"]),
        "threshold_z": float(chosen["threshold_z"]),
        "threshold_value": float(threshold_val),
        "map_mask_vox": int(chosen["map_mask_vox"]),
        "model_mask_vox": int(chosen["model_mask_vox"]),
        "union_vox": int(chosen["union_vox"]),
        "constrained": bool(best is not None),
    }
    if best is None and best_any is not None:
        _fallback_notice(
            "vof_optimized_unconstrained",
            "VOF constrained mask criteria not met; using best unconstrained candidate.",
        )
    return float(chosen["vof"]), details

_CUBE24_ROTATIONS_BASE: Optional[np.ndarray] = None
_CUBE24_AXES_ANGLES_BASE: Optional[tuple[np.ndarray, np.ndarray]] = None


def _rotation_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a /= (np.linalg.norm(a) + 1e-12)
    b /= (np.linalg.norm(b) + 1e-12)
    c = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if c >= 1.0 - 1e-10:
        return np.eye(3, dtype=np.float64)
    if c <= -1.0 + 1e-10:
        ortho = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = np.cross(a, ortho)
        axis /= (np.linalg.norm(axis) + 1e-12)
        return R.from_rotvec(axis * math.pi).as_matrix()
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = v / s
    ang = math.atan2(s, c)
    return R.from_rotvec(axis * ang).as_matrix()


def _align_rotation_axes_to_target(axes: np.ndarray, target_axis: np.ndarray) -> np.ndarray:
    axes = np.asarray(axes, dtype=np.float64)
    if axes.size == 0:
        return axes
    t = np.asarray(target_axis, dtype=np.float64)
    tn = np.linalg.norm(t)
    if not np.isfinite(tn) or tn < 1e-12:
        _fallback_notice(
            "align_rotset_target_invalid",
            "Target axis for coarse rotation-set alignment is invalid; leaving rotation set orientation unchanged.",
        )
        return axes
    t = t / tn
    idx = int(np.argmax(axes @ t))
    Q = _rotation_from_a_to_b(axes[idx], t)
    out = (Q @ axes.T).T
    out /= (np.linalg.norm(out, axis=1, keepdims=True) + 1e-12)
    return out


def _cube24_rotation_axis_angles() -> tuple[np.ndarray, np.ndarray]:
    global _CUBE24_AXES_ANGLES_BASE
    if _CUBE24_AXES_ANGLES_BASE is not None:
        return _CUBE24_AXES_ANGLES_BASE

    mats = _cube24_rotation_matrices()
    axes: list[np.ndarray] = []
    angles: list[float] = []
    for M in mats:
        rv = R.from_matrix(M).as_rotvec()
        ang = float(np.linalg.norm(rv))
        if ang < 1e-12:
            axes.append(np.array([0.0, 0.0, 1.0], dtype=np.float64))
            angles.append(0.0)
        else:
            axes.append(np.asarray(rv / ang, dtype=np.float64))
            angles.append(ang)
    _CUBE24_AXES_ANGLES_BASE = (np.stack(axes, axis=0), np.asarray(angles, dtype=np.float64))
    return _CUBE24_AXES_ANGLES_BASE


def _cube24_rotation_matrices() -> np.ndarray:
    """
    Return the 24 proper signed-permutation rotations (cube rotation group).

    These are applied as perturbations around the PCA seed frame (R_delta @ R_base),
    which gives a deterministic PCA-aligned coarse rotation search without Fibonacci
    sampling.
    """
    global _CUBE24_ROTATIONS_BASE
    if _CUBE24_ROTATIONS_BASE is not None:
        return _CUBE24_ROTATIONS_BASE

    mats: list[np.ndarray] = []
    I = np.eye(3, dtype=np.float64)
    for perm in itertools.permutations(range(3)):
        P = I[:, perm]
        for signs in itertools.product([-1.0, 1.0], repeat=3):
            M = P * np.asarray(signs, dtype=np.float64)[None, :]
            if np.linalg.det(M) > 0.5:  # proper rotations only (det=+1)
                mats.append(np.asarray(M, dtype=np.float64))
    if len(mats) != 24:
        _fallback_notice(
            "cube24_count_unexpected",
            f"Expected 24 cube rotations, got {len(mats)}; falling back to identity only.",
        )
        _CUBE24_ROTATIONS_BASE = np.eye(3, dtype=np.float64).reshape(1, 3, 3)
        return _CUBE24_ROTATIONS_BASE

    _CUBE24_ROTATIONS_BASE = np.stack(mats, axis=0)
    return _CUBE24_ROTATIONS_BASE


def global_rotation_align(
    mapinfo: MapInfo,
    model_pts_world: np.ndarray,
    max_dim: int = GLOBAL_SEARCH_MAX_DIM,
    z_thresh: float = GLOBAL_SEARCH_Z,
    radius_ang: float = GLOBAL_SEARCH_RADIUS_A,
    n_rot: int = GLOBAL_SEARCH_ROTATIONS,
    seed: int = GLOBAL_SEARCH_SEED,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Coarse global alignment (single method):
      - Downsample map to max_dim
      - Threshold map in Z-score space (>= z_thresh)
      - Sample 24 PCA-aligned cube rotations; for each, translate by map/model COM
      - Choose best base rotation by coarse score on the downsampled grid
      - Spin this base around the model's own PCA axis at model COM
      - Refine translation around the best spun seed
      - Coarse score uses IoU * max(meanZ, 0)
    Returns (R_best, t_best, best_iou) in world coordinates.
    """
    # Downsample map + mask
    m = mapinfo.fit_map
    m_ds = _downsample_to_maxdim(m, max_dim)
    mZ_ds = (m_ds - mapinfo.mu) / mapinfo.sd
    map_mask = (mZ_ds >= float(z_thresh))
    map_sum = float(np.count_nonzero(map_mask))
    if map_sum <= 0:
        _fallback_notice(
            "global_align_empty_map_mask",
            "Coarse alignment map mask is empty at current z-threshold; returning identity pose.",
        )
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 0.0

    # Scales to downsampled grid
    scale_z = m.shape[0] / m_ds.shape[0]
    scale_y = m.shape[1] / m_ds.shape[1]
    scale_x = m.shape[2] / m_ds.shape[2]
    scale_mean = float(np.mean([scale_z, scale_y, scale_x]))

    # Occupancy radius in downsampled voxels
    vs_mean = float((mapinfo.sx + mapinfo.sy + mapinfo.sz) / 3.0)
    r_vox = float(radius_ang) / max(vs_mean, 1e-6)
    rad_ds = r_vox / max(scale_mean, 1e-6)

    # Build map/model PCA frames (with skew-sign rule) and initialize a PCA->PCA seed pose.
    map_pts, map_wts, map_com = sample_map_points(mapinfo, stride=3, target_min=2500)
    model_com = model_pts_world.mean(axis=0)
    map_frame: Optional[np.ndarray] = None
    model_frame: Optional[np.ndarray] = None
    model_pc1_local: Optional[np.ndarray] = None
    rotset_target_axis: Optional[np.ndarray] = None
    R_base = np.eye(3, dtype=np.float64)
    try:
        map_frame, map_center = _build_signed_pca_frame(map_pts, weights=map_wts)
        map_com = map_center
        rotset_target_axis = map_frame[:, 0].copy()
    except Exception:
        _fallback_notice(
            "global_align_map_pca_failed",
            "Map PCA frame construction failed; using non-PCA base rotation fallback.",
        )
        map_frame = None
    try:
        model_frame, model_center = _build_signed_pca_frame(model_pts_world, weights=None)
        model_com = model_center
        model_pc1_local = model_frame[:, 0].copy()
    except Exception:
        _fallback_notice(
            "global_align_model_pca_failed",
            "Model PCA frame construction failed; using non-PCA base rotation fallback.",
        )
        model_frame = None

    if GLOBAL_SEARCH_USE_PCA_SEED and map_frame is not None and model_frame is not None:
        # PCA-to-PCA rigid seed: aligns COM + full principal frame (PC1/PC2 => spin included).
        R_base = map_frame @ model_frame.T
        if np.linalg.det(R_base) < 0.0:
            _fallback_notice(
                "global_align_seed_handedness_fix",
                "PCA seed rotation had negative determinant; applying handedness fix.",
            )
            # Safety guard for numerical edge cases.
            fix = model_frame.copy()
            fix[:, 1] *= -1.0
            fix[:, 2] = _safe_normalize(np.cross(fix[:, 0], fix[:, 1]), fix[:, 2])
            fix[:, 1] = _safe_normalize(np.cross(fix[:, 2], fix[:, 0]), fix[:, 1])
            R_base = map_frame @ fix.T
    elif GLOBAL_SEARCH_USE_PCA_SEED:
        _fallback_notice(
            "global_align_seed_identity",
            "PCA seeding enabled but unavailable; using identity base rotation before cube-rotation perturbations.",
        )

    # Fixed 24 signed-permutation rotations (cube rotation group), oriented to map PCA PC1
    # and applied around the PCA seed. This preserves deterministic coarse coverage while
    # matching the previous PCA-oriented rotation-set behavior.
    # `n_rot`/`seed` are retained in the signature for API compatibility but are no longer used.
    cube_axes, cube_angles = _cube24_rotation_axis_angles()
    if GLOBAL_SEARCH_USE_PCA_SEED and rotset_target_axis is not None:
        cube_axes = _align_rotation_axes_to_target(cube_axes, rotset_target_axis)
    rotations_delta = np.stack(
        [R.from_rotvec(axis * angle).as_matrix() for axis, angle in zip(cube_axes, cube_angles)],
        axis=0,
    )

    # Preselect points for occupancy + alignment scoring to keep this coarse stage fast
    n_occ = model_pts_world.shape[0]
    if n_occ > 5000:
        idx = np.linspace(0, n_occ - 1, 5000, dtype=int)
        model_pts_occ = model_pts_world[idx]
    else:
        model_pts_occ = model_pts_world

    n_pts = model_pts_world.shape[0]
    if n_pts > 2000:
        idx = np.linspace(0, n_pts - 1, 2000, dtype=int)
        model_pts_score = model_pts_world[idx]
    else:
        model_pts_score = model_pts_world

    # Local spin axis in model frame (principal axis 1).
    if model_pc1_local is not None:
        spin_axis_local = model_pc1_local.copy()
    else:
        _fallback_notice(
            "global_align_spin_axis_default",
            "Model PC1 unavailable; using default spin axis [0,0,1].",
        )
        spin_axis_local = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    spin_axis_local /= (np.linalg.norm(spin_axis_local) + 1e-12)

    # Rotate about model COM, then place at map COM.
    model_pts_occ_ctr = model_pts_occ - model_com
    model_pts_score_ctr = model_pts_score - model_com

    # Translation offsets around COM (coarse grid)
    base_step_ang = max(vs_mean * scale_mean, 1.0) * float(GLOBAL_SEARCH_TRANS_STEP_SCALE)
    r_fine = int(GLOBAL_SEARCH_TRANS_RANGE)
    offsets_fine = [i * base_step_ang for i in range(-r_fine, r_fine + 1)]
    trans_offsets_fine = np.array(list(itertools.product(offsets_fine, offsets_fine, offsets_fine)), dtype=np.float64)

    use_coarse = bool(GLOBAL_SEARCH_COARSE_TRANSLATION)
    r_coarse = int(GLOBAL_SEARCH_COARSE_RANGE)
    coarse_mult = float(GLOBAL_SEARCH_COARSE_MULT)
    if use_coarse:
        step_coarse = base_step_ang * coarse_mult
        offsets_coarse = [i * step_coarse for i in range(-r_coarse, r_coarse + 1)]
        trans_offsets_coarse = np.array(list(itertools.product(offsets_coarse, offsets_coarse, offsets_coarse)), dtype=np.float64)
    else:
        trans_offsets_coarse = None

    def _score_pose(pts_world: np.ndarray, pts_world_score: Optional[np.ndarray]):
        fz, fy, fx = mapinfo.world_to_array_frac_many(pts_world)
        pts_zyx_ds = np.column_stack([fz/scale_z, fy/scale_y, fx/scale_x])
        model_occ = _model_occupancy_from_points(m_ds.shape, pts_zyx_ds, rad_ds)
        model_sum = float(np.count_nonzero(model_occ))
        if model_sum <= 0:
            return float("-inf"), 0.0
        inter = float(np.logical_and(map_mask, model_occ > 0).sum())
        union = map_sum + model_sum - inter
        iou = (inter / union) if union > 0 else 0.0
        mean_z = _score_meanZ(mapinfo, pts_world_score)
        score = iou * max(mean_z, 0.0)
        return score, iou

    # First pass: score each rotation at COM translation only.
    rot_scores: list[tuple[float, float, np.ndarray, np.ndarray]] = []
    for R_delta in rotations_delta:
        Rm = R_delta @ R_base
        t0 = map_com - (Rm @ model_com)
        pts_world = (Rm @ model_pts_occ_ctr.T).T + map_com
        pts_world_score = (Rm @ model_pts_score_ctr.T).T + map_com if model_pts_score is not None else None
        score, iou = _score_pose(pts_world, pts_world_score)
        rot_scores.append((score, iou, Rm, t0))

    # Pick best base candidate by coarse score.
    if not rot_scores:
        _fallback_notice(
            "global_align_no_rot_scores",
            "No coarse rotation scores produced; returning identity pose.",
        )
        return np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 0.0
    _, _, R_seed, t0_seed = max(rot_scores, key=lambda x: (x[0], x[1]))

    # Spin around model's own axis (about model COM) for this score-best seed.
    spin_steps = max(1, int(GLOBAL_SEARCH_SELF_SPIN_STEPS))
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
        R_spin = R.from_rotvec(axis_world * ang).as_matrix()
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

    # Translation refinement for the spun seed
    best_iou = float("-inf")
    best_score = float("-inf")
    best_R = best_spin_R
    best_t = best_spin_t0
    pts_rot = best_spin_pts_rot
    pts_rot_score = best_spin_pts_rot_score

    # Coarse translation sweep (optional)
    delta_seed = np.zeros(3, dtype=np.float64)
    if trans_offsets_coarse is not None:
        local_best = (float("-inf"), 0.0, delta_seed)
        for delta in trans_offsets_coarse:
            pts_world = pts_rot + delta
            pts_world_score = pts_rot_score + delta if pts_rot_score is not None else None
            score, iou = _score_pose(pts_world, pts_world_score)
            if score > local_best[0]:
                local_best = (score, iou, delta)
        delta_seed = local_best[2]

    # Fine translation sweep around best coarse
    for delta in trans_offsets_fine:
        delta_total = delta_seed + delta
        pts_world = pts_rot + delta_total
        pts_world_score = pts_rot_score + delta_total if pts_rot_score is not None else None
        score, iou = _score_pose(pts_world, pts_world_score)
        if score > best_score:
            best_score = score
            best_iou = iou
            best_R = best_spin_R
            best_t = best_spin_t0 + delta_total

    return best_R, best_t, best_iou

# ====================== Exported API =============================
def fit_model_and_score(
    map_path: Path | str,
    model_path: Path | str,
    out_pdb_path: Path | str | None = None,
    recommended_contour_level: Optional[float] = None,
) -> Tuple[Optional[float], Optional[Path]]:
    """
    Fit `model_path` into `map_path` (rigid body), evaluate VOF and mean-Z, and write the fitted structure.
    Returns `(score, out_path)` where `score = raw_vof * max(mean_z, 0)`.
    On any exception, returns `(None, None)` and
    cleans temporary outputs.

    """
    out_final: Optional[Path] = None
    out_tmp: Optional[Path] = None
    try:
        map_path = Path(map_path); model_path = Path(model_path)
        if not map_path.exists(): raise FileNotFoundError(f"Map not found: {map_path}")
        if not model_path.exists(): raise FileNotFoundError(f"Model not found: {model_path}")

        # Fixed parameters BEFORE building MapInfo
        vof_max_dim = VOF_MAX_DIM

        # Build/reuse map variants.
        # Default behavior: denoise-only branch.
        map_variants: list[tuple[str, MapInfo]] = []
        if GLOBAL_MAP_DENOISE_ENABLED and GLOBAL_MAP_DENOISE_SIGMA > 0:
            den_tag = f"gaussian_sigma{GLOBAL_MAP_DENOISE_SIGMA:g}"
            if GLOBAL_MAP_DENOISE_CENTER_BLOB_ONLY:
                den_tag += "_center_blob"
            den_variant = (
                den_tag,
                _get_cached_mapinfo(
                    map_path,
                    denoise_sigma=GLOBAL_MAP_DENOISE_SIGMA,
                    denoise_center_blob_only=GLOBAL_MAP_DENOISE_CENTER_BLOB_ONLY,
                    denoise_center_blob_level=recommended_contour_level,
                    denoise_center_blob_z_fallback=GLOBAL_MAP_DENOISE_CENTER_BLOB_Z_FALLBACK,
                    denoise_center_blob_cluster_disconnected=GLOBAL_MAP_DENOISE_CENTER_BLOB_CLUSTER_DISCONNECTED,
                    denoise_center_blob_link_factor=GLOBAL_MAP_DENOISE_CENTER_BLOB_LINK_FACTOR,
                    denoise_center_blob_center_factor=GLOBAL_MAP_DENOISE_CENTER_BLOB_CENTER_FACTOR,
                    denoise_center_blob_min_component_vox=GLOBAL_MAP_DENOISE_CENTER_BLOB_MIN_COMPONENT_VOX,
                ),
            )
            if GLOBAL_MAP_USE_DENOISED_ONLY:
                map_variants = [den_variant]
            else:
                map_variants = [("raw", _get_cached_mapinfo(map_path, denoise_sigma=0.0)), den_variant]
        else:
            map_variants = [("raw", _get_cached_mapinfo(map_path, denoise_sigma=0.0))]

        # Load model coords (representative atoms) for alignment
        st, model_coords_full = read_model_coords(model_path)
        if VOF_SCORE_ATOMS == "backbone":
            score_coords_full = _collect_backbone_coords(st)
        elif VOF_SCORE_ATOMS == "heavy":
            score_coords_full = _collect_non_h_coords(st)
        else:
            score_coords_full = model_coords_full
        if score_coords_full.size == 0:
            score_coords_full = model_coords_full

        # Use the exact same (full) scoring atom set for local refinement and final scoring.
        model_pts_score = score_coords_full

        # Use scoring atoms for global search to better match density
        align_pts = score_coords_full if score_coords_full.size else model_coords_full

        def _refine_and_score(mapinfo: MapInfo, R_init: np.ndarray, t_init: np.ndarray):
            longest = max(mapinfo.longest_extent, 1e-6)
            trans_bound_ang = GLOBAL_JOINT_TRANS_FACTOR * longest
            R_best, t_best, _ = joint_powell_refine(
                mapinfo,
                model_pts_score,
                R_init,
                t_init,
                trans_bound_ang=trans_bound_ang,
            )
            pts_world = (R_best @ score_coords_full.T).T + t_best
            vof, vof_meta = compute_vof_optimized(
                mapinfo,
                pts_world,
                max_dim=vof_max_dim,
                recommended_contour_level=recommended_contour_level,
            )
            mean_z = _score_meanZ(mapinfo, pts_world)
            score_used = _score_from_vof_meanz(float(vof), mean_z)

            return {
                "R": R_best,
                "t": t_best,
                "vof": float(vof),
                "vof_meta": vof_meta,
                "score": float(score_used),
                "mean_z": mean_z,
            }

        best = None
        for variant_name, mapinfo in map_variants:
            R0, t0, _ = global_rotation_align(
                mapinfo,
                align_pts,
                max_dim=GLOBAL_SEARCH_MAX_DIM,
                z_thresh=GLOBAL_SEARCH_Z,
                radius_ang=GLOBAL_SEARCH_RADIUS_A,
                n_rot=GLOBAL_SEARCH_ROTATIONS,
                seed=GLOBAL_SEARCH_SEED,
            )
            cand = _refine_and_score(mapinfo, R0, t0)
            cand["map_variant"] = variant_name
            if best is None or cand["score"] > best["score"]:
                best = cand
        if best is None:
            raise RuntimeError("No fit result produced for any map variant.")

        R_best = best["R"]
        t_best = best["t"]
        vof_meta = best["vof_meta"]
        score_used = best["score"]
        if vof_meta and vof_meta.get("radius_ang") is not None:
            r_sel = vof_meta.get("radius_ang")
            z_sel = vof_meta.get("threshold_z")
            thr_val = vof_meta.get("threshold_value")
            constrained = vof_meta.get("constrained")
            map_variant = str(best.get("map_variant", "raw"))
            msg = f"[fit_v5] map={map_variant}; VOF params: radius={r_sel:.2f} A, z={z_sel:.2f}, thr={thr_val:.4g}"
            if constrained is False:
                _fallback_notice(
                    "fit_vof_unconstrained",
                    "Final VOF selection used unconstrained fallback candidate.",
                )
                msg += " (fallback unconstrained)"
            print(msg)

        # Write fitted structure (PDB or CIF) to a temp, then atomically move
        if out_pdb_path is None:
            out_pdb_path = model_path.with_name(model_path.stem + "_fitted_v5.pdb")
        out_final = Path(out_pdb_path)
        out_final.parent.mkdir(parents=True, exist_ok=True)
        out_tmp = out_final.with_suffix(out_final.suffix + ".tmp")

        apply_transform_to_structure(st, R_best, t_best)
        suffix = out_final.suffix.lower()
        if suffix in (".cif", ".mmcif"):
            doc = st.make_mmcif_document()
            doc.write_file(str(out_tmp))
        else:
            st.write_pdb(str(out_tmp))
        if not out_tmp.exists() or out_tmp.stat().st_size == 0:
            raise RuntimeError("Temporary fitted structure was not written correctly.")
        os.replace(str(out_tmp), str(out_final))

        return float(score_used), out_final

    except Exception as e:
        print(f"[fit_v5] Error during fit: {e}")
        try:
            if out_tmp and Path(out_tmp).exists():
                Path(out_tmp).unlink(missing_ok=True)
        except Exception: pass
        return None, None
