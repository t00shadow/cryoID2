# =============================
# File: fit_v5.py
# =============================
#!/usr/bin/env python3
"""
Rigid-body fit MODEL+MAP, then evaluate with:
  - VOF (Volume Overlap Fraction) from a 3D voxel IoU between a thresholded map and
    a fixed-radius model occupancy (no resolution scaling).

New:
  - Fast single-map resolution estimator (estimate_resolution_single_map)
  - Resolution-aware parameters use MAP resolution with precedence, then model res.

Returns (VOF, fitted_path).

Deps: gemmi, numpy, scipy, mrcfile
  pip install gemmi numpy scipy mrcfile
"""

from pathlib import Path
import math, itertools, os
from typing import Tuple, Optional
import numpy as np
import mrcfile, gemmi
from scipy.ndimage import gaussian_filter, map_coordinates, binary_dilation, binary_erosion
from scipy.spatial.transform import Rotation as R

# -------------------- MRC helpers --------------------
def _safe_get_origin(mrc: mrcfile.mrcfile.MrcFile):
    h = mrc.header
    try:
        return float(h.xorigin), float(h.yorigin), float(h.zorigin)
    except Exception:
        try:
            arr = np.asarray(h.origin, dtype=np.float64).ravel()
            if arr.size >= 3:
                return float(arr[0]), float(arr[1]), float(arr[2])
        except Exception:
            pass
    return 0.0, 0.0, 0.0

def _safe_get_step(mrc: mrcfile.mrcfile.MrcFile):
    try:
        vs = mrc.voxel_size
        return float(vs.x), float(vs.y), float(vs.z)
    except Exception:
        h = mrc.header
        cella = np.asarray(h.cella, dtype=np.float64).ravel()
        mx, my, mz = int(getattr(h, "mx", 0) or 0), int(getattr(h, "my", 0) or 0), int(getattr(h, "mz", 0) or 0)
        nx, ny, nz = int(h.nx), int(h.ny), int(h.nz)
        sx = cella[0] / (mx if mx else nx)
        sy = cella[1] / (my if my else ny)
        sz = cella[2] / (mz if mz else nz)
        return float(sx), float(sy), float(sz)

class MapInfo:
    """MRC mapping + vectorized interpolation on a blurred copy."""
    def __init__(self, mrc: mrcfile.mrcfile.MrcFile, blur_sigma_vox: float):
        raw = mrc.data.astype(np.float32, copy=False)
        if not np.isfinite(raw).all():
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

        x0, y0, z0 = _safe_get_origin(mrc)
        self.eff_x0 = x0 + self._start_for(self.mapc) * self.sx
        self.eff_y0 = y0 + self._start_for(self.mapr) * self.sy
        self.eff_z0 = z0 + self._start_for(self.maps) * self.sz

        self.cx = self._count_for(self.mapc)
        self.cy = self._count_for(self.mapr)
        self.cz = self._count_for(self.maps)

        self.axX = self._axis_from_code(self.mapc)
        self.axY = self._axis_from_code(self.mapr)
        self.axZ = self._axis_from_code(self.maps)

        self.fit_map = gaussian_filter(self.raw, sigma=(blur_sigma_vox,)*3, mode='nearest') if blur_sigma_vox>0 else self.raw
        self.mu = float(np.mean(self.fit_map))
        self.sd = float(np.std(self.fit_map, ddof=0)) + 1e-12

        self.extent_x = self.sx * (self.cx - 1)
        self.extent_y = self.sy * (self.cy - 1)
        self.extent_z = self.sz * (self.cz - 1)
        self.longest_extent = float(max(self.extent_x, self.extent_y, self.extent_z))

    def _count_for(self, code: int) -> int:
        return {1: self.nx, 2: self.ny, 3: self.nz}[code]
    def _start_for(self, code: int) -> int:
        return {1: self.nxstart, 2: self.nystart, 3: self.nzstart}[code]
    @staticmethod
    def _axis_from_code(code: int) -> int:
        return {1: 2, 2: 1, 3: 0}[code]

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

    def interp_many(self, pts_world: np.ndarray) -> np.ndarray:
        fz, fy, fx = self.world_to_array_frac_many(pts_world)
        coords = np.vstack([fz, fy, fx])
        vals = map_coordinates(self.fit_map, coords, order=1, mode='nearest', cval=0.0)
        return vals.astype(np.float64)

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

def proper_rotation(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    S = np.eye(3)
    if np.linalg.det(U @ Vt) < 0: S[2, 2] = -1.0
    return (U @ S @ Vt).astype(np.float64)

def all_24_axis_maps():
    mats = []
    for perm in itertools.permutations(range(3)):
        P = np.eye(3)[:, perm]
        for signs in itertools.product([-1, 1], repeat=3):
            S = np.diag(signs); M = P @ S
            if np.linalg.det(M) > 0.5: mats.append(M.astype(np.float64))
    uniq = []
    for M in mats:
        if not any(np.allclose(M, N) for N in uniq):
            uniq.append(M)
    return uniq

def sample_map_points(mapinfo: MapInfo, stride: int, target_min: int):
    A = mapinfo.fit_map
    sub = A[::stride, ::stride, ::stride]
    mu, sd = float(sub.mean()), float(sub.std(ddof=0))
    thresholds = [mu + 1.5*sd, mu + 1.0*sd, np.quantile(sub, 0.98), np.quantile(sub, 0.95), mu + 0.5*sd]
    pts = wts = None

    def grid_to_world(ix, iy, iz):
        xs = mapinfo.eff_x0 + (ix if mapinfo.mapc == 1 else iy if mapinfo.mapc == 2 else iz) * (mapinfo.sx if mapinfo.mapc == 1 else mapinfo.sy if mapinfo.mapc == 2 else mapinfo.sz)
        ys = mapinfo.eff_y0 + (ix if mapinfo.mapr == 1 else iy if mapinfo.mapr == 2 else iz) * (mapinfo.sx if mapinfo.mapr == 1 else mapinfo.sy if mapinfo.mapr == 2 else mapinfo.sz)
        zs = mapinfo.eff_z0 + (ix if mapinfo.maps == 1 else iy if mapinfo.maps == 2 else iz) * (mapinfo.sx if mapinfo.maps == 1 else mapinfo.sy if mapinfo.maps == 2 else mapinfo.sz)
        return xs, ys, zs

    for th in thresholds:
        mask = sub >= th
        if int(mask.sum()) >= target_min:
            iz, iy, ix = np.nonzero(mask)
            iz *= stride; iy *= stride; ix *= stride
            vals = A[iz, iy, ix].astype(np.float64, copy=False)
            xs, ys, zs = grid_to_world(ix, iy, iz)
            pts = np.column_stack([xs, ys, zs]); wts = vals
            break

    if pts is None:
        iz, iy, ix = np.nonzero(np.ones_like(sub, dtype=bool))
        iz *= stride; iy *= stride; ix *= stride
        vals = A[iz, iy, ix].astype(np.float64, copy=False)
        xs, ys, zs = grid_to_world(ix, iy, iz)
        pts = np.column_stack([xs, ys, zs]); wts = vals

    wsum = float(wts.sum()) or 1.0
    com = (pts * (wts.reshape(-1, 1))).sum(axis=0) / wsum
    return pts, wts, com

def coarse_align_with_24(mapinfo: MapInfo, model_pts: np.ndarray, atoms_sample: int = 3000):
    mpts = model_pts[np.linspace(0, model_pts.shape[0]-1, min(model_pts.shape[0], atoms_sample), dtype=int)]
    map_pts, map_wts, map_com = sample_map_points(mapinfo, stride=3, target_min=2500)
    U_model, m_centroid = pca(mpts)
    V_map, _ = pca(map_pts, map_wts)
    candidates = all_24_axis_maps()
    best_R = None; best_t = None; best_s = -np.inf
    for A in candidates:
        R0 = proper_rotation(V_map @ A @ U_model.T)
        t0 = map_com - (R0 @ m_centroid)
        z = (mapinfo.interp_many((R0 @ mpts.T).T + t0) - mapinfo.mu) / mapinfo.sd
        s = float(np.mean(z))
        if s > best_s: best_s, best_R, best_t = s, R0, t0
    return best_R, best_t, m_centroid, map_com, best_s

# -------------------- Greedy refine --------------------
def _score_meanZ(mapinfo: MapInfo, pts_world: np.ndarray) -> float:
    vals = mapinfo.interp_many(pts_world)
    return float(np.mean((vals - mapinfo.mu) / mapinfo.sd))

def greedy_refine(mapinfo: MapInfo, model_pts_scoring: np.ndarray,
                  R_init: np.ndarray, t_init: np.ndarray,
                  rot_steps_deg: list[float], trans_steps_ang: list[float],
                  max_sweeps_per_step: int, trans_bound_ang: float) -> tuple[np.ndarray, np.ndarray, float]:
    R_best = R_init.copy(); t_best = t_init.copy()
    s_best = _score_meanZ(mapinfo, (R_best @ model_pts_scoring.T).T + t_best)

    def try_move(Rc, tc, best):
        pts = (Rc @ model_pts_scoring.T).T + tc
        s = _score_meanZ(mapinfo, pts)
        return (Rc, tc, s, True) if s > best else (R_best, t_best, best, False)

    for rstep in rot_steps_deg:
        for tstep in trans_steps_ang:
            sweeps = 0
            while sweeps < max_sweeps_per_step:
                improved = False; sweeps += 1
                for axis in [(1,0,0),(0,1,0),(0,0,1)]:
                    for sgn in (-1.0, 1.0):
                        Rstep = R.from_rotvec(np.array(axis, float)*math.radians(sgn*rstep)).as_matrix()
                        R_best, t_best, s_best, did = try_move(Rstep @ R_best, t_best, s_best)
                        improved |= did
                for ax in range(3):
                    for sgn in (-1.0,1.0):
                        t_try = t_best.copy(); t_try[ax] += sgn*tstep
                        t_try = np.clip(t_try, -abs(trans_bound_ang), +abs(trans_bound_ang))
                        R_best, t_best, s_best, did = try_move(R_best, t_try, s_best)
                        improved |= did
                if not improved: break
    return R_best, t_best, s_best

# -------------------- VOF (3D IoU) --------------------
def _downsample_to_maxdim(vol: np.ndarray, max_dim: int) -> np.ndarray:
    z, y, x = vol.shape
    scale = max(1, int(np.ceil(max([z,y,x]) / max_dim)))
    if scale == 1: return vol
    return vol[::scale, ::scale, ::scale]

def _model_occupancy_from_points(shape: tuple[int,int,int], pts_zyx: np.ndarray, rad_vox: float) -> np.ndarray:
    occ = np.zeros(shape, dtype=np.uint8)
    if pts_zyx.size == 0: return occ
    zmax,ymax,xmax = shape
    rr = int(max(1, np.ceil(rad_vox)))
    offsets = []
    for dz in range(-rr, rr+1):
        for dy in range(-rr, rr+1):
            for dx in range(-rr, rr+1):
                if (dx*dx + dy*dy + dz*dz) <= rad_vox*rad_vox + 1e-6:
                    offsets.append((dz,dy,dx))
    offsets = np.array(offsets, dtype=int)
    idx = np.round(pts_zyx).astype(int)
    for z,y,x in idx:
        zz = np.clip(z + offsets[:,0], 0, zmax-1)
        yy = np.clip(y + offsets[:,1], 0, ymax-1)
        xx = np.clip(x + offsets[:,2], 0, xmax-1)
        occ[zz,yy,xx] = 1
    return occ

def compute_vof(mapinfo: MapInfo, model_pts_world: np.ndarray,
                sphere_radius_vox: float,
                max_dim: int = 128) -> float:
    mZ = (mapinfo.fit_map - mapinfo.mu) / mapinfo.sd
    mZ_ds = _downsample_to_maxdim(mZ, max_dim)

    map_mask = mZ_ds >= 1.0

    fz, fy, fx = mapinfo.world_to_array_frac_many(model_pts_world)
    scale_z = mZ.shape[0] / mZ_ds.shape[0]
    scale_y = mZ.shape[1] / mZ_ds.shape[1]
    scale_x = mZ.shape[2] / mZ_ds.shape[2]
    pts_zyx_ds = np.column_stack([fz/scale_z, fy/scale_y, fx/scale_x])

    scale_mean = float(np.mean([scale_z, scale_y, scale_x]))
    rad_ds = sphere_radius_vox / max(scale_mean, 1e-6)
    model_occ = _model_occupancy_from_points(mZ_ds.shape, pts_zyx_ds, rad_ds)

    inter = np.logical_and(map_mask, model_occ > 0).sum()
    union = np.logical_or(map_mask, model_occ > 0).sum()
    return float(inter) / float(union) if union > 0 else 0.0

# -------------------- Resolution-aware parameterization --------------------
def _clamp(x, lo, hi): return float(min(max(x, lo), hi))

def _derive_params_from_resolution(vox_size_xyz: tuple[float,float,float],
                                   effective_resolution_A: Optional[float]):
    sx, sy, sz = vox_size_xyz
    vs_mean = float((sx + sy + sz) / 3.0)

    if effective_resolution_A and effective_resolution_A > 0:
        fwhm_A = 0.5 * effective_resolution_A
        sigma_A = fwhm_A / 2.355
        blur_sigma_vox = _clamp(sigma_A / max(vs_mean, 1e-6), 0.2, 3.0)
    else:
        blur_sigma_vox = 0.8

    return {
        "blur_sigma_vox": blur_sigma_vox,
        "sphere_radius_vox": 6.0 / max(vs_mean, 1e-6),  # fixed 5.5 Å converted to vox
        "vof_max_dim": 128,
    }

# -------------------- Fast single-map resolution estimate --------------------
def estimate_resolution_single_map(map_path: Path | str,
                                   tau: float = 1.0,
                                   max_dim: int = 256) -> Optional[float]:
    """
    Fast spectral SNR crossing on a single map:
      1) auto soft-mask
      2) FFT signal (masked) vs background (1-mask)
      3) find k where As/Ab = tau; return d = 1/k (Å)

    Returns None if no crossing found.
    """
    map_path = Path(map_path)
    try:
        with mrcfile.open(str(map_path), mode="r") as mrc:
            vol = mrc.data.astype(np.float32, copy=False)
            apix = float(mrc.voxel_size.x)  # assume cubic voxels
        # downsample (keeps runtime small)
        z,y,x = vol.shape
        scale = int(np.ceil(max(z,y,x)/max_dim))
        if scale > 1:
            vol = vol[::scale, ::scale, ::scale].copy()

        v = vol - float(vol.mean())
        thr = float(v.mean() + 0.5*v.std(ddof=0))
        hard = (v >= thr)
        hard = binary_dilation(binary_erosion(hard, iterations=1), iterations=3)
        soft = gaussian_filter(hard.astype(np.float32), sigma=3.0)
        soft -= soft.min()
        if soft.max() > 0: soft /= soft.max()
        mask = np.clip(soft, 0.0, 1.0).astype(np.float32, copy=False)

        sig = v * mask
        bkg = v * (1.0 - mask)

        F_sig = np.fft.fftn(sig)
        F_bkg = np.fft.fftn(bkg)

        nz,ny,nx = sig.shape
        kz = np.fft.fftfreq(nz, d=apix)
        ky = np.fft.fftfreq(ny, d=apix)
        kx = np.fft.fftfreq(nx, d=apix)
        KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
        kr = np.sqrt(KZ*KZ + KY*KY + KX*KX)  # Å^-1
        nbins = int(np.ceil(min(nx, ny, nz) / 2))
        edges = np.linspace(0.0, float(np.max(kr)), nbins+1, dtype=np.float64)
        centers = 0.5*(edges[:-1] + edges[1:])

        def radavg_amp(F):
            amp = np.abs(F).astype(np.float64, copy=False)
            idx = np.digitize(kr.ravel(), edges) - 1
            nb = len(edges) - 1
            s = np.bincount(idx, weights=amp.ravel(), minlength=nb).astype(np.float64)
            c = np.bincount(idx, minlength=nb).astype(np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(c>0, s/c, 0.0)

        As = radavg_amp(F_sig)
        Ab = radavg_amp(F_bkg)
        ratio = As / (Ab + 1e-12)

        above = ratio >= tau
        if not np.any(above): return None
        i = int(np.max(np.where(above)[0]))
        if i >= len(centers)-1: return None
        x0,x1 = centers[i], centers[i+1]
        y0,y1 = ratio[i], ratio[i+1]
        if y1 == y0: k_cross = x1
        else:
            t = (tau - y0)/(y1 - y0)
            k_cross = float(x0 + t*(x1 - x0))
        return (1.0/k_cross) if (k_cross and k_cross>0) else None
    except Exception:
        return None

# ====================== Exported API =============================
def fit_model_and_score(
    map_path: Path | str,
    model_path: Path | str,
    out_pdb_path: Path | str | None = None,
    model_resolution_A: Optional[float] = None,
    map_resolution_A: Optional[float] = None,
) -> Tuple[Optional[float], Optional[Path]]:
    """
    Fit `model_path` into `map_path` (rigid body), evaluate VOF, and write fitted PDB.
    Returns (VOF, out_pdb_path). On any exception, returns (None,None) and
    cleans temporary outputs.

    Resolution precedence:
      effective_resolution = map_resolution_A (if given) else model_resolution_A (if given) else defaults.
    """
    out_final: Optional[Path] = None
    out_tmp: Optional[Path] = None
    try:
        map_path = Path(map_path); model_path = Path(model_path)
        if not map_path.exists(): raise FileNotFoundError(f"Map not found: {map_path}")
        if not model_path.exists(): raise FileNotFoundError(f"Model not found: {model_path}")

        # Read voxel size to derive resolution-aware params BEFORE building MapInfo
        with mrcfile.open(str(map_path), mode="r") as mrc_probe:
            sx, sy, sz = _safe_get_step(mrc_probe)
            effective_res = map_resolution_A if (map_resolution_A and map_resolution_A>0) else model_resolution_A
            params = _derive_params_from_resolution((sx, sy, sz), effective_res)

        # Build MapInfo using derived blur sigma
        with mrcfile.open(str(map_path), mode="r") as mrc:
            mapinfo = MapInfo(mrc, blur_sigma_vox=params["blur_sigma_vox"])

        # Load model coords (representative atoms)
        st, model_coords_full = read_model_coords(model_path)

        # Coarse align + refine
        R0, t0, _, _, _ = coarse_align_with_24(mapinfo, model_coords_full, atoms_sample=3000)
        if model_coords_full.shape[0] > 2500:
            idx = np.linspace(0, model_coords_full.shape[0]-1, 2500, dtype=int)
            model_pts_score = model_coords_full[idx]
        else:
            model_pts_score = model_coords_full

        longest = max(mapinfo.longest_extent, 1e-6)
        trans_bound_ang = 0.60 * longest
        R_best, t_best, _ = greedy_refine(
            mapinfo, model_pts_score, R0, t0,
            rot_steps_deg=[4.0, 2.0, 1.0],
            trans_steps_ang=[2.0, 1.0, 0.5],
            max_sweeps_per_step=2,
            trans_bound_ang=trans_bound_ang
        )

        # Random restarts (choose by internal mean-Z)
        rng = np.random.default_rng(0)
        s_best = _score_meanZ(mapinfo, (R_best @ model_pts_score.T).T + t_best)
        for _ in range(3):
            axis = rng.normal(size=3); n = np.linalg.norm(axis); axis = axis/n if n else np.array([1.0,0.0,0.0])
            ang = math.radians(rng.uniform(-20.0, 20.0))
            Rrand = R.from_rotvec(axis * ang).as_matrix()
            ttry = t0 + rng.uniform(-0.08*longest, 0.08*longest, size=3)
            Rc, tc, sc = greedy_refine(
                mapinfo, model_pts_score, Rrand @ R0, ttry,
                rot_steps_deg=[4.0, 2.0, 1.0],
                trans_steps_ang=[2.0, 1.0, 0.5],
                max_sweeps_per_step=2,
                trans_bound_ang=trans_bound_ang
            )
            if sc > s_best:
                R_best, t_best, s_best = Rc, tc, sc

        # Evaluate on final placement FIRST (avoid writing files if scoring fails)
        pts_world = (R_best @ model_coords_full.T).T + t_best
        vof = compute_vof(mapinfo, pts_world,
                          sphere_radius_vox=params["sphere_radius_vox"],
                          max_dim=params["vof_max_dim"])

        # Write PDB atom coordinates to a temp, then atomically move
        if out_pdb_path is None:
            out_pdb_path = model_path.with_name(model_path.stem + "_fitted_v5.pdb")
        out_final = Path(out_pdb_path)
        out_final.parent.mkdir(parents=True, exist_ok=True)
        out_tmp = out_final.with_suffix(out_final.suffix + ".tmp")

        apply_transform_to_structure(st, R_best, t_best)
        st.write_pdb(str(out_tmp))
        if not out_tmp.exists() or out_tmp.stat().st_size == 0:
            raise RuntimeError("Temporary fitted PDB was not written correctly.")
        if out_final.exists():
            try: out_final.unlink()
            except Exception: pass
        os.replace(str(out_tmp), str(out_final))

        return float(vof), out_final

    except Exception as e:
        print(f"[fit_v5] Error during fit: {e}")
        try:
            if out_tmp and Path(out_tmp).exists():
                Path(out_tmp).unlink(missing_ok=True)
        except Exception: pass
        try:
            if out_final and Path(out_final).exists():
                Path(out_final).unlink(missing_ok=True)
        except Exception: pass
        return None, None
