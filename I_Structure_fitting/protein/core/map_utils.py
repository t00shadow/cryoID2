"""
Map I/O, geometry, and scoring helpers for the rawfield aligner.

Wraps MRC volume reading, world ↔ array coordinate transforms, vectorized
trilinear interpolation, structure I/O via gemmi, model occupancy rasterization,
and the joint Powell refinement used by ``rawfield.run_raw_field``. All
geometry-aware paths go through :class:`MapInfo`, which normalizes non-standard
map axis orderings and absorbs origin/nxstart conventions so the alignment code
can pretend every map is a simple (z, y, x)-indexed array in world units.
"""
from __future__ import annotations

import math
import hashlib
import os
import shutil
from pathlib import Path
from typing import Any

import gemmi
import mrcfile
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as SciRot


# Per-process LRU caches; bounded by simple max-size logic in get_mapinfo.
_MAPINFO_CACHE: dict[tuple[Path, float], tuple[int, int, "MapInfo"]] = {}
_SPHERE_OFFSETS_CACHE: dict[tuple[int, float], np.ndarray] = {}
_NOTICE_SEEN: set[str] = set()
def _default_local_cache_dir() -> Path:
    env_tmp = os.environ.get("CRYOID_STRUCT_FETCH_TMP_DIR")
    if env_tmp:
        return Path(env_tmp).expanduser() / "mapinfo_local_cache"
    return Path(__file__).resolve().parents[1] / ".mapinfo_local_cache"


_MAPINFO_LOCAL_CACHE = _default_local_cache_dir()
_STRUCTURE_WRITE_TMP_DIR: Path | None = None


def configure_local_cache_dir(cache_dir: Path | str) -> None:
    """Route local map staging into a run-local temp dir."""
    global _MAPINFO_LOCAL_CACHE
    _MAPINFO_LOCAL_CACHE = Path(cache_dir).expanduser().resolve()


def configure_structure_write_tmp_dir(tmp_dir: Path | str | None) -> None:
    """Route atomic structure-write temp files into a run-local temp dir."""
    global _STRUCTURE_WRITE_TMP_DIR
    _STRUCTURE_WRITE_TMP_DIR = Path(tmp_dir).expanduser().resolve() if tmp_dir else None


def clear_per_case_caches() -> None:
    """Drop caches that hold case-specific data; called between cases.

    Clears :data:`_MAPINFO_CACHE` (the real cryo-EM MapInfo for the just-
    finished case is never reused by the next case). Leaves
    :data:`_SPHERE_OFFSETS_CACHE` (tiny, keyed by radius which is constant
    across cases) and :data:`_NOTICE_SEEN` (one-time warnings should stay
    suppressed across cases) intact.
    """
    _MAPINFO_CACHE.clear()


def clear_caches() -> None:
    """Drop every module-level cache. Called by pipeline.run_csv at end of run.

    The end-of-run wipe is broader than :func:`clear_per_case_caches`: it also
    clears the sphere-offset cache and the one-time-notice set, so a fresh
    Python process starts with absolutely no residual state.
    """
    _MAPINFO_CACHE.clear()
    _SPHERE_OFFSETS_CACHE.clear()
    _NOTICE_SEEN.clear()


def _notice(tag: str, message: str, *, once: bool = True) -> None:
    """Print a one-time map-loader warning unless ``once=False``."""
    if once and tag in _NOTICE_SEEN:
        return
    _NOTICE_SEEN.add(tag)
    print(f"[struct_fetch_new_method][map] {message}", flush=True)


def _local_stage_path_for_map(map_path: Path) -> Path:
    digest = hashlib.sha1(str(map_path).encode("utf-8")).hexdigest()[:12]
    stem = map_path.stem.replace(".", "_")
    return _MAPINFO_LOCAL_CACHE / f"{stem}_{digest}{map_path.suffix}"


def _load_mapinfo_with_local_fallback(map_path: Path) -> MapInfo:
    staged = _local_stage_path_for_map(map_path)
    last_exc: Exception | None = None
    for use_staged in (False, True):
        current = staged if use_staged else map_path
        try:
            if use_staged:
                _MAPINFO_LOCAL_CACHE.mkdir(parents=True, exist_ok=True)
                if (not staged.exists()) or staged.stat().st_size <= 0:
                    _notice("stage_map", f"Staging map locally after direct read failed: {map_path}", once=False)
                    shutil.copyfile(str(map_path), str(staged))
            with mrcfile.open(str(current), mode="r", permissive=True) as mrc:
                return MapInfo(mrc)
        except Exception as exc:
            last_exc = exc
            if use_staged and staged.exists():
                staged.unlink(missing_ok=True)
    if last_exc is None:
        raise RuntimeError(f"Failed to read map: {map_path}")
    raise last_exc


def _safe_get_origin(mrc: object) -> tuple[float, float, float]:
    """Return the explicit MRC world-space origin as an ``(x, y, z)`` tuple.

    ``mrcfile`` exposes ``header.origin`` as a structured record with named
    ``x``/``y``/``z`` fields.  Casting that record directly to a float array
    raises ``TypeError`` and previously caused non-zero origins to be silently
    replaced by zero.  Read named fields first, while retaining compatibility
    with header implementations that expose scalar ``xorigin`` attributes or
    a plain three-element array.
    """
    header = getattr(mrc, "header")
    try:
        values = (float(header.xorigin), float(header.yorigin), float(header.zorigin))
    except Exception:
        try:
            origin = header.origin
            values = (float(origin.x), float(origin.y), float(origin.z))
        except Exception:
            try:
                origin = header.origin
                values = (float(origin["x"]), float(origin["y"]), float(origin["z"]))
            except Exception:
                try:
                    arr = np.asarray(header.origin).ravel()
                    if arr.size < 3:
                        return 0.0, 0.0, 0.0
                    values = (float(arr[0]), float(arr[1]), float(arr[2]))
                except Exception:
                    return 0.0, 0.0, 0.0
    if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        return 0.0, 0.0, 0.0
    return values


def _safe_get_step(mrc: object) -> tuple[float, float, float]:
    """Return per-axis voxel size (sx, sy, sz) handling both ``voxel_size`` and ``cella/mx`` MRC variants."""
    try:
        voxel_size = getattr(mrc, "voxel_size")
        return float(voxel_size.x), float(voxel_size.y), float(voxel_size.z)
    except Exception:
        header = getattr(mrc, "header")
        cella = np.asarray(getattr(header, "cella", [1.0, 1.0, 1.0]), dtype=np.float64).ravel()
        nx, ny, nz = int(header.nx), int(header.ny), int(header.nz)
        mx, my, mz = int(getattr(header, "mx", 0) or nx), int(getattr(header, "my", 0) or ny), int(getattr(header, "mz", 0) or nz)

        def _axis_step(cell: float, samples: int, fallback_samples: int) -> float:
            denom = samples if samples > 0 else fallback_samples
            return abs(float(cell) / float(max(denom, 1)))

        cellx = float(cella[0]) if cella.size > 0 else 1.0
        celly = float(cella[1]) if cella.size > 1 else 1.0
        cellz = float(cella[2]) if cella.size > 2 else 1.0
        return _axis_step(cellx, mx, nx), _axis_step(celly, my, ny), _axis_step(cellz, mz, nz)


class MapInfo:
    """MRC map geometry and vectorized interpolation for alignment scoring."""

    def __init__(self, mrc: object):
        raw = np.asarray(getattr(mrc, "data"), dtype=np.float32)
        if raw.ndim == 4 and raw.shape[0] == 1:
            raw = raw[0]
        if raw.ndim != 3:
            raise ValueError(f"Expected a 3D map volume, got shape {raw.shape}")
        if not np.isfinite(raw).all():
            finite = np.isfinite(raw)
            fill = float(np.nanmedian(raw[finite])) if np.any(finite) else 0.0
            raw = np.where(finite, raw, fill).astype(np.float32, copy=False)

        self.raw = np.asarray(raw, dtype=np.float32, order="C")
        self.nz, self.ny, self.nx = self.raw.shape
        self.sx, self.sy, self.sz = _safe_get_step(mrc)

        header = getattr(mrc, "header")
        self.nxstart = int(getattr(header, "nxstart", 0) or 0)
        self.nystart = int(getattr(header, "nystart", 0) or 0)
        self.nzstart = int(getattr(header, "nzstart", 0) or 0)
        self.mapc = int(getattr(header, "mapc", 1) or 1)
        self.mapr = int(getattr(header, "mapr", 2) or 2)
        self.maps = int(getattr(header, "maps", 3) or 3)
        if {self.mapc, self.mapr, self.maps} != {1, 2, 3}:
            _notice("axis_order", f"Non-standard map axis order ({self.mapc},{self.mapr},{self.maps}); using (1,2,3).")
            self.mapc, self.mapr, self.maps = 1, 2, 3

        x0, y0, z0 = _safe_get_origin(mrc)
        explicit_origin = np.asarray([x0, y0, z0], dtype=np.float64)
        # A non-zero explicit ORIGIN is the map's authoritative placement.
        # NX/NY/NZSTART is an older alternative convention and must not be
        # added on top of ORIGIN, which would double-translate such maps.
        # Retain START as a fallback for maps whose ORIGIN is zero/unset.
        if np.any(np.abs(explicit_origin) > 1e-6):
            self.eff_x0, self.eff_y0, self.eff_z0 = map(float, explicit_origin)
            self.placement_source = "origin"
        else:
            self.eff_x0 = self._start_for_world_axis(1) * self.sx
            self.eff_y0 = self._start_for_world_axis(2) * self.sy
            self.eff_z0 = self._start_for_world_axis(3) * self.sz
            self.placement_source = "start"
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
        self._coords_buf = np.empty((3, 0), dtype=np.float64)
        self._set_fit_map(self.raw)
        self.extent_x = self.sx * max(self.cx - 1, 0)
        self.extent_y = self.sy * max(self.cy - 1, 0)
        self.extent_z = self.sz * max(self.cz - 1, 0)
        self.longest_extent = float(max(self.extent_x, self.extent_y, self.extent_z))

    def _set_fit_map(self, fit_map: np.ndarray) -> None:
        arr = np.asarray(fit_map, dtype=np.float32, order="C")
        if arr.shape != self.raw.shape:
            raise ValueError(f"fit_map shape mismatch: got {arr.shape}, expected {self.raw.shape}")
        if not np.isfinite(arr).all():
            finite = np.isfinite(arr)
            fill = float(np.nanmedian(arr[finite])) if np.any(finite) else 0.0
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
        return self.eff_x0 + idx_x * self.sx, self.eff_y0 + idx_y * self.sy, self.eff_z0 + idx_z * self.sz

    def world_to_array_frac_many(self, pts: np.ndarray):
        pts = np.asarray(pts, dtype=np.float64)
        idx_x = (pts[:, 0] - self.eff_x0) / self.sx
        idx_y = (pts[:, 1] - self.eff_y0) / self.sy
        idx_z = (pts[:, 2] - self.eff_z0) / self.sz
        out = [None, None, None]
        out[self.axX] = idx_x
        out[self.axY] = idx_y
        out[self.axZ] = idx_z
        return out

    def _world_to_coords(self, pts_world: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_world, dtype=np.float64)
        n = int(pts.shape[0])
        if self._coords_buf.shape[1] < n:
            self._coords_buf = np.empty((3, n), dtype=np.float64)
        coords = self._coords_buf[:, :n]
        coords[self.axX] = (pts[:, 0] - self.eff_x0) / self.sx
        coords[self.axY] = (pts[:, 1] - self.eff_y0) / self.sy
        coords[self.axZ] = (pts[:, 2] - self.eff_z0) / self.sz
        return coords

    def interp_many(self, pts_world: np.ndarray) -> np.ndarray:
        vals = map_coordinates(self.fit_map, self._world_to_coords(pts_world), order=1, mode="nearest", cval=0.0)
        return vals.astype(np.float64, copy=False)


def get_mapinfo(map_path: Path, denoise_sigma: float = 0.0) -> MapInfo:
    """Load (and cache) the :class:`MapInfo` for ``map_path``, optionally Gaussian-smoothed.

    The cache key is ``(absolute_path, denoise_sigma)`` and tracks the file's
    ``st_mtime_ns`` so an in-place re-write of the map invalidates the entry.
    """
    map_path = Path(map_path).resolve()
    stat = map_path.stat()
    sigma = float(max(0.0, denoise_sigma))
    key = (map_path, sigma)
    cached = _MAPINFO_CACHE.get(key)
    if cached and (cached[0], cached[1]) == (int(stat.st_mtime_ns), int(stat.st_size)):
        return cached[2]
    mapinfo = _load_mapinfo_with_local_fallback(map_path)
    if sigma > 0.0:
        mapinfo._set_fit_map(gaussian_filter(mapinfo.raw, sigma=sigma).astype(np.float32, copy=False))
    _MAPINFO_CACHE[key] = (int(stat.st_mtime_ns), int(stat.st_size), mapinfo)
    return mapinfo


def read_model_coords(st_path: Path):
    """Read a structure file and return ``(gemmi.Structure, representative_coords)``.

    The representative coord per residue is CA → P → first non-hydrogen atom.
    Raises if the structure has no residues with usable atoms.
    """
    st = gemmi.read_structure(str(st_path))
    if len(st) == 0:
        raise RuntimeError(f"Empty structure: {st_path}")
    coords: list[tuple[float, float, float]] = []
    for model in st:
        for chain in model:
            for res in chain:
                atom = None
                for candidate in res:
                    if candidate.name.strip().upper() == "CA":
                        atom = candidate
                        break
                if atom is None:
                    for candidate in res:
                        if candidate.name.strip().upper() == "P":
                            atom = candidate
                            break
                if atom is None:
                    for candidate in res:
                        if candidate.element and candidate.element.name != "H":
                            atom = candidate
                            break
                if atom is not None:
                    p = atom.pos
                    coords.append((p.x, p.y, p.z))
    if not coords:
        raise RuntimeError(f"No representative atoms found in {st_path}")
    return st, np.asarray(coords, dtype=np.float64)


def collect_ca_coords(st: gemmi.Structure) -> np.ndarray:
    """Return the first CA-atom position from each residue as an Nx3 float64 array."""
    coords: list[list[float]] = []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    if atom.name.strip().upper() == "CA":
                        p = atom.pos
                        coords.append([p.x, p.y, p.z])
                        break
    return np.asarray(coords, dtype=np.float64)


def collect_non_h_coords(st: gemmi.Structure) -> np.ndarray:
    """Return all non-hydrogen atom positions as an Nx3 float64 array."""
    coords: list[tuple[float, float, float]] = []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    if atom.element and atom.element.name == "H":
                        continue
                    p = atom.pos
                    coords.append((p.x, p.y, p.z))
    return np.asarray(coords, dtype=np.float64) if coords else np.empty((0, 3), dtype=np.float64)


def collect_backbone_coords(st: gemmi.Structure) -> np.ndarray:
    """Return backbone atom positions (N/CA/C/O for proteins, P for nucleic acids); falls back to non-H atoms."""
    coords: list[tuple[float, float, float]] = []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    if atom.name.strip().upper() in {"N", "CA", "C", "O", "P"}:
                        if atom.element and atom.element.name == "H":
                            continue
                        p = atom.pos
                        coords.append((p.x, p.y, p.z))
    return np.asarray(coords, dtype=np.float64) if coords else collect_non_h_coords(st)


def apply_transform_to_structure(st: gemmi.Structure, rotation: np.ndarray, translation: np.ndarray) -> None:
    """In-place apply ``q = R p + t`` to every atom in ``st``."""
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    p = np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=np.float64)
                    q = rotation @ p + translation
                    atom.pos = gemmi.Position(float(q[0]), float(q[1]), float(q[2]))


def _extended_path(path: Path) -> Path:
    """Allow file I/O on Windows even when long paths are disabled globally."""
    if os.name != "nt":
        return path
    value = str(path.resolve())
    if value.startswith("\\\\?\\"):
        return Path(value)
    if value.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + value[2:])
    return Path("\\\\?\\" + value)


def write_transformed_structure(in_model: Path, out_model: Path, rotation: np.ndarray, translation: np.ndarray) -> None:
    """Read ``in_model``, apply ``R p + t`` to every atom, write ``out_model``.

    Output format follows the destination suffix (``.cif`` / ``.mmcif`` → mmCIF,
    otherwise PDB). Writes via a ``.tmp`` rename so a partial failure can't
    leave a half-written file in place.
    """
    st = gemmi.read_structure(str(in_model))
    apply_transform_to_structure(st, rotation, translation)
    out_model = _extended_path(out_model)
    out_model.parent.mkdir(parents=True, exist_ok=True)
    if _STRUCTURE_WRITE_TMP_DIR is not None:
        write_tmp_dir = _extended_path(_STRUCTURE_WRITE_TMP_DIR)
        write_tmp_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(str(out_model.resolve()).encode("utf-8")).hexdigest()[:12]
        tmp = write_tmp_dir / f"{out_model.stem}_{digest}{out_model.suffix}.tmp"
    else:
        tmp = out_model.with_suffix(out_model.suffix + ".tmp")
    # Gemmi's native file writers can reject Windows paths longer than MAX_PATH.
    # Serialize with Gemmi, then use Python's long-path-aware file I/O.
    if out_model.suffix.lower() in {".cif", ".mmcif"}:
        tmp.write_bytes(st.make_mmcif_document().as_string().encode("utf-8"))
    else:
        tmp.write_bytes(st.make_pdb_string().encode("utf-8"))
    if not tmp.exists() or tmp.stat().st_size <= 0:
        raise RuntimeError(f"Failed to write transformed structure: {tmp}")
    tmp.replace(out_model)


def score_mean_z(mapinfo: MapInfo, pts_world: np.ndarray) -> float:
    """Return the mean z-score of map intensities sampled at ``pts_world``."""
    vals = mapinfo.interp_many(np.asarray(pts_world, dtype=np.float64))
    return float(np.mean((vals - mapinfo.mu) / mapinfo.sd))


def downsample_to_maxdim(volume: np.ndarray, max_dim: int) -> np.ndarray:
    """Strided downsample so the longest axis fits in ``max_dim`` voxels; returns the input unchanged if already small enough."""
    z, y, x = np.asarray(volume).shape
    scale = max(1, int(np.ceil(max(z, y, x) / max(1, int(max_dim)))))
    return volume if scale == 1 else volume[::scale, ::scale, ::scale]


def bounded_powell_refine(
    rotation_seed: np.ndarray,
    translation_seed: np.ndarray,
    scorer,
    *,
    model_center: np.ndarray,
    rot_bound_deg: float,
    trans_bound_ang: float,
    maxiter: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Bounded Powell refinement of a rigid pose around a seed.

    Parameterizes the perturbation as a 6-vector (rotation-vector + translation)
    clipped to ``[-rot_bound, rot_bound]`` and ``[-trans_bound, trans_bound]``,
    keeping the seed's world-space center of mass anchored so the rotation
    happens around the model centroid rather than around the world origin.

    ``scorer(rotation, translation)`` is called with each candidate pose and
    must return a float to maximize (the helper negates internally for
    ``scipy.optimize.minimize``).

    Returns ``(rotation_best, translation_best, meta)`` where ``meta`` has
    ``score`` plus scipy's ``nit``/``nfev``/``success``/``message`` fields.
    """
    model_center = np.asarray(model_center, dtype=np.float64)
    rotation_seed = np.asarray(rotation_seed, dtype=np.float64)
    translation_seed = np.asarray(translation_seed, dtype=np.float64)
    rot_bound = math.radians(float(rot_bound_deg))
    trans_bound = abs(float(trans_bound_ang))
    lower = np.array([-rot_bound, -rot_bound, -rot_bound, -trans_bound, -trans_bound, -trans_bound], dtype=np.float64)
    upper = np.array([rot_bound, rot_bound, rot_bound, trans_bound, trans_bound, trans_bound], dtype=np.float64)
    seed_world_center = np.asarray(rotation_seed @ model_center + translation_seed, dtype=np.float64)

    def _project(v: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(v, dtype=np.float64), lower, upper)

    def _score(v: np.ndarray) -> float:
        v = _project(v)
        delta = SciRot.from_rotvec(v[:3]).as_matrix()
        rotation_try = np.asarray(delta @ rotation_seed, dtype=np.float64)
        translation_try = np.asarray(seed_world_center + v[3:] - (rotation_try @ model_center), dtype=np.float64)
        return float(scorer(rotation_try, translation_try))

    best_x = np.zeros(6, dtype=np.float64)
    best_score = _score(best_x)
    res = minimize(
        lambda v: -_score(v),
        best_x,
        method="Powell",
        bounds=list(zip(lower, upper)),
        options={"maxiter": int(maxiter), "xtol": 1e-6, "ftol": 1e-6, "disp": False},
    )
    if res.x is not None:
        candidate_x = _project(np.asarray(res.x, dtype=np.float64))
        candidate_score = _score(candidate_x)
        if candidate_score > best_score:
            best_score = candidate_score
            best_x = candidate_x
    delta = SciRot.from_rotvec(best_x[:3]).as_matrix()
    rotation_best = np.asarray(delta @ rotation_seed, dtype=np.float64)
    translation_best = np.asarray(seed_world_center + best_x[3:] - (rotation_best @ model_center), dtype=np.float64)
    meta = {
        "score": float(best_score),
        "nit": int(getattr(res, "nit", -1)),
        "nfev": int(getattr(res, "nfev", -1)),
        "success": bool(getattr(res, "success", True)),
        "message": str(getattr(res, "message", "")),
    }
    return rotation_best, translation_best, meta


def joint_powell_refine(
    mapinfo: MapInfo,
    model_pts_scoring: np.ndarray,
    rotation_seed: np.ndarray,
    translation_seed: np.ndarray,
    *,
    trans_bound_ang: float,
    maxiter: int,
    rot_bound_deg: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Refine a pose against the map mean-z score using :func:`bounded_powell_refine`.

    Returns ``(rotation_best, translation_best, best_score)``.
    """
    model_pts = np.asarray(model_pts_scoring, dtype=np.float64)
    if model_pts.ndim != 2 or model_pts.shape[0] == 0 or model_pts.shape[1] != 3:
        raise ValueError("Expected non-empty Nx3 model points for joint Powell refinement")
    model_center = np.mean(model_pts, axis=0)

    def _scorer(rotation: np.ndarray, translation: np.ndarray) -> float:
        pts_world = (rotation @ model_pts.T).T + translation
        return score_mean_z(mapinfo, pts_world)

    rotation_best, translation_best, meta = bounded_powell_refine(
        rotation_seed,
        translation_seed,
        _scorer,
        model_center=model_center,
        rot_bound_deg=rot_bound_deg,
        trans_bound_ang=trans_bound_ang,
        maxiter=maxiter,
    )
    return rotation_best, translation_best, float(meta["score"])


def model_occupancy_from_points(shape: tuple[int, int, int], pts_zyx: np.ndarray, rad_vox: float) -> np.ndarray:
    """Rasterize a binary occupancy volume by stamping a voxel sphere of radius ``rad_vox`` around each point.

    Sphere offset tables are cached per radius for speed.
    """
    occ = np.zeros(shape, dtype=np.uint8)
    if pts_zyx.size == 0:
        return occ
    zmax, ymax, xmax = shape
    radius = float(rad_vox)
    radius_i = int(max(1, np.ceil(radius)))
    key = (radius_i, round(radius, 6))
    offsets = _SPHERE_OFFSETS_CACHE.get(key)
    if offsets is None:
        offsets_raw = []
        radius_sq = radius * radius + 1e-6
        for dz in range(-radius_i, radius_i + 1):
            for dy in range(-radius_i, radius_i + 1):
                for dx in range(-radius_i, radius_i + 1):
                    if dx * dx + dy * dy + dz * dz <= radius_sq:
                        offsets_raw.append((dz, dy, dx))
        offsets = np.asarray(offsets_raw, dtype=np.int64)
        _SPHERE_OFFSETS_CACHE[key] = offsets
    idx = np.round(np.asarray(pts_zyx, dtype=np.float64)).astype(np.int64)
    all_coords = (idx[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    all_coords[:, 0] = np.clip(all_coords[:, 0], 0, zmax - 1)
    all_coords[:, 1] = np.clip(all_coords[:, 1], 0, ymax - 1)
    all_coords[:, 2] = np.clip(all_coords[:, 2], 0, xmax - 1)
    occ[all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]] = 1
    return occ


def prepare_density_inputs(mapinfo: MapInfo, model_pts_world: np.ndarray, max_dim: int):
    """Downsample the map and remap world points into the downsampled (z,y,x) grid for occupancy scoring."""
    density = mapinfo.fit_map
    density_ds = downsample_to_maxdim(density, max_dim)
    fz, fy, fx = mapinfo.world_to_array_frac_many(np.asarray(model_pts_world, dtype=np.float64))
    scale_z = density.shape[0] / density_ds.shape[0]
    scale_y = density.shape[1] / density_ds.shape[1]
    scale_x = density.shape[2] / density_ds.shape[2]
    pts_zyx_ds = np.column_stack([fz / scale_z, fy / scale_y, fx / scale_x])
    return density_ds, pts_zyx_ds, float(np.mean([scale_z, scale_y, scale_x]))


def load_structure_coords(path: Path) -> tuple[list[tuple[object, ...]], np.ndarray]:
    """Read a structure and return parallel ``(atom_id_tuples, coords)`` arrays preserving file order."""
    st = gemmi.read_structure(str(path))
    ids: list[tuple[object, ...]] = []
    coords: list[list[float]] = []
    for model_i, model in enumerate(st):
        for chain in model:
            for res in chain:
                for atom in res:
                    p = atom.pos
                    ids.append((model_i, chain.name, res.seqid.num, str(res.seqid.icode), res.name, atom.name, str(atom.altloc)))
                    coords.append([p.x, p.y, p.z])
    return ids, np.asarray(coords, dtype=np.float64)


def pose_metrics_vs_gt(gt_model: Path, test_model: Path) -> dict[str, object]:
    """Compare two structures atom-by-atom, returning direct RMSD and best-rigid-superposition RMSD.

    Requires both files to contain atoms in the same order with the same atom
    identifiers; otherwise returns ``{"ok": False, "error": ...}``.
    """
    ids_gt, coords_gt = load_structure_coords(gt_model)
    ids_test, coords_test = load_structure_coords(test_model)
    if coords_gt.shape != coords_test.shape:
        return {"ok": False, "error": f"atom_count_mismatch {coords_gt.shape} vs {coords_test.shape}"}
    delta = coords_test - coords_gt
    direct_rmsd = float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))
    centered_gt = coords_gt - coords_gt.mean(axis=0)
    centered_test = coords_test - coords_test.mean(axis=0)
    u, _, vt = np.linalg.svd(centered_test.T @ centered_gt)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1
        rotation = vt.T @ u.T
    fitted = (centered_test @ rotation.T) + coords_gt.mean(axis=0)
    superposed_delta = fitted - coords_gt
    cos_angle = max(-1.0, min(1.0, float((np.trace(rotation) - 1.0) * 0.5)))
    return {
        "ok": True,
        "same_atom_ids_order": bool(ids_gt == ids_test),
        "direct_rmsd_A": direct_rmsd,
        "max_abs_coord_delta_A": float(np.max(np.abs(delta))),
        "rmsd_after_best_rigid_superposition_A": float(np.sqrt(np.mean(np.sum(superposed_delta * superposed_delta, axis=1)))),
        "pose_rotation_deg": float(math.degrees(math.acos(cos_angle))),
        "close_to_gt_direct_rmsd_le_2A": bool(direct_rmsd <= 2.0),
    }
