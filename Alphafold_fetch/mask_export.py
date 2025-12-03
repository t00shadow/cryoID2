#!/usr/bin/env python3
"""
Export binary masks (map mask and model occupancy mask) as MRC files.

Example:
  python Alphafold_fetch/mask_export.py \
    --map Alphafold_fetch/test_files/emd_29659.map \
    --model Alphafold_fetch/fitted_models_v6/CDABP0047__PDB_1U8F_fitted_v5.pdb \
    --radius-ang 3.5 \
    --out-dir Alphafold_fetch/masks

Outputs:
  - <out_dir>/<model_stem>__map_mask.mrc      (thresholded map Z>=1.0)
  - <out_dir>/<model_stem>__model_mask.mrc    (voxelized model occupancy)
"""

import argparse
from pathlib import Path
import numpy as np
import mrcfile

from fit_v5 import (
    MapInfo,
    read_model_coords,
    _safe_get_step,
    _derive_params_from_resolution,
    _model_occupancy_from_points,
)

# ---- Hardcoded defaults (edit these to your preferred map and model) ----
DEFAULT_MAP_PATH = Path("Alphafold_fetch/test_files/emd_29659.map")
DEFAULT_MODEL_PATH = Path("Alphafold_fetch/fitted_models_v6/CDABP0047__PDB_8G17_fitted_v5.pdb")


def build_mapinfo(map_path: Path, map_resolution_A: float | None) -> tuple[MapInfo, float, dict]:
    with mrcfile.open(str(map_path), mode="r") as mrc:
        vs = _safe_get_step(mrc)
        params = _derive_params_from_resolution(vs, map_resolution_A)
        mapinfo = MapInfo(mrc, blur_sigma_vox=params["blur_sigma_vox"])
    vs_mean = float(sum(vs) / 3.0)
    return mapinfo, vs_mean, params


def write_mrc(path: Path, data: np.ndarray, voxel_size: tuple[float, float, float]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with mrcfile.new(str(path), overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32, copy=False))
        mrc.voxel_size = voxel_size


def make_masks(mapinfo: MapInfo, model_coords_world: np.ndarray, radius_ang: float, vs_mean: float):
    mZ = (mapinfo.fit_map - mapinfo.mu) / mapinfo.sd
    map_mask = (mZ >= 1.0).astype(np.uint8)

    fz, fy, fx = mapinfo.world_to_array_frac_many(model_coords_world)
    pts_zyx = np.column_stack([fz, fy, fx])
    r_vox = radius_ang / max(vs_mean, 1e-6)
    model_mask = _model_occupancy_from_points(map_mask.shape, pts_zyx, r_vox)
    return map_mask, model_mask


def main():
    parser = argparse.ArgumentParser(description="Generate binary map/model masks as MRC files.")
    parser.add_argument("--map", default=str(DEFAULT_MAP_PATH), type=str, help="Cryo-EM map path")
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH), type=str, help="Fitted PDB/CIF path (aligned to the map)")
    parser.add_argument("--radius-ang", default=3.5, type=float, help="Sphere radius in Angstroms for model voxelization")
    parser.add_argument("--map-res", default=None, type=float, help="Map resolution in Angstroms (optional override)")
    parser.add_argument("--out-dir", default="Alphafold_fetch/masks", type=str, help="Output directory for masks")
    args = parser.parse_args()

    map_path = Path(args.map)
    model_path = Path(args.model)
    out_dir = Path(args.out_dir)

    if not map_path.exists():
        raise SystemExit(f"Map not found: {map_path}")
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    mapinfo, vs_mean, params = build_mapinfo(map_path, args.map_res)
    _, coords = read_model_coords(model_path)

    map_mask, model_mask = make_masks(mapinfo, coords, args.radius_ang, vs_mean)

    stem = model_path.stem
    write_mrc(out_dir / f"{stem}__map_mask.mrc", map_mask, (mapinfo.sx, mapinfo.sy, mapinfo.sz))
    write_mrc(out_dir / f"{stem}__model_mask.mrc", model_mask, (mapinfo.sx, mapinfo.sy, mapinfo.sz))

    print(f"Saved map mask to   {out_dir / f'{stem}__map_mask.mrc'}")
    print(f"Saved model mask to {out_dir / f'{stem}__model_mask.mrc'}")
    print(f"Used radius={args.radius_ang:.2f} Å (voxel ~{args.radius_ang / max(vs_mean, 1e-6):.2f}) at map sigma blur {params['blur_sigma_vox']:.2f} vox.")


if __name__ == "__main__":
    main()
