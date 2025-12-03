#!/usr/bin/env python3
"""
Grid-search VOF over sphere radii for two fitted models (e.g., 1U8F and AlphaFold).

Usage:
  python Alphafold_fetch/vof_radius_grid.py \
    --map Alphafold_fetch/test_files/emd_29659.map \
    --pdb-fitted Alphafold_fetch/fitted_models_v6/CDABP0047__PDB_1U8F_fitted_v5.pdb \
    --af-fitted  Alphafold_fetch/fitted_models_v6/CDABP0047__AF_P04406_fitted_v5.pdb \
    --out-csv results.csv

Notes:
  - Models must already be fitted into the map (aligned coordinates). If a path
    is missing the script will skip that entry.
  - Radii are in Angstroms and sampled from 3.5 to 10.0 (step 0.5).
"""

import argparse
from pathlib import Path
import numpy as np
import mrcfile

from fit_v5 import (
    MapInfo,
    compute_vof,
    read_model_coords,
    _safe_get_step,
    _derive_params_from_resolution,
)

# ---- Hardcoded defaults (edit these paths to point at your fitted models) ----
DEFAULT_PDB_FITTED = Path("Alphafold_fetch/fitted_models_v6/CDABP0047__PDB_8G14_fitted_v5.pdb")
DEFAULT_AF_FITTED  = Path("Alphafold_fetch/fitted_models_v6/CDABP0047__AF_P04406_fitted_v5.pdb")


def build_mapinfo(map_path: Path, map_resolution_A: float | None) -> tuple[MapInfo, float, dict]:
    with mrcfile.open(str(map_path), mode="r") as mrc:
        vs = _safe_get_step(mrc)
        params = _derive_params_from_resolution(vs, map_resolution_A)
        mapinfo = MapInfo(mrc, blur_sigma_vox=params["blur_sigma_vox"])
    vs_mean = float(sum(vs) / 3.0)
    return mapinfo, vs_mean, params


def evaluate_radii(label: str, fitted_path: Path, mapinfo: MapInfo, vs_mean: float, max_dim: int) -> list[tuple[float, float]]:
    if not fitted_path.exists():
        print(f"[skip] {label}: fitted model not found at {fitted_path}")
        return []
    _, coords = read_model_coords(fitted_path)
    radii_ang = np.arange(1.0, 10.0 + 0.25, 0.5)
    results: list[tuple[float, float]] = []
    for r_ang in radii_ang:
        r_vox = r_ang / max(vs_mean, 1e-6)
        vof = compute_vof(mapinfo, coords, sphere_radius_vox=r_vox, max_dim=max_dim)
        results.append((float(r_ang), float(vof)))
    return results


def main():
    parser = argparse.ArgumentParser(description="Grid-search VOF across radii for two fitted models.")
    parser.add_argument("--map", default="Alphafold_fetch/test_files/emd_29659.map", type=str, help="Cryo-EM map path")
    parser.add_argument("--map-res", default=None, type=float, help="Map resolution in Angstroms (optional)")
    parser.add_argument("--pdb-fitted", default=str(DEFAULT_PDB_FITTED), type=str, help="Fitted PDB/CIF for experimental structure (1U8F)")
    parser.add_argument("--af-fitted", default=str(DEFAULT_AF_FITTED), type=str, help="Fitted PDB/CIF for AlphaFold model")
    parser.add_argument("--out-csv", default=None, type=str, help="Optional CSV output path")
    args = parser.parse_args()

    map_path = Path(args.map)
    if not map_path.exists():
        raise SystemExit(f"Map not found: {map_path}")

    mapinfo, vs_mean, params = build_mapinfo(map_path, args.map_res)

    all_rows: list[tuple[str, float, float]] = []
    for label, fpath in [("PDB_1U8F", Path(args.pdb_fitted)), ("AF_model", Path(args.af_fitted))]:
        res = evaluate_radii(label, fpath, mapinfo, vs_mean, max_dim=params["vof_max_dim"])
        if not res:
            continue
        print(f"\n{label} ({fpath}):")
        for r_ang, vof in res:
            print(f"  radius={r_ang:4.1f} Å -> VOF={vof:.4f}")
            all_rows.append((label, r_ang, vof))

    if args.out_csv and all_rows:
        import csv

        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "radius_A", "vof"])
            writer.writerows(all_rows)
        print(f"\nSaved grid results to {out_path}")


if __name__ == "__main__":
    main()
