"""Mask clustered protein C-alpha positions with the predicted scaffold."""

from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree

from H_Backbone_tracing.common.clustering import save_pdb
from H_Backbone_tracing.common.volume import load_thresholded_coords


def _read_calpha_coords(path: str | Path) -> np.ndarray:
    structure = PDBParser(QUIET=True).get_structure("trace", path)
    return np.asarray(
        [atom.get_coord() for atom in structure.get_atoms() if atom.get_name() == "CA"],
        dtype=np.float32,
    )


def mask_pdb_with_mrc(
    pdb_file: str | Path,
    mrc_file: str | Path,
    mask_threshold: float,
    radius: float,
    percent: float,
    *,
    verbose: bool = False,
) -> np.ndarray:
    """Keep C-alpha atoms with sufficient nearby above-threshold voxels."""

    ca_coords = _read_calpha_coords(pdb_file)
    voxel_indices = load_thresholded_coords(mrc_file, mask_threshold)
    with mrcfile.open(mrc_file, permissive=True) as mrc:
        voxel_size = np.array(mrc.voxel_size.tolist(), dtype=np.float32)
        if voxel_size.size == 1:
            voxel_size = np.repeat(voxel_size, 3)
    voxel_coords = voxel_indices[:, ::-1] * voxel_size
    tree = cKDTree(voxel_coords)
    expected_voxels = ((4.0 / 3.0) * np.pi * float(radius) ** 3) / float(np.prod(voxel_size))
    kept = [
        coord
        for coord in ca_coords
        if len(tree.query_ball_point(coord, radius)) / expected_voxels >= percent
    ]
    if verbose:
        print(f"Protein scaffold mask kept {len(kept)}/{len(ca_coords)} C-alpha positions.")
    return np.asarray(kept, dtype=np.float32)


def mask_and_save_pdb(
    pdb_file: str | Path,
    mrc_file: str | Path,
    output_dir: str | Path,
    output_name: str,
    mask_threshold: float,
    radius: float,
    percent: float,
    timestamp: bool = False,
    verbose: bool = False,
) -> Path | None:
    """Apply the scaffold mask and write retained coordinates as a PDB."""

    filtered_xyz = mask_pdb_with_mrc(
        pdb_file,
        mrc_file,
        mask_threshold,
        radius,
        percent,
        verbose=verbose,
    )
    if len(filtered_xyz) == 0:
        return None
    return save_pdb(filtered_xyz[:, ::-1], output_dir, output_name, timestamp=timestamp)
