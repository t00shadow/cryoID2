#!/usr/bin/env python3
"""
Visualize discrete voxel spheres for a 6 Å desired radius at two voxel sizes.

Example:
  python Alphafold_fetch/voxel_sphere_demo.py --voxels 1.5 2.0 --radius-ang 6 --out sphere_demo.png
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


def voxel_offsets(rad_vox: float) -> np.ndarray:
    """Return integer (dx, dy, dz) offsets whose squared distance <= rad_vox^2."""
    r_int = int(np.ceil(max(1.0, rad_vox)))
    offsets = []
    cutoff = rad_vox * rad_vox + 1e-6
    for dz in range(-r_int, r_int + 1):
        for dy in range(-r_int, r_int + 1):
            for dx in range(-r_int, r_int + 1):
                if (dx * dx + dy * dy + dz * dz) <= cutoff:
                    offsets.append((dx, dy, dz))
    return np.asarray(offsets, dtype=int)


def build_mask(rad_vox: float) -> np.ndarray:
    """Binary mask large enough to contain the sphere for visualization."""
    r_int = int(np.ceil(max(1.0, rad_vox)))
    size = 2 * r_int + 1
    mask = np.zeros((size, size, size), dtype=np.uint8)
    center = np.array([r_int, r_int, r_int])
    for dx, dy, dz in voxel_offsets(rad_vox):
        z, y, x = center + np.array([dz, dy, dx])
        mask[z, y, x] = 1
    return mask


def plot_sphere(ax, mask: np.ndarray, title: str):
    """Simple 3D scatter of occupied voxels."""
    z, y, x = np.nonzero(mask)
    ax.scatter(x, y, z, s=40, alpha=0.7, edgecolors="k")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_box_aspect([1, 1, 1])
    lim = [mask.shape[0] - 0.5, -0.5]
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_zlim(lim)


def main(voxel_sizes: Iterable[float], radius_ang: float, out: Path | None):
    voxel_sizes = list(voxel_sizes)
    fig = plt.figure(figsize=(5 * len(voxel_sizes), 5))
    for i, vs in enumerate(voxel_sizes, start=1):
        rad_vox = radius_ang / vs
        mask = build_mask(rad_vox)
        ax = fig.add_subplot(1, len(voxel_sizes), i, projection="3d")
        plot_sphere(ax, mask, f"{radius_ang:.1f} Å / {vs:.2f} Å/vox → {rad_vox:.2f} vox\nvoxels filled: {mask.sum()}")
    plt.tight_layout()
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
        print(f"Saved visualization to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voxel sphere visualization for different voxel sizes.")
    parser.add_argument("--radius-ang", type=float, default=6.0, help="Desired sphere radius in Angstroms.")
    parser.add_argument("--voxels", type=float, nargs="+", default=[1.5, 2.0], help="Voxel sizes (Angstroms per voxel).")
    parser.add_argument("--out", type=Path, default=None, help="Optional output image path (PNG).")
    args = parser.parse_args()
    main(args.voxels, args.radius_ang, args.out)
