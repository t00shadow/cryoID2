"""Volume helpers shared by the protein and RNA tracing branches."""

from __future__ import annotations

from pathlib import Path

import mrcfile
import numpy as np


def load_volume_metadata(path: str | Path) -> tuple[float, tuple[int, ...]]:
    """Return the x-axis voxel size and data shape for a prediction map."""

    with mrcfile.open(path, permissive=True) as mrc:
        return float(mrc.voxel_size.x), tuple(int(value) for value in mrc.data.shape)


def load_thresholded_coords(
    path: str | Path,
    threshold: float,
    z_slice: tuple[int, int] | None = None,
    y_slice: tuple[int, int] | None = None,
    x_slice: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return z/y/x voxel indices whose prediction exceeds ``threshold``."""

    with mrcfile.open(path, permissive=True) as mrc:
        data = np.squeeze(mrc.data)
        z_start, z_stop = z_slice or (0, data.shape[0])
        y_start, y_stop = y_slice or (0, data.shape[1])
        x_start, x_stop = x_slice or (0, data.shape[2])
        chunk = data[z_start:z_stop, y_start:y_stop, x_start:x_stop]
        coords = np.argwhere(chunk > float(threshold))
    if coords.size:
        coords += np.array((z_start, y_start, x_start), dtype=coords.dtype)
    return coords
