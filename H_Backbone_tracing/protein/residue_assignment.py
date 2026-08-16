"""Protein trace coordinate and residue-assignment helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import mrcfile


def get_voxel_size(mrc_path: str | Path) -> tuple[tuple[float, float, float], tuple[int, int, int]]:
    """Return voxel size and map shape in the order used by tracing."""

    with mrcfile.open(mrc_path, permissive=True) as mrc:
        header = mrc.header
        nx, ny, nz = int(header.nx), int(header.ny), int(header.nz)
        voxel_size = (
            float(header.cella.x / nx),
            float(header.cella.y / ny),
            float(header.cella.z / nz),
        )
    return voxel_size, (nx, ny, nz)


def read_pdb_coords(pdb_file: str | Path) -> list[tuple[float, float, float]]:
    """Read trace atoms as z/y/x coordinates in PDB record order."""

    coords: list[tuple[float, float, float]] = []
    with Path(pdb_file).open(encoding="ascii") as handle:
        for line in handle:
            is_calpha = line.startswith("ATOM") and line[12:16].strip() == "CA"
            is_trace_atom = line.startswith("HETATM") and line[12:16].strip() == "C"
            if is_calpha or is_trace_atom:
                coords.append(
                    (float(line[46:54]), float(line[38:46]), float(line[30:38]))
                )
    return coords


def coords_to_voxels(
    coords: Iterable[tuple[float, float, float]],
    voxel_size: Sequence[float],
    radius: int = 1,
) -> dict[int, list[tuple[int, int, int]]]:
    """Convert trace coordinates to local prediction-map neighborhoods."""

    neighborhoods: dict[int, list[tuple[int, int, int]]] = {}
    for atom_id, coordinate in enumerate(coords):
        center = [int(round(value / size)) for value, size in zip(coordinate, voxel_size)]
        neighborhoods[atom_id] = [
            (center[0] + dz, center[1] + dy, center[2] + dx)
            for dz in range(-radius, radius + 1)
            for dy in range(-radius, radius + 1)
            for dx in range(-radius, radius + 1)
        ]
    return neighborhoods


def update_pdb_residues(
    input_pdb: str | Path,
    output_pdb: str | Path,
    residue_sequence: Sequence[str],
) -> Path:
    """Replace each trace atom's residue field with its selected class."""

    updated: list[str] = []
    atom_index = 0
    with Path(input_pdb).open(encoding="ascii") as handle:
        for line in handle:
            is_calpha = line.startswith("ATOM") and line[12:16].strip() == "CA"
            is_trace_atom = line.startswith("HETATM") and line[12:16].strip() == "C"
            if is_calpha or is_trace_atom:
                if atom_index >= len(residue_sequence):
                    raise ValueError("Number of trace atoms exceeds the residue sequence length.")
                line = line[:17] + f"{residue_sequence[atom_index]:>3}" + line[20:]
                atom_index += 1
            updated.append(line)
    if atom_index != len(residue_sequence):
        raise ValueError("Number of trace atoms does not match the residue sequence length.")
    output = Path(output_pdb)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(updated), encoding="ascii", newline="\n")
    return output
