"""Distance-graph primitives shared by protein and RNA tracing."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.neighbors import NearestNeighbors


def build_distance_graph(
    coords: np.ndarray,
    max_dist: float,
    *,
    return_edge_lengths: bool = False,
) -> tuple[dict[int, list], list[tuple]]:
    """Connect every coordinate pair within ``max_dist`` Angstroms."""

    if len(coords) == 0:
        return {}, []
    neighbors = NearestNeighbors(radius=max_dist)
    neighbors.fit(coords)
    graph = neighbors.radius_neighbors_graph(coords, mode="distance").tocoo()
    adjacency: dict[int, list] = defaultdict(list)
    edges: list[tuple] = []
    for i, j, distance in zip(graph.row, graph.col, graph.data):
        if i >= j:
            continue
        if return_edge_lengths:
            adjacency[int(i)].append((int(j), float(distance)))
            adjacency[int(j)].append((int(i), float(distance)))
            edges.append((int(i), int(j), float(distance)))
        else:
            adjacency[int(i)].append(int(j))
            adjacency[int(j)].append(int(i))
            edges.append((int(i), int(j)))
    return dict(adjacency), edges


def save_graph_pdb(coords: np.ndarray, edges: list[tuple], filename: str | Path) -> Path:
    """Write z/y/x nodes and zero-based edges as a connected PDB graph."""

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for index, (z, y, x) in enumerate(coords, start=1):
            handle.write(
                f"HETATM{index:5d}  C   UNK     1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
        for edge in edges:
            i, j = edge[:2]
            handle.write(f"CONECT{int(i) + 1:5d}{int(j) + 1:5d}\n")
    return path
