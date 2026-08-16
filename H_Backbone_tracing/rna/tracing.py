"""Paper-compatible phosphate/sugar tracing for RNA query generation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from H_Backbone_tracing.common.graph import build_distance_graph, save_graph_pdb
from H_Backbone_tracing.common.volume import load_thresholded_coords


def edge_density_score(
    coords: np.ndarray,
    em_voxels: np.ndarray,
    i: int,
    j: int,
    radius: float,
    metric: str = "raw_count",
) -> float:
    """Score the scaffold support inside the finite cylinder for one edge."""

    a, b = coords[i], coords[j]
    segment = b - a
    length = float(np.linalg.norm(segment))
    if length == 0:
        return 0.0
    unit = segment / length
    offsets = np.asarray(em_voxels) - a
    projections = offsets @ unit
    bounded = (projections >= 0) & (projections <= length)
    if np.any(bounded):
        perpendicular = offsets[bounded] - projections[bounded, None] * unit
        count = int(np.count_nonzero(np.linalg.norm(perpendicular, axis=1) <= radius))
    else:
        count = 0
    if metric == "raw_count":
        return float(count)
    if metric == "normalized_by_length":
        return count / length
    raise ValueError("metric must be 'raw_count' or 'normalized_by_length'")


def limit_edges_global(
    coords: np.ndarray,
    edges: list[tuple],
    em_voxels: np.ndarray,
    *,
    radius: float = 2.0,
    max_degree: int = 2,
) -> list[tuple[int, int]]:
    """Select the scaffold-supported degree-two graph used by the paper run.

    Edges are considered globally in descending scaffold-density order.  An
    edge is retained only while both endpoints remain below ``max_degree``.
    The deterministic tuple sort is intentional: it preserves the historical
    tie breaking that produced the seven EMD-47655 queries.
    """

    edge_pairs = [(int(edge[0]), int(edge[1])) for edge in edges]
    voxels = np.asarray(em_voxels)
    scored: list[tuple[float, int, int]] = []
    for i, j in edge_pairs:
        score = edge_density_score(
            coords,
            voxels,
            i,
            j,
            radius,
            metric="normalized_by_length",
        )
        scored.append((score, i, j))
    scored.sort(reverse=True)
    degree: dict[int, int] = defaultdict(int)
    kept: list[tuple[int, int]] = []
    for _score, i, j in scored:
        if degree[i] < max_degree and degree[j] < max_degree:
            kept.append((i, j))
            degree[i] += 1
            degree[j] += 1
    return kept


def load_coords_from_pdb(path: str | Path, atom_name: str = "CA") -> np.ndarray:
    """Read selected PDB atoms as z/y/x coordinates."""

    coords: list[tuple[float, float, float]] = []
    with Path(path).open(encoding="ascii") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")) and line[12:16].strip() == atom_name:
                coords.append((float(line[46:54]), float(line[38:46]), float(line[30:38])))
    if not coords:
        raise ValueError(f"No {atom_name} atoms found in {path}")
    return np.asarray(coords, dtype=np.float32)


def get_connected_components(node_count: int, edges: list[tuple]) -> list[list[int]]:
    adjacency = [[] for _ in range(node_count)]
    for edge in edges:
        i, j = (int(value) for value in edge[:2])
        adjacency[i].append(j)
        adjacency[j].append(i)
    visited = [False] * node_count
    components: list[list[int]] = []
    for start in range(node_count):
        if visited[start]:
            continue
        stack = [start]
        component: list[int] = []
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            component.append(node)
            stack.extend(neighbor for neighbor in adjacency[node] if not visited[neighbor])
        components.append(component)
    return components


def _save_components(
    coords: np.ndarray,
    edges: list[tuple],
    components: list[list[int]],
    output_dir: str | Path,
) -> list[Path]:
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for component_index, component in enumerate(components):
        members = set(component)
        index_map = {old: new for new, old in enumerate(component)}
        component_edges = [
            (index_map[int(edge[0])], index_map[int(edge[1])])
            for edge in edges
            if int(edge[0]) in members and int(edge[1]) in members
        ]
        outputs.append(
            save_graph_pdb(
                coords[component],
                component_edges,
                output_root / f"cc_{component_index}.pdb",
            )
        )
    return outputs


def trace_rna_components(
    *,
    phosphate_pdb_path: str | Path,
    sugar_pdb_path: str | Path,
    scaffold_path: str | Path,
    scaffold_threshold: float,
    scaffold_cylinder_radius: float,
    max_distance: float,
    max_degree: int,
    min_component_size: int,
    graph_output_dir: str | Path,
    components_output_dir: str | Path,
    verbose: bool = False,
) -> list[Path]:
    """Trace combined phosphate/sugar centroids using scaffold support."""

    phosphate = load_coords_from_pdb(phosphate_pdb_path)
    sugar = load_coords_from_pdb(sugar_pdb_path)
    centroids = np.vstack((phosphate, sugar))
    _adjacency, edges = build_distance_graph(
        centroids,
        max_distance,
        return_edge_lengths=True,
    )
    scaffold_voxels = load_thresholded_coords(scaffold_path, scaffold_threshold)
    # ``load_coords_from_pdb`` and ``load_thresholded_coords`` both expose the
    # historical internal z/y/x order.  Do not reorder scaffold coordinates:
    # doing so changes graph topology and no longer reproduces the seven paper
    # queries.  The production RNA maps are isotropic, approximately 1 A/voxel,
    # matching the coordinate convention of the validated paper workflow.
    limited_edges = limit_edges_global(
        centroids,
        edges,
        scaffold_voxels,
        radius=scaffold_cylinder_radius,
        max_degree=max_degree,
    )
    graph_root = Path(graph_output_dir)
    graph_root.mkdir(parents=True, exist_ok=True)
    outputs = [
        save_graph_pdb(centroids, edges, graph_root / "superset_graph.pdb"),
        save_graph_pdb(centroids, limited_edges, graph_root / "superset_graph_degree_limited.pdb"),
    ]
    components = [
        component
        for component in get_connected_components(len(centroids), limited_edges)
        if len(component) >= min_component_size
    ]
    outputs.extend(_save_components(centroids, limited_edges, components, components_output_dir))
    if verbose:
        print(f"RNA tracing retained {len(components)} component(s).")
    return outputs
