"""Degree-limited protein backbone tracing."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from H_Backbone_tracing.common.graph import build_distance_graph, save_graph_pdb
from H_Backbone_tracing.common.density_pruning import limit_edges_global, repair_cycles
from H_Backbone_tracing.common.volume import load_thresholded_coords, load_volume_metadata


def limit_node_degree(coords: np.ndarray, edges: list[tuple], max_degree: int = 3) -> list[tuple]:
    """Keep the shortest incident edges at each node."""

    neighbors: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for edge in edges:
        i, j = edge[:2]
        distance = float(edge[2]) if len(edge) == 3 else float(np.linalg.norm(coords[i] - coords[j]))
        neighbors[int(i)].append((int(j), distance))
        neighbors[int(j)].append((int(i), distance))
    kept: set[tuple[int, int, float]] = set()
    for i, candidates in neighbors.items():
        for j, distance in sorted(candidates, key=lambda item: item[1])[:max_degree]:
            kept.add((min(i, j), max(i, j), distance))
    return sorted(kept)


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


def trace_backbone_components(
    *,
    clustered_pdb_path: str | Path,
    scaffold_path: str | Path,
    max_distance: float,
    max_degree: int,
    min_component_size: int,
    graph_output_dir: str | Path,
    components_output_dir: str | Path,
    scaffold_threshold: float = 0.97,
    scaffold_cylinder_radius: float = 1.0,
    verbose: bool = False,
) -> list[Path]:
    """Trace clustered C-alpha positions and write retained components."""

    centroids = load_coords_from_pdb(clustered_pdb_path)
    _adjacency, edges = build_distance_graph(
        centroids,
        max_distance,
        return_edge_lengths=True,
    )
    scaffold_voxels = load_thresholded_coords(scaffold_path, scaffold_threshold)
    voxel_size, _shape = load_volume_metadata(scaffold_path)
    # Tracing coordinates are map-relative z/y/x Angstroms; do not apply an
    # x/y/z reversal. Production prediction maps are isotropic 1 A grids.
    scaffold_voxels = scaffold_voxels * voxel_size
    limited_edges, superset_edges, edge_scores = limit_edges_global(
        centroids, edges, scaffold_voxels,
        radius=scaffold_cylinder_radius, max_degree=max_degree,
    )
    limited_edges = repair_cycles(
        len(centroids), limited_edges, superset_edges, edge_scores, max_degree=max_degree,
    )
    graph_root = Path(graph_output_dir)
    graph_root.mkdir(parents=True, exist_ok=True)
    outputs = [
        save_graph_pdb(centroids, edges, graph_root / "superset_graph.pdb"),
        save_graph_pdb(centroids, limited_edges, graph_root / "superset_graph_degree_limited.pdb"),
    ]
    all_components = get_connected_components(len(centroids), limited_edges)
    components = [
        component
        for component in all_components
        if len(component) >= min_component_size
    ]
    fallback = bool(all_components) and not components
    if fallback:
        components = all_components
    length_filter = {
        "requested_min_trace_length": min_component_size,
        "effective_min_trace_length": 0 if fallback else min_component_size,
        "fallback_applied": fallback,
        "component_count_before_filter": len(all_components),
        "longest_component_length": max(map(len, all_components), default=0),
        "retained_component_count": len(components),
    }
    filter_path = graph_root / "trace_length_filter.json"
    filter_path.write_text(json.dumps(length_filter, indent=2) + "\n", encoding="utf-8")
    outputs.append(filter_path)
    outputs.extend(_save_components(centroids, limited_edges, components, components_output_dir))
    if verbose:
        if fallback:
            print(
                f"No protein trace reached the requested minimum length {min_component_size}; "
                "using minimum length 0 and retaining all available components."
            )
        print(f"Protein tracing retained {len(components)} component(s).")
    return outputs
