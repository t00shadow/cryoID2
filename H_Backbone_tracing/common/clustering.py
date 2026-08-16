"""Prediction-volume clustering shared by protein and RNA query generation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering, MeanShift
from sklearn.neighbors import NearestNeighbors

from .volume import load_thresholded_coords, load_volume_metadata


@dataclass(frozen=True)
class ClusteringConfig:
    input_mrc: str
    clustering_method: str = "agglomerative"
    output_dir: str = "."
    threshold: float = 0.9
    min_cluster_size: int = 5
    min_spacing: float = 3.0
    connect_dist: float = 4.0
    bandwidth: float = 1.9
    manual_file_name: str | None = None
    timestamp: bool = False
    z_slice: tuple[int, int] = (0, 999)
    y_slice: tuple[int, int] = (0, 999)
    x_slice: tuple[int, int] = (0, 999)


def save_pdb(
    centroids: np.ndarray,
    output_dir: str | Path,
    filename: str,
    *,
    timestamp: bool = False,
) -> Path:
    """Write z/y/x centroid coordinates as PDB C-alpha atoms."""

    suffix = f"_{datetime.now():%Y%m%d_%H%M%S}" if timestamp else ""
    path = Path(output_dir) / f"{filename}{suffix}.pdb"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        for index, (z, y, x) in enumerate(centroids, start=1):
            residue_number = (index - 1) % 9999 + 1
            handle.write(
                f"ATOM  {index:5d}  CA  UNK A{residue_number:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
    return path


def _output_name(config: ClusteringConfig) -> str:
    emdb_id = config.input_mrc.split("emd_")[-1].split(".")[0]
    base = config.manual_file_name or f"emd_{emdb_id}"
    if config.clustering_method == "agglomerative":
        return f"{base}_agglomerative_cd_{config.connect_dist}"
    return f"{base}_meanshift_bw_{config.bandwidth}"


def _thin_centroids(centroids: np.ndarray, minimum_distance: float) -> np.ndarray:
    if len(centroids) == 0:
        return centroids
    neighbors = NearestNeighbors(radius=minimum_distance)
    neighbors.fit(centroids)
    adjacency = neighbors.radius_neighbors_graph(centroids, mode="connectivity")
    component_count, labels = connected_components(csgraph=adjacency, directed=False)
    return np.asarray([centroids[labels == label].mean(axis=0) for label in range(component_count)])


def _cluster_coords(coords: np.ndarray, voxel_size: float, config: ClusteringConfig) -> np.ndarray:
    if len(coords) == 0:
        return np.empty((0, 3))
    real_coords = np.empty_like(coords, dtype=float)
    real_coords[:, 0] = coords[:, 2] * voxel_size
    real_coords[:, 1] = coords[:, 1] * voxel_size
    real_coords[:, 2] = coords[:, 0] * voxel_size
    if config.clustering_method == "meanshift":
        labels = MeanShift(bandwidth=config.bandwidth, bin_seeding=True).fit_predict(real_coords)
    elif config.clustering_method == "agglomerative":
        labels = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=config.connect_dist,
            linkage="complete",
        ).fit_predict(real_coords)
    else:
        raise ValueError(f"Unsupported clustering method: {config.clustering_method}")
    clusters: dict[int, list[np.ndarray]] = defaultdict(list)
    for coord, label in zip(coords, labels):
        clusters[int(label)].append(coord)
    centroids = [
        np.mean(np.asarray(points) * voxel_size, axis=0)
        for points in clusters.values()
        if len(points) >= config.min_cluster_size
    ]
    return np.asarray(centroids) if centroids else np.empty((0, 3))


def run_clustering(config: ClusteringConfig) -> dict[str, object]:
    """Cluster one prediction channel and write its centroid PDB."""

    voxel_size, _shape = load_volume_metadata(config.input_mrc)
    coords = load_thresholded_coords(
        config.input_mrc,
        config.threshold,
        config.z_slice,
        config.y_slice,
        config.x_slice,
    )
    centroids = _thin_centroids(_cluster_coords(coords, voxel_size, config), config.min_spacing)
    output_path = save_pdb(
        centroids,
        config.output_dir,
        f"{_output_name(config)}_CA_only",
        timestamp=config.timestamp,
    )
    return {"centroids": centroids, "output_path": output_path}
