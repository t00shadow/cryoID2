import os
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from sklearn.cluster import MeanShift, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components

from R_A0_utils import (
    load_thresholded_coords,
)


@dataclass
class ClusteringConfig:
    input_mrc: str
    clustering_method: str = "agglomerative"
    output_dir: str = "H_Backbone_tracing/misc_outputs"

    threshold: float = 0.9
    min_cluster_size: int = 5
    min_spacing: float = 3.0

    connect_dist: float = 4.0   # for agglomerative
    bandwidth: float = 1.9      # for meanshift

    manual_file_name: str | None = None
    timestamp: bool = False

    z_slice: tuple = (0, 999)
    y_slice: tuple = (0, 999)
    x_slice: tuple = (0, 999)


# ----------------------------- Util: load volume file metadata -----------------------------
#! move to utils file later
def load_volume_metadata(input_mrc):
    import mrcfile
    with mrcfile.open(input_mrc, permissive=True) as mrc:
        voxel_size = float(mrc.voxel_size.x)
        shape = mrc.data.shape
    return voxel_size, shape

# ----------------------------- Util: save pdb -----------------------------
def save_pdb(centroids, output_dir, filename, timestamp=False):
    if timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename}_{ts}.pdb"
    else:
        filename = f"{filename}.pdb"

    path = os.path.join(output_dir, filename)

    # with open(path, "w") as f:
    #     for i, (z, y, x) in enumerate(centroids, 1):
    #         f.write(
    #             f"ATOM  {i:5d}  CA  UNK A{i:4d}    "
    #             f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
    #         )

    with open(path, "w") as f:
        for i, (z, y, x) in enumerate(centroids, 1):
            atom_serial = i
            resseq = (i - 1) % 9999 + 1
            chain_id = "A"

            f.write(
                f"ATOM  {atom_serial:5d}  CA  UNK {chain_id}{resseq:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00           C\n"
            )

    return path

# ----------------------------- Helper: build output file name based on configs -----------------------------
def build_output_name(config: ClusteringConfig):
    emdb_id = config.input_mrc.split("emd_")[-1].split(".")[0]
    base = config.manual_file_name or f"emd_{emdb_id}"

    if config.clustering_method == "agglomerative":
        return f"{base}_agglomerative_cd_{config.connect_dist}"
    else:
        return f"{base}_meanshift_bw_{config.bandwidth}"

# ----------------------------- Helper: local thinning -----------------------------
def local_thinning(centroids, min_dist):
    """
    Replace blobs of centroids closer than min_dist with a single representative.
    """
    if len(centroids) == 0:
        return centroids

    nn = NearestNeighbors(radius=min_dist)
    nn.fit(centroids)
    adjacency = nn.radius_neighbors_graph(centroids, mode='connectivity')

    n_components, labels = connected_components(csgraph=adjacency, directed=False)
    thinned = []
    for lbl in range(n_components):
        blob_points = centroids[labels == lbl]
        # pick centroid of blob as representative
        rep = blob_points.mean(axis=0)
        thinned.append(rep)
    return np.array(thinned)

# ----------------------------- Clustering -----------------------------
def cluster_coords(coords, voxel_size, config: ClusteringConfig):
    if len(coords) == 0:
        return np.empty((0, 3))

    coords_real = np.empty_like(coords, dtype=float)
    coords_real[:, 0] = coords[:, 2] * voxel_size
    coords_real[:, 1] = coords[:, 1] * voxel_size
    coords_real[:, 2] = coords[:, 0] * voxel_size

    if config.clustering_method == "meanshift":
        clustering = MeanShift(
            bandwidth=config.bandwidth,
            bin_seeding=True
        )
        labels = clustering.fit_predict(coords_real)

    elif config.clustering_method == "agglomerative":
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=config.connect_dist,
            linkage='complete'
        )
        labels = clustering.fit_predict(coords_real)
    else:
        labels = np.zeros(len(coords_real), dtype=int)

    clusters = defaultdict(list)
    for coord, lbl in zip(coords, labels):
        clusters[lbl].append(coord)

    filtered = [
        np.array(pts)
        for pts in clusters.values()
        if len(pts) >= config.min_cluster_size
    ]

    centroids = [
        np.mean(pts * voxel_size, axis=0)
        for pts in filtered
    ]

    return np.array(centroids) if centroids else np.empty((0, 3))

# ----------------------------- Main function -----------------------------
def run_clustering(config: ClusteringConfig):
    os.makedirs(config.output_dir, exist_ok=True)

    voxel_size, shape = load_volume_metadata(config.input_mrc)
    print(f"Volume shape: {shape}, voxel size: {voxel_size}")

    coords = load_thresholded_coords(
        config.input_mrc,
        config.threshold,
        config.z_slice,
        config.y_slice,
        config.x_slice
    )

    centroids = cluster_coords(coords, voxel_size, config)

    centroids = local_thinning(centroids, config.min_spacing)    # comment this single line out to disable local thinning. should keep it tho

    output_name = build_output_name(config)
    output_path = save_pdb(
        centroids,
        config.output_dir,
        f"{output_name}_CA_only",
        config.timestamp
    )

    return {
        "centroids": centroids,
        "output_path": output_path,
    }


if __name__ == "__main__":
    config = ClusteringConfig(
        # input_mrc="C:/Users/noelu/programming/python_projects/cryoID_v2_DATA/UnetPrediction/EMD-34843_re_3.2/possibility/emd_34843.map_key_atoms_1_possibility.mrc",
        input_mrc=r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\real_malaria\3-E1_D3\possibility\E1_D3_2_7A_cryosparc_P8_J562_008_volume_map_sharp_key_atoms_1_possibility.mrc",
        # input_mrc="G:/.shortcut-targets-by-id/1VHPX-gidYYLEewDizIvgtImMbleuqE0r/CryoID2/WorkFolder/BackboneTracing/TestSet/Raw/EMD-6553_re_7.06/possibility/emd_6553.map_key_atoms_1_possibility.mrc",
        clustering_method="meanshift",
        # manual_file_name="refactor2",
        manual_file_name="E1_D3",
        threshold=0.97    # R_B8_clustering_eval.py's results suggest threshold=0.97 is better (higher metrics)
    )
    results = run_clustering(config)
    print(len(results["centroids"]))
    print(results["output_path"])