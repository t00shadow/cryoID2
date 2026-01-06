# NOTE (12/17/25 - 9:41pm) TO SELF, WAS SPLITING A1_clustering.py (yes that version not the newer version) into 2 files, put the util stuff in its own file and clustering only in this file.
# The R in the prefix stands for rewrite.


import os
import numpy as np
from collections import defaultdict
from datetime import datetime
from sklearn.cluster import MeanShift, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components

from R_A0_utils import (
    load_thresholded_coords,
    intersect_coords,
    save_mask_mrc_from_coords,
    save_mask_pdb,
    remove_duplicates
)

# ----------------------------- Configuration -----------------------------
from A0_config_loader import load_sample_paths, load_config_section

config_file = "H_Backbone_tracing/small_prototypes/A0_config.yaml"

FILE_TOGGLE = load_config_section(config_file, "sample_toggle")    # use config file, no longer hard coded, but can hardcode if wanna test the same sample without worrying about config file
# print(FILE_TOGGLE); assert(False)

paths_cfg = load_sample_paths(config_file, FILE_TOGGLE)
INPUT_MRC = paths_cfg["CA_mrc"]
MASK_HELIX_MRC = paths_cfg["mask_helix_mrc"]
MASK_SHEET_MRC = paths_cfg["mask_sheet_mrc"]
MASK_RING_MRC = paths_cfg["mask_ring_mrc"]
ORIGINAL_MAP_PATH = paths_cfg["original_map"]    # not used

clustering_cfg = load_config_section(config_file, "clustering")
OUTPUT_DIR = clustering_cfg["output_dir"]
THRESHOLD = clustering_cfg["threshold"]
CLUSTERING_METHOD = clustering_cfg["method"]
if CLUSTERING_METHOD == "agglomerative":
    CONNECT_DIST = clustering_cfg["connect_dist"]
    OUTPUT_FILE_NAME = f"{FILE_TOGGLE}_agglomerative_cd_{CONNECT_DIST}"
elif CLUSTERING_METHOD == "meanshift":
    BANDWIDTH = clustering_cfg["bandwidth"]
    OUTPUT_FILE_NAME = f"{FILE_TOGGLE}_meanshift_bw_{BANDWIDTH}"
MIN_CLUSTER_SIZE = clustering_cfg["min_cluster_size"]

# ----------------------------- Slice configuration -----------------------------
z_slice = (0, 999)
y_slice = (0, 999)
x_slice = (0, 999)

# ----------------------------- Main clustering function -----------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load voxel size and shape from INPUT_MRC header
    import mrcfile
    with mrcfile.open(INPUT_MRC, permissive=True) as mrc:
        voxel_size = float(mrc.voxel_size.x)
        data_shape = mrc.data.shape

    print(f"Volume shape: {data_shape}, voxel size: {voxel_size}")

    # Load thresholded coords of base volume chunk
    print("Loading volume coords (chunked)...")
    volume_coords = load_thresholded_coords(INPUT_MRC, THRESHOLD, z_slice, y_slice, x_slice)
    print(f"Volume voxels above threshold: {len(volume_coords)}")

    def cluster_and_get_centroids(coords, tag):
        if len(coords) == 0:
            print(f"No voxels to cluster for {tag}")
            return np.empty((0, 3), dtype=float)

        coords_real = np.empty_like(coords, dtype=float)
        coords_real[:, 0] = coords[:, 2] * voxel_size
        coords_real[:, 1] = coords[:, 1] * voxel_size
        coords_real[:, 2] = coords[:, 0] * voxel_size

        if CLUSTERING_METHOD == "meanshift":
            clustering = MeanShift(bandwidth=BANDWIDTH, bin_seeding=True)
        elif CLUSTERING_METHOD == "agglomerative":
            clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=CONNECT_DIST, linkage='complete')
        labels = clustering.fit_predict(coords_real)
        print(f"Clusters found in {tag}: {len(set(labels))}")

        clusters = defaultdict(list)
        for coord, label in zip(coords, labels):
            clusters[label].append(coord)

        filtered_clusters = {
            label: np.array(pts)
            for label, pts in clusters.items()
            if len(pts) >= MIN_CLUSTER_SIZE
        }
        print(f"Filtered clusters in {tag} (size >= {MIN_CLUSTER_SIZE}): {len(filtered_clusters)}")

        # Compute centroids
        centroids = []
        for pts in filtered_clusters.values():
            centroid = np.mean(pts * voxel_size, axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids) if centroids else np.empty((0, 3), dtype=float)

        # Save PDB of centroids
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdb_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE_NAME}_{tag}_{timestamp}.pdb")
        with open(pdb_path, 'w') as f:
            for i, (z, y, x) in enumerate(centroids, 1):
                f.write(f"HETATM{i:5d}  C   UNK     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        print(f"Saved PDB with {len(centroids)} centroids for {tag} to {pdb_path}")

        return centroids, pdb_path

    # Cluster data alone
    print("\nClustering data alone...")
    data_centroids, all_ca_pdb_path = cluster_and_get_centroids(volume_coords, "data_alone")

    # Masks
    ss_masks = {
        'loop': MASK_RING_MRC,
    }

    mask_centroids_list = []

    for name, path in ss_masks.items():
        print(f"\nProcessing data + {name} mask, path: {path} ...")
        mask_coords = load_thresholded_coords(path, THRESHOLD, z_slice, y_slice, x_slice)
        print(f"Mask '{name}' voxels above threshold: {len(mask_coords)}")
        filtered_coords = np.array(list(set(map(tuple, volume_coords)) & set(map(tuple, mask_coords))), dtype=int)
        removed = len(volume_coords) - len(filtered_coords)
        print(f"Voxels removed after applying {name} mask: {removed}")
        centroids, _ = cluster_and_get_centroids(filtered_coords, f"data_plus_{name}")
        mask_centroids_list.append(centroids)

    # Combine all centroids
    all_centroids = [data_centroids] + mask_centroids_list
    combined_centroids = np.vstack([c for c in all_centroids if c.size > 0])
    data_centroids = np.vstack([c for c in data_centroids if c.size > 0])   # convert to (N, 3) too

    print(f"\nTotal combined centroids before connected components: {len(combined_centroids)}")
    if combined_centroids.size == 0:
        print("No centroids to process in combined data. Exiting.")
        return

    print(f"Before deduplication: {len(combined_centroids)}")
    combined_centroids = remove_duplicates(combined_centroids, voxel_size, tol=1.0)
    print(f"After deduplication: {len(combined_centroids)}")

    nn = NearestNeighbors(radius=CONNECT_DIST)
    nn.fit(combined_centroids)
    adjacency = nn.radius_neighbors_graph(combined_centroids, mode='connectivity')

    n_components, comp_labels = connected_components(csgraph=adjacency, directed=False)
    print(f"Found {n_components} connected components among combined centroids")

    component_sizes = np.bincount(comp_labels)
    largest_component_label = np.argmax(component_sizes)
    largest_mask = comp_labels == largest_component_label
    largest_centroids = combined_centroids[largest_mask]

    adjacency = adjacency.tocoo()
    gcc_edges = [
        (i, j)
        for i, j in zip(adjacency.row, adjacency.col)
        if i < j and largest_mask[i] and largest_mask[j]
    ]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdb_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE_NAME}_combined_largestCC_{timestamp}.pdb")

    old_to_new = {old_idx: new_idx+1 for new_idx, old_idx in enumerate(np.where(largest_mask)[0])}

    with open(pdb_path, 'w') as f:
        for i, (z, y, x) in enumerate(largest_centroids, 1):
            f.write(f"HETATM{i:5d}  C   UNK     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        for i, j in gcc_edges:
            f.write(f"CONECT{old_to_new[i]:5d}{old_to_new[j]:5d}\n")

    print(f"Saved final combined largest connected component PDB with {len(largest_centroids)} centroids to {pdb_path}")
    print(f"All CAs saved to {all_ca_pdb_path}")

    # return pdb_path

    return {
        # "centroids": combined_centroids,
        "centroids": data_centroids,
        "connect_dist": CONNECT_DIST,
        "output_dir": OUTPUT_DIR,
        "output_name": all_ca_pdb_path,  # dont rly need this
    }



if __name__ == "__main__":
    main()
