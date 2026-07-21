# might not be the best fix but seems to work better than the original version, doesnt change the already good parts (hope thats true for other samples too)
# edit: this actually doesnt work that well on the malaria sample cuz so dense, and this is a global treatment, local cleanup would be better, cuz dont wanna ruin the already good parts

# NOTE: Agglomerative clustering with second-layer merge, preserves original coordinate order
import os
import numpy as np
from collections import defaultdict
from datetime import datetime
from sklearn.cluster import MeanShift, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from H_Backbone_tracing.R_A0_utils import (
    load_thresholded_coords,
    intersect_coords,
    save_mask_mrc_from_coords,
    save_mask_pdb,
    remove_duplicates
)

# ----------------------------- Configuration -----------------------------
from H_Backbone_tracing.A0_config_loader import load_sample_paths, load_config_section

config_file = "H_Backbone_tracing/small_prototypes/A0_config.yaml"
FILE_TOGGLE = load_config_section(config_file, "sample_toggle")
paths_cfg = load_sample_paths(config_file, FILE_TOGGLE)

INPUT_MRC = paths_cfg["CA_mrc"]
MASK_HELIX_MRC = paths_cfg["mask_helix_mrc"]
MASK_SHEET_MRC = paths_cfg["mask_sheet_mrc"]
MASK_RING_MRC = paths_cfg["mask_ring_mrc"]
ORIGINAL_MAP_PATH = paths_cfg["original_map"]

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

MIN_CUTOFF = 3.0  # Å, strict minimum spacing for second-layer merge

# ----------------------------- Slice configuration -----------------------------
z_slice = (0, 999)
y_slice = (0, 999)
x_slice = (0, 999)

# ----------------------------- Helper: second-layer merge -----------------------------
def merge_close_centroids(centroids, merge_threshold=MIN_CUTOFF):
    """Merge centroids that are closer than merge_threshold Å."""
    if len(centroids) < 2:
        return centroids.copy()
    tree = cKDTree(centroids)
    keep = np.ones(len(centroids), dtype=bool)
    for i, pt in enumerate(centroids):
        if not keep[i]:
            continue
        neighbors = tree.query_ball_point(pt, merge_threshold)
        for j in neighbors:
            if j > i:
                # Merge neighbor into i
                centroids[i] = (centroids[i] + centroids[j]) / 2
                keep[j] = False
    return centroids[keep]

# ----------------------------- Main clustering function -----------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import mrcfile
    with mrcfile.open(INPUT_MRC, permissive=True) as mrc:
        voxel_size = float(mrc.voxel_size.x)
        data_shape = mrc.data.shape

    print(f"Volume shape: {data_shape}, voxel size: {voxel_size}")

    print("Loading volume coords (chunked)...")
    volume_coords = load_thresholded_coords(INPUT_MRC, THRESHOLD, z_slice, y_slice, x_slice)
    print(f"Volume voxels above threshold: {len(volume_coords)}")

    def cluster_and_get_centroids(coords, tag):
        if len(coords) == 0:
            return np.empty((0,3)), None

        coords_real = np.empty_like(coords, dtype=float)
        coords_real[:, 0] = coords[:, 2] * voxel_size  # preserve original flip order
        coords_real[:, 1] = coords[:, 1] * voxel_size
        coords_real[:, 2] = coords[:, 0] * voxel_size

        if CLUSTERING_METHOD == "meanshift":
            clustering = MeanShift(bandwidth=BANDWIDTH, bin_seeding=True)
        else:
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=CONNECT_DIST,
                linkage='complete'
            )

        labels = clustering.fit_predict(coords_real)
        clusters = defaultdict(list)
        for coord, label in zip(coords, labels):
            clusters[label].append(coord)

        filtered_clusters = {
            label: np.array(pts)
            for label, pts in clusters.items()
            if len(pts) >= MIN_CLUSTER_SIZE
        }

        centroids = []
        for pts in filtered_clusters.values():
            centroid = np.mean(pts*voxel_size, axis=0)  # preserve flip
            centroids.append(centroid)
        centroids = np.array(centroids) if centroids else np.empty((0,3))

        # --- SECOND-LAYER MERGE ---
        if CLUSTERING_METHOD == "agglomerative" and centroids.shape[0] > 1:
            centroids = merge_close_centroids(centroids, merge_threshold=MIN_CUTOFF)
            print(f"{tag}: {len(centroids)} centroids after second-layer merge (min {MIN_CUTOFF} Å)")

        # Save PDB
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdb_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE_NAME}_{tag}_{timestamp}.pdb")
        with open(pdb_path, 'w') as f:
            for i, (z,y,x) in enumerate(centroids,1):
                f.write(f"HETATM{i:5d}  C   UNK     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 0.00           C\n")
        print(f"Saved PDB for {tag} with {len(centroids)} centroids to {pdb_path}")

        return centroids, pdb_path

    # Cluster main data
    print("\nClustering data alone...")
    data_centroids, all_ca_pdb_path = cluster_and_get_centroids(volume_coords, "data_alone")

    # Masks
    ss_masks = {"loop": MASK_RING_MRC}
    mask_centroids_list = []

    for name, path in ss_masks.items():
        mask_coords = load_thresholded_coords(path, THRESHOLD, z_slice, y_slice, x_slice)
        filtered_coords = np.array(
            list(set(map(tuple, volume_coords)) & set(map(tuple, mask_coords))),
            dtype=int
        )
        centroids, _ = cluster_and_get_centroids(filtered_coords, f"data_plus_{name}")
        mask_centroids_list.append(centroids)

    # Combine all centroids
    all_centroids = [data_centroids] + mask_centroids_list
    combined_centroids = np.vstack([c for c in all_centroids if c.size > 0])
    combined_centroids = remove_duplicates(combined_centroids, voxel_size, tol=1.0)

    nn = NearestNeighbors(radius=CONNECT_DIST)
    nn.fit(combined_centroids)
    adjacency = nn.radius_neighbors_graph(combined_centroids, mode='connectivity')
    n_components, comp_labels = connected_components(adjacency, directed=False)
    largest_label = np.argmax(np.bincount(comp_labels))
    largest_mask = comp_labels == largest_label
    largest_centroids = combined_centroids[largest_mask]

    # Save final largest CC
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdb_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE_NAME}_combined_largestCC_{timestamp}.pdb")
    with open(pdb_path,'w') as f:
        for i,(z,y,x) in enumerate(largest_centroids,1):
            f.write(f"HETATM{i:5d}  C   UNK     1    {x:8.3f}{y:8.3f}{z:8.3f} 1.00 0.00           C\n")
    print(f"Saved final largest CC PDB with {len(largest_centroids)} nodes to {pdb_path}")

    return {
        "centroids": data_centroids,
        "connect_dist": CONNECT_DIST,
        "output_dir": OUTPUT_DIR,
        "output_name": all_ca_pdb_path,
    }

if __name__ == "__main__":
    main()
