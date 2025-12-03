### About
# [This is NOT the original for backbone C-alphas]
# This file is for clustering C-alpha to help with clustering residues
# There's 3 clustering files (the original for backbone C-alphas, the other derivative for residue atoms only, and this derivative which is bbasically the original with more stuff)

import numpy as np
import mrcfile
import os
from sklearn.cluster import MeanShift, AgglomerativeClustering
from collections import defaultdict
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components

# ----------------------------- Configuration -----------------------------
from A0_config_loader import load_sample_paths, load_config_section

config_file = "H_Backbone_tracing/A0_config.yaml"

FILE_TOGGLE = "emd_34843"

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
CONNECT_DIST = clustering_cfg["connect_dist"]   # need it for both because connected graph uses this too, # TODO: should maybe give it its own parameter
if CLUSTERING_METHOD == "agglomerative":
    OUTPUT_FILE_NAME = f"{FILE_TOGGLE}_agglomerative_cd_{CONNECT_DIST}"
elif CLUSTERING_METHOD == "meanshift":
    BANDWIDTH = clustering_cfg["bandwidth"]
    OUTPUT_FILE_NAME = f"{FILE_TOGGLE}_meanshift_bw_{BANDWIDTH}"
MIN_CLUSTER_SIZE = clustering_cfg["min_cluster_size"]

#^ emd_21887
# z_slice = (150, 200)
z_slice = (200, 250)
y_slice = (150, 200)
x_slice = (150, 200)

z_slice = (190, 250)
# z_slice = (100, 250)    # runs in about the same amt of time
y_slice = (140, 200)
x_slice = (140, 200)

#^ emd_34843
z_slice = (50, 150)
# z_slice = (100, 250)    # runs in about the same amt of time
y_slice = (50, 150)
x_slice = (50, 150)

z_slice = (0, 999)    # max size is 208 but going over is fine since the code will fix it
y_slice = (0, 999)
x_slice = (0, 999)

# ----------------------------- Helper functions -----------------------------

def load_thresholded_coords(path, threshold=THRESHOLD, z_slice=None, y_slice=None, x_slice=None):
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.squeeze(mrc.data)      # removes the singleton (1,) dimension
        z_slice = z_slice or (0, data.shape[0])
        y_slice = y_slice or (0, data.shape[1])
        x_slice = x_slice or (0, data.shape[2])
        data_chunk = data[z_slice[0]:z_slice[1],
                          y_slice[0]:y_slice[1],
                          x_slice[0]:x_slice[1]]
        mask = data_chunk > threshold
        coords = np.argwhere(mask)
        coords[:, 0] += z_slice[0]
        coords[:, 1] += y_slice[0]
        coords[:, 2] += x_slice[0]
    return coords

def intersect_coords(coords_list):
    if not coords_list:
        return np.empty((0, 3), dtype=int)
    sets = [set(map(tuple, coords)) for coords in coords_list]
    intersected = set.intersection(*sets)
    return np.array(list(intersected), dtype=int)

def save_mask_mrc_from_coords(coords, shape, voxel_size, filename):
    mask = np.zeros(shape, dtype=np.uint8)
    if coords.size > 0:
        mask[tuple(coords.T)] = 1
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(mask)
        mrc.voxel_size = (voxel_size, voxel_size, voxel_size)

def save_mask_pdb(coords, voxel_size, filename):
    coords_real = np.empty_like(coords, dtype=float)
    coords_real[:, 0] = coords[:, 2] * voxel_size
    coords_real[:, 1] = coords[:, 1] * voxel_size
    coords_real[:, 2] = coords[:, 0] * voxel_size
    with open(filename, 'w') as f:
        for i, (x, y, z) in enumerate(coords_real, 1):
            f.write(f"HETATM{i:5d}  C   UNK     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")



def remove_duplicates(coords, voxel_size, tol=1.0):
    """
    Remove near-duplicate centroids within 'tol' angstroms. (More flexible than np.unique)
    Uses voxel_size to keep units consistent.
    """
    if len(coords) == 0:
        return coords

    nn = NearestNeighbors(radius=tol)
    nn.fit(coords)
    adjacency = nn.radius_neighbors_graph(coords).tocoo()

    n_components, labels = connected_components(csgraph=adjacency, directed=False)
    deduped = []
    for comp_id in range(n_components):
        members = coords[labels == comp_id]
        deduped.append(members.mean(axis=0))  # replace cluster of duplicates with average
    return np.array(deduped)



# ----------------------------- main: Improved clustering with centroids -----------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load voxel size and shape from INPUT_MRC header
    with mrcfile.open(INPUT_MRC, permissive=True) as mrc:
        voxel_size = float(mrc.voxel_size.x)
        data_shape = mrc.data.shape

    print(f"Volume shape: {data_shape}, voxel size: {voxel_size}")

    # Load thresholded coords of base volume chunk (same for all masks)
    print("Loading volume coords (chunked)...")
    volume_coords = load_thresholded_coords(INPUT_MRC, THRESHOLD, z_slice, y_slice, x_slice)
    print(f"Volume voxels above threshold: {len(volume_coords)}")

    # Helper to cluster coords and return centroids without filtering largest connected component
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
            # clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=CONNECT_DIST, linkage='average')
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

        # Compute centroids of filtered clusters in real-world coords
        centroids = []
        for pts in filtered_clusters.values():
            centroid = np.mean(pts * voxel_size, axis=0)  # shape (3,)
            centroids.append(centroid)
        centroids = np.array(centroids) if centroids else np.empty((0, 3), dtype=float)

        # Save PDB of all centroids for this tag
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdb_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE_NAME}_{tag}_{timestamp}.pdb")
        with open(pdb_path, 'w') as f:
            for i, (z, y, x) in enumerate(centroids, 1):
                f.write(f"HETATM{i:5d}  C   UNK     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        print(f"Saved PDB with {len(centroids)} centroids for {tag} to {pdb_path}")

        return centroids

    # 1) Data alone (no mask)
    print("\nClustering data alone...")
    data_centroids = cluster_and_get_centroids(volume_coords, "data_alone")

    # Masks
    ss_masks = {
        # 'helix': MASK_HELIX_MRC,
        # 'sheet': MASK_SHEET_MRC,
        'loop': MASK_RING_MRC,  # assuming loop = ring mask
    }

    mask_centroids_list = []

    for name, path in ss_masks.items():
        print(f"\nProcessing data + {name} mask, path: {path} ...")

        # Load and threshold mask chunk
        mask_coords = load_thresholded_coords(path, THRESHOLD, z_slice, y_slice, x_slice)
        print(f"Mask '{name}' voxels above threshold: {len(mask_coords)}")

        # Intersect mask coords with volume coords to keep only voxels present in both
        volume_set = set(map(tuple, volume_coords))
        mask_set = set(map(tuple, mask_coords))
        filtered_coords = np.array(list(volume_set & mask_set), dtype=int)

        removed = len(volume_coords) - len(filtered_coords)
        print(f"Voxels removed after applying {name} mask: {removed}")

        # Save PDB of filtered coords (optional)
        # save_mask_pdb(filtered_coords, voxel_size, os.path.join(OUTPUT_DIR, f"{name}_filtered.pdb"))     #! can toggle off for now, it's just the mask

        # Cluster filtered coords and get centroids
        centroids = cluster_and_get_centroids(filtered_coords, f"data_plus_{name}")
        mask_centroids_list.append(centroids)

    # Combine all centroids: data alone + each mask centroids
    all_centroids = [data_centroids] + mask_centroids_list      # effectively uses all coords rn, since LHS has all coords, and RHS is just a subset* of LHS (*well maybe not perfectly aligned since I threshold first then cluster)
    # all_centroids = [data_centroids]    #! right now its equivalent to this
    # all_centroids = mask_centroids_list    #! use this if want ONLY the masked stuff
    combined_centroids = np.vstack([c for c in all_centroids if c.size > 0])

    print(f"\nTotal combined centroids before connected components: {len(combined_centroids)}")

    if combined_centroids.size == 0:
        print("No centroids to process in combined data. Exiting.")
        return

    # Remove duplicate coords
    print(f"Before deduplication: {len(combined_centroids)}")
    combined_centroids = remove_duplicates(combined_centroids, voxel_size, tol=1.0)
    print(f"After deduplication: {len(combined_centroids)}")

    # Build connectivity graph on combined centroids
    nn = NearestNeighbors(radius=CONNECT_DIST)
    nn.fit(combined_centroids)
    adjacency = nn.radius_neighbors_graph(combined_centroids, mode='connectivity')

    n_components, comp_labels = connected_components(csgraph=adjacency, directed=False)

    # 9/27/25 addition START (only for helping with residue clustering the amino nitrogen and carboxylic carbon)
    # 🔹 Save full connected graph (all centroids + edges)
    adjacency_coo = adjacency.tocoo()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_graph_pdb = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE_NAME}_combined_fullGraph_{timestamp}.pdb")

    with open(full_graph_pdb, "w") as f:
        # Write all centroids as atoms
        for i, (z, y, x) in enumerate(combined_centroids, 1):
            f.write(f"HETATM{i:5d}  C   UNK     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")

        # Write edges from adjacency
        for i, j in zip(adjacency_coo.row, adjacency_coo.col):
            if i < j:  # avoid duplicate edges
                f.write(f"CONECT{i+1:5d}{j+1:5d}\n")

    print(f"Saved full connected graph PDB with {len(combined_centroids)} centroids to {full_graph_pdb}")
    # 9/27/25 addition END


    print(f"Found {n_components} connected components among combined centroids")

    component_sizes = np.bincount(comp_labels)

    sorted_components = sorted(
        enumerate(component_sizes),
        key=lambda x: x[1],
        reverse=True
    )
    for label, size in sorted_components:
        if size > 10:
            print(f"Component {label}: size {size}")
    print("...")

    largest_component_label = np.argmax(component_sizes)
    print(f"Largest connected component label: {largest_component_label} with size {component_sizes[largest_component_label]}")

    largest_mask = comp_labels == largest_component_label
    largest_centroids = combined_centroids[largest_mask]

    # # remove duplicates   (#! 9/01/25: changed so that now we remove duplicates BEFORE building connected graph)
    # print(f"Largest CC centroids before deduplication: {len(largest_centroids)}")
    # largest_centroids = remove_duplicates(largest_centroids, voxel_size, tol=1.0)
    # print(f"Largest CC centroids after deduplication: {len(largest_centroids)}")


    # 8/18/25 addition
    # 🔹 Store edges of the GCC
    adjacency = adjacency.tocoo()
    gcc_edges = [
        (i, j)
        for i, j in zip(adjacency.row, adjacency.col)
        if i < j and largest_mask[i] and largest_mask[j]
    ]

    # Save largest connected component centroids as final PDB
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdb_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE_NAME}_combined_largestCC_{timestamp}.pdb")

    # 8/18/25 addition
    # Map old indices → new sequential atom indices
    old_to_new = {old_idx: new_idx+1 for new_idx, old_idx in enumerate(np.where(largest_mask)[0])}

    with open(pdb_path, 'w') as f:
        for i, (z, y, x) in enumerate(largest_centroids, 1):
            f.write(f"HETATM{i:5d}  C   UNK     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")

        # 8/18/25 addition
        # Write edges for GCC
        for i, j in gcc_edges:
            f.write(f"CONECT{old_to_new[i]:5d}{old_to_new[j]:5d}\n")

    print(f"Saved final combined largest connected component PDB with {len(largest_centroids)} centroids to {pdb_path}")

    return pdb_path


if __name__ == "__main__":
    main()
