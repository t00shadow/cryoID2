### About
# [This is NOT the original for backbone C-alphas]
# This file is for clustering residue atoms only
# There's 3 clustering files (the original for backbone C-alphas, this derivative for residue atoms only, and the other derivative which is bbasically the original with more stuff)

import numpy as np
import mrcfile
import os
from sklearn.cluster import MeanShift, AgglomerativeClustering
from collections import defaultdict
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components

# ----------------------------- Configuration -----------------------------
from H_Backbone_tracing.archived.old_scheduled_for_deletion.A0_config_loader import load_sample_paths, load_config_section

config_file = "H_Backbone_tracing/A0_config.yaml"

FILE_TOGGLE = "emd_34843"

paths_cfg = load_sample_paths(config_file, FILE_TOGGLE)
INPUT_MRC = paths_cfg["CA_mrc"]    #! this gets overrided in main() btw, cuz need to use a loop
MASK_HELIX_MRC = paths_cfg["mask_helix_mrc"]
MASK_SHEET_MRC = paths_cfg["mask_sheet_mrc"]
MASK_RING_MRC = paths_cfg["mask_ring_mrc"]
ORIGINAL_MAP_PATH = paths_cfg["original_map"]    # ... this isnt even used anymore, BUT can add it back as a mask to remove artifacts. should help for clustering the residues

clustering_cfg = load_config_section(config_file, "clustering")
# OUTPUT_DIR = clustering_cfg["output_dir"]
OUTPUT_DIR = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\RESIDUE_ATOMS"    #! manual override
THRESHOLD = clustering_cfg["threshold"]
CLUSTERING_METHOD = clustering_cfg["method"]
if CLUSTERING_METHOD == "agglomerative":
    # CONNECT_DIST = clustering_cfg["connect_dist"]
    CONNECT_DIST = 1.5    #! manual override, use less than the value used for c-alpha clustering cuz c-alpha to c-alpha skips two atoms
    OUTPUT_FILE_NAME = f"{FILE_TOGGLE}_agglomerative_cd_{CONNECT_DIST}"
elif CLUSTERING_METHOD == "meanshift":
    BANDWIDTH = clustering_cfg["bandwidth"]
    OUTPUT_FILE_NAME = f"{FILE_TOGGLE}_meanshift_bw_{BANDWIDTH}"
# MIN_CLUSTER_SIZE = clustering_cfg["min_cluster_size"]
MIN_CLUSTER_SIZE = 4    #! manual override

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
def main_helper(residue_number=1):
    INPUT_MRC = f"D:/NEW_TEST_DATA/UnetPrediction/EMD-34843_re_3.2/possibility/emd_34843.map_residue_types_{residue_number}_possibility.mrc"

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
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # pdb_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE_NAME}_{tag}_{timestamp}.pdb")
        pdb_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE_NAME}_{tag}_residue_{residue_number}.pdb")
        with open(pdb_path, 'w') as f:
            for i, (z, y, x) in enumerate(centroids, 1):
                f.write(f"HETATM{i:5d}  C   UNK     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        print(f"Saved PDB with {len(centroids)} centroids for {tag} to {pdb_path}")

        return centroids

    # 1) Data alone (no mask)
    print("\nClustering data alone...")
    data_centroids = cluster_and_get_centroids(volume_coords, "data_alone")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for i in range(1, 21):
        print(f"\n> Residue {i}:")
        main_helper(residue_number=i)


if __name__ == "__main__":
    main()
