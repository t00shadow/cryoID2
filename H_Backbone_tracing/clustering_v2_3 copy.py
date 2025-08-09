import numpy as np
import mrcfile
import os
from sklearn.cluster import MeanShift
from collections import defaultdict
from datetime import datetime
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components

# -----------------------------
# CONFIGURATION
# -----------------------------
INPUT_MRC = r"D:\NEW_TEST_DATA\UnetPrediction\EMD-27018_re_3.0\possibility\emd_27018.map_key_atoms_1_possibility.mrc"
THRESHOLD = 0.8                # Density threshold
BANDWIDTH = 1.9               # MeanShift bandwidth
MIN_CLUSTER_SIZE = 5           # Minimum voxels per cluster to keep
CONNECT_DIST = 4.0             # Max distance to connect clusters (Angstroms)
OUTPUT_DIR = 'H_Backbone_tracing/outputs/clustering_v2'
OUTPUT_FILE_NAME = f"meanshift_largestCC_bw{BANDWIDTH}"
SAVE_INDIVIDUAL_MRCS = False
MAX_CLUSTERS_TO_SAVE = 50

# Define slices (Z, Y, X)
z_slice = (200, 250)
y_slice = (150, 200)
x_slice = (150, 200)

# -----------------------------
# STEP 1: Load and threshold
# -----------------------------
with mrcfile.open(INPUT_MRC) as mrc:
    data = mrc.data.copy()
    voxel_size = float(mrc.voxel_size.x)  # isotropic

print(f"Original data shape: {data.shape}")

if data.ndim == 4:
    data = data[0]
    print(f"Reduced data shape: {data.shape}")

coords = np.argwhere(data > THRESHOLD)
print(f"Total coords above threshold: {len(coords)}")

# -----------------------------
# STEP 2: Slice filtering
# -----------------------------
mask = (
    (coords[:, 0] >= z_slice[0]) & (coords[:, 0] < z_slice[1]) &
    (coords[:, 1] >= y_slice[0]) & (coords[:, 1] < y_slice[1]) &
    (coords[:, 2] >= x_slice[0]) & (coords[:, 2] < x_slice[1])
)
coords_filtered = coords[mask]
print(f"Coords inside slice: {len(coords_filtered)}")

if len(coords_filtered) == 0:
    raise ValueError("No voxels in the selected slice.")

# -----------------------------
# STEP 3: Convert to real-world coords (x, y, z)
# -----------------------------
coords_real = np.empty_like(coords_filtered, dtype=float)
coords_real[:, 0] = coords_filtered[:, 2] * voxel_size  # x
coords_real[:, 1] = coords_filtered[:, 1] * voxel_size  # y
coords_real[:, 2] = coords_filtered[:, 0] * voxel_size  # z

# -----------------------------
# STEP 4: MeanShift clustering
# -----------------------------
clustering = MeanShift(bandwidth=BANDWIDTH, bin_seeding=True)
labels = clustering.fit_predict(coords_real)
print(f"Total clusters found (including small): {len(set(labels))}")

# -----------------------------
# STEP 5: Group points by cluster and filter by size
# -----------------------------
clusters = defaultdict(list)
for coord, label in zip(coords_filtered, labels):
    clusters[label].append(coord)

filtered_clusters = {
    label: np.array(pts)
    for label, pts in clusters.items()
    if len(pts) >= MIN_CLUSTER_SIZE
}
print(f"Clusters after filtering by size: {len(filtered_clusters)}")

# Compute centroids of filtered clusters in real-world coords (x,y,z)
centroids = {
    label: np.mean(pts * voxel_size, axis=0)
    for label, pts in filtered_clusters.items()
}
centroid_coords = np.array(list(centroids.values()))
print(f"Number of centroids: {len(centroid_coords)}")

# -----------------------------
# STEP 6: Build connectivity graph on centroids
# -----------------------------
nn = NearestNeighbors(radius=CONNECT_DIST)
nn.fit(centroid_coords)
adjacency = nn.radius_neighbors_graph(centroid_coords, mode='connectivity')

# Find connected components
n_components, comp_labels = connected_components(csgraph=adjacency, directed=False)
print(f"Found {n_components} connected components among clusters")

# -----------------------------
# STEP 7: Keep only the largest connected component
# -----------------------------
component_sizes = np.bincount(comp_labels)
largest_component_label = np.argmax(component_sizes)
print(f"Largest connected component label: {largest_component_label} with size {component_sizes[largest_component_label]}")

largest_mask = comp_labels == largest_component_label
largest_centroids = centroid_coords[largest_mask]

# -----------------------------
# STEP 8: Save largest component centroids as PDB
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
pdb_path = os.path.join(OUTPUT_DIR, OUTPUT_FILE_NAME + "_" + timestamp + ".pdb")

with open(pdb_path, 'w') as f:
    for i, (z, y, x) in enumerate(largest_centroids, 1):
        f.write(f"HETATM{i:5d}  C   UNK     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")

print(f"Saved PDB with {len(largest_centroids)} centroids (largest connected component) to {pdb_path}")

# -----------------------------
# STEP 9: Optional - Save individual cluster masks as .mrc files
# -----------------------------
# if SAVE_INDIVIDUAL_MRCS:
#     for i, (label, voxels) in enumerate(filtered_clusters.items()):
#         if i >= MAX_CLUSTERS_TO_SAVE:
#             break
#         cluster_mask = np.zeros_like(data, dtype=np.uint8)
#         cluster_mask[tuple(voxels.T)] = 1
#         out_path = os.path.join(OUTPUT_DIR, f'cluster_{label:04d}.mrc')
#         with mrcfile.new(out_path, overwrite=True) as mrc:
#             mrc.set_data(cluster_mask.astype(np.uint8))
#             mrc.voxel_size = (voxel_size, voxel_size, voxel_size)
#     print(f"Saved individual cluster .mrcs (up to {MAX_CLUSTERS_TO_SAVE})")
