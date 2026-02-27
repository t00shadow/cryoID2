import os
import numpy as np
from collections import defaultdict
from datetime import datetime
from sklearn.cluster import MeanShift, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components

from A0_utils import (
    load_thresholded_coords,
)

# ----------------------------- Configuration -----------------------------
# REQUIRED: change this path to the ...key_atoms_1_possibility.mrc of desired sample.
# Also that long string after shortcut-targets-by-id might differ for you
INPUT_MRC = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\TestSet\Raw\EMD-15903_re_3.0\possibility\emd_15903.map_key_atoms_1_possibility.mrc"

# Optional: change to wherever you want the output saved
OUTPUT_DIR = "H_Backbone_tracing/misc_outputs"

# Optional: can try meanshift if agglomerative is too slow
CLUSTERING_METHOD = "agglomerative"   # chose from ["agglomerative", "meanshift"]


# You can leave the rest as is
THRESHOLD = 0.9     # ranges from 0.0 to 1.0, possibility threshold for key_atoms_1_possibility.mrc
emdb_id = INPUT_MRC.split("emd_")[-1].split(".")[0]
# print(emdb_id); assert(False)
TIMESTAMP = False     # adds a timestamp to the file if True
if CLUSTERING_METHOD == "agglomerative":
    CONNECT_DIST = 4.0   # for agglomerative only
    OUTPUT_FILE_NAME = f"emd_{emdb_id}_agglomerative_cd_{CONNECT_DIST}"
elif CLUSTERING_METHOD == "meanshift":
    BANDWIDTH = 1.9      # for meanshift only
    OUTPUT_FILE_NAME = f"emd_{emdb_id}_meanshift_bw_{BANDWIDTH}"
MIN_CLUSTER_SIZE = 5

MIN_SPACING = 3.0  # Å minimum allowed distance between centroids

# ----------------------------- Slice configuration -----------------------------
z_slice = (0, 999)
y_slice = (0, 999)
x_slice = (0, 999)

# ----------------------------- Helper: local thinning -----------------------------
def local_thinning(centroids, min_dist=MIN_SPACING):
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

# ----------------------------- Main clustering function -----------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    import mrcfile
    with mrcfile.open(INPUT_MRC, permissive=True) as mrc:
        voxel_size = float(mrc.voxel_size.x)
        data_shape = mrc.data.shape

    print(f"Volume shape: {data_shape}, voxel size: {voxel_size}")

    # ----------------- Load thresholded coordinates -----------------
    print("Loading volume coords (chunked)...")
    volume_coords = load_thresholded_coords(INPUT_MRC, THRESHOLD, z_slice, y_slice, x_slice)
    print(f"Volume voxels above threshold: {len(volume_coords)}")

    def cluster_and_get_centroids(coords, tag):
        if len(coords) == 0:
            print(f"No voxels to cluster for {tag}")
            return np.empty((0,3), dtype=float), None

        coords_real = np.empty_like(coords, dtype=float)
        # Preserve original mapping: z, y, x -> x, y, z
        coords_real[:,0] = coords[:,2] * voxel_size
        coords_real[:,1] = coords[:,1] * voxel_size
        coords_real[:,2] = coords[:,0] * voxel_size

        # Initial clustering
        if CLUSTERING_METHOD == "meanshift":
            clustering = MeanShift(bandwidth=BANDWIDTH, bin_seeding=True)
            labels = clustering.fit_predict(coords_real)
        elif CLUSTERING_METHOD == "agglomerative":
            clustering = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=CONNECT_DIST,
                linkage='complete'
            )
            labels = clustering.fit_predict(coords_real)
        else:
            labels = np.zeros(len(coords_real), dtype=int)

        clusters = defaultdict(list)
        for coord, lbl in zip(coords, labels):
            clusters[lbl].append(coord)

        filtered_clusters = {
            lbl: np.array(pts)
            for lbl, pts in clusters.items()
            if len(pts) >= MIN_CLUSTER_SIZE
        }

        centroids = []
        for pts in filtered_clusters.values():
            centroid = np.mean(pts * voxel_size, axis=0)
            centroids.append(centroid)
        centroids = np.array(centroids) if centroids else np.empty((0,3), dtype=float)
        print(f"{tag}: {len(centroids)} centroids before local thinning")

        # ----------------- Local thinning -----------------
        centroids = local_thinning(centroids, min_dist=MIN_SPACING)
        print(f"{tag}: {len(centroids)} centroids after local thinning")

        # Save PDB
        if TIMESTAMP:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            pdb_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE_NAME}_{tag}_{timestamp}.pdb")
        else:
            pdb_path = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE_NAME}_{tag}.pdb")
        with open(pdb_path,'w') as f:
            for i, (z,y,x) in enumerate(centroids,1):
                f.write(f"HETATM{i:5d}  CA  UNK     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        print(f"Saved PDB with {len(centroids)} centroids for {tag} to {pdb_path}")

        return centroids, pdb_path

    # ----------------- Cluster CA -----------------
    print("\nClustering CA ...")
    ca_centroids, all_ca_pdb_path = cluster_and_get_centroids(volume_coords, "CA_only")

    return {
        "centroids": ca_centroids,
        # "connect_dist": CONNECT_DIST,
        # "output_dir": OUTPUT_DIR,
        "output_path": all_ca_pdb_path,
    }

if __name__ == "__main__":
    main()