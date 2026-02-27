import numpy as np
import mrcfile
from sklearn.neighbors import NearestNeighbors
from scipy.sparse.csgraph import connected_components

# ----------------------------- File I/O / helper functions -----------------------------

def load_thresholded_coords(path, threshold, z_slice=None, y_slice=None, x_slice=None):
    """
    Load MRC volume, threshold it, and return coordinates above threshold.
    """
    print(path)
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.squeeze(mrc.data)      # removes singleton dimension
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
    """
    Return the intersection of multiple sets of coordinates.
    """
    if not coords_list:
        return np.empty((0, 3), dtype=int)
    sets = [set(map(tuple, coords)) for coords in coords_list]
    intersected = set.intersection(*sets)
    return np.array(list(intersected), dtype=int)

def save_mask_mrc_from_coords(coords, shape, voxel_size, filename):
    """
    Save a binary mask to an MRC file from coordinates.
    """
    mask = np.zeros(shape, dtype=np.uint8)
    if coords.size > 0:
        mask[tuple(coords.T)] = 1
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(mask)
        mrc.voxel_size = (voxel_size, voxel_size, voxel_size)

def save_mask_pdb(coords, voxel_size, filename):
    """
    Save coordinates to a PDB file as HETATM atoms.
    """
    coords_real = np.empty_like(coords, dtype=float)
    coords_real[:, 0] = coords[:, 2] * voxel_size
    coords_real[:, 1] = coords[:, 1] * voxel_size
    coords_real[:, 2] = coords[:, 0] * voxel_size
    with open(filename, 'w') as f:
        for i, (x, y, z) in enumerate(coords_real, 1):
            f.write(f"HETATM{i:5d}  C   UNK     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")

def remove_duplicates(coords, voxel_size, tol=1.0):
    """
    Remove near-duplicate centroids within 'tol' angstroms.
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
