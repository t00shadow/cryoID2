import numpy as np
import mrcfile
from scipy.ndimage import gaussian_filter

def read_mrc(input_mrc, pre_threshold=0.0):
    """
    Read a 3D density map from an MRC file and apply optional pre-thresholding.

    Parameters
    ----------
    input_mrc : str
        Path to the input MRC file.
    pre_threshold : float
        Density values below this are set to zero.

    Returns
    -------
    volume : np.ndarray
        3D density map after squeezing and thresholding.
    """
    with mrcfile.open(input_mrc, permissive=True) as mrc:
        volume = np.copy(mrc.data)
        print("Original shape:", volume.shape)
        volume = np.squeeze(volume)
        print("Squeezed shape:", volume.shape)

    # Apply pre-threshold
    if pre_threshold > 0.0:
        volume[volume < pre_threshold] = 0.0
        print(f"Applied pre-threshold: values below {pre_threshold} set to 0")

    return volume


def immersive_watershed(density_map, connectivity=26):
    """
    3D immersive watershed for a density map.
    """
    shape = density_map.shape
    labels = np.zeros(shape, dtype=int)
    current_label = 1

    # Flatten and sort voxels by descending density
    indices = np.array(np.unravel_index(np.argsort(density_map.ravel())[::-1], shape)).T

    # Define neighbor offsets
    if connectivity == 6:
        neighbors = [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]
    elif connectivity == 18:
        neighbors = [(i,j,k) for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1]
                     if (i,j,k)!=(0,0,0) and abs(i)+abs(j)+abs(k)<=2]
    elif connectivity == 26:
        neighbors = [(i,j,k) for i in [-1,0,1] for j in [-1,0,1] for k in [-1,0,1]
                     if (i,j,k)!=(0,0,0)]
    else:
        raise ValueError("Connectivity must be 6, 18, or 26")

    # Immersive flooding
    for idx in indices:
        x, y, z = idx
        if density_map[x, y, z] == 0:
            continue  # Skip masked-out voxels
        neighbor_labels = []
        for dx, dy, dz in neighbors:
            nx, ny, nz = x + dx, y + dy, z + dz
            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                if labels[nx, ny, nz] > 0:
                    neighbor_labels.append(labels[nx, ny, nz])
        if not neighbor_labels:
            labels[x, y, z] = current_label
            current_label += 1
        else:
            # Assign to the neighbor region with most adjacent voxels
            counts = {l: neighbor_labels.count(l) for l in set(neighbor_labels)}
            max_label = max(counts, key=counts.get)
            labels[x, y, z] = max_label

    return labels


def segment_mrc(input_mrc, output_mrc, sigma=2, threshold=0.0, pre_threshold=0.9, connectivity=26):
    # Read MRC file
    volume = read_mrc(input_mrc, pre_threshold=pre_threshold)

    # Smooth
    smoothed = gaussian_filter(volume, sigma=sigma)

    # Threshold mask
    mask = smoothed > threshold
    smoothed[~mask] = 0.0

    # Run immersive watershed
    labels = immersive_watershed(smoothed, connectivity=connectivity)
    print("Segmentation done. Unique regions:", np.unique(labels).size)

    # Save segmented map
    with mrcfile.new(output_mrc, overwrite=True) as mrc:
        mrc.set_data(labels.astype(np.int16))

    print(f"Saved segmentation to {output_mrc}")


# Example usage
if __name__ == "__main__":
    input_file = r"D:\NEW_TEST_DATA\UnetPrediction\EMD-21887_re_3.2\possibility\emd_21887.map_secondary_strctures_3_possibility.mrc"
    # segment_mrc(input_file, "z_segmentation.mrc", sigma=9, threshold=0.1, pre_threshold=0.9)
    # segment_mrc(input_file, "z_segmentation.mrc", sigma=8, threshold=0.1, pre_threshold=0.7, connectivity=6)
    segment_mrc(input_file, "z_segmentation.mrc", sigma=8, threshold=0.1, pre_threshold=0.7)
