import os
import mrcfile
import numpy as np
import clustering_v2_3__copy__3_2__copy__8_25_25 as clustering  # <-- this is your existing file
import traceback

def run_sliding_window(cube_size=50, overlap=20, output_dir=None):
    """
    Wrapper to run clustering_main.main3() over sliding windows of the volume.
    Args:
        cube_size (int): size of each cubic chunk in voxels
        overlap (int): overlap between cubes in voxels
        output_dir (str): optional override for clustering.OUTPUT_DIR
    """

    # Override output dir if specified
    if output_dir is not None:
        clustering.OUTPUT_DIR = output_dir
        os.makedirs(clustering.OUTPUT_DIR, exist_ok=True)

    # Load shape + voxel size from INPUT_MRC in clustering
    with mrcfile.open(clustering.INPUT_MRC, permissive=True) as mrc:
        data_shape = np.squeeze(mrc.data).shape
        print("Data shape:", data_shape)
        voxel_size = float(mrc.voxel_size.x)
        print("single dimension:", voxel_size)

    print(f"Full volume shape: {data_shape}, voxel size: {voxel_size}")
    print(f"Sliding window with cube_size={cube_size}, overlap={overlap}")

    z_max, y_max, x_max = data_shape

    # Step size = cube_size - overlap
    step = cube_size - overlap

    # Iterate over sliding windows
    for z_start in range(0, z_max, step):
        for y_start in range(0, y_max, step):
            for x_start in range(0, x_max, step):
                z_end = min(z_start + cube_size, z_max)
                y_end = min(y_start + cube_size, y_max)
                x_end = min(x_start + cube_size, x_max)

                clustering.z_slice = (z_start, z_end)
                clustering.y_slice = (y_start, y_end)
                clustering.x_slice = (x_start, x_end)

                print(f"\nRunning window: "
                      f"z={clustering.z_slice}, y={clustering.y_slice}, x={clustering.x_slice}")

                # Run clustering
                # clustering.main3()
                try:
                    clustering.main3()
                except ValueError as e:
                    if "need at least one array to concatenate" in str(e):
                        print("  ⚠️ No centroids in this window, skipping.")
                        continue
                    else:
                        print("  ❌ Unexpected error in this window:")
                        traceback.print_exc()
                        continue




def main():
    import datetime

    # Base output directory
    base_output_dir = "H_Backbone_tracing/outputs/clustering_v2/full_input_file"

    # Create timestamped subfolder
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Outputs will be saved in: {output_dir}")

    # Example usage of wrapper
    run_sliding_window(
        cube_size=50,
        overlap=20,
        output_dir=output_dir
    )


if __name__ == "__main__":
    main()