import os
import datetime
import numpy as np
import mrcfile
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree
from R_A0_utils import parse_calpha_PDB, load_thresholded_coords

def mask_pdb_with_mrc(pdb_file, mrc_file, mask_threshold, radius, percent):
    """
    Mask PDB CA atoms based on nearby MRC voxel density.

    Args:
        pdb_file (str): path to PDB
        mrc_file (str): path to MRC
        mask_threshold (float): density threshold for MRC
        radius (float): radius (in Å) around each atom
        percent (float): fraction [0,1] of voxels required inside sphere

    Returns:
        np.ndarray: filtered CA coordinates (N, 3)
    """

    # --- load CA atoms ---
    ca_coords = parse_calpha_PDB(pdb_file)
    print(len(ca_coords))

    # --- load thresholded voxel indices ---
    voxel_indices = load_thresholded_coords(mrc_file, mask_threshold)

    # --- get voxel size ---
    with mrcfile.open(mrc_file, permissive=True) as mrc:
        voxel_size = np.array(mrc.voxel_size.tolist(), dtype=np.float32)
        # handle scalar voxel size (common case)
        if voxel_size.size == 1:
            voxel_size = np.array([voxel_size[0]] * 3)

    # --- convert voxel indices (z,y,x) → real coordinates (x,y,z) ---
    voxel_coords = voxel_indices[:, ::-1] * voxel_size  # flip to (x,y,z)

    # --- build KD-tree ---
    tree = cKDTree(voxel_coords)

    # --- precompute sphere volume estimate ---
    sphere_volume = (4/3) * np.pi * radius**3
    voxel_volume = np.prod(voxel_size)
    expected_voxels = sphere_volume / voxel_volume
    print(sphere_volume)
    print(voxel_volume)
    print(expected_voxels)

    kept_atoms = []

    for coord in ca_coords:
        neighbors = tree.query_ball_point(coord, radius)
        count = len(neighbors)

        if count / expected_voxels >= percent:
            kept_atoms.append(coord)

    print(len(kept_atoms))
    return np.array(kept_atoms, dtype=np.float32)


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


def mask_and_save_pdb(pdb_file, mrc_file, output_dir, output_name, mask_threshold, radius, percent, timestamp=False):
    """
    Runs masking and saves result as a new PDB.

    Returns:
        path to saved pdb
    """

    # --- run masking ---
    filtered_coords_xyz = mask_pdb_with_mrc(
        pdb_file=pdb_file, 
        mrc_file=mrc_file, 
        mask_threshold=mask_threshold, 
        radius=radius, 
        percent=percent
    )

    if len(filtered_coords_xyz) == 0:
        print("Warning: no atoms passed the mask.")
        return None

    # --- convert (x,y,z) -> (z,y,x) for save_pdb ---
    filtered_coords_zyx = filtered_coords_xyz[:, ::-1]

    # --- ensure output dir exists ---
    os.makedirs(output_dir, exist_ok=True)

    # --- save ---
    output_path = save_pdb(
        filtered_coords_zyx,
        output_dir,
        output_name,
        timestamp=timestamp,
    )

    print(output_path)
    return output_path


if __name__ == "__main__":
    pdb_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\E1_D3_meanshift_bw_1.9_CA_only.pdb"
    # pdb_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\superset_graph_degree_limited.pdb"
    mrc_file = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\real_malaria\3-E1_D3\possibility\E1_D3_2_7A_cryosparc_P8_J562_008_volume_map_sharp_secondary_strctures_1_possibility.mrc"
    mask_threshold = 0.7
    radius = 1.5
    percent = 0.5
    
    # result = mask_pdb_with_mrc(pdb_file=pdb_file, mrc_file=mrc_file, mask_threshold=mask_threshold, radius=radius, percent=percent)

    out_path = mask_and_save_pdb(
        pdb_file=pdb_file,
        mrc_file=mrc_file,
        output_dir=r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs",
        output_name="E1_D3_calpha_clustered_then_masked",
        mask_threshold=mask_threshold,
        radius=radius,
        percent=percent,
        timestamp=False,
    )