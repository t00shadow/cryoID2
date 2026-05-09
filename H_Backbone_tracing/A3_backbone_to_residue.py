import mrcfile
import numpy as np
import pandas as pd
import os

def get_voxel_size(mrc_path):
    with mrcfile.open(mrc_path, permissive=True) as mrc:
        header = mrc.header
        # box dimensions in Angstroms
        cella = header.cella
        nx, ny, nz = header.nx, header.ny, header.nz
        voxel_size = (
            cella.x / nx,
            cella.y / ny,
            cella.z / nz
        )
        print(f"Main map dimensions: {nx} x {ny} x {nz}")
        print(f"Voxel size: {voxel_size}")
        return voxel_size, (nx, ny, nz)

def read_pdb_coords(pdb_file):
    coords = []
    with open(pdb_file, "r") as f:
        for line in f:
            # if line.startswith("ATOM") and line[13:15].strip() == "CA":
            if ( line.startswith("ATOM") and line[13:15].strip() == "CA" ) or \
                ( line.startswith("HETATM") and line[13:15].strip() == "C" ):    # A1_clustering.py does this (this is if you skip A2_....py), need to modify A1_clustering.py later
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                # coords.append((x, y, z))
                coords.append((z, y, x))    # flip from (x, y, z) -> (z, y, x) since the .mrc files are z. y, x
    return coords

def coords_to_voxels(coords, voxel_size, radius=1):
    """Convert Å coords → voxel indices and expand to neighborhood."""
    voxel_dict = {}
    for atom_id, (x, y, z) in enumerate(coords):
        i = int(round(x / voxel_size[0]))
        j = int(round(y / voxel_size[1]))
        k = int(round(z / voxel_size[2]))

        neighborhood = []
        for di in range(-radius, radius+1):
            for dj in range(-radius, radius+1):
                for dk in range(-radius, radius+1):
                    ni, nj, nk = i+di, j+dj, k+dk
                    neighborhood.append((ni, nj, nk))

        voxel_dict[atom_id] = neighborhood
    return voxel_dict

# TODO: this part is fine, but could also rewrite with Pathlib glob and my util fxns i just maed today (10/9/25). But wtv, gonna do a full refactor this weekend anyways.
def sample_residue_maps(voxel_dict, residue_dir, shape):    # TODO: maybe rename this fxn
    """
    For each atom (from voxel_dict), sample values from all residue_type .mrc maps.

    Args:
        voxel_dict (dict): {atom_id: [(i,j,k), ...]} voxel indices around each CA atom
        residue_dir (str): directory containing the residue_type maps
        shape (tuple): expected (x,y,z) shape of the maps (from header of original MRC)

    Returns:
        dict: {atom_id: {residue_file: [values...]}}
    """
    results = {atom_id: {} for atom_id in voxel_dict}

    # Only use residue_type files
    residue_files = [
        f for f in os.listdir(residue_dir)
        if f.endswith(".mrc") and "residue_types" in f
    ]

    for res_file in residue_files:
        print("-", res_file)
        res_path = os.path.join(residue_dir, res_file)
        with mrcfile.open(res_path, permissive=True) as mrc:
            data = mrc.data
            data = np.squeeze(data)
            print(f"{res_file}: {data.shape}")
            print("shape 1:", data.shape, ", shape 2:", shape)
            assert data.shape == shape, "Residue map shape mismatch!"

            for atom_id, voxels in voxel_dict.items():
                values = []
                for (i, j, k) in voxels:
                    if (0 <= i < shape[0] and
                        0 <= j < shape[1] and
                        0 <= k < shape[2]):
                        values.append(data[i, j, k])
                results[atom_id][res_file] = values

    return results

def average_residues(results, save_file=False, file_name="residue_sequence.txt"):
    """
    For each atom, average the probabilities from each residue_type map,
    print the averages, and return the predicted residue sequence.
    
    Args:
        results (dict): output from sample_residue_maps, 
                        {atom_id: {res_file: [values...]}}
    
    Returns:
        sequence (list of str): predicted residue for each atom
    """
    sequence = []
    for atom_id, res_data in results.items():
        avg_probs = {}
        for res_file, vals in res_data.items():
            avg_probs[res_file] = np.mean(vals) if vals else 0.0

        # Print averaged probabilities for debugging
        print(f"Residue {atom_id + 1}:")
        for res_file, avg in avg_probs.items():
            print(f"  {res_file}: {avg:.4f}")

        # Choose residue with highest average probability
        best_residue = max(avg_probs, key=avg_probs.get)
        sequence.append(best_residue)

    seq_str = "-".join(sequence)
    print("\nPredicted residue sequence:")
    print(seq_str)

    # Optionally save to file
    if save_file:
        with open(file_name, "w") as f:
            f.write(seq_str + "\n")

    return sequence





# v2 of average_residues that uses the actual residue names
import re

residues_protein = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
residues_RNA = ['A', 'G', 'C', 'U']

def map_residue_file_to_code(res_file):
    """
    Convert a residue_type filename to actual residue code.
    Assumes files are numbered like ..._1_, ..._2_, etc.
    """
    # match = re.search(r"_([0-9]+)_", res_file)
    match = re.search(r"_([0-9]+)_possibility\.mrc$", res_file)    # more robust regex (keeping $ anchor since all res possibilities as of now end like this, also enforces a more consistent naming convention. All naming conventions should be controlled by a config file once all backend is integrated together), cuz new file names have ints (from 1-20, ex. 006) surrounded by underscores too. Alternatively, can consider tokenization instead of regex (like how I do in extract_number_from_file_name() of A0_utils_pdbtofasta.py)
    if not match:
        raise ValueError(f"Cannot extract index from filename: {res_file}")
    idx = int(match.group(1)) - 1  # convert to 0-based
    if idx < 20:
        return residues_protein[idx]
    elif idx < 24:
        return residues_RNA[idx - 20]
    else:
        raise ValueError(f"Index {idx} out of range for known residues")

def average_residues_AND_USE_RESIDUE_NAMES(results):
    """
    For each atom, average the probabilities from residue maps,
    print the averages, and return the predicted residue sequence
    using actual residue codes.
    """
    sequence = []
    for atom_id, res_data in results.items():
        avg_probs = {}
        for res_file, vals in res_data.items():
            avg_probs[res_file] = np.mean(vals) if vals else 0.0

        # Print averaged probabilities for debugging
        print(f"Residue {atom_id + 1}:")
        for res_file, avg in avg_probs.items():
            code = map_residue_file_to_code(res_file)
            print(f"  {code}: {avg:.4f}")

        # Choose residue with highest average probability
        best_file = max(avg_probs, key=avg_probs.get)
        best_residue = map_residue_file_to_code(best_file)
        sequence.append(best_residue)

    seq_str = "-".join(sequence)
    print("\nPredicted residue sequence:")
    print(seq_str)

    # Save to file
    with open("residue_sequence.txt", "w") as f:
        f.write(seq_str + "\n")

    return sequence

def average_residues_matrix(results):
    residues = residues_protein + residues_RNA
    n_atoms = len(results)
    
    # Initialize matrix (n_atoms x 24)
    matrix = np.zeros((n_atoms, len(residues)))
    
    for atom_id, res_data in results.items():
        for res_file, vals in res_data.items():
            avg = np.mean(vals) if vals else 0.0
            code = map_residue_file_to_code(res_file)
            col_idx = residues.index(code)
            matrix[atom_id, col_idx] = avg  # row = atom, col = residue

    # Optional: create DataFrame for nicer display / CSV export
    df = pd.DataFrame(matrix, columns=residues, index=[f'Residue {i+1}' for i in range(n_atoms)])
    
    # Predicted sequence
    pred_sequence = [residues[np.argmax(matrix[i])] for i in range(n_atoms)]
    print("Predicted residue sequence:")
    print(pred_sequence)
    
    return df, pred_sequence

def update_pdb_residues(input_pdb, output_pdb, residue_sequence):
    """
    Update residue types in a PDB file for CA atoms only.

    Args:
        input_pdb (str): path to original PDB
        output_pdb (str): path to save updated PDB
        residue_sequence (list of str): predicted residues for each atom
    """
    updated_lines = []
    ca_index = 0  # counter for CA atoms

    with open(input_pdb, 'r') as f:
        for line in f:
            # if line.startswith("ATOM") and line[13:15].strip() == "CA":
            if ( line.startswith("ATOM") and line[13:15].strip() == "CA" ) or \
                ( line.startswith("HETATM") and line[13:15].strip() == "C" ):    # A1_clustering.py does this (this is if you skip A2_....py), need to modify A1_clustering.py later
                if ca_index >= len(residue_sequence):
                    raise ValueError("Number of CA atoms does not match the residue sequence")

                # Replace residue name (columns 17-20)
                new_line = line[:17] + f"{residue_sequence[ca_index]:>3}" + line[20:]
                updated_lines.append(new_line)
                ca_index += 1
            else:
                updated_lines.append(line)

    with open(output_pdb, 'w') as f:
        f.writelines(updated_lines)

    print(f"Updated PDB saved to {output_pdb}")



def main(input_file_path = None, main_map_path = None, residue_dir_path = None):
    # main_map = r"D:\NEW_TEST_DATA\UnetPrediction\EMD-21887_re_3.2\possibility\emd_21887.map_key_atoms_1_possibility.mrc"
    # TODO: rename main_map to smth else because it is NOT a map. Also just fyi, this file is only for getting the dimensions (assumes all unet possiblity mrcs have same dimensions, for the same sample).
    main_map = "C:/Users/noelu/programming/python_projects/cryoID_v2_DATA/UnetPrediction/EMD-34843_re_3.2/possibility/emd_34843.map_key_atoms_1_possibility.mrc"
    pdb_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\compare_dif_versions\superset_graph_degree_limited.pdb"
    ### meeting figure
    main_map = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\real_malaria\3-E1_D3\possibility\E1_D3_2_7A_cryosparc_P8_J562_008_volume_map_sharp_key_atoms_1_possibility.mrc"
    pdb_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\results_3_26_26\superset_graph_degree_limited.pdb"
    # pdb_file = "traced_backbone3.pdb"
    # main_map = r"C:\Users\noelu\programming\python_projects\cryoID_v2_DATA\UnetPrediction\real_malaria\1\possibility\cryosparc_P164_J166_006_volume_map_key_atoms_1_possibility.mrc"
    # pdb_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\compare_dif_versions\superset_graph.pdb"
    # 10/1/25 one off files for michael
    # pdb_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\emd_34843_agglomerative_cd_4.2_combined_fullGraph_20250927_235412.pdb"
    if input_file_path:
        pdb_file = input_file_path
    if main_map_path:
        main_map = main_map_path
    coords = read_pdb_coords(pdb_file)
    print("Read", len(coords), "CA coordinates:")
    for i, (x, y, z) in enumerate(coords, start=1):
        print(f"{i:3d}: {x:.3f}, {y:.3f}, {z:.3f}")
    # assert(False)
    # residue_dir = r"D:\NEW_TEST_DATA\UnetPrediction\EMD-21887_re_3.2\possibility"
    # BUG: bruh
    # TODO: fix this so can pass in residue_dir, should just build it from config file, it's based on base_dir, literally just Path(base_dir) / possibility
    # TODO: minor consideration, rename to residue_possibility_dir? longer tho so maybe no. tho regardless itll get autofilled eventually based on the config file. user should NOT have to set this variable manually
    residue_dir = "C:/Users/noelu/programming/python_projects/cryoID_v2_DATA/UnetPrediction/EMD-34843_re_3.2/possibility"
    residue_dir = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\real_malaria\3-E1_D3\possibility"
    # residue_dir = r"C:\Users\noelu\programming\python_projects\cryoID_v2_DATA\UnetPrediction\real_malaria\1\possibility"
    if residue_dir_path:
        residue_dir = residue_dir_path

    voxel_size, shape = get_voxel_size(main_map)
    voxel_dict = coords_to_voxels(coords, voxel_size, radius=1)
    residue_dict = sample_residue_maps(voxel_dict, residue_dir, shape)

    ###! These are OLD versions of the average_residues function with dif output formats. Comment out so code runs faster (redundant anyways).
    # # print residue possibilities for each atom
    # for atom_id, res_data in residue_dict.items():
    #     print(f"\nResidue {atom_id}:")
    #     for res_file, vals in res_data.items():
    #         print(f"  {res_file}: {vals[:5]} ...")

    # residue_sequence = average_residues(residue_dict, save_file=True, file_name="residue_sequence_OLD.txt")
    residue_sequence = average_residues_AND_USE_RESIDUE_NAMES(residue_dict)
    print(residue_sequence)


    ###^ NEW OUTPUT FORMAT
    # === Replace old average_residues calls with matrix version ===
    df_matrix, residue_sequence = average_residues_matrix(residue_dict)

    print("\nResidue probabilities matrix:")
    print(df_matrix)

    print("\nPredicted residue sequence:")
    print(residue_sequence)

    print("\nFormatted as a string:")
    seq_str = "-".join(residue_sequence)
    print(seq_str)

    # Optional: save probability matrix to CSV
    df_matrix.to_csv("z_residue_prob_matrix.csv")
    print("\nResidue probability matrix saved to residue_prob_matrix.csv")


    ###^ Update the input pdb file to the correct residues (only need to change residue column)
    output_pdb = "traced_backbone3_updated_residues.pdb"
    # 10/1/25 one off files for michael
    # base, ext = os.path.splitext(os.path.basename(pdb_file))
    # output_pdb = f"{base}_WITH_RESIDUES{ext}"
    # output_pdb = os.path.join("H_Backbone_tracing/outputs/one_off_files_for_michael", output_pdb)
    update_pdb_residues(pdb_file, output_pdb, residue_sequence)

    return output_pdb


if __name__ == "__main__":
    main()