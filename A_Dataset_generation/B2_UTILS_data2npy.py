# import math
import os
import shutil

import gemmi
# import matplotlib.pyplot as plt
import mrcfile
import numpy as np

import sys
from pathlib import Path
import random

sys.path.append(str(Path(__file__).parent.parent))

from U_UTILS.MRC import MRC

# import pandas as pd

# Generate taged files (num = CLASSES) of mrc format from atomic model data
GENERATET_MRC_TEST = False
CLASSES = 24

# RNA bases
atoms_base_A = ['N9', 'C8', 'N7', 'C5', 'C6', 'N6', 'N1', 'C2', 'N3', 'C4']
atoms_base_G = [
    'N9', 'C8', 'N7', 'C5', 'C6', 'O6', 'N1', 'C2', 'N2', 'N3', 'C4'
]
atoms_base_C = ['N1', 'C2', 'O2', 'N3', 'C4', 'N4', 'C5', 'C6']
atoms_base_U = ['N1', 'C2', 'O2', 'N3', 'C4', 'O4', 'C5', 'C6']
atoms_nuc_base = list(set(atoms_base_A + atoms_base_G + atoms_base_C + atoms_base_U))

# Protein_residues
atoms_alanine_ALA = ['CB']
atoms_arginine_ARG = ['CB', 'CG', 'CD', 'NE', 'CZ', 'NH1', 'NH2']
atoms_asparagine_ASN = ['CB', 'CG', 'OD1', 'ND2']
atoms_aspartic_acid_ASP = ['CB', 'CG', 'OD1', 'OD2']
atoms_cysteine_CYS = ['CB', 'SG']
atoms_glutamic_acid_GLU = ['CB', 'CG', 'CD', 'OE1', 'OE2']
atoms_glutamine_GLN = ['CB', 'CG', 'CD', 'OE1', 'NE2']
atoms_glycine_GLY = []
atoms_histidine_HIS = ['CB', 'CG', 'ND1', 'CD2', 'CE1', 'NE2']
atoms_isoleucine_ILE = ['CB', 'CG1', 'CG2', 'CD1']
atoms_leucine_LEU = ['CB', 'CG', 'CD1', 'CD2']
atoms_lysine_LYS = ['CB', 'CG', 'CD', 'CE', 'NZ']
atoms_methionine_MET = ['CB', 'CG', 'SD', 'CE']
atoms_phenylalanine_PHE = ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']
atoms_proline_PRO = ['CB', 'CG', 'CD']
atoms_serine_SER = ['CB', 'OG']
atoms_threonine_THR = ['CB', 'OG1', 'CG2']
atoms_tryptophan_TRP = ['CB', 'CG', 'CD1', 'CD2', 'NE1', 'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2']
atoms_tyrosine_TYR = ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ']
atoms_valine_VAL = ['CB', 'CG1', 'CG2']
atoms_protein_residue = list(set(atoms_alanine_ALA+atoms_arginine_ARG+atoms_asparagine_ASN+atoms_aspartic_acid_ASP+atoms_cysteine_CYS+
                         atoms_glutamic_acid_GLU+atoms_glutamine_GLN+atoms_glycine_GLY+atoms_histidine_HIS+atoms_isoleucine_ILE+
                         atoms_leucine_LEU+atoms_lysine_LYS+atoms_methionine_MET+atoms_phenylalanine_PHE+atoms_proline_PRO+
                         atoms_serine_SER+atoms_threonine_THR+atoms_tryptophan_TRP+atoms_tyrosine_TYR+atoms_valine_VAL))

# Protein and rna residues
residues_protein = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
]
residues_rna = ['A', 'G', 'C', 'U']

# Tagged atom groups
atoms_sugar_ring_new = ["C4'", "O4'", "C3'", "C2'", "C1'"]
key_atoms = [["CA"], ['P'], atoms_sugar_ring_new, atoms_nuc_base, atoms_protein_residue]
atoms_protein_backbone = ['CA', 'C', 'N']
atoms_rna_backbone = [
    "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'",
    "C1'"
]


def data_to_npy(map_path: str,
                model_path: str,
                model_parts: list,
                sample_dir: str,
                sample_num: int = 0,
                npy_size: int = 64):
    """
    Read an MRC map file and a PDB model file, and create 3D numpy arraies with data from them.

    Args:
    - map_path (str): path to the MRC map file
    - model_path (str): path to the PDB model file
    - model_parts (list): list of strings representing the parts of the model to sample (e.g., ["key_atoms", "nitrogenous_bases"])
    - sample_dir (str): path to a directory where the samples will be saved
    - sample_num (int): used to name the out put samples
    - npy_size (int = 64):

    """
    # Save map data and tagged data
    data = []
    part_names = []

    # Read the MRC map file and change the coordinates order to [z, y, x]
    mrc = MRC(map_path)
    map_data = mrc.swap_back_coordinates(mrc.dens, mrc.ordermode)
    map_size = [int(mrc.nz), int(mrc.ny), int(mrc.nx)]

    # Read the PDB model file
    structure = gemmi.read_structure(model_path)

    # Ensure the size of the model is not larger than the map
    box = structure.calculate_box()
    # box coord is [x, y, z] so change the order to [z, y, x]
    box_min = list(box.minimum)[::-1]
    box_max = list(box.maximum)[::-1]
    if box_max[2] > map_size[2] or box_max[1] > map_size[2] or box_max[
            0] > map_size[0]:
        raise ValueError("The model box size exceeds that of the map.")

    # Compute the grid parameters for splitting the volume
    start_coords, n_samples = compute_grid_params(box_min, box_max, map_size,
                                                  npy_size)

    # Sample the selected parts of the model to create the numpy array, record the number of each tag in each part
    num_of_tag_in_each_part = {}  # Save number of each tag in each part

    for part_name in model_parts:
        model_data = np.zeros(map_size, np.int8)
        part_coords = None
        dis_array = np.array([])

        print(f"Sampling part: {part_name}")

        # Tag protein backbone atoms(CA+C+N) in helices with 1, in sheets with 2, in loops with 3
        # Tag protein backbone atoms in rna with 4
        if part_name == "secondary_strctures":
            # for helices, sheets and loops
            helices, sheets = protein_2nd_structure_lists(structure)
            protein_2nd_structure_coords = atom_coord_cif_protein_secondary(
                structure, helices, sheets, residues_protein,
                atoms_protein_backbone)
            for tag in range(1, len(protein_2nd_structure_coords) + 1):
                part_coords = protein_2nd_structure_coords[tag - 1]
                model_data, dis_array = tag_npy(model_data, part_coords, tag,
                                                dis_array)

            # for rna
            part_coords = atom_coord_cif(structure, residues_rna,
                                         atoms_rna_backbone)
            model_data, dis_array = tag_npy(model_data, part_coords, 4,
                                            dis_array)
            data.append(model_data)
            part_names.append(part_name)

        # Tag CA, P, sugar, rna base, protein side chain atoms with 1, 2, 3, 4, 5 repectively
        elif part_name == "key_atoms":
            for tag in range(1, len(key_atoms) + 1):
                if key_atoms[tag - 1] in [["CA"]] or key_atoms[tag - 1] in [atoms_protein_residue]:
                    part_coords = atom_coord_cif(structure, residues_protein,
                                                 key_atoms[tag - 1])
                else:
                    part_coords = atom_coord_cif(structure, residues_rna,
                                                 key_atoms[tag - 1])
                model_data, dis_array = tag_npy(model_data, part_coords, tag,
                                                dis_array)
            data.append(model_data)
            part_names.append(part_name)

        # Tag key atoms with 1-24 for different residue types
        elif part_name == "residue_types":
            # Tag CA atoms with 1, 2, 3, ..., 20 for 20 types of amino acids repectively
            for tag in range(1, len(residues_protein) + 1):
                part_coords = atom_coord_cif(structure,
                                             [residues_protein[tag - 1]],
                                             ['CA'])
                model_data, dis_array = tag_npy(model_data, part_coords, tag,
                                                dis_array)
            # Tag sugar_ring atoms with 21, ..., 24 for 4 types of rna bases repectively
            for base_tag in range(1, len(residues_rna) + 1):
                tag = base_tag + len(residues_protein)
                part_coords = atom_coord_cif(structure,
                                             [residues_rna[base_tag - 1]],
                                             atoms_sugar_ring_new)
                model_data, dis_array = tag_npy(model_data, part_coords, tag,
                                                dis_array)
            data.append(model_data)
            part_names.append(part_name)

        else:
            raise ValueError(f"Unknown model part: {part_name}")

        if GENERATET_MRC_TEST is True:
            # # Generate TEST.mrc for one part
            # out_map = f"{map_path.split('.mrc')[0]}_EXAMPLE_{part_name}.mrc"
            # print("=> Writing new map")
            # shutil.copy(map_path, out_map)
            # with mrcfile.open(out_map, mode='r+') as mrc:
            #     TEST_data = model_data
            #     mrc.set_data(TEST_data)
            #     # mrc.header.mz = model_data.shape[0]
            # print("New map is writen.")
            # continue

            # Generate TEST.mrc for every value in one part
            for tag in range(1, CLASSES + 1):
                out_map = f"{map_path.split('.mrc')[0]}_EXAMPLE_{part_name}_{tag}.mrc"
                print("=> Writing new map")
                shutil.copy(map_path, out_map)
                with mrcfile.open(out_map, mode='r+') as mrc:
                    TEST_data = np.zeros_like(model_data)
                    TEST_data = np.where(model_data == tag, model_data, 0)
                    mrc.set_data(TEST_data)
                    # mrc.header.mz = model_data.shape[0]
                print("New map is writen.")
                continue
            exit

    # Create npy files from model_data of each part and map_data
    # Calcutate num of different tags
    data.append(map_data)
    part_names.append('map_sample')
    num_tags, sample_num = split_to_npy(data, sample_dir, start_coords,
                                        n_samples, npy_size, sample_num,
                                        part_names)

    for idx in range(len(part_names) - 1):
        part_name = part_names[idx]
        num_of_tag_in_each_part[part_name] = num_tags[idx]

    # print(num_of_tag_in_each_part)

    return sample_num, num_of_tag_in_each_part


def compute_grid_params(box_min_list, box_max_list, axis_length_list,
                        grid_size):  #
    """
    Computes the starting coordinate and number of grid samples along three axis for a given box.

    Args:
        box_min_list (List): Minimum coordinates of the box.
        box_max_list (List): Maximum coordinates of the box.
        axis_length_list (List): Lengths of the axes.
        grid_size (int): Size of the grid.

    Returns:
        List[int], List[int]: Starting coordinates and numbers of grid samples
        when sampling stride = 0.
    """
    start_coords = []
    n_samples = []
    for box_min, box_max, axis_length in zip(box_min_list, box_max_list,
                                             axis_length_list):
        box_mid = (box_min + box_max) // 2

        # num of samples along one axis
        n_samples_axis = (box_max - box_min) // grid_size + 1

        if not (32 * n_samples_axis <= box_mid <
                axis_length - 32 * n_samples_axis):
            n_samples_axis -= 1
        start_coord = box_mid - 32 * n_samples_axis
        start_coords.append(int(start_coord))
        n_samples.append(int(n_samples_axis))
    return start_coords, n_samples


def split_to_npy(data,
                 sample_dir,
                 start_coords,
                 n_samples,
                 npy_size,
                 sample_num,
                 part_names,
                 extract_stride=32):
    """
    Extracts sub-volumes of size npy_size from the input data array, starting from the given start coordinates and generates npy files for each sub-volume.

    Parameters:
        data (list of ndarray): Input data arrays [tag_data1, tag_data2, ..., map_data]
        sample_dir (str): Directory to save the npy files
        start_coords (tuple): Tuple of start coordinates (z,y,x)
        n_samples (tuple): Tuple of number of samples to extract (z,y,x)
        npy_size (int): Size of the sub-volume to extract
        extract_stride (int): Stride length of the stepping sample
        sample_num (int): Number to use as suffix in the npy file names
        part_names (str): Name of the subdirectory to save the npy files

    Returns:
        None
    """
    num_tags = [0, 0, 0]
    sample_start_z, sample_start_y, sample_start_x = start_coords
    for i in range(3):
        n_samples[i] = int(n_samples[i] * npy_size / extract_stride) - 1
    for part_name in part_names:
        os.makedirs(os.path.join(sample_dir, part_name), exist_ok=True)
    for n_z in range(n_samples[0]):
        idx_z = sample_start_z + extract_stride * n_z
        for n_y in range(n_samples[1]):
            idx_y = sample_start_y + extract_stride * n_y
            for n_x in range(n_samples[2]):
                idx_x = sample_start_x + extract_stride * n_x
                samples = []
                counts = []
                # ratio_tags = []
                for idx in range(0, len(data) - 1):
                    sample = data[idx][idx_z:idx_z + npy_size,
                                       idx_y:idx_y + npy_size,
                                       idx_x:idx_x + npy_size]
                    samples.append(sample)
                    count = np.bincount(sample.flatten())
                    counts.append(count)
                #     ratio_tag = 1 - count[0] / count.sum()
                #     ratio_tags.append(ratio_tag)
                # # print(ratio_tags)
                # if max(ratio_tags) < 0.01:
                #     random_number = random.randint(1, 10)
                #     if random_number < 6:
                #         continue
                # Save seperated files for model tags
                for idx in range(0, len(data) - 1):
                    file_name = os.path.join(
                        sample_dir, part_names[idx],
                        f"model_{part_names[idx]}.{sample_num}.npy")
                    np.save(file_name, samples[idx])
                    # print(np.max(samples[idx]))
                    count = counts[idx]
                    count = np.pad(count, (0, max(0, 27 - len(count))),
                                   'constant')
                    num_tags[idx] += count

                # Save seperated files for map data
                sample = data[len(data) - 1][idx_z:idx_z + npy_size,
                                             idx_y:idx_y + npy_size,
                                             idx_x:idx_x + npy_size]
                file_name = os.path.join(sample_dir, part_names[len(data) - 1],
                                         f"map.{sample_num}.npy")
                np.save(file_name, sample)

                sample_num += 1

    return num_tags, sample_num


def atom_coord_cif(structure, RESIDUE=None, ATOM=None):
    """
    Returns the atomic coordinates from a PDB structure for specific residues and atoms.

    Args:
        structure (Structure): PDB structure.
        RESIDUE (list, optional): List of residue names to select. Defaults to None (all residues).
        ATOM (list, optional): List of atom names to select. Defaults to None (all atoms).

    Returns:
        list: List of atomic coordinates as lists [z, y, x].
    """
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if RESIDUE is not None and residue.name not in RESIDUE:
                    continue
                for atom in residue:
                    if ATOM is not None and atom.name not in ATOM:
                        continue
                    # coords.append(
                    #     (int(round(atom.pos.z)), int(round(atom.pos.y)),
                    #      int(round(atom.pos.x)))
                    # )
                    # coords.append(atom.pos),
                    coords.append([atom.pos.z, atom.pos.y, atom.pos.x])
    return coords


def tag_npy(model_data,
            part_coords,
            tag_id,
            dis_array=np.array([]),
            atom_grid_radius=1.5):
    """
    Add tags to 3D grid points surrounding given part coordinates.

    Args:
        model_data (ndarray): A 3D numpy array representing the 3D grid.
        part_coords (list of tuples): A list of tuples representing the part coordinates (z, y, x).
        tag_id (int): The tag value to apply to the tagged points.
        atom_grid_radius (int, optional): The radius of the grid to tag around the part coordinates. Defaults to 2.

    Returns:
        model_data (ndarray): The modified model data with the tagged points.
        dis_array (ndarray): The modified model data with the tagged points.
    """

    dis_array = np.full(model_data.shape, atom_grid_radius**2)

    index_grid = int(atom_grid_radius) + 1
    extension = range(-index_grid, index_grid + 1)
    arounds = []
    for dz in extension:
        for dy in extension:
            for dx in extension:
                dist = dz**2 + dy**2 + dx**2
                if dist <= atom_grid_radius**2:
                    arounds.append([dz, dy, dx])
    arounds = np.array(arounds)
    arounds = np.vstack((arounds, arounds + np.array([0, 0, 1])))
    arounds = np.vstack((arounds, arounds + np.array([0, 1, 0])))
    arounds = np.vstack((arounds, arounds + np.array([1, 0, 0])))
    arounds = np.unique(arounds, axis=0).astype(int)

    for coord in np.array(part_coords):
        floor_coord = np.floor(coord).astype(int)
        for around in arounds:
            around_coord = floor_coord + around
            dist = np.sum((around_coord - coord)**2)
            if dist <= dis_array[around_coord[0], around_coord[1],
                                 around_coord[2]]:
                model_data[around_coord[0], around_coord[1],
                           around_coord[2]] = tag_id
                dis_array[around_coord[0], around_coord[1],
                          around_coord[2]] = dist

    return model_data, dis_array


def protein_2nd_structure_lists(structure):
    helices, sheets, = [], []
    # helices = [[hlx_chain1, start.res_id.seqid.num, end.res_id.seqid.num],
    #            [hlx_chain2, start.res_id.seqid.num, end.res_id.seqid.num],
    #            ... ]

    for helix_index in range(len(structure.helices)):
        start = structure.helices[helix_index].start.res_id.seqid.num
        end = structure.helices[helix_index].end.res_id.seqid.num
        chain_name = structure.helices[helix_index].start.chain_name
        helices.append([chain_name, start, end])

    for sheet_index in range(len(structure.sheets)):
        for strand_index in range(len(structure.sheets[sheet_index].strands)):
            start = structure.sheets[sheet_index].strands[
                strand_index].start.res_id.seqid.num
            end = structure.sheets[sheet_index].strands[
                strand_index].end.res_id.seqid.num
            chain_name = structure.sheets[sheet_index].strands[
                strand_index].start.chain_name
            sheets.append([chain_name, start, end])

    return helices, sheets


def atom_coord_cif_protein_secondary(structure,
                                     helices,
                                     sheets,
                                     RESIDUE=None,
                                     ATOM=None):
    coords, coords_helices, coords_sheets = [], [], []
    chain_helices = [row[0] for row in helices]
    chain_sheets = [row[0] for row in sheets]
    chain_name = list(set(chain_helices + chain_sheets))

    coords = atom_coord_cif(structure, RESIDUE, ATOM)

    for model in structure:
        for chain in model:
            if chain_name is not None and chain.name not in chain_name:
                continue
            # helices
            for index, helix_chian_name in enumerate(chain_helices):
                if chain.name != helix_chian_name:
                    continue
                chain_info = helices[index]
                for residue in chain:
                    if residue.seqid.num in range(chain_info[1],
                                                  chain_info[2] + 1):
                        for atom in residue:
                            if ATOM is not None and atom.name not in ATOM:
                                continue
                            atom_coord = [atom.pos.z, atom.pos.y, atom.pos.x]
                            coords_helices.append(atom_coord)
            # sheets
            for index, sheet_chian_name in enumerate(chain_sheets):
                if chain.name != sheet_chian_name:
                    continue
                chain_info = sheets[index]
                for residue in chain:
                    if residue.seqid.num in range(chain_info[1],
                                                  chain_info[2] + 1):
                        for atom in residue:
                            if ATOM is not None and atom.name not in ATOM:
                                continue
                            atom_coord = [atom.pos.z, atom.pos.y, atom.pos.x]
                            coords_sheets.append(atom_coord)

    coords_loops = [
        x for x in coords if not (x in coords_sheets or x in coords_helices)
    ]
    # coords_others1 = [x for x in coords_sheets if x not in coords]
    # coords_others2 = [x for x in coords_helices if x not in coords]
    # print(coords_others1, coords_others2)

    return [coords_helices, coords_sheets, coords_loops]


if __name__ == "__main__":
    map_path = '/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/Filtered_Dateset/Raw/EMD-11893_re_3.33/emd_11893_normalized.mrc'
    model_path = '/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/Filtered_Dateset/Raw/EMD-11893_re_3.33/7ase.cif'
    model_parts = ['secondary_strctures', 'key_atoms', 'residue_types']
    sample_dir = '/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/Filtered_Dateset/Training/ready_to_train_and_val'
    npy_size = 64

    sample_num, num_of_tag_in_each_part = data_to_npy(map_path, model_path,
                                                      model_parts, sample_dir)
    print(sample_num)
    print(num_of_tag_in_each_part)
