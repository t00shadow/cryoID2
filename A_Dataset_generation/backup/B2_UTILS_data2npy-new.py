import os
import shutil
import math
import gemmi
import matplotlib.pyplot as plt
import mrcfile
import numpy as np

# import pandas as pd

GENERATET_MRC_TEST = True
# If True, generate taged file of mrc format from atomic model data, for tag visualization


key_atoms = [["CA"], ["C"], ["N"]]


residues_rna = ['A', 'G', 'C', 'U',]

atom_p = ['P']
atom_c1 = ["C1'"]
atoms_sugar_ring_new = ["C4'", "O4'", "C3'", "C2'", "C1'"]                            # sugar_ring_old = ["C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]
atoms_base_A = ["N9", "C8", "N7", "C5", "C6", "N6", "N1", "C2", "N3", "C4"]
atoms_base_G = ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4"]
atoms_base_C = ["N1", "C2", "O2", "N3", "C4", "N4", "C5", "C6"]
atoms_base_U = ["N1", "C2", "O2", "N3", "C4", "O4", "C5", "C6"]
atoms_rna_backbone = ["P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'"]
atoms_rna_sidechain = ["N9", "C8", "N7", "C5", "C6", "O6", "N1", "C2", "N2", "N3", "C4", "O2", "O4", "N4",]
atoms_bases = [atoms_base_A, atoms_base_G, atoms_base_C, atoms_base_U]

# Protein backbone
atoms_protein_backbone = ["CA","C","N"]
# protein_recidues
residues_protein = ["ALA","ARG","ASN","ASP","CYS","GLU","GLN","GLY","HIS","ILE","LEU","LYS","MET","PHE","PRO","SER","THR","TRP","TYR","VAL"]
atoms_alanine_ALA = ["CB"]
atoms_arginine_ARG = ["CB","CG","CD","NE","CZ","NH1","NH2"]
atoms_asparagine_ASN = ["CB","CG","OD1","ND2"]
atoms_aspartic_acid_ASP = ["CB","CG","OD1","OD2"]
atoms_cysteine_CYS = ["CB","SG"]
atoms_glutamic_acid_GLU = ["CB","CG","CD","OE1","OE2"]
atoms_glutamine_GLN = ["CB","CG","CD","OE1","NE2"]
atoms_glycine_GLY = []
atoms_histidine_HIS = ["CB","CG","ND1","CD2","CE1","NE2"]
atoms_isoleucine_ILE = ["CB","CG1","CG2","CD1"]
atoms_leucine_LEU = ["CB","CG","CD1","CD2"]
atoms_lysine_LYS = ["CB","CG","CD","CE","NZ"]
atoms_methionine_MET = ["CB","CG","SD","CE"]
atoms_phenylalanine_PHE = ["CB","CG","CD1","CD2","CE1","CE2","CZ"]
atoms_proline_PRO = ["CB","CG","CD"]
atoms_serine_SER = ["CB","OG"]
atoms_threonine_THR = ["CB","OG1","CG2"]
atoms_tryptophan_TRP = ["CB","CG","CD1","CD2","NE1","CE2","CE3","CZ2","CZ3","CH2"]
atoms_tyrosine_TYR = ["CB","CG","CD1","CD2","CE1","CE2","CZ"]
atoms_valine_VAL = ["CB","CG1","CG2"]
atoms_side_chain = [atoms_alanine_ALA, atoms_arginine_ARG, atoms_asparagine_ASN, atoms_aspartic_acid_ASP, atoms_cysteine_CYS, atoms_glutamic_acid_GLU, atoms_glutamine_GLN,
atoms_glycine_GLY, atoms_histidine_HIS, atoms_isoleucine_ILE, atoms_leucine_LEU, atoms_lysine_LYS, atoms_methionine_MET, atoms_phenylalanine_PHE, atoms_proline_PRO,
atoms_serine_SER, atoms_threonine_THR, atoms_tryptophan_TRP, atoms_tyrosine_TYR, atoms_tyrosine_TYR]


def atom_coord_cif(structure, RESIDUE=None, ATOM=None):
    """
    Returns the atomic coordinates from a PDB structure for specific residues and atoms.

    Args:
        structure (Structure): PDB structure.
        RESIDUE (list, optional): List of residue names to select. Defaults to None (all residues).
        ATOM (list, optional): List of atom names to select. Defaults to None (all atoms).

    Returns:
        list: List of atomic coordinates as tuples (z, y, x).
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
                    coords.append((int(round(atom.pos.z)), int(round(atom.pos.y)), int(round(atom.pos.x))))            # coords.append(atom.pos), # coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
    return coords                                                                                       # return np.array(coords)


def atoms_mass_center_cif(structure, RESIDUE=None, ATOM=None,):
    """
    Returns the center-of-mass coordinates for specific residues and atoms in a PDB structure.

    Args:
        structure (Structure): PDB structure.
        RESIDUE (list, optional): List of residue names to select. Defaults to None (all residues).
        ATOM (list, optional): List of atom names to select. Defaults to None (all atoms).

    Returns:
        list: List of center-of-mass coordinates as tuples (z, y, x).
    """
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if RESIDUE is not None and residue.name not in RESIDUE:
                    continue
                res_coords = []
                for atom in residue:
                    if ATOM is not None and atom.name not in ATOM:
                        continue
                    res_coords.append([atom.pos.z, atom.pos.y, atom.pos.x])
                if res_coords:
                    com = np.mean(res_coords, axis=0)
                    coords.append((int(round(com[0])), int(round(com[1])), int(round(com[2]))))
    return coords


def tag_npy(model_data, part_coords, tag_id, dis_dict = dict(), atom_grid_radius=2):
    """
    Add tags to 3D grid points surrounding given part coordinates.

    Args:
        model_data (ndarray): A 3D numpy array representing the 3D grid.
        part_coords (list of tuples): A list of tuples representing the part coordinates (z, y, x).
        tag_id (int): The tag value to apply to the tagged points.
        atom_grid_radius (int, optional): The radius of the grid to tag around the part coordinates. Defaults to 2.

    Returns:
        ndarray: The modified model data with the tagged points.
    """
    extension = range(-atom_grid_radius, atom_grid_radius + 1)

    for z, y, x in part_coords:
        for dz in extension:
            for dy in extension:
                for dx in extension:
                    #if dz**2 + dy**2 + dx**2 <= atom_grid_radius**2:
                    #    model_data[z+dz, y+dy, x+dx] = tag_id
                    if dis_dict == {}:
                        if dz ** 2 + dy ** 2 + dx ** 2 <= atom_grid_radius ** 2:
                            model_data[z + dz, y + dy, x + dx] = tag_id
                            dis_dict[f"[{z + dz}, {y + dy}, {x + dx}]"] = math.sqrt(dz ** 2 + dy ** 2 + dx ** 2)
                    else:
                        #print(dis_dict.keys())
                        if f"[{z+dz}, {y+dy}, {x+dx}]" in dis_dict.keys():
                            if dis_dict.get(f"[{z+dz}, {y+dy}, {x+dx}]") > math.sqrt(dz**2 + dy**2 + dx**2) and dz**2 + dy**2 + dx**2 <= atom_grid_radius**2:
                                dis_dict[f"[{z + dz}, {y + dy}, {x + dx}]"] = math.sqrt(dz ** 2 + dy ** 2 + dx ** 2)
                                model_data[z + dz, y + dy, x + dx] = tag_id
                        else:
                            if dz**2 + dy**2 + dx**2 <= atom_grid_radius**2:
                                model_data[z + dz, y + dy, x + dx] = tag_id
                                dis_dict[f"[{z + dz}, {y + dy}, {x + dx}]"] = math.sqrt(dz ** 2 + dy ** 2 + dx ** 2)
    return model_data, dis_dict


def compute_grid_params(box_min_list, box_max_list, axis_length_list, grid_size):
    """
    Computes the starting coordinate and number of grid samples along three axis for a given box.

    Args:
        box_min_list (List): Minimum coordinates of the box.
        box_max_list (List): Maximum coordinates of the box.
        axis_length_list (List): Lengths of the axes.
        grid_size (int): Size of the grid.

    Returns:
        Tuple[List[int], List[int]]: Starting coordinates and numbers of grid samples.
    """
    start_coords = []
    n_samples = []
    for box_min, box_max, axis_length in zip(box_min_list, box_max_list, axis_length_list):
        box_mid = (box_min + box_max) // 2
        n_samples_axis = (box_max - box_min) // grid_size + 1
        if not (32 * n_samples_axis <= box_mid < axis_length - 32 * n_samples_axis):
            n_samples_axis -= 1
        start_coord = box_mid - 32 * n_samples_axis
        start_coords.append(int(start_coord))
        n_samples.append(int(n_samples_axis))
    return start_coords, n_samples


def split_to_npy(data, sample_dir, start_coords, n_samples, npy_size, sample_num, part_name):
    """
    Extracts sub-volumes of size npy_size from the input data array, starting from the given start coordinates and generates npy files for each sub-volume.

    Parameters:
        data (ndarray): Input data array
        sample_dir (str): Directory to save the npy files
        start_coords (tuple): Tuple of start coordinates (z,y,x)
        n_samples (tuple): Tuple of number of samples to extract (z,y,x)
        npy_size (int): Size of the sub-volume to extract
        sample_num (int): Number to use as suffix in the npy file names
        part_name (str): Name of the subdirectory to save the npy files

    Returns:
        None
    """
    sample_start_z, sample_start_y, sample_start_x = start_coords
    os.makedirs(os.path.join(sample_dir, part_name), exist_ok=True)
    for i_z in range(n_samples[0]*2-1):
        idx_z = sample_start_z + int(npy_size/2) * i_z
        for i_y in range(n_samples[1]*2-1):
            idx_y = sample_start_y + int(npy_size/2) * i_y
            for i_x in range(n_samples[2]*2-1):
                idx_x = sample_start_x + int(npy_size/2) * i_x
                #print(idx_z,idx_y,idx_x)
                sample = data[idx_z:idx_z+npy_size, idx_y:idx_y+npy_size, idx_x:idx_x+npy_size]
                if part_name == 'map_sample':
                    np.save(os.path.join(sample_dir, part_name, f"map.{sample_num}.npy"), sample)
                else:
                    np.save(os.path.join(sample_dir, part_name, f"model_{part_name}.{sample_num}.npy"), sample)

                sample_num += 1


def test_map_model_fit(box, map_data, model_data, map_path, part_name):
    """
    Plots and saves a comparison of a map and model, sliced along the z-axis.

    Args:
        box (Box): a bounding box object defining the region of interest
        map_data (ndarray): a 3D numpy array containing map data
        model_data (ndarray): a 3D numpy array containing model data
        map_path (str): the path to the original map file

    Returns:
        None
    """

    # Create a directory to store the output image if it doesn't exist
    path = os.path.join(os.path.split(map_path)[0], 'TEST', 'z_plane')
    os.makedirs(path, exist_ok=True)

    # Determine the number of slices to plot (maximum 4)
    n = min(int(box.get_size()[2] / 20), 4)

    # Create a new figure and plot each slice
    fig, axs = plt.subplots(n, 2, figsize=(8, 3 * n))
    for i in range(n):
        n_slice = int(box.minimum[2]) + 20 * i
        axs[i, 0].imshow(map_data[n_slice, :, :], cmap='gray', origin='lower')
        axs[i, 0].set_title(f'Map slice {i}')
        axs[i, 1].imshow(model_data[n_slice, :, :], cmap='gray', origin='lower')
        axs[i, 1].set_title(f'Model slice {i}')

    # Save the figure to a file
    fig.savefig(os.path.join(path, f"{os.path.basename(map_path)}_zplane_{part_name}.png"))


def helices_list(structure):
    """
    Store the chain and position info of all the helices in the strucuture into a list: [chain name, start pos, end pos]
    """
    list = []
    for helices_index in range(len(structure.helices)):
        list.append([structure.helices[helices_index].start.chain_name,
                    structure.helices[helices_index].start.res_id.seqid.num,
                    structure.helices[helices_index].end.res_id.seqid.num])
    return list


def sheets_list(structure):
    """
    Store the chain and position info of all the sheets in the strucuture into a list: [chain name, start pos, end pos]
    """
    list = []
    for sheetslist_index in range(len(structure.sheets)):
        for strand_index in range(len(structure.sheets[sheetslist_index].strands)):
            list.append([structure.sheets[sheetslist_index].strands[strand_index].start.chain_name,
                        structure.sheets[sheetslist_index].strands[strand_index].start.res_id.seqid.num,
                        structure.sheets[sheetslist_index].strands[strand_index].end.res_id.seqid.num])
    return list


def loop_list(structure,helix,sheet):
    """
    Store the chain and position info of all the sheets in the strucuture into a list: [chain name, loop atom pos_1, loop atom pos_2, ...]
    The atom positions are obtained from removing helices and sheets positions from the chain
    """
    list = []

    for model in structure:
        for chain in model:
            chain_list = [chain.name]
            for residue in chain:
                chain_list.append(residue.seqid.num)
            list.append(chain_list)

    "Remove helices positions from the chain"
    for i in range(len(list)):
        for j in range(len(helix)):
            if list[i][0] == helix[j][0]:
                for k in range(helix[j][1],helix[j][2]):
                    list[i].remove(k)

    "Remove sheet positions from the chain"
    for i in range(len(list)):
        for j in range(len(sheet)):
            if list[i][0] == sheet[j][0]:
                for k in range(sheet[j][1],sheet[j][2]):
                    if k in list[i]:
                        list[i].remove(k)

    return list


def atom_coord_cif_protein_secondary(structure,loop=None,helix=None,sheet=None):
    helix_coords = []
    sheet_coords = []
    loop_coords = []
    "Calculate the atom positions of helices"
    if helix != None:
        for model in structure:
            for chain in model:
                for helix_index in range(len(helix)):
                    if chain.name == helix[helix_index][0]:
                        for residue in chain:
                            if residue.seqid.num in range(helix[helix_index][1],helix[helix_index][2]+1):
                                for atom in residue:
                                    if atom.name in atoms_protein_backbone:
                                        helix_coords.append((int(round(atom.pos.z)), int(round(atom.pos.y)), int(round(atom.pos.x))))

    "Calculate the atom positions of sheets"
    if sheet != None:
        for model in structure:
            for chain in model:
                for sheet_index in range(len(sheet)):
                    if chain.name == sheet[sheet_index][0]:
                        for residue in chain:
                            if residue.seqid.num in range(sheet[sheet_index][1],sheet[sheet_index][2]+1):
                                for atom in residue:
                                    if atom.name in atoms_protein_backbone:
                                        sheet_coords.append((int(round(atom.pos.z)), int(round(atom.pos.y)), int(round(atom.pos.x))))

    "Calculate the atom positions of loops"
    if loop != None:
        for model in structure:
            for chain in model:
                for loop_index in range(len(loop)):
                    if chain.name == loop[loop_index][0]:
                        for residue in chain:
                            if residue.seqid.num in loop[loop_index][1:-1]:
                                for atom in residue:
                                    if atom.name in atoms_protein_backbone:
                                        loop_coords.append((int(round(atom.pos.z)), int(round(atom.pos.y)), int(round(atom.pos.x))))

    return [loop_coords,helix_coords,sheet_coords]


def data_to_npy(map_path: str, model_path: str, model_parts: list, sample_dir: str, sample_num: int, npy_size: int = 64):
    """
    Read an MRC map file and a PDB model file, and create 3D numpy arraies with data from them.

    Args:
    - map_path (str): path to the MRC map file
    - model_path (str): path to the PDB model file
    - model_parts (list): list of strings representing the parts of the model to sample (e.g., ["key_atoms", "nitrogenous_bases"])
    - sample_dir (str): path to a directory where the samples will be saved
    - sample_num (int): 
    - npy_size (int = 64): 

    """
    with mrcfile.mmap(map_path) as mrc:
        map_data = mrc.data

        # Ensure the MRC file has a cube shape
        if not (mrc.header.nz == mrc.header.ny == mrc.header.nx):
            raise ValueError("MRC file is not cubic.")
        cube_size = [mrc.header.nz, mrc.header.ny, mrc.header.nx]

        # Read the PDB model file
        structure = gemmi.read_structure(model_path)

        # Ensure the size of the model is not larger than the map
        box = structure.calculate_box()
        if box.maximum[2] > mrc.header.nz or box.maximum[1] > mrc.header.ny or box.maximum[0] > mrc.header.nx:
            raise ValueError("The model box size exceeds that of the map.")
            print(box.minimum, box.maximum, box.get_size())
            print(mrc.header.nz, mrc.header.ny, mrc.header.nx)
            print(map_path)

        # Compute the grid parameters for splitting the volume
        start_coords, n_samples = compute_grid_params(list(box.minimum)[::-1], list(box.maximum)[::-1], cube_size, npy_size)

        # Create npy files from map_data
        split_to_npy(map_data, sample_dir, start_coords, n_samples, npy_size, sample_num, 'map_sample')


        # Sample the selected parts of the model to create the numpy array, record the number of each tag in each part
        num_of_tag_in_each_part = {}           # Save number of each tag in each part

        for part_name in model_parts:

            part_coords = None
            model_data = np.zeros(cube_size, np.int8)
            num_of_tag = []

            print(f"Sampling part: {part_name}")

            if part_name == "phosphorus_C1":
                # Tag 'P', 'C1'' with 1, 2, respectively
                part_coords = atom_coord_cif(structure, residues_rna, atom_p)
                model_data = tag_npy(model_data, part_coords, 1)
                part_coords = atom_coord_cif(structure, residues_rna, atom_c1)
                model_data = tag_npy(model_data, part_coords, 2)

            elif part_name == "sugar_ring_new":
                # Tag atoms in sugar_ring with 1
                part_coords = atom_coord_cif(structure, residues_rna, atoms_sugar_ring_new)
                model_data = tag_npy(model_data, part_coords, 1)

            elif part_name == "nitrogenous_bases_in_sugar_new":
                # Tag suagr_ring in 'A', 'G', 'C', 'U' with 1, 2, 3, 4, respectively
                for base_tag, residue_base in enumerate(residues_rna):
                    part_coords = atom_coord_cif(structure, [residue_base], atoms_sugar_ring_new)
                    model_data = tag_npy(model_data, part_coords, base_tag+1)

            elif part_name == "RNA_backbone":
                # Tag RNA backbone, sidechain with 1, 2, respectively
                part_coords = atom_coord_cif(structure, residues_rna, atoms_rna_backbone)
                model_data = tag_npy(model_data, part_coords, 1)
                part_coords = atom_coord_cif(structure, residues_rna, atoms_rna_sidechain)
                model_data = tag_npy(model_data, part_coords, 2)

            elif part_name == "sugar_ring_center":
                # Calculate then tag the mass center of sugar_ring with 1,
                part_coords = atoms_mass_center_cif(structure, residues_rna, atoms_sugar_ring_new)
                model_data = tag_npy(model_data, part_coords, 1)

            elif part_name == "nitrogenous_bases":
                # Tag 'A', 'G', 'C', 'U' with 1, 2, 3, 4, respectively
                for base_tag, residue_base in enumerate(residues_rna):
                    part_coords = atom_coord_cif(structure, [residue_base], atoms_bases[base_tag])
                    model_data = tag_npy(model_data, part_coords, base_tag+1)

            # protein
            # "protein_atoms","protein_backbone","protein_amino_acid","protein_secondary"

            elif part_name == "protein_atoms":
                # Tag CA, C, N atoms with 1, 2, 3 repectively
                for base_tag, residue_base in enumerate(residues_protein):
                    for i in range(3):
                        part_coords = atom_coord_cif(structure, [residue_base], [atoms_protein_backbone[i]])
                        model_data = tag_npy(model_data, part_coords, i+1)

            elif part_name == "protein_atoms_CA":
                part_coords = atom_coord_cif(structure, residues_protein, [atoms_protein_backbone[0]])
                model_data = tag_npy(model_data, part_coords, 1)

            elif part_name == "protein_atoms_C":
                part_coords = atom_coord_cif(structure, residues_protein, [atoms_protein_backbone[1]])
                model_data = tag_npy(model_data, part_coords, 1)

            elif part_name == "protein_atoms_N":
                part_coords = atom_coord_cif(structure, residues_protein, [atoms_protein_backbone[2]])
                model_data = tag_npy(model_data, part_coords, 1)

            elif part_name == "protein_backbone":
                # Tag backbone with 1, side chains with 2
                part_coords = atom_coord_cif(structure, residues_protein, atoms_protein_backbone)
                model_data = tag_npy(model_data, part_coords, 1)
                for base_tag, residue_base in enumerate(residues_rna):
                    part_coords = atom_coord_cif(structure, residues_protein, atoms_side_chain[base_tag])
                    model_data = tag_npy(model_data, part_coords, 2)

            elif part_name == "protein_amino_acid":
                # Tag CA atoms in amino acids with 1-20 respectively
                for base_tag, residue_base in enumerate(residues_protein):
                    part_coords = atom_coord_cif(structure, [residue_base], ["CA"])
                    model_data = tag_npy(model_data, part_coords, base_tag+1)

            elif part_name == "protein_secondary":
                # Tag protein secondary structures: one for loop, two for helix,
                # and three for sheet
                helix = helices_list(structure)
                sheet = sheets_list(structure)
                loop = loop_list(structure,helix,sheet)
                for i in range(3):
                    part_coords = atom_coord_cif_protein_secondary(structure,loop,helix,sheet)[i]
                    model_data = tag_npy(model_data, part_coords, i+1, atom_grid_radius=2)

                # RNA backbone
                part_coords = atom_coord_cif(structure, residues_rna, atoms_rna_backbone)
                model_data = tag_npy(model_data, part_coords, 4)


            elif part_name == "TEST_MAP_VS_MODEL_ALL_ATOMS":
                # Tag all atoms with 1,
                part_coords = atom_coord_cif(structure, None, None)
                model_data = tag_npy(model_data, part_coords, 1)
                test_map_model_fit(box, map_data, model_data, map_path, part_name)    # compare between model and map
                exit()

            elif part_name == "unit_1":
                # Tag CA, C, N atoms with 1, 2, 3 repectively
                for base_tag, residue_base in enumerate(residues_protein):
                    for i in range(3):
                        part_coords = atom_coord_cif(structure, [residue_base], [atoms_protein_backbone[i]])
                        if i == 0:
                            model_data, dis_dict = tag_npy(model_data, part_coords, i + 1)
                        else:
                            model_data, dis_dict = tag_npy(model_data, part_coords, i + 1, dis_dict)

                # Tag atoms in sugar_ring with 4
                part_coords = atom_coord_cif(structure, residues_rna, atoms_sugar_ring_new)
                model_data, dis_dict = tag_npy(model_data, part_coords, 4, dis_dict)

                # Tag P atom with 5
                part_coords = atom_coord_cif(structure, residues_rna, atom_p)
                model_data, dis_dict = tag_npy(model_data, part_coords,5, dis_dict)

            elif part_name == "unit_2":
                # Tag suagr_ring in 'A', 'G', 'C', 'U' with 1, 2, 3, 4, respectively
                for base_tag, residue_base in enumerate(residues_rna):
                    part_coords = atom_coord_cif(structure, [residue_base], atoms_sugar_ring_new)
                    if i == 0:
                        model_data, dis_dict = tag_npy(model_data, part_coords, base_tag + 1)
                    else:
                        model_data, dis_dict = tag_npy(model_data, part_coords, base_tag + 1, dis_dict)

                # Tag CA atoms in amino acids with 5-24 respectively
                for base_tag, residue_base in enumerate(residues_protein):
                    part_coords = atom_coord_cif(structure, [residue_base], ["CA"])
                    if i == 0:
                        model_data, dis_dict = tag_npy(model_data, part_coords, base_tag + 4 + 1)
                    else:
                        model_data, dis_dict = tag_npy(model_data, part_coords, base_tag + 4 + 1, dis_dict)


            else:
                raise ValueError(f"Unknown model part: {part_name}")

            if GENERATET_MRC_TEST is True:
                # Generate TEST.mrc for every part
                out_map = f"{map_path.split('.mrc')[0]}_EXAMPLE_{part_name}.mrc"
                print("=> Writing new map")
                shutil.copy(map_path, out_map)
                with mrcfile.open(out_map, mode='r+') as mrc:
                    mrc.set_data(model_data)
                    # mrc.header.mz = model_data.shape[0]
                print("New map is writen.")
                continue


            # Calculate numbers of tag in each part
            counts = np.bincount(model_data.flatten())
            num_of_tag_in_each_part[part_name] = counts.tolist()

            # Create npy files from model_data of each part
            # split_to_npy(model_data, sample_dir, start_coords, n_samples, npy_size, sample_num, part_name)

    sample_num += (n_samples[0]*2-1) * (n_samples[1]*2-1) * (n_samples[2]*2-1)
    # print(sample_num)
    # print(num_of_tag_in_each_part)

    return sample_num, num_of_tag_in_each_part
