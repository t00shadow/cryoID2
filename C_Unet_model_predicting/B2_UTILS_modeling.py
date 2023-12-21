import os
# import statistics
import subprocess

# import gemmi
# import mrcfile
import numpy as np
# import pandas as pd
# import torch
from Bio.PDB import PDBIO
from Bio.PDB.Atom import Atom
from Bio.PDB.Chain import Chain
from Bio.PDB.Model import Model
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure
from scipy.ndimage import maximum_filter


def refine_mass_centers(confidence_map, radius=2, threshold=0.9):
    """
    This function refines mass centers given a confidence map, radius and a threshold.

    Args:
        confidence_map : ndarray
            A 3D ndarray representing a confidence map.
        radius : int
            An integer representing the radius of the maximum filter. Default is 2.
        threshold : float
            A float representing the threshold value. Default is 0.5.

    Returns:
        refined_mass_centers : list of ndarray
            A list of ndarrays representing the refined mass centers.
        local_maxima_indices : ndarray
            A 3D ndarray representing the indices of the local maxima.            
    """

    thresholded_map = confidence_map > threshold
    neighborhood = maximum_filter(confidence_map, size=radius*2+1, mode="constant")

    # Find the indices where thresholded map and neighborhood are equal
    local_maxima_indices_raw = np.squeeze(np.argwhere(np.logical_and(thresholded_map, confidence_map == neighborhood)))
    # local_maxima_indices_raw = np.argwhere(np.logical_and(thresholded_map, confidence_map == neighborhood))

    refined_mass_centers = []

    dd = 2

    local_maxima_indices = []
    for position in local_maxima_indices_raw:
        z, y, x = position
        weighted_sum = np.zeros(3)
        total_weight = 0.0
        num_volex = 0

        for dz in range(-dd, dd+1):
            for dy in range(-dd, dd+1):
                for dx in range(-dd, dd+1):
                    if dz**2 + dy**2 + dx**2 <= dd:
                        indices = (z+dz, y+dy, x+dx)
                        if all(0 <= index < size for index, size in zip(indices, confidence_map.shape)):
                            weight = confidence_map[indices]
                            weighted_sum += weight * np.array(indices)
                            total_weight += weight
                            num_volex += 1

                        # weight = confidence_map[z+dz, y+dy, x+dx]
                        # weighted_sum += weight * np.array([z+dz, y+dy, x+dx])
                        # total_weight += weight


# 11/15/2023 TO BE CHECED
        if total_weight / num_volex > 0.5:
            refined_mass_centers.append(weighted_sum / total_weight)
            local_maxima_indices.append([z, y, x])

    return refined_mass_centers, local_maxima_indices


def write_pdb(atom_name, element, residue_lists, atom_positions, pdb_filename):
    """
    This function creates a PDB file given an atom name, an element symbol, a list of residue names,
    a list of atom positions, and a file name.

    Args:
        atom_name : str
            A string representing the atom name.
        element : str
            A string representing the element symbol.
        residue_lists : list of str, or str
            A list of strings representing the residue names.
        atom_positions : list of ndarray
            A list of ndarrays representing the atom positions.
        pdb_filename : str
            A string representing the name of the PDB file.
    """
    if os.path.exists(pdb_filename):
        status = subprocess.call(['rm', pdb_filename])
    print(f"Writing model: {os.path.basename(pdb_filename)}")
    # Create a PDB structure
    structure = Structure("example")
    model = Model(0)
    chain = Chain("A")
    for idx, atom_position in enumerate(atom_positions):
        if len(residue_lists) == len(atom_positions):
            residue = Residue((' ', idx+1, ' '), residue_lists[idx], ' ')
        elif len(residue_lists) == 1:
            residue = Residue((' ', idx+1, ' '), residue_lists, ' ')
        z, y, x = atom_position
        atom = Atom(atom_name, (x, y, z), 1.0, 0.0, ' ', atom_name, idx+1, element)
        # atom = Atom(atom_name, (z, y, x), 1.0, 0.0, ' ', atom_name, idx+1, element)

        residue.add(atom)
        chain.add(residue)
    model.add(chain)
    structure.add(model)

    # Write the PDB file
    pdb_io = PDBIO()
    pdb_io.set_structure(structure)
    pdb_io.save(pdb_filename)
    # print("New model is writen.")


# test


# map_path = "/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/RawData_testing/EMD-26818_re_2.37/emd_26818.map_sugar_ring_new_1_possibility.mrc"
# map_path = "/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/RawData_testing/EMD-16016_re_2.8-GOOD/possibility/emd_16016.map_sugar_ring_new_1_possibility.mrc"
# pdb_filename = map_path.split("possibility.mrc")[0]+".pdb"

# with mrcfile.mmap(map_path) as mrc:
#     map_data = np.array(mrc.data, dtype=np.float32)[0]
#     # refined_mass_centers, local_maxima_indices = refine_mass_centers(map_data, radius=2, threshold=0.5)
#     refined_mass_centers, local_maxima_indices = refine_mass_centers(map_data, radius=2, threshold=0.5)

#     base_types_indices = []
#     for local_maxima_index in local_maxima_indices:
#         type = []
#         z, y, x = local_maxima_index
#         dd = 1
#         for dz in range(-dd, dd+1):
#             for dy in range(-dd, dd+1):
#                 for dx in range(-dd, dd+1):
#                     if dz**2 + dy**2 + dx**2 <= dd:
#                         type.append(type_map_data[z+dz, y+dy, x+dx])
#                         most_common_number = statistics.mode(type)
#                         base_types_indices.append(most_common_number)


#     residue_lists = ['O' if base_types_indice == 0 \
#                      else 'A' if base_types_indice == 1 \
#                         else 'G' if base_types_indice == 2 \
#                             else 'C' if base_types_indice == 3 \
#                                 else 'U' if base_types_indice == 4 \
#                                     else 'X' for base_types_indice in base_types_indices]
# residue_lists = ["A"]*len(local_maxima_indices)
# write_pdb("C1'", "C", residue_lists, refined_mass_centers, pdb_filename)


def track_5_3(P5, P_position, ribose_position, C1_position):
    a1, a2, a3 = -0.186, 1.623, -0.124
    b1, b2 = 0.440, 0.910

    P3s = [pos for pos in P_position if 4 < np.linalg.norm(pos - P5) < 9]

    if len(P3s) == 0:
        return 9999, None, None, None

    score_R_P3 = []
    for P3 in P3s:
        d = np.linalg.norm(P3 - P5)
        z = b1 * d + b2
        origin = (P3 - P5) * z/d + P5
        r = a1 * d**2 + a2 * d + a3
        axis = (P3 - P5) / np.linalg.norm(P3 - P5)

        Rs = [pos for pos in ribose_position if np.linalg.norm(pos - origin) < r+3]
        score_R = []

        for R in Rs:
            vector = R - origin
            projection_z = abs(np.dot(vector, axis))
            projection_r = np.linalg.norm(np.cross(vector, axis))

            score = ((projection_z - z)**2 + (projection_r - r)**2) ** 0.5
            score_R.append([score, R])

            # if abs(projection_z - z) < 2 and abs(projection_r - r) < 2.5:
            #     score_R.append([((projection_z - z)**2 + (projection_r - r)**2) **0.5, R])
            # else:
            #     score_R.append([9999, R])

        if len(Rs) == 0:
            score_R.append([9999, None])

        # print(score_R)

        min_score_index = min(range(len(score_R)), key=lambda i: score_R[i][0])
        score_R_P3.append([score_R[min_score_index], P3])

    min_score_index = np.argmin([inner_list[0][0] for inner_list in score_R_P3])
    [[score, R], P3] = score_R_P3[min_score_index]


    C1s = [pos for pos in C1_position if np.linalg.norm(pos - R) < 2.5] if R is not None else []

    # ADD the experience formula of C1 position
    if len(C1s) == 0 :
        C1 = None
    elif len(C1s) == 1:
        C1 = C1s[0]
    elif len(C1s) > 1:
        C1 = C1s[0]
        print("len(C1s) == 2: to be fixed in the future.")


    return score, R, P3, C1
