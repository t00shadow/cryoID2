import pickle
import os
from B2_UTILS_modeling import (refine_mass_centers, write_pdb, track_5_3)
from B2_UTILS_modeling_backbone_track import backbone_track
import numpy as np
import pandas as pd


MAP_PATH = "/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/LabData_testing/Fumi/Nsp15-RNA_complex-unsharpen/cryosparc_P53_cryosparc_P53_J358_010_volume_map.mrc"
# MAP_PATH = "/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/RawData_testing/EMD-14956_re_2.7-GOOD/emd_14956.map"
# MAP_PATH = "/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/RawData_testing/EMD-16016_re_2.8-GOOD/emd_16016.map"
# MAP_PATH = "/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/RawData_testing/EMD-34905_re_2.9-GOOD/emd_34905.map"
# MAP_PATH = "/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/test_for_dirrerentiaiton_ssRNA/spliceosome_re2_3/ribosome/EMD-13112_re_2.4 - GOOD - ribosome/emd_13112.map"


folder_path = os.path.dirname(MAP_PATH)

chain_id = 1
with open(os.path.join(folder_path, "temp_in_chain", f"R_in_chain_{chain_id}"), "rb") as file:
    refined_mass_centers_sugar_ring_new_in_chain = pickle.load(file)
with open(os.path.join(folder_path, "temp_in_chain", f"residue_in_chain_{chain_id}"), "rb") as file:
    residue_lists_in_chain = pickle.load(file)
with open(os.path.join(folder_path, "temp_in_chain", f"P_in_chain_{chain_id}"), "rb") as file:
    refined_mass_centers_P_in_chain = pickle.load(file)
with open(os.path.join(folder_path, "temp_in_chain", f"C1_in_chain_{chain_id}"), "rb") as file:
    refined_mass_centers_C1_in_chain = pickle.load(file)

with open(os.path.join(folder_path, "temp_in_chain", f"possibility_in_chain"), "rb") as file:
    backbone_possibility = pickle.load(file)



# Read score list
current_directory = os.path.dirname(os.path.abspath(__file__))
score_folder = os.path.join(current_directory, "score_distribution")
df = pd.read_csv(os.path.join(current_directory, "score_distribution", "Scores_all.csv"))
s_dist, s_angleP, s_angleS, s_pseudoX = dict(zip(df["dist_phosDistSmoothed"], df["likelihood_phosDistSmoothed"])), \
    dict(zip(df["angle_phosAngleSmoothed"], df["likelihood_phosAngleSmoothed"])), \
    dict(zip(df["angel_sugarPhosSugarAngleSmoothed"], df["likelihood_sugarPhosSugarAngleSmoothed"])), \
    dict(zip(df["angle_pseudoChiSmoothed"], df["likelihood_pseudoChiSmoothed"]))

# connection 
P5s, Rs, C1s, P3s, residues, scores = [], [], [], [], [], []
for P5 in refined_mass_centers_P_in_chain:
    score, R, P3, C1 = track_5_3(P5, refined_mass_centers_P_in_chain, refined_mass_centers_sugar_ring_new_in_chain, refined_mass_centers_C1_in_chain)
    # P5s.append(P5.tolist() if P5 is not None else None)
    # Rs.append(R.tolist() if R is not None else None)
    # C1s.append(C1.tolist() if C1 is not None else None)
    # P3s.append(P3.tolist() if P3 is not None else None)

    P5s.append(P5 if P5 is not None else None)
    Rs.append(R if R is not None else None)
    C1s.append(C1 if C1 is not None else None)
    P3s.append(P3 if P3 is not None else None)

    list_refined_mass_centers_sugar_ring_new_in_chain = [list(arr) for arr in refined_mass_centers_sugar_ring_new_in_chain]
    residue = residue_lists_in_chain[list_refined_mass_centers_sugar_ring_new_in_chain.index(R.tolist())] if R is not None else None
    residues.append(residue)
    scores.append(score)


# # unique method
# unique_indexes = []
# for i, P3 in enumerate(P3s):
#     if P3s.count(P3) == 1:
#         unique_indexes.append(i)
# P5s_U = [P5s[i] for i in unique_indexes]
# connection_indexes = []
# for i, index in enumerate(unique_indexes):
#     if index in connection_indexes:
#         continue
#     segement = []
#     while P3s[index] in P5s_U:
#         segement.append(index)
#         index = P5s.index(P3s[index])
#         print(index)
#     segement.append(index)
    
#     connection_indexes.append(segement)
# A = connection_indexes


# score method

# P5s_track, Rs_track, C1s_track, P3s_track, residues_track, scores_track = [], [], [], [], [], []

# print(len(P5s), len(residues))

backbone = []


# start_indexes = [i for i, score in enumerate(scores) if score in sorted(scores)[:10]]


start_indexes = range(0, len(P5s), len(P5s)//20)


for start_index in start_indexes:
    # this can be modified
    # if C1s[start_index] is None:
    #     continue
    # backbone_piece = [[P5s[start_index], C1s[start_index]], [P3s[start_index]]]
    # backbone_track(P5s, C1s, residues, backbone_piece)


    if Rs[start_index] is None:
        continue
    backbone_piece = [[P5s[start_index], Rs[start_index], residues[start_index]], [P3s[start_index]]]
    backbone_track(P5s, Rs, residues, backbone_piece, backbone_possibility)
    backbone.append(backbone_piece)
# print(backbone)


os.makedirs(os.path.join(folder_path, "chains_pdb"), exist_ok=True)



for i in range(len(backbone)):
    PP =[]
    res = []
    
    for j in range(len(backbone[i])):
        PP.append(backbone[i][j][0])
        res.append(backbone[i][j][2])
    out_pdb = os.path.join(folder_path, "chains_pdb", f"backbone.A.A.A.A.__{i}.pdb")
    print(res)
    write_pdb("P", "P", res, PP, out_pdb)

