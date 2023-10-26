import os
import pickle

# import numpy as np
import pandas as pd
from B2_UTILS_modeling import write_pdb  # refine_mass_centers, track_5_3,
from B2_UTILS_modeling_backbone_track import backbone_track

HOME_PATH = "/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/"
LAB_DATA = HOME_PATH + "LabData_testing/"
RAW_RATA = HOME_PATH + "RawData_testing/"
# MAP_PATH = LAB_DATA + "Alex/segement/postprocess.mrc"
# MAP_PATH = LAB_DATA + "Yuntao/02/after-ISN/post_isonet_J498.mrc"
# MAP_PATH = LAB_DATA + "Jane/J498/cryosparc_P2_J498_010_volume_map_sharp.mrc"
# MAP_PATH = LAB_DATA + "Jensen/RECS-B/treated/post_run2ar1_1sharpen.mrc"
# MAP_PATH = LAB_DATA + "Fumi/emd_27875/emd_27875.map"
# MAP_PATH = LAB_DATA + "Fumi/Nsp15-RNA_complex/cryosparc_P53_cryosparc_P53_J358_010_volume_map_sharp.mrc"
MAP_PATH = LAB_DATA + "Fumi/Nsp15-RNA_complex-unsharpen/cryosparc_P53_cryosparc_P53_J358_010_volume_map.mrc"
# MAP_PATH = LAB_DATA + "Yao/emd_26865/emd_26865.map"

# MAP_PATH = RAW_RATA + "EMD-16016_re_2.8-GOOD/emd_16016.map"
# MAP_PATH = RAW_RATA + "EMD-14956_re_2.7-GOOD/emd_14956.map"
# MAP_PATH = RAW_RATA + "EMD-34905_re_2.9-GOOD/emd_34905.map"

# MAP_PATH = HOME_PATH + "test_for_dirrerentiaiton_ssRNA/spliceosome_re2_3/ribosome/EMD-13112_re_2.4 - GOOD - ribosome/emd_13112.map"


folder_path = os.path.dirname(MAP_PATH)
temp_folder = os.path.join(folder_path, "temp_in_chain")

chain_id = 1
with open(os.path.join(temp_folder, f"R_in_chain_{chain_id}"), "rb") as file:
    refined_mass_centers_sugar_ring_new_in_chain = pickle.load(file)
with open(os.path.join(temp_folder, f"residue_in_chain_{chain_id}"), "rb") as file:
    residue_lists_in_chain = pickle.load(file)
with open(os.path.join(temp_folder, f"P_in_chain_{chain_id}"), "rb") as file:
    refined_mass_centers_P_in_chain = pickle.load(file)
with open(os.path.join(temp_folder, f"C1_in_chain_{chain_id}"), "rb") as file:
    refined_mass_centers_C1_in_chain = pickle.load(file)
with open(os.path.join(temp_folder, "possibility_in_chain"), "rb") as file:
    backbone_probability = pickle.load(file)

# Read score list
current_directory = os.path.dirname(os.path.abspath(__file__))
score_folder = os.path.join(current_directory, "score_distribution")
df = pd.read_csv(
    os.path.join(current_directory, "score_distribution", "Scores_all.csv")
    )
s_dist = dict(zip(df["dist_phosDistSmoothed"], df["likelihood_phosDistSmoothed"])),
s_angleP = dict(zip(df["angle_phosAngleSmoothed"], df["likelihood_phosAngleSmoothed"])),
s_angleS = dict(zip(
    df["angel_sugarPhosSugarAngleSmoothed"],
    df["likelihood_sugarPhosSugarAngleSmoothed"]
    )),
s_pseudoX = dict(zip(df["angle_pseudoChiSmoothed"], df["likelihood_pseudoChiSmoothed"]))

# delete this part on 10/4/2023
# connection
# P5s, Rs, C1s, P3s, residues, scores = [], [], [], [], [], []
# for P5 in refined_mass_centers_P_in_chain:
#     score, R, P3, C1 = track_5_3(
#         P5,
#         refined_mass_centers_P_in_chain,
#         refined_mass_centers_sugar_ring_new_in_chain,
#         refined_mass_centers_C1_in_chain
#     )
#     # P5s.append(P5.tolist() if P5 is not None else None)
#     # Rs.append(R.tolist() if R is not None else None)
#     # C1s.append(C1.tolist() if C1 is not None else None)
#     # P3s.append(P3.tolist() if P3 is not None else None)

#     P5s.append(P5 if P5 is not None else None)
#     Rs.append(R if R is not None else None)
#     C1s.append(C1 if C1 is not None else None)
#     P3s.append(P3 if P3 is not None else None)

#     list_refined_mass_centers_sugar_ring_new_in_chain = [list(arr) for arr in refined_mass_centers_sugar_ring_new_in_chain]
#     residue = residue_lists_in_chain[list_refined_mass_centers_sugar_ring_new_in_chain.index(R.tolist())] if R is not None else None
#     residues.append(residue)
#     scores.append(score)


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

# backbone = []


# start_indexes = [i for i, score in enumerate(scores) if score in sorted(scores)[:10]]


# start_indexes = range(0, len(P5s), len(P5s)//20)


chains = backbone_track(
    refined_mass_centers_P_in_chain,
    refined_mass_centers_sugar_ring_new_in_chain,
    residue_lists_in_chain,
    refined_mass_centers_C1_in_chain,
    backbone_probability
)


os.makedirs(os.path.join(folder_path, "chains_pdb"), exist_ok=True)


for i in range(len(chains)):

    RR = []
    res = []

    chain = chains[i]

    for mono in chain:
        if mono.R is not None:
            RR.append(mono.R)
            res.append(mono.base)
        elif mono.C1 is not None:
            RR.append(mono.C1)
            res.append(mono.base)
    out_pdb = os.path.join(folder_path, "chains_pdb", f"chains.A.A.A.A.__{i}.pdb")
    print(res)
    write_pdb("R", "R", res, RR, out_pdb)
