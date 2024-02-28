import os
from collections import deque

import numpy as np
import pandas as pd

# import pickle

# from B2_UTILS_modeling import refine_mass_centers, track_5_3, write_pdb

# Read score list
current_directory = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(current_directory, "score_distribution", "Scores_all.csv")
df = pd.read_csv(csv_file)
S_DIST = dict(
    zip(df["dist_phosDistSmoothed"],
        df["likelihood_phosDistSmoothed"]))
S_ANGLEP = dict(
    zip(df["angle_phosAngleSmoothed"],
        df["likelihood_phosAngleSmoothed"])
    )
S_ANGLES = dict(
    zip(df["angel_sugarPhosSugarAngleSmoothed"],
        df["likelihood_sugarPhosSugarAngleSmoothed"])
    )
# S_PESUDOX = dict(
#     zip(df["angle_pseudoChiSmoothed"], df["likelihood_pseudoChiSmoothed"])
#     )
SCORE_LIST = [S_DIST, S_ANGLES]


class Monomer:
    def __init__(self, P, R, C1, base):
        self.P = P
        self.R = R
        self.C1 = C1
        self.base = base


def rm_atom(atom_list, atom1):
    return [x for x in atom_list if not (x == atom1).all()]


def idx(atom_list, atom1):
    return [i for i in range(len(atom_list)) if (atom_list[i] == atom1).all()]


# C1 can be R
def backbone_track(P_atoms, R_atoms, residue_types, C1_atoms, backbone_probability):
    # backbone_probability is the probability of part backbone == 1

    Ps, Rs, res, C1s = P_atoms[:], R_atoms[:], residue_types[:], C1_atoms[:]

    chains = []

    for p in Ps:
        Ps = rm_atom(Ps, p)
        chain = deque()

        # 1. one side of p00
        p_00 = p
        while p_00 is not None:
            next = next_monomer(p_00, Ps, Rs, res, C1s, backbone_probability)
            chain.append(next)
            if next.P is not None:
                Ps = rm_atom(Ps, next.P)
            if next.C1 is not None:
                C1s = rm_atom(C1s, next.C1)
            if next.R is not None:
                idx_R = idx(Rs, next.R)[0]
                res.pop(idx_R)
                Rs.pop(idx_R)

            p_00 = next.P

        # 2. another side of p_00
        p_00 = chain[0].P
        Ps.append(p)
        while p_00 is not None:
            next = next_monomer(p_00, Ps, Rs, res, C1s, backbone_probability)
            chain.appendleft(next)
            if next.P is not None:
                Ps = rm_atom(Ps, next.P)
            if next.C1 is not None:
                C1s = rm_atom(C1s, next.C1)
            if next.R is not None:
                idx_R = idx(Rs, next.R)[0]
                res.pop(idx_R)
                Rs.pop(idx_R)

            p_00 = next.P

        if len(chain) >= 2:
            chains.append([chain, len(chain)])
        else:
            Ps = rm_atom(Ps, p)

    chains = sorted(chains, key=lambda x: x[1], reverse=True)
    chains = [x for x, _ in chains]

    return chains


def next_monomer(p_00, Ps, Rs, res, C1s, backbone_probability):

    mono_scores = []

    p_01s = find_atoms(Ps, p_00, [2.1, 9.8])
    r_01s = find_atoms(Rs, p_00, [2., 7.])
    c1_01s = find_atoms(C1s, p_00, [2.1, 9.8])

    r_temps = examine_prob(r_01s, p_00, backbone_probability)
    c1_temps = examine_prob(c1_01s, p_00, backbone_probability)
    r_to_c1 = False if r_temps else True
    r_temps = c1_temps if r_to_c1 else r_temps

    if r_temps:
        for r_temp in r_temps:
            if not r_to_c1:
                res_temp = res[idx(Rs, r_temp)[0]]
            else:
                res_temp = "X"

            c_temps = find_atoms(c1_01s, r_temp, [0., 3.])
            c_temps = examine_prob(c1_01s, r_temp, backbone_probability)
            c_temp = c_temps[0] if c_temps else None
            p_temps = find_atoms(p_01s, r_temp, [2., 7.])
            p_temps = examine_prob(p_temps, r_temp, backbone_probability)
            if p_temps:
                for p_temp in p_temps:
                    score = evaluate(p_00, r_temp, p_temp)
                    mono = Monomer(p_temp, r_temp, c_temp, res_temp)
                    mono.R = None if r_to_c1 else mono.R
                    mono_scores.append([mono, score])
            else:
                mono = Monomer(None, r_temp, c_temp, res_temp)
                mono.R = None if r_to_c1 else mono.R
                mono_scores.append([mono, 0.])

        # find best monomer (do we need to add some limit to this score?)
        mono = max(mono_scores, key=lambda x: x[1])
        mono = mono[0] if mono[1] > 0. else Monomer(None, None, None, None)

    # elif p_01s:
    #     for p_temp in p_01s:
    #         score = evaluate(p_00, None, p_temp)
    #         mono = Monomer(p_temp, None, None, "X")
    #         mono_scores.append([mono, score])

    else:
        mono = Monomer(None, None, None, None)
        # mono_scores.append([mono, float('-inf')])

    print(len(res), len(Rs), len(Ps), len(C1s))
    return mono


def find_atoms(atom_list, atom, range):
    atom_dists = []

    for pos in atom_list:
        distance = np.linalg.norm(pos - atom)
        if range[0] <= distance <= range[1]:
            atom_dists.append([pos, distance])

    atom_dists = sorted(atom_dists, key=lambda x: x[1])
    atoms = [x for x, _ in atom_dists]
    return atoms


def examine_prob(atom_list, atom, probability):
    atom_probs = []
    for pos in atom_list:
        vec = pos - atom
        probs = []
        # for i in range(10):
        for i in range(2, 9):
            pos_1 = np.round(atom + i * vec / 10).astype(int)
            z1, y1, x1 = pos_1
            probs.append(probability[z1][y1][x1])
        probs.sort()
        # if min(probs) > 0.5:
        if probs[0] > 0.5:
            atom_probs.append([pos, probs[0]])
    atom_probs = sorted(atom_probs, key=lambda x: x[1], reverse=True)

    atoms = [x for x, _ in atom_probs]
    return atoms


def evaluate(p0, r0, p1, SCORE_LIST=SCORE_LIST):
    s_dist, s_angleS = SCORE_LIST
    dist_pp = round(np.linalg.norm(p1 - p0), 2)
    s_P = 5 * np.log(s_dist[dist_pp])

    if r0 is not None:
        r0p1 = p1 - r0
        r0p0 = p0 - r0
        cosine_angle = np.dot(r0p1, r0p0) / (np.linalg.norm(r0p1) * np.linalg.norm(r0p0))
        angle_pcp = int(np.arccos(cosine_angle) * 180/np.pi)
        s_R = np.log(s_angleS[angle_pcp])
    else:
        s_R = 0.

    return s_P + s_R

    # max_index = max(range(len(score_P_Rs)), key=lambda i: score_P_Rs[i][0])
    # # print(score_P_Rs[max_index][0])

    # if score_P_Rs[max_index][0] < 0.:
    #     backbone_piece[-1].extend([None, "X"])
    #     break
    # P3, R2 = score_P_Rs[max_index][1], score_P_Rs[max_index][2]
    # backbone_piece[-1].append(R2)

    # if R2 is not None:
    #     idx_R2 = idx(Rs, R2)[0]
    #     Rs = rm_atom(Rs, R2)
    #     if 0 <= idx_R2 < len(res):
    #         # del res[idx_R2]
    #         res2 = res.pop(idx_R2)
    #         backbone_piece[-1].append(res2)
    # else:
    #     backbone_piece[-1].append("X")
    # # print(backbone_piece[-1])

    # backbone_piece.append([P3])
    # Ps = rm_atom(Ps, P3)
    # P1 = P2
    # P2 = P3
    # R1 = R2


def find_Rs(P5, P3, R_position):

    a1, a2, a3 = -0.186, 1.623, -0.124
    b1, b2 = 0.440, 0.910

    d = np.linalg.norm(P3 - P5)
    z = b1 * d + b2
    origin = (P3 - P5) * z/d + P5
    r = a1 * d**2 + a2 * d + a3
    axis = (P3 - P5) / np.linalg.norm(P3 - P5)

    # Rs: dis(R, origin)<r+4
    Rs = [pos for pos in R_position if pos is not None and np.linalg.norm(pos - origin) < r + 4]

    # score_R: score = dis(R, calculated_R)
    score_Rs = []
    for R in Rs:
        vector = R - origin
        projection_z = abs(np.dot(vector, axis))
        projection_r = np.linalg.norm(np.cross(vector, axis))
        score = ((projection_z - z)**2 + (projection_r - r)**2) ** 0.5
        score_Rs.append([score, R])

        # if abs(projection_z - z) < 2 and abs(projection_r - r) < 3:
        #     score_Rs.append([score , R])
        # else:
        #     score_Rs.append([9999, R])

    score_Rs = sorted(score_Rs, key=lambda x: x[0])
    score_Rs = [R for R in score_Rs if R[0] <= 2]

    return score_Rs


# def backbone_track(P_atoms, C1_atoms, residue_types, backbone_piece):
#     # backbone_piece = [[P5_1, C1_1], [P3_1=P5_2]]

#     # Read score list
#     current_directory = os.path.dirname(os.path.abspath(__file__))
#     df = pd.read_csv(os.path.join(current_directory, "score_distribution", "Scores_all.csv"))
#     s_dist, s_angleP, s_angleS, score_pseudoX = dict(zip(df["dist_phosDistSmoothed"], df["likelihood_phosDistSmoothed"])), \
#         dict(zip(df["angle_phosAngleSmoothed"], df["likelihood_phosAngleSmoothed"])), \
#         dict(zip(df["angel_sugarPhosSugarAngleSmoothed"], df["likelihood_sugarPhosSugarAngleSmoothed"])), \
#         dict(zip(df["angle_pseudoChiSmoothed"], df["likelihood_pseudoChiSmoothed"]))


#     # backbone_piece = [[P5_1, C1_1], [P3_1=P5_2]]

#     # can add a varification of C12 here

#     Ps, C1s, res = P_atoms, C1_atoms, residue_types
#     P1, C11, P2 = backbone_piece[0][0], backbone_piece[0][1], backbone_piece[1][0]
#     Ps = rm_atom(Ps, P1)
#     Ps = rm_atom(Ps, P2)
#     C1s = rm_atom(C1s, C11)
#     # C1s = rm_atom(C1s, C12)

#     while True:

#         P3s = [pos for pos in Ps if 2.1 < np.linalg.norm(pos - P2) < 9.8]
#         if len(P3s) == 0:
#             break

#         score_Ps = []
#         for P3 in P3s:

#             dist_PP = round(np.linalg.norm(P3 - P2), 2)
#             P2P1 = P1 - P2
#             P2P3 = P3 - P2
#             cosine_angle = np.dot(P2P1, P2P3) / (np.linalg.norm(P2P1) * np.linalg.norm(P2P3))
#             angle_PPP = int(np.arccos(cosine_angle) * 180/np.pi)

#             # determine C12
#             C12s = find_C1s(P2, P3, C1s)
#             for C12 in C12s:
#                 P2C11 = C11 - P2
#                 P2C12 = C12 - P2
#                 cosine_angle = np.dot(P2C11, P2C12) / (np.linalg.norm(P2C11) * np.linalg.norm(P2C12))
#                 angle_CPC = int(np.arccos(cosine_angle) * 180/np.pi)

#                 s_P = 5 * np.log(s_dist[dist_PP]) + np.log(s_angleP[angle_PPP]) + np.log(s_angleS[angle_CPC])
#                 score_Ps.append([s_P, P3, C12])

#         max_score = max(score_Ps, key=lambda x: x[0])
#         if max_score < 0:
#             break
#         max_index = max(range(len(score_Ps)), key=lambda i: score_Ps[i][0])
#         P3, C12 = score_Ps[max_index][1], score_Ps[max_index][2]
#         backbone_piece[-1].append(C12)
#         backbone_piece.append([P3])

#         Ps = rm_atom(Ps, P3)
#         C1s = rm_atom(C1s, C12)
#         P1 = P2
#         P2 = P3
#         C11 = C12


# def find_C1s(P5, P3, C1_position):

#     a1, a2, a3 = -0.186, 1.623, -0.124
#     b1, b2 = 0.440, 0.910

#     d = np.linalg.norm(P3 - P5)
#     z = b1 * d + b2
#     origin = (P3 - P5) *z/d + P5
#     r = a1 * d**2 + a2 * d + a3
#     axis = (P3 - P5) / np.linalg.norm(P3 - P5)

#     # C1s: dis(C1, origin)<r+4
#     C1s = [pos for pos in C1_position if pos is not None and np.linalg.norm(pos - origin) < r+4 ]

#     # C1_candidates: score = dis_min(C1, calculated_G)  [notice only for x, z distance]
#     C1_candidates = []
#     for C1 in C1s:
#         vector = C1 - origin
#         projection_z = abs(np.dot(vector, axis))
#         projection_r = np.linalg.norm(np.cross(vector, axis))
#         score = ((projection_z - z)**2 + (projection_r - r)**2) **0.5
#         C1_candidates.append([score , C1])

#         # if abs(projection_z - z) < 2 and abs(projection_r - r) < 3:
#         #     C1_candidates.append([score , C1])
#         # else:
#         #     C1_candidates.append([9999, R])

#     C1_candidates = sorted(C1_candidates, key = lambda x: x[0])
#     C1_candidates = [C1 for C1 in C1_candidates if C1[0]<=3]

#     return C1_candidates


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
    if len(C1s) == 0:
        C1 = None
    elif len(C1s) == 1:
        C1 = C1s[0]
    elif len(C1s) > 1:
        C1 = C1s[0]
        print("len(C1s) == 2: to be fixed in the future.")

    return score, R, P3, C1
