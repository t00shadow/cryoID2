import pickle
import os
from B2_UTILS_modeling import (refine_mass_centers, write_pdb, track_5_3)
import numpy as np
import pandas as pd


def rm_np(Ps, P1):
    return [ x for x in Ps if not (x == P1).all()]

def idx(Ps, P1):
    return [i for i in range(len(Ps)) if (Ps[i] == P1).all()]




ALPHA = 15.

# C1 can be R
def backbone_track(P_in_chain, R_in_chain, residue_in_chain, backbone_piece, backbone_possibility):
    # backbone_piece = [[P1, R1, residues1], [P2]]


    # Read score list
    current_directory = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(current_directory, "score_distribution", "Scores_all.csv"))
    s_dist, s_angleP, s_angleS, score_pseudoX = dict(zip(df["dist_phosDistSmoothed"], df["likelihood_phosDistSmoothed"])), \
        dict(zip(df["angle_phosAngleSmoothed"], df["likelihood_phosAngleSmoothed"])), \
        dict(zip(df["angel_sugarPhosSugarAngleSmoothed"], df["likelihood_sugarPhosSugarAngleSmoothed"])), \
        dict(zip(df["angle_pseudoChiSmoothed"], df["likelihood_pseudoChiSmoothed"]))

    # can add a varification of P2 here

    Ps, Rs, res = P_in_chain[:], R_in_chain[:], residue_in_chain[:]
    P1, R1, P2 = backbone_piece[0][0], backbone_piece[0][1], backbone_piece[1][0]
    Ps = rm_np(Ps, P1)
    Ps = rm_np(Ps, P2)
    idx_R1 = idx(Rs, R1)[0]
    Rs = rm_np(Rs, R1)
    if 0 <= idx_R1 < len(res):
        res1 = res.pop(idx_R1)
    # Rs = rm_np(Rs, R2)
    
    while True:
        P3s = [pos for pos in Ps if 2.1 < np.linalg.norm(pos - P2) < 9.8]        
        if len(P3s) == 0:
            backbone_piece[-1].extend([None, "X"])
            break 
        
        score_P_Rs = []

        for P3 in P3s:
            dist_PP = round(np.linalg.norm(P3 - P2), 2)
            P2P1 = P1 - P2 
            P2P3 = P3 - P2 
            cosine_angle = np.dot(P2P1, P2P3) / (np.linalg.norm(P2P1) * np.linalg.norm(P2P3))
            angle_PPP = int(np.arccos(cosine_angle) * 180/np.pi)

            s_P = 5 * np.log(s_dist[dist_PP]) + np.log(s_angleP[angle_PPP])

            # determine R2
            # score_Rs = find_Rs(P2, P3, Rs)
            # R2s = [score_Rs[i][1] for i in range(len(score_Rs))]

            R2s = [pos for pos in Rs if pos is not None and np.linalg.norm(pos - P2) < 9.8]

            if R2s == []:
                s_P_R = s_P
                score_P_Rs.append([s_P_R, P3, None])

            else:
                for R2 in R2s:
                    P3R2 = R2 - P3
                    R2P2 = P2 - R2
                    s_possible = 0
                    for i in range(10):
                        pos_1 = P3 + i*P3R2/10
                        pos_2 = R2 + i*R2P2/10
                        s_possible += 30 * ( backbone_possibility[round(pos_1[0]), round(pos_1[1]), round(pos_1[2])]\
                                        +backbone_possibility[round(pos_2[0]), round(pos_2[1]), round(pos_2[2])] ) 
                    if R1 is None:
                        s_P_R = s_P + ALPHA * np.log(s_possible)
                        # R2 = R2s[0]
                        # score_P_Rs.append([s_P_R, P3, R2])
                        # continue

                    else:
                        P2R1 = R1 - P2 
                        P2R2 = R2 - P2 
                        cosine_angle = np.dot(P2R1, P2R2) / (np.linalg.norm(P2R1) * np.linalg.norm(P2R2))
                        angle_CPC = int(np.arccos(cosine_angle) * 180/np.pi)
                        s_R = np.log(s_angleS[angle_CPC])
                        s_P_R = s_P + s_R + s_possible
                    
                    score_P_Rs.append([s_P_R, P3, R2])

        max_index = max(range(len(score_P_Rs)), key=lambda i: score_P_Rs[i][0])        
        # print(score_P_Rs[max_index][0])

        if score_P_Rs[max_index][0] < 0:
            backbone_piece[-1].extend([None, "X"])
            break
        P3, R2 = score_P_Rs[max_index][1], score_P_Rs[max_index][2]
        backbone_piece[-1].append(R2)

        if R2 is not None:
            idx_R2 = idx(Rs, R2)[0]
            Rs = rm_np(Rs, R2)
            if 0 <= idx_R2 < len(res):
                # del res[idx_R2]
                res2 = res.pop(idx_R1)
                backbone_piece[-1].append(res2)
        else:
            backbone_piece[-1].append("X")
        # print(backbone_piece[-1])

        backbone_piece.append([P3])
        Ps = rm_np(Ps, P3)
        P1 = P2 
        P2 = P3
        R1 = R2


def find_Rs(P5, P3, R_position):

    a1, a2, a3 = -0.186, 1.623, -0.124
    b1, b2 = 0.440, 0.910

    d = np.linalg.norm(P3 - P5)
    z = b1 * d + b2
    origin = (P3 - P5) *z/d + P5
    r = a1 * d**2 + a2 * d + a3 
    axis = (P3 - P5) / np.linalg.norm(P3 - P5)

    # Rs: dis(R, origin)<r+4
    Rs = [pos for pos in R_position if pos is not None and np.linalg.norm(pos - origin) < r+4 ]
    
    # score_R: score = dis(R, calculated_R)
    score_Rs = []
    for R in Rs:
        vector = R - origin
        projection_z = abs(np.dot(vector, axis))
        projection_r = np.linalg.norm(np.cross(vector, axis))
        score = ((projection_z - z)**2 + (projection_r - r)**2) **0.5
        score_Rs.append([score , R])

        # if abs(projection_z - z) < 2 and abs(projection_r - r) < 3:
        #     score_Rs.append([score , R])
        # else:
        #     score_Rs.append([9999, R])

    score_Rs = sorted(score_Rs, key = lambda x: x[0])
    score_Rs = [R for R in score_Rs if R[0]<=2]

    return score_Rs


# def backbone_track(P_in_chain, C1_in_chain, residue_in_chain, backbone_piece):
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

#     Ps, C1s, res = P_in_chain, C1_in_chain, residue_in_chain
#     P1, C11, P2 = backbone_piece[0][0], backbone_piece[0][1], backbone_piece[1][0]
#     Ps = rm_np(Ps, P1)
#     Ps = rm_np(Ps, P2)
#     C1s = rm_np(C1s, C11)
#     # C1s = rm_np(C1s, C12)
    
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

#         Ps = rm_np(Ps, P3)
#         C1s = rm_np(C1s, C12)
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
        origin = (P3 - P5) *z/d + P5
        r = a1 * d**2 + a2 * d + a3 
        axis = (P3 - P5) / np.linalg.norm(P3 - P5)

        Rs = [pos for pos in ribose_position if np.linalg.norm(pos - origin) < r+3]
        score_R = []

        for R in Rs:
            vector = R - origin
            projection_z = abs(np.dot(vector, axis))
            projection_r = np.linalg.norm(np.cross(vector, axis))

            score = ((projection_z - z)**2 + (projection_r - r)**2) **0.5
            score_R.append([score , R])

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