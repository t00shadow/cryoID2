# Compare clustered CA atoms with depostied model CA atoms for a single sample

# NOTE based on how this eval is done, the order of atoms doesnt even matter, so don't worry if it's somehow different (cif vs pdb).

import numpy as np
from R_A0_utils import parse_calpha_CIF, parse_calpha_PDB
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from collections import defaultdict


#1
# Kd-tree + greedy
def compute_metrics_greedy(gt_coords, pred_coords, radius=1.0):
    gt_coords = np.asarray(gt_coords)
    pred_coords = np.asarray(pred_coords)

    tree = cKDTree(gt_coords)

    # For each prediction, find all GT points within radius
    matches = tree.query_ball_point(pred_coords, r=radius)

    matched_gt = set()
    tp = 0

    for pred_idx, gt_indices in enumerate(matches):
        for gt_idx in gt_indices:
            if gt_idx not in matched_gt:
                matched_gt.add(gt_idx)
                tp += 1
                break  # ensure one-to-one

    fp = len(pred_coords) - tp
    fn = len(gt_coords) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)

    return precision, recall, f1, tp, fp, fn




#2
# Optimal approach using Hungarian algo
def compute_metrics_optimal(gt_coords, pred_coords, radius=1.0):
    gt_coords = np.asarray(gt_coords)
    pred_coords = np.asarray(pred_coords)

    dist_matrix = cdist(pred_coords, gt_coords)

    # Mask distances beyond threshold
    large_value = 1e6
    cost_matrix = np.where(dist_matrix <= radius, dist_matrix, large_value)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Count valid matches
    valid = dist_matrix[row_ind, col_ind] <= radius
    tp = np.sum(valid)

    fp = len(pred_coords) - tp
    fn = len(gt_coords) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)

    return precision, recall, f1, tp, fp, fn




#3
# modified version of compute_metrics_greedy that tracks the amount of many-to-one collisions during matching
def compute_metrics_greedy_with_manytoone_count(gt_coords, pred_coords, radius=1.0):
    from scipy.spatial import cKDTree
    import numpy as np

    gt_coords = np.asarray(gt_coords)
    pred_coords = np.asarray(pred_coords)

    tree = cKDTree(gt_coords)
    matches = tree.query_ball_point(pred_coords, r=radius)

    gt_to_preds = defaultdict(list)

    # Build reverse mapping
    for pred_idx, gt_indices in enumerate(matches):
        for gt_idx in gt_indices:
            gt_to_preds[gt_idx].append(pred_idx)

    matched_gt = set()
    used_pred = set()
    tp = 0

    # Standard greedy matching (same as before)
    for pred_idx, gt_indices in enumerate(matches):
        for gt_idx in gt_indices:
            if gt_idx not in matched_gt:
                matched_gt.add(gt_idx)
                used_pred.add(pred_idx)
                tp += 1
                break

    fp = len(pred_coords) - tp
    fn = len(gt_coords) - tp

    # --- NEW: collision stats ---
    multi_match_gt = sum(1 for preds in gt_to_preds.values() if len(preds) > 1)
    total_collisions = sum(len(preds) - 1 for preds in gt_to_preds.values() if len(preds) > 1)

    percent_gt_with_collisions = multi_match_gt / len(gt_coords) if len(gt_coords) else 0
    percent_extra_preds = total_collisions / len(pred_coords) if len(pred_coords) else 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "multi_match_gt": multi_match_gt,
        "total_collisions": total_collisions,
        "percent_gt_with_collisions": percent_gt_with_collisions,
        "percent_extra_preds": percent_extra_preds,
    }




if __name__ == "__main__":
    cif_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\8hk3.cif"
    # clustered_CA_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\8HK3.pdb"    #* pdb version of ground truth
    clustered_CA_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\refactor2_agglomerative_cd_4.0_CA_only.pdb"
    clustered_CA_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\refactor2_DBSCAN_meanshift_bw_1.9_CA_only.pdb"
    clustered_CA_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\refactor2_meanshift_bw_1.9_CA_only.pdb"
    
    ca_coords_gt = parse_calpha_CIF(cif_file)
    ca_coords_pred = parse_calpha_PDB(clustered_CA_file)
    
    print("ground truth:")
    # print(ca_coords_gt)
    print(ca_coords_gt.shape)
    print("predicted")
    # print(ca_coords_pred)
    print(ca_coords_pred.shape)




    tolerance_radius = 1.5
    np.set_printoptions(legacy='1.25')

    print("METRIC ORDER: precision, recall, f1, tp, fp, fn")
    print("compute_metrics_greedy:")
    print(compute_metrics_greedy(gt_coords=ca_coords_gt, pred_coords=ca_coords_pred, radius=tolerance_radius))

    print("compute_metrics_optimal:")
    print(compute_metrics_optimal(gt_coords=ca_coords_gt, pred_coords=ca_coords_pred, radius=tolerance_radius))

    print("compute_metrics_greedy_with_manytoone_count:")
    print(tuple(compute_metrics_greedy_with_manytoone_count(gt_coords=ca_coords_gt, pred_coords=ca_coords_pred, radius=tolerance_radius).values()))
    # ^ oh this overlap should be low bc i have that local_thinning function in my clustering script that prevents atoms from getting too bunched up, so unless you increase the tolerance radius to like 3 or 5, you should almost never see any many-to-one matches

    assert(False)


    print("check if cif and pdb have same data, should be the same")
    cif_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\3JCH_helix_CA.cif"
    pdb_version_of_cif_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\3JCH_helix_CA.pdb"
    ca_coords_gt_cif = parse_calpha_CIF(cif_file)
    ca_coords_gt_pdb = parse_calpha_PDB(pdb_version_of_cif_file)
    print(ca_coords_gt_cif.shape)
    print(ca_coords_gt_pdb.shape)
    print(np.array_equal(ca_coords_gt_cif, ca_coords_gt_pdb))






    print("\nEMD-6553, re=7.06")
    cif_file = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\TestSet\Raw\EMD-6553_re_7.06\3JCH.cif"
    # cif_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\3JCH_helix_CA.cif"
    # clustered_CA_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\R_A7_experiments_ca_candidates.pdb"
    clustered_CA_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\refactor2_DBSCAN_meanshift_bw_1.9_CA_only.pdb"
    # clustered_CA_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\refactor2_DBSCAN_agglomerative_cd_4.0_CA_only.pdb"
    # clustered_CA_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\refactor2_DBSCAN_kmeans_CA_only.pdb"

    ca_coords_gt = parse_calpha_CIF(cif_file)
    ca_coords_pred = parse_calpha_PDB(clustered_CA_file)

    print("ground truth (note this is without selecting only the helix regions, since dont have DSSP installed rn):")
    print("edit: nvm just use chimerax to select helix regions")
    # print(ca_coords_gt)
    print(ca_coords_gt.shape)
    print("predicted")
    # print(ca_coords_pred)
    print(ca_coords_pred.shape)

    print("compute_metrics_optimal:")
    print(compute_metrics_optimal(gt_coords=ca_coords_gt, pred_coords=ca_coords_pred, radius=1.9))
    # print(tuple(compute_metrics_greedy_with_manytoone_count(gt_coords=ca_coords_gt, pred_coords=ca_coords_pred, radius=tolerance_radius).values()))
