# running a single or a few samples for CLUSTERING

import numpy as np
from R_B8_clustering_eval_driver import find_emdb_dirs, find_cif_file, clustering_result_exists
from R_A8_refactor2_A1_clustering import run_clustering, ClusteringConfig
# from R_A8_refactor2_A1_clustering_CVT import run_clustering, ClusteringConfig
from R_A0_utils import parse_calpha_CIF, parse_calpha_PDB
from R_B8_clustering_eval import compute_metrics_optimal

base_input_dir = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\TestSet\Raw"
emdb_ids = [6553]
threshold = 0.97
clustering_method = "meanshift"
output_dir = "H_Backbone_tracing/misc_outputs"
# output_dir = "H_Backbone_tracing/misc_outputs/test_DELETE_THESE3"
use_cache = False
cache_dir = output_dir
tolerance_radius = 1.9
np.set_printoptions(legacy='1.25')

#& Find files
emdb_folders = find_emdb_dirs(base_input_dir, emdb_ids)
print(emdb_folders)

#& Cluster for all entries in list
for id, subfolder in emdb_folders.items():
    print(f"EMD-{id}: {subfolder}")
    group_mrc_dir = subfolder + r"\possibility"
    ca_input_file = find_cif_file(subfolder)
    # main_mrc_map = group_mrc_dir + fr"\emd_{id}.map_residue_types_6letter_1_possibility.mrc"    #! this is from my residues driver file
    main_mrc_map = group_mrc_dir + fr"\emd_{id}.map_key_atoms_1_possibility.mrc"     #! key atom 1 aka CA
    print(f"group_mrc_dir: {group_mrc_dir}\nca_input_file: {ca_input_file}\nmain_mrc_map: {main_mrc_map}")

    # Optionally, use cached results
    cached_clustered_file = clustering_result_exists(cache_dir, id)
    if not (use_cache and cached_clustered_file):
        config = ClusteringConfig(
            input_mrc=main_mrc_map,
            clustering_method=clustering_method,
            output_dir=output_dir,
            threshold=threshold,
            manual_file_name=f"emd_{id}_{threshold}"
        )
        #~ CHANGE CLUSTERING FXN HERE
        results = run_clustering(config)
        clustered_file = results["output_path"]
        print(clustered_file)
    else:
        print("using cached result:", cached_clustered_file)
        clustered_file = cached_clustered_file

    #& Evaluate clustering results
    ca_gt = parse_calpha_CIF(ca_input_file)
    ca_pred = parse_calpha_PDB(clustered_file)
    metrics = compute_metrics_optimal(gt_coords=ca_gt, pred_coords=ca_pred, radius=tolerance_radius)
    print(metrics)