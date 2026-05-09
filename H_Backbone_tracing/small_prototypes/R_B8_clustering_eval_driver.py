import csv
import os
import re
import datetime
import numpy as np


# ============================================================
# UTIL FXN
# ============================================================
def get_column_values(csv_file_path, column_name):
    values = []
    
    with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        if column_name not in reader.fieldnames:
            raise ValueError(f"Column '{column_name}' not found in CSV headers.")
        
        for row in reader:
            values.append(row[column_name])
    
    return values

# ============================================================
# UTIL FXN    #TODO: move into utils file (in the refactored version)
# ============================================================
def find_emdb_dirs(base_dir: str, id_list: list) -> dict[str, str | None]:
    """
    base_dir: some directory to start searching at
    id_list: iterable of ints or strings
    returns: dict {id: full_path_or_None}
    """
    # Convert IDs to strings
    id_list = {str(i) for i in id_list}
    # Build the dictionary
    results = {i: None for i in id_list}

    # Regex: EMD-<id>_re_<anything>
    pattern = re.compile(r"^EMD-(\d+)_re_")

    for entry in os.listdir(base_dir):
        full_path = os.path.join(base_dir, entry)

        if not os.path.isdir(full_path):
            continue

        match = pattern.match(entry)
        if not match:
            continue

        emd_id = match.group(1)

        if emd_id in id_list:
            results[emd_id] = full_path

    return results


# ============================================================
# UTIL FXN    #TODO: move into utils file (in the refactored version)
# ============================================================
# NOTE: due to how our files are organized, there's only one cif file in each dir, 
# but if this changes in the future, just read in that column data too and check that it's the right cif file.
def find_cif_file(base_dir: str) -> str | None:
    """
    base_dir: str
    returns: path of found cif file or None
    """
    for name in os.listdir(base_dir):
        path = os.path.join(base_dir, name)
        if os.path.isfile(path) and name.lower().endswith(".cif"):
            return path
    return None

# ============================================================
# UTIL FXN    find if a clustering result file exists, helps with the "caching" logic
# ============================================================
def clustering_result_exists(cache_dir: str, emdb_id: int) -> bool:     # TODO: consider renaming to find_clustering_file to match above fxns naming scheme
    target = f"_{emdb_id}_"
    for name in os.listdir(cache_dir):
        path = os.path.join(cache_dir, name)
        if os.path.isfile(path) and target in name:
            return path
    return None


def run_analysis_driver(sample_list: str | list[int], base_input_dir: str, clustering_method: str, threshold: float, use_cache: bool, cache_dir: str, output_dir: str, log_file: str, log_mode: str, tolerance_radius: float) -> None:
    """Run the analysis pipeline.

    Args:
        sample_list: Either the path to a CSV file containing sample IDs or a list of sample IDs.

        base_input_dir: Path to the base input directory.

        clustering_method: Clustering algorithm to use

        threshold: Threshold for the mrc files. 0.9 is the default #TODO: add the default

        use_cache: Use previously clustered results (i.e. skip cllustering if file already exists)

        cache_dir: Directory of existing clustered results to use, often the same as output_dir. But use a dif output_dir if you don't want to affect log files, etc (i.e. if you want "read-only").

        output_dir: Directory to store results, logs, and other output files.

        log_file: Name of the log file.    #TODO: add a default file name. Actually should add defaults for most of these.

        log_mode: Same as standard python file modes. https://www.geeksforgeeks.org/python/file-mode-in-python/

        tolerance_radius: Tolerance radius to use when evaluating clustering metrics. Larger radius improves metrics, but too large is meaningless.
    """

    ### Extracing sample ids from CSV file
    if type(sample_list) is str:
        emdb_ids = get_column_values(sample_list, "emdb_id")
        # print(emdb_ids)
        emdb_ids = [x.removeprefix('EMD-') for x in emdb_ids]
        # print(emdb_ids)
    elif type(sample_list) is list:
        if not all(type(x) is int for x in sample_list):
            raise TypeError("sample_list list must only contain ints")
        emdb_ids = sample_list
    else:
        raise TypeError("sample_list needs to be a string OR list of ints")


    ### Find the subfolder for each sample
    emdb_folders = find_emdb_dirs(base_input_dir, emdb_ids)
    # for k, v in emdb_folders.items(): print(f"{k}: {v}")   # single line print
    found_count = sum(value is not None for value in emdb_folders.values())
    print(f"{found_count}/{len(emdb_folders)} sample subfolders found")


    ### The meat of the code
    # from A4_analysis_ALT_6letterinput import run_analysis
    # from R_A8_refactor1_A1_clustering import main, ClusteringConfig
    from R_A8_refactor2_A1_clustering import run_clustering, ClusteringConfig
    from R_A0_utils import parse_calpha_CIF, parse_calpha_PDB
    from R_B8_clustering_eval import compute_metrics_optimal

    with open(log_file, log_mode) as file:
        timestamp = datetime.datetime.now()
        file.write(f"{timestamp}\n")
        file.write(f"parameters: sample_list={sample_list}, base_input_dir={base_input_dir}, clustering_method={clustering_method}, threshold={threshold}, use_cache={use_cache}, cache_dir={cache_dir}, output_dir={output_dir}, log_file={log_file}, log_mode={log_mode}, tolerance_radius={tolerance_radius}\n\n")

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
            # results, overall_accuracy = run_analysis(ca_input_file, group_mrc_dir, main_mrc_map)
            config = ClusteringConfig(
                input_mrc=main_mrc_map,
                clustering_method=clustering_method,
                output_dir=output_dir,
                threshold=threshold
            )
            # results = main(config)
            results = run_clustering(config)
            clustered_file = results["output_path"]
            print(clustered_file)
        else:
            print("using cached result:", cached_clustered_file)
            clustered_file = cached_clustered_file

        # Call the function that compares the clustering output with the corresponding cif file
        ca_gt = parse_calpha_CIF(ca_input_file)
        ca_pred = parse_calpha_PDB(clustered_file)
        metrics = compute_metrics_optimal(gt_coords=ca_gt, pred_coords=ca_pred, radius=tolerance_radius)
        print(metrics)

        #TODO replace with logging module
        with open(log_file, "a") as file:
            file.write(f"EMD-{id}: {subfolder}\ndeposited_model: {os.path.basename(ca_input_file)}\nclustered_file: {clustered_file}\n")
            file.write(f"metrics (precision, recall, f1, tp, fp, fn): {metrics}\n\n")



if __name__ == "__main__":
    csv_file = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\TestSet\New\ready_to_test_new.csv"
    sample_list = csv_file
    # sample_list = [123]

    # base_input_dir = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction"
    base_input_dir = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\TestSet\Raw"

    # clustering_method="agglomerative"
    clustering_method = "meanshift"   #* use meanshift since some are too large for agglomerative
    threshold = 0.97
    use_cache = True   # set to False if change threshold, since theyll still have the same file name
    # output_dir = "H_Backbone_tracing/misc_outputs/test_DELETE_THESE2"
    output_dir = "H_Backbone_tracing/misc_outputs/clustering_figure"
    cache_dir = output_dir
    log_file = output_dir + "/clustering_eval_log2.txt"
    log_mode = "a"
    log_mode = "w"   # https://www.geeksforgeeks.org/python/file-mode-in-python/
    # tolerance_radius = 1.5
    tolerance_radius = 1.9
    np.set_printoptions(legacy='1.25')


    run_analysis_driver(sample_list=sample_list, base_input_dir=base_input_dir, clustering_method=clustering_method, threshold=threshold, use_cache=use_cache, cache_dir=cache_dir, output_dir=output_dir, log_file=log_file, log_mode=log_mode, tolerance_radius=tolerance_radius)



# TODO: could consider adding verbose flag and then printing out all the debug stuff (and reformatting some of them), but low prio