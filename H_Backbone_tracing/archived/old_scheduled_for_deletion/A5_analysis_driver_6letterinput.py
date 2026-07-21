# from A4_analysis_ALT_6letterinput import main as run_analysis

import csv
import os
import re


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


def run_analysis_driver(sample_list: str | list[int], base_folder: str) -> None:
    """Run the analysis pipeline.

    Args:
        sample_list: Either the path to a CSV file containing sample IDs or a list of sample IDs.

        base_folder: Path to the base output directory.
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
    emdb_folders = find_emdb_dirs(base_folder, emdb_ids)
    # for k, v in emdb_folders.items(): print(f"{k}: {v}")   # single line print
    found_count = sum(value is not None for value in emdb_folders.values())
    print(f"{found_count}/{len(emdb_folders)} sample subfolders found")


    ### The meat of the code
    from H_Backbone_tracing.archived.old_scheduled_for_deletion.A4_analysis_ALT_6letterinput import run_analysis
    for id, subfolder in emdb_folders.items():
        print(f"EMD-{id}: {subfolder}")
        group_mrc_dir = subfolder + r"\possibility"
        ca_input_file = find_cif_file(subfolder)
        main_mrc_map = group_mrc_dir + fr"\emd_{id}.map_residue_types_6letter_1_possibility.mrc"
        print(f"group_mrc_dir: {group_mrc_dir}\nca_input_file: {ca_input_file}\nmain_mrc_map: {main_mrc_map}")

        # results, overall_accuracy = run_analysis(ca_input_file, group_mrc_dir, main_mrc_map)

        #TODO: continue from this for loopthe l
        with open("H_Backbone_tracing/A6_results_dump.txt", "a") as file:
            file.write(f"EMD-{id}: {subfolder}\n{os.path.basename(ca_input_file)}\n\n")



if __name__ == "__main__":
    csv_file = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\TestSet\New\ready_to_test_new.csv"
    sample_list = csv_file
    # sample_list = [123]

    # base_folder = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction"
    base_folder = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\TestSet\Raw"

    run_analysis_driver(sample_list=sample_list, base_folder=base_folder)



# TODO: could consider adding verbose flag and then printing out all the debug stuff (and reformatting some of them), but low prio