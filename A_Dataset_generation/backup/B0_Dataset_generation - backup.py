import json
import os
import shutil

import pandas as pd
from B1_UTILS import download_map_model, map_normalizing, map_output
from B2_UTILS_data2npy import data_to_npy

# from scipy.ndimage import zoom

DOWNLOAD = True
NORMALIZATION = False
MAP2NPY = False
# MODEL_PARTS = ["phosphorus_C1", "sugar_ring_new", "nitrogenous_bases_in_sugar_new",
#                "RNA_backbone", "sugar_ring_center"]
# MODEL_PARTS = ["TEST_MAP_VS_MODEL_ALL_ATOMS"]
MODEL_PARTS = ["unit_1", "unit_2"]

# MAIN_HOME_PATH = '/amber/electron/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net'
MAIN_HOME_PATH = '/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net'
PATH_SETTINGS = {
    "Filtered_Dateset": [
        os.path.join(MAIN_HOME_PATH, 'Filtered_Dateset', 'Raw'),
        os.path.join(MAIN_HOME_PATH, 'Filtered_Dateset', 'Training'),
        os.path.join(MAIN_HOME_PATH, 'Filtered_Dateset',
                     'Final_Filtered_Data.csv'),
    ],
    "EXAMPLE_KEYS": [
        "DATA_PATH",
        "HOME_PATH",
        "CSV_PATH",
    ],
    "training2": [
        MAIN_HOME_PATH + '/RawData',
        MAIN_HOME_PATH + '/training/training2',
        MAIN_HOME_PATH +
        '/training/training2/TEST_EMDB_download_resolution_1-4_link_refined_unique.csv',
    ],
    "training3": [
        MAIN_HOME_PATH + '/RawData',
        MAIN_HOME_PATH + '/training/training3',
        MAIN_HOME_PATH +
        '/training/training3/TEST_EMDB_download_resolution_1-4_link_refined_unique.csv',
    ],
    "testing1": [
        MAIN_HOME_PATH + '/RawData_testing',
        MAIN_HOME_PATH + '/to_be_tested/Internet/testing1',
        MAIN_HOME_PATH +
        '/to_be_tested/Internet/testing1/test_cvs_1-3A_20230518-refine.csv',
    ],
}

PATH_KEYS = "Filtered_Dateset"  # "training3" # Possible values: "training_data", "test_ribosome_cryoET", "test_ribosome_cryoEM"
DATA_PATH, HOME_PATH, csv_path = PATH_SETTINGS[PATH_KEYS]
temp_sample_path = os.path.join(HOME_PATH, "ready_to_train_and_val")
os.makedirs(temp_sample_path, exist_ok=True)


def generate_dataset(HOME_PATH, csv_path):
    # Read map list and generate raw_map and model downloading paths
    df = pd.read_csv(csv_path)
    emdbs, pdbs = df["emdb_id"], df["fitted_pdbs"]
    resolutions = df["resolution"].astype(str)
    emdb_ids = [emdb.split("-")[1] for emdb in emdbs]
    folders = [f"{emdb}_re_{resolution}" for emdb, resolution in zip(emdbs, resolutions)]
    raw_maps = [f"emd_{emdb_id}.map" for emdb_id in emdb_ids]
    models = [f"{pdb}.cif" for pdb in pdbs]
    raw_map_paths = [f"{DATA_PATH}/{folder}/{raw_map}" for folder, raw_map in zip(folders, raw_maps)]
    model_paths = [f"{DATA_PATH}/{folder}/{model}" for folder, model in zip(folders, models)]


    # Download cryoEM map file and correspoding atomic model file, then create resampled and normalized map file
    map_paths = [f"{raw_map_path.split('.map')[0]}_normalized.mrc" for raw_map_path in raw_map_paths]
    for idx, emdb_id in enumerate(emdb_ids):
        if DOWNLOAD:
            download_map_model(emdb=emdbs[idx], pdb=pdbs[idx], resolution=resolutions[idx], directory=DATA_PATH)
            # if idx == 20:   # for download limited number of testing data
            #     exit()
        if NORMALIZATION and not os.path.exists(map_paths[idx]):
            map_data = map_normalizing(raw_map_paths[idx])
            map_output(raw_map_paths[idx], map_data, map_paths[idx], is_model = False)

    if MAP2NPY:
        # Create training and testing dataset (3d numpy arraies) from map file and model file
        sample_num = 0                                              # idx of npy file
        num_of_tag_in_each_part_for_all_models = {}                 # number of tag in each part for all models

        for idx, emdb_id in enumerate(emdb_ids):
            # for idx in [0,1]:
            print(f"Generating dataset from EMDB-{emdb_id}... ")
            sample_num, num_of_tag_in_each_part = data_to_npy(map_paths[idx], model_paths[idx], MODEL_PARTS, temp_sample_path, sample_num)

            # Add number of each tag in new sampled model
            for key, value_list in num_of_tag_in_each_part.items():
                num_of_tag_in_each_part_for_all_models[key] = [num_1 + num_2 for num_1, num_2 in zip(num_of_tag_in_each_part_for_all_models.get(key, [0]*len(value_list)), value_list)]

        # Delete 0s in tag number list and calculate 1/ratio of tag number
        ratio_of_tag = {}
        for key, value_list in num_of_tag_in_each_part_for_all_models.items():
            if 0 in value_list:
                print(f'There is missing groups in part {key}, numbers of each class are {value_list}.')
            ratio_of_tag[key] = [value_list[0]//value_x for value_x in value_list if value_x != 0]

        # Print statistical results
        print(f"The number of .npy file: {sample_num}")
        print("Num of each tag: ")
        for key, value in num_of_tag_in_each_part_for_all_models.items():
            print(f'{key}: {value}')
        print("\nRatio of tags: ")
        for key, value in ratio_of_tag.items():
            print(f'{key}: {value}')




        # split data into training and testing dataset
        import splitfolders
        sample_path = os.path.join(HOME_PATH, "TEST", "train_val_data")
        os.makedirs(sample_path, exist_ok=True)
        splitfolders.ratio(input=temp_sample_path, output=sample_path,
                           seed=44, ratio=(.8, .2), group_prefix=None, move=True)
        shutil.rmtree(temp_sample_path)


        # Calculate and print weight (ratio of tags)
        os.makedirs(os.path.join(HOME_PATH, "TEST"), exist_ok=True)
        with open(f"{HOME_PATH}/TEST/train_val_data/class_weight_for_training.txt", "w") as file:
            json.dump(ratio_of_tag, file)


if __name__ == "__main__":
    generate_dataset(HOME_PATH, csv_path)



# OLD TRANNING DATASET：
# sugar_ring_new: 1:85
# nitrogenous_bases_in_sugar_new: 1:344:279:366:419

# RNA_backbone: 1:40:51;
# sugar_ring_old: 1:63;
# nitrogenous_bases_in_sugar_old: 1:255:206:270:309
# nitrogenous_bases: 1:200:149:247:282
# phosphorus_C1: 1:315:315
# sugar_ring_center: 1:316
