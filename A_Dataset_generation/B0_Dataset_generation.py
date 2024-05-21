import json
import os
import shutil

import numpy as np
import pandas as pd
from B1_UTILS import download_map_model, map_normalizing, map_output
from B2_UTILS_data2npy import data_to_npy

# from scipy.ndimage import zoom

DOWNLOAD = False
NORMALIZATION = False
MAP2NPY = True

MODEL_PARTS = ['secondary_strctures', 'key_atoms', 'residue_types']

MAIN_HOME_PATH = '/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/Filtered_Dateset'
PATH_SETTINGS = {
    "Filtered_Dateset": [
        os.path.join(MAIN_HOME_PATH, 'Raw'),
        os.path.join(MAIN_HOME_PATH, 'Training'),
        # os.path.join(MAIN_HOME_PATH, 'Raw', 'final-20240212.csv'),
        os.path.join(MAIN_HOME_PATH, 'Raw', 'final-20240520.csv'),
    ],
}

PATH_KEYS = "Filtered_Dateset"
DATA_PATH, HOME_PATH, csv_path = PATH_SETTINGS[PATH_KEYS]
temp_sample_path = os.path.join(HOME_PATH, "ready_to_train_and_val")
os.makedirs(temp_sample_path, exist_ok=True)


def generate_dataset(HOME_PATH, csv_path):
    # Read map list and generate raw_map and model downloading paths
    df = pd.read_csv(csv_path)
    emdbs, pdbs = df["emdb_id"], df["fitted_pdbs"]
    resolutions = df["resolution"].astype(str)
    emdb_ids = [emdb.split("-")[1] for emdb in emdbs]
    folders = [
        f"{emdb}_re_{resolution}"
        for emdb, resolution in zip(emdbs, resolutions)
    ]
    raw_maps = [f"emd_{emdb_id}.map" for emdb_id in emdb_ids]
    models = [f"{pdb}.cif" for pdb in pdbs]
    raw_map_paths = [
        f"{DATA_PATH}/{folder}/{raw_map}"
        for folder, raw_map in zip(folders, raw_maps)
    ]
    model_paths = [
        f"{DATA_PATH}/{folder}/{model}"
        for folder, model in zip(folders, models)
    ]

    # Download cryoEM map file and correspoding atomic model file,
    # then create resampled and normalized map file
    map_paths = [
        f"{raw_map_path.split('.map')[0]}_normalized.mrc"
        for raw_map_path in raw_map_paths
    ]
    for idx, emdb_id in enumerate(emdb_ids):
        if DOWNLOAD:
            download_map_model(emdb=emdbs[idx],
                               pdb=pdbs[idx],
                               resolution=resolutions[idx],
                               directory=DATA_PATH)
        if NORMALIZATION and not os.path.exists(map_paths[idx]):
            map_data = map_normalizing(raw_map_paths[idx])
            map_output(raw_map_paths[idx],
                       map_data,
                       map_paths[idx],
                       is_model=False)

    if MAP2NPY:
        # Create training and testing dataset (3d numpy arraies) from map file and model file

        # idx of npy file
        sample_num = 0

        # num of tag in each part for all models
        num_of_tag_in_each_part_for_all_models = {}
        for part in MODEL_PARTS:
            num_of_tag_in_each_part_for_all_models[part] = 0

        for idx, emdb_id in enumerate(emdb_ids):
            # test for idx in [0,1]
            # for idx, emdb_id in enumerate(emdb_ids[0:1]):
            print(
                f"[{idx+1}/{len(emdb_ids)}] Generating dataset from EMDB-{emdb_id}... "
            )
            sample_num, num_of_tag_in_each_part = data_to_npy(
                map_paths[idx], model_paths[idx], MODEL_PARTS,
                temp_sample_path, sample_num)
            # Add number of each tag in new sampled model
            for key, value_list in num_of_tag_in_each_part.items():
                num_of_tag_in_each_part_for_all_models[key] += value_list

        # Delete 0s in tag number list and calculate 1/ratio of tag number
        ratio_of_tag = {}
        for key, value_list in num_of_tag_in_each_part_for_all_models.items():
            value_list = np.trim_zeros(value_list)
            num_of_tag_in_each_part_for_all_models[key] = value_list
            if 0 in value_list:
                print(
                    f'There is missing groups in part {key}, numbers of each class are {value_list}.'
                )
            ratio_of_tag[key] = [
                int(value_list[0] // value_x) for value_x in value_list
                if value_x != 0
            ]

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
        # sample_path = os.path.join(HOME_PATH, "TEST_with_blank", "train_val_data")
        # sample_path = os.path.join(HOME_PATH, "TEST_with_all", "train_val_data")
        sample_path = os.path.join(HOME_PATH, "TEST_20240520", "train_val_data")


        os.makedirs(sample_path, exist_ok=True)
        splitfolders.ratio(input=temp_sample_path,
                           output=sample_path,
                           seed=44,
                           ratio=(.8, .2),
                           group_prefix=None,
                           move=True)
        shutil.rmtree(temp_sample_path)

        # Calculate weight, create weight file and save it (ratio of tags)
        weight_path = os.path.join(sample_path,
                                   'class_weight_for_training.txt')
        with open(weight_path, "w") as file:
            json.dump(ratio_of_tag, file)


if __name__ == "__main__":
    generate_dataset(HOME_PATH, csv_path)
