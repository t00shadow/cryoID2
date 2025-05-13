import json
import os
import pickle

import numpy as np
# import pandas
from pandas import read_csv



# import pandas as pd
import torch
from B0_MODEL import UNET
from B1_UTILS import (
    identify_chain,
    load_checkpoint,
    map_normalizing,
    map_writing,
    predict_location,
    smooth_location,
)
from B2_UTILS_modeling import refine_mass_centers, write_pdb  # track_5_3,


# ====================== Hyperparameters etc. ========================
# MAIN_PATH = '/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/Filtered_Dateset/Training/TEST_with_blank'
# MAIN_PATH = '/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/Filtered_Dateset/Training/TEST_with_all'
# MAIN_PATH = '/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/Filtered_Dateset/Training/TEST_20240520'

EX_OR_CO = 'experimental'  # 'control'
MAIN_PATH = f'/home/qiboxu/Database/CryoDataBot_Data/Training/{EX_OR_CO}-dataset'



MODEL_TYPES = ['secondary_strctures', 'key_atoms', 'residue_types']
# MAP_DIRS_TYPES = [['map_sample'],
#                   ['map_sample', 'secondary_strctures_pred_prob'],
#                   [
#                       'map_sample', 'secondary_strctures_pred_prob',
#                       'key_atoms_pred_prob'
#                   ]]
MAP_DIRS_TYPES = [["map_sample"], ["map_sample"], ["map_sample"]]

# IN_CHANNELS = [1, 6, 10]

# data_path = os.path.join(MAIN_PATH, 'train_val_data')
data_path = MAIN_PATH


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# batch_size (vs GPU Memory: 1vs2877; 2vs3485; 4vs4668; 8vs7162)
# batch_size = 8
# num_workers = 2
# learning_rate = 1e-4
# pin_memory = True


def initial_setting(model_index, data_path, device):
    """
    This function performs the initial setup before training process:
    1. Read the class weights of the UNET model,
    2. Read the saved epoch number,
    3. Generate paths of file that will be loaded to UNET model.
    """

    model_type = MODEL_TYPES[model_index]
    # Load the class weights of all model types
    weight_path = os.path.join(data_path, 'class_weight_for_training.txt')
    try:
        # Load the weight for training
        with open(weight_path, "r") as file:
            weight_all = json.load(file)

        if model_type not in weight_all.keys():
            print(f"Weight for {model_type} is not found.")
            exit()
        else:
            c_w = weight_all[model_type]
            class_weights = torch.FloatTensor(c_w).to(device)

            # Calculate the number of output channels based on the number of tags in model_type
            out_channels = len(c_w)
            in_channels = 1
            if model_index > 0:
                for ii in range(model_index):
                    in_channels = in_channels + len(
                        weight_all[MODEL_TYPES[ii]])

    except FileNotFoundError:
        print("Weight file does not exist.")
        exit()

    # Load the load_epoch_num data or initialize it if the file doesn't exist
    load_epoch_num_path = os.path.join(data_path, "epoch_num_path.txt")

    if not os.path.exists(load_epoch_num_path):
        load_epoch_num_all = {model_type: 0}
    else:
        with open(load_epoch_num_path, "r") as file:
            load_epoch_num_all = json.load(file)
            if model_type not in load_epoch_num_all.keys():
                load_epoch_num_all[model_type] = 0
    load_epoch_num = load_epoch_num_all[model_type]

    # Save the updated load_epoch_num_all data
    with open(load_epoch_num_path, "w") as file:
        json.dump(load_epoch_num_all, file)

    load_model_path = os.path.join(
        data_path, 'checkpoint', model_type,
        f'{model_type}.epoch{load_epoch_num}.pth.tar')

    return class_weights, in_channels, out_channels, load_epoch_num, load_model_path


def output(MAP_PATH, out_channels, part_possibility, model_data, model_type,
           zoom_factors):

    folder_path = os.path.dirname(MAP_PATH)
    map_name = os.path.basename(MAP_PATH)

    print(f"\n==> Start map writing of {model_type}: ")
    for channel_num in range(1, out_channels):

        possibility = part_possibility[channel_num, :, :, :]
        # out_map_file = f"{map_name.split('.mrc')[0]}_{model_type}_{channel_num}_possibility.mrc"
        # out_map = os.path.join(folder_path, "possibility", out_map_file)

        out_map_file = f"{map_name.split('.mrc')[0]}_{EX_OR_CO}_{model_type}_{channel_num}_possibility.mrc"
        out_map = os.path.join(folder_path, out_map_file)

        map_writing(MAP_PATH, possibility, out_map, zoom_factors)

        # label_data = np.where(model_data == channel_num, 1, 0)
        # out_map_file = f"{map_name.split('.mrc')[0]}_{model_type}_{channel_num}.mrc"
        # out_map = os.path.join(folder_path, "unrefined", out_map_file)
        # map_writing(MAP_PATH, label_data, out_map, zoom_factors, is_int=True)

        # smooth_loc = smooth_location(label_data, channel_num)
        # out_map_file = f"{map_name.split('.mrc')[0]}_{model_type}_{channel_num}_refined.mrc"
        # out_map = os.path.join(folder_path, "refined", out_map_file)
        # map_writing(MAP_PATH, smooth_loc, out_map, zoom_factors, is_int=True)


def main(MAP_PATH, recl):

    # Create sub-folders
    folder_path = os.path.dirname(MAP_PATH)
    # created_folders = ["possibility", "unrefined"]  #, "refined", "chains", "temp_in_chain"]
    # for created_folder in created_folders:
    #     os.makedirs(os.path.join(folder_path, created_folder), exist_ok=True)

    # Read map file and normalize it
    map_data, zoom_factors = map_normalizing(MAP_PATH, recl)

    for model_index in range(1):

        # Choose 3 U-nets for different model types
        model_type = MODEL_TYPES[model_index]

        # Initialize parameters
        class_weights, in_channels, out_channels, load_epoch_num, load_model_path = (
            initial_setting(model_index, data_path, device=DEVICE)
        )
        print(f"\n==> Start {model_type} prediction: ")

        # Load U-net model
        model = UNET(in_channels, out_channels).to(DEVICE)
        load_checkpoint(load_model_path, model, map_location=DEVICE)

        # Predict atom group's location
        if model_index == 0:
            map_data_ = np.expand_dims(map_data, axis=0)

        part_possibility = predict_location(map_data_, model, out_channels, DEVICE)
        model_data = np.argmax(part_possibility, 0)

        # Save the results
        output(
            MAP_PATH,
            out_channels,
            part_possibility,
            model_data,
            model_type,
            zoom_factors,
        )

        # Input map_data for next prediction
        map_data_ = np.concatenate((map_data_, part_possibility), axis=0)


def read_csv_info(csv_path, raw_dir):
    df = read_csv(csv_path)
    emdbs, pdbs = df["emdb_id"], df["fitted_pdbs"]
    resolutions = df["resolution"].astype(str)
    recls = df["recommended_contour_level"].astype(float)
    # print(emdbs)
    emdb_ids = [emdb.split("-")[1] for emdb in emdbs]
    folders = [
        f"{emdb}_re_{resolution}"
        for emdb, resolution in zip(emdbs, resolutions)
    ]
    raw_maps = [f"emd_{emdb_id}.map" for emdb_id in emdb_ids]
    models = [f"{pdb}.cif" for pdb in pdbs]

    raw_map_paths = [os.path.join(raw_dir, folder, raw_map) for folder, raw_map in zip(folders, raw_maps)]
    model_paths = [os.path.join(raw_dir, folder, model) for folder, model in zip(folders, models)]
    normalized_map_paths = [os.path.join(raw_dir, folder, f"{os.path.splitext(os.path.basename(raw_map_path))[0]}_normalized.mrc")\
                             for folder, raw_map_path in zip(folders, raw_map_paths)]

    csv_info = (emdb_ids, pdbs, resolutions, folders, raw_maps, models, recls)
    path_info = (raw_map_paths, model_paths, normalized_map_paths)
    return csv_info, path_info



if __name__ == "__main__":
    csv_path = '/home/qiboxu/MyProject/test/test_dataset_metadata.csv'
    RAW_DATA_PATH = '/home/qiboxu/Database/CryoDataBot_Data/Test-cryodatabot'
    csv_info, path_info = read_csv_info(csv_path, RAW_DATA_PATH)
    (raw_map_paths, model_paths, normalized_map_paths) = path_info
    emdb_ids, pdbs, resolutions, folders, raw_maps, models, recls = csv_info


    for i in range(len(raw_map_paths)):
        main(raw_map_paths[i], recls[i])
