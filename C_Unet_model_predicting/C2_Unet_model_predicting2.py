import json
import os
import pickle

import numpy as np

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

import pandas as pd


# ====================== Hyperparameters etc. ========================
MAIN_PATH = (
    "/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/Filtered_Dateset/Training/TEST_with_BLANK"
)
MODEL_TYPES = ["secondary_strctures", "key_atoms", "residue_types"]
MAP_DIRS_TYPES = [
    ["map_sample"],
    ["map_sample", "secondary_strctures_pred_prob"],
    ["map_sample", "secondary_strctures_pred_prob", "key_atoms_pred_prob"],
]
# IN_CHANNELS = [1, 6, 10]

data_path = os.path.join(MAIN_PATH, "train_val_data")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
    weight_path = os.path.join(data_path, "class_weight_for_training.txt")
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
                    in_channels = in_channels + len(weight_all[MODEL_TYPES[ii]])

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
        data_path,
        "checkpoint",
        model_type,
        f"{model_type}.epoch{load_epoch_num}.pth.tar",
    )

    return class_weights, in_channels, out_channels, load_epoch_num, load_model_path


def output(
    MAP_PATH, out_channels, part_possibility, model_data, model_type, zoom_factors
):

    folder_path = os.path.dirname(MAP_PATH)
    map_name = os.path.basename(MAP_PATH)

    for channel_num in range(1, out_channels):
        print(f"\n==> Start map writing of {model_type}: ")

        possibility = part_possibility[channel_num, :, :, :]
        out_map_file = (
            f"{map_name.split('.mrc')[0]}_{model_type}_{channel_num}_possibility.mrc"
        )
        out_map = os.path.join(folder_path, "possibility", out_map_file)
        map_writing(MAP_PATH, possibility, out_map, zoom_factors)

        label_data = np.where(model_data == channel_num, 1, 0)
        out_map_file = f"{map_name.split('.mrc')[0]}_{model_type}_{channel_num}.mrc"
        out_map = os.path.join(folder_path, "unrefined", out_map_file)
        map_writing(MAP_PATH, label_data, out_map, zoom_factors, is_int=True)

        # smooth_loc = smooth_location(label_data, channel_num)
        # out_map_file = f"{map_name.split('.mrc')[0]}_{model_type}_{channel_num}_refined.mrc"
        # out_map = os.path.join(folder_path, "refined", out_map_file)
        # map_writing(MAP_PATH, smooth_loc, out_map, zoom_factors, is_int=True)


def main(MAP_PATH):

    # Create sub-folders
    folder_path = os.path.dirname(MAP_PATH)
    created_folders = ["possibility", "unrefined", "refined", "chains", "temp_in_chain"]
    for created_folder in created_folders:
        os.makedirs(os.path.join(folder_path, created_folder), exist_ok=True)

    # Read map file and normalize it
    map_data, zoom_factors = map_normalizing(MAP_PATH)

    for model_index in range(3):

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


#### stop here 3/11/2024

#### Following are old code in 2023

# for idx, part_name in enumerate(MODEL_PARTS):
#     # # Step1: Load U-net model
#     # # Load model settings
#     # load_epoch_num = load_epoch_num_all[part_name]
#     # load_model_path = os.path.join(
#     #     UNET_PATH, 'TEST', 'checkpoint', part_name,
#     #     f'{part_name}.epoch{load_epoch_num}.pth.tar')
#     # out_channels = len(weight_all[part_name])
#     # DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
#     # in_channels = 1

#     # # Load model
#     # model = UNET(in_channels, out_channels).to(DEVICE)
#     # load_checkpoint(load_model_path, model, map_location=DEVICE)

#     # # Step2: Read map file and normalize it
#     # map_data = map_normalizing(MAP_PATH)

#     # # Step3: Predict atom group's location using U-Nets (64*64*64 per sample cube)
#     # part_possibility = predict_location(map_data, model, out_channels,
#     #                                     DEVICE)
#     # model_data = np.argmax(part_possibility, 0)

#     # Step4: Save the results
#     if part_name == "RNA_backbone":
#         pos_tag = []
#         ModelData, longest_segments, segments_data = identify_chain(
#             model_data, pos_tag=pos_tag)
#         # out_map = MAP_PATH.split('.mrc')[0]+'_segments.mrc'
#         # map_writing(MAP_PATH, segments_data, out_map)
#         out_map = MAP_PATH.split('.mrc')[0] + '_segments_refine.mrc'
#         map_writing(MAP_PATH, specify_location(segments_data, 1), out_map)

#     for channel_num in range(1, out_channels):
#         out_map = f"{MAP_PATH.split('.mrc')[0]}_{part_name}_{channel_num}_possibility.mrc"
#         write_data = part_possibility[channel_num, :, :, :]
#         map_writing(MAP_PATH, write_data, out_map)

#         out_map = f"{MAP_PATH.split('.mrc')[0]}_{part_name}_{channel_num}.mrc"
#         write_data = np.where(model_data == channel_num, channel_num, 0)
#         map_writing(MAP_PATH, write_data, out_map, is_int=True)

#         out_map = f"{MAP_PATH.split('.mrc')[0]}_{part_name}_{channel_num}_refine.mrc"
#         map_writing(MAP_PATH,
#                     specify_location(write_data, channel_num),
#                     out_map,
#                     is_int=True)

#     # else:
#     #     out_map = MAP_PATH.split('.mrc')[0]+"_"+part_name+'.mrc'
#     #     write_data = model_data
#     #     map_writing(MAP_PATH, write_data, out_map)


if __name__ == "__main__":
    RAW_DATA_PATH = (
        "/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/Filtered_Dateset/Raw"
    )
    MAP_PATH = os.path.join(RAW_DATA_PATH, "EMD-0065_re_3.27", "emd_0065.map")
    # for MAP_PATH in raw_map_paths:
    main(MAP_PATH)
