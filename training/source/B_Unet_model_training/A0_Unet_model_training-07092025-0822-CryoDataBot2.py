from B0_MODEL import UNET
from B2_UTILS import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
    print_and_write,
)
import torch
from tqdm import tqdm  # makes your loops show a smart progress meter
import torch.nn as nn
import torch.optim as optim
import os
import json
import numpy as np

import splitfolders
import shutil

# Test train? Train? Predict?
BATCH_1_TEST = 0 # If True, do a batch 1 test to check if the U-Net model works.
TRAIN = 1 # If True, train the U-Net model.
MODEL_INDEX = 0  # 0: secondary_strctures; 1: key_atoms; 2: residue_types
PREDICT = 0 # If True, predict the U-Net model and save the results.

# Set training epoch numbers, early stopping epoch num
training_epoch_num = 150


early_stopping_patience = 10

# batch_size (vs GPU Memory: 1vs2877; 2vs3485; 4vs4668; 8vs7162)
batch_size = 8 #16
learning_rate = 1e-4

# ====================== Hyperparameters etc. ========================
# MAIN_PATH = "/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/Filtered_Dateset/Training/TEST_with_blank"
# MAIN_PATH = "/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/Filtered_Dateset/Training/TEST_with_all"
# MAIN_PATH = "/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/Filtered_Dateset/Training/TEST_20240520"
# MAIN_PATH = "/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/Filtered_Dateset/Training/cryoID2_metadata_Final-VOF-filter"
# MAIN_PATH = '/home/qiboxu/Database/CryoDataBot_Data/Training/train_experiment_GPCR_38_entries'
# MAIN_PATH = '/home/qiboxu/Database/CryoDataBot_Data/Training/train_raw_GPCR_126_entries'
# MAIN_PATH = '/home/qiboxu/Database/CryoDataBot_Data/Training/train_Q0.5_GPCR_16_entries'
MAIN_PATH = '/home/qiboxu/Database/CryoDataBot_Data/Training/train_Q0.3_GPCR_48_entries'




# MODEL_TYPES = ["secondary_strctures", "key_atoms", "residue_types"]
MODEL_TYPES = ["CA_atoms"]
# MAP_DIRS_TYPES = [
#     ["map_sample"],
#     ["map_sample", "secondary_strctures_pred_prob"],
#     ["map_sample", "secondary_strctures_pred_prob", "key_atoms_pred_prob"],
# ]
MAP_DIRS_TYPES = [["map_sample"]]
# IN_CHANNELS = [1, 6, 10]

data_path = MAIN_PATH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

num_workers = 8 #12 #16
pin_memory = True


def initial_setting(model_index, data_path, map_dirs, device):
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

    # Set file paths and make some directories when not existed
    train_map_dirs = [os.path.join(data_path, "train", map_dir) for map_dir in map_dirs]
    val_map_dirs = [os.path.join(data_path, "val", map_dir) for map_dir in map_dirs]
    train_model_dir = os.path.join(data_path, "train", model_type)
    val_model_dir = os.path.join(data_path, "val", model_type)

    load_model_path = os.path.join(
        data_path,
        "checkpoint",
        model_type,
        f"{model_type}.epoch{load_epoch_num}.pth.tar",
    )

    evl_folder_path = os.path.join(data_path, "evaluation", model_type)

    os.makedirs(os.path.dirname(load_model_path), exist_ok=True)
    os.makedirs(evl_folder_path, exist_ok=True)

    return (
        class_weights,
        in_channels,
        out_channels,
        load_epoch_num,
        train_map_dirs,
        val_map_dirs,
        train_model_dir,
        val_model_dir,
        load_model_path,
        evl_folder_path,
    )


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data, targets = data.to(device=DEVICE), targets.long().to(device=DEVICE)
        print(batch_idx)

        # forward
        with torch.amp.autocast('cuda'):
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            total_loss += loss.item()

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
    average_loss = total_loss / (batch_idx + 1)

    return round(average_loss, 4)


def batch_1_test(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    data, targets = next(iter(loader))
    data, targets = data.to(device=DEVICE), targets.long().to(device=DEVICE)

    for epoch in range(500):
        # forward
        with torch.amp.autocast('cuda'):
            predictions = model(data)
            # print(predictions.shape)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def predict(model, pre_map_dirs, model_index):

    model_type = MODEL_TYPES[model_index]
    temp_sample_path = os.path.join(os.path.dirname(pre_map_dirs[0]), f'{model_type}_pred_prob')
    os.makedirs(temp_sample_path, exist_ok=True)

    map_names = [sorted(os.listdir(map_dir)) for map_dir in pre_map_dirs]

    model.eval()
    with torch.no_grad():

        count = 0

        for index in range(len(map_names[0])):
            count += 1
            print(f'{count}/{len(map_names[0])}', end='\r', flush=True)
            for i in range(len(pre_map_dirs)):
                map_dir = pre_map_dirs[i]
                map_name = map_names[i][index]
                map_path = os.path.join(map_dir, map_name)
                map_part = np.load(map_path)

                # concatenate maps in different features
                if i == 0:
                    # add a dimension of feature to original map
                    map = np.expand_dims(map_part, axis=0)
                else:
                    map = np.concatenate((map, map_part), axis=0)
            map = np.expand_dims(map, axis=0)

            # Move map to the specified device and perform prediction
            map = torch.from_numpy(map)
            map = map.to(DEVICE)
            # pred_prob = model(map)
            pred_prob = torch.softmax(model(map), dim=1)
            writen = pred_prob[0, :, :, :, :].cpu().numpy()

            file_name = f"{model_type}_pred_prob.{map_name.split('.')[-2]}.npy"
            file_path = os.path.join(temp_sample_path, file_name)
            np.save(file_path, writen)

    model.train()


def main():

    for model_index, model_type in enumerate(MODEL_TYPES):
        if model_index != MODEL_INDEX:
            continue
        map_dirs = MAP_DIRS_TYPES[model_index]

        (
            class_weights,
            in_channels,
            out_channels,
            load_epoch_num,
            train_map_dirs,
            val_map_dirs,
            train_model_dir,
            val_model_dir,
            load_model_path,
            evl_folder_path,
        ) = initial_setting(model_index, data_path, map_dirs, DEVICE)

        model = UNET(in_channels, out_channels).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=0.5,
                    patience=5,
                )
        scaler = torch.amp.GradScaler()

        # Generate data loaders
        train_loader, val_loader = get_loaders(
            train_map_dirs,
            train_model_dir,
            val_map_dirs,
            val_model_dir,
            batch_size,
            num_workers,
            pin_memory,
        )

        # Do batch 1 test to evaluate the U-Net model first
        if BATCH_1_TEST:
            batch_1_test(train_loader, model, optimizer, loss_fn, scaler)
            exit()

        if TRAIN:
            print(f"Start training for {model_type} U-net:")

            # Early stopping initialization
            best_loss, no_improvement_count = 10., 0
            # best_loss, no_improvement_count = 0.1757, 0

            if not load_epoch_num == 0:
                # Load previous check point
                load_checkpoint(load_model_path, model, map_location=DEVICE)

            # # Check accuracy
            # check_accuracy(val_loader, model, out_channels, loss_fn,
            #                 evl_folder_path, DEVICE, load_epoch_num, class_weights)
            # save_predictions_as_imgs(val_loader, model, evl_folder_path,
            #                             DEVICE, load_epoch_num, model_type, batch_size)

            # exit()

            previous_lr = optimizer.param_groups[0]['lr']

            for epoch in range(
                load_epoch_num + 1, load_epoch_num + 1 + training_epoch_num
            ):
                # Train U-net model and get the average loss
                average_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
                print_and_write(
                    f"{evl_folder_path}/check_accuracy.log",
                    f"EPOCH {epoch} of {model_type}: \nAverage Loss = {average_loss}.",
                )

                # Check accuracy
                val_average_loss = check_accuracy(val_loader, model, out_channels, loss_fn,
                               evl_folder_path, DEVICE, epoch, class_weights)
                save_predictions_as_imgs(val_loader, model, evl_folder_path,
                                         DEVICE, epoch, model_type, batch_size)

                scheduler.step(val_average_loss)
                current_lr = optimizer.param_groups[0]['lr']

                if current_lr != previous_lr:
                    print_and_write(
                        f"{evl_folder_path}/check_accuracy.log",
                        f"=========\nLearning rate changed from {previous_lr:.8f} to {current_lr:.8f} at Epoch {epoch + 1}\n=========\n",
                    )
                    previous_lr = current_lr

                # Check if the current loss is better than the previous best loss
                if val_average_loss < best_loss:
                    best_loss = val_average_loss
                    no_improvement_count = 0

                    # Save model and update the loading epoch number file
                    checkpoint = {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    }
                    save_model_name = f"{model_type}.epoch{epoch}.pth.tar"
                    save_checkpoint(
                        checkpoint, os.path.dirname(load_model_path), save_model_name
                    )

                    # Update the epoch number in file
                    load_epoch_num_path = os.path.join(data_path, "epoch_num_path.txt")

                    with open(load_epoch_num_path, "r") as file:
                        load_epoch_num_all = json.load(file)
                        load_epoch_num_all[model_type] = epoch
                    with open(load_epoch_num_path, "w") as file:
                        json.dump(load_epoch_num_all, file)

                else:
                    no_improvement_count += 1

                # Check if early stopping criteria are met
                if no_improvement_count >= early_stopping_patience:
                    print(
                        "Early stopping! No improvement for",
                        early_stopping_patience,
                        "epochs.",
                    )
                    break

        if PREDICT:
            load_checkpoint(load_model_path, model, map_location=DEVICE)
            predict(model, train_map_dirs, model_index)
            predict(model, val_map_dirs, model_index)


if __name__ == "__main__":
    main()
