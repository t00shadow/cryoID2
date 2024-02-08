from B0_MODEL import UNET
from B2_UTILS import (load_checkpoint, save_checkpoint, get_loaders,
                      check_accuracy, save_predictions_as_imgs,
                      print_and_write)
import torch
from tqdm import tqdm  # makes your loops show a smart progress meter
import torch.nn as nn
import torch.optim as optim
import os, json

i = 0
# MMOODDEELL = ["phosphorus_C1", "sugar_ring_new", "nitrogenous_bases_in_sugar_new", "RNA_backbone", "sugar_center", ]
MMOODDEELL = [
    "sugar_center",
]

MODEL_PARTS = [MMOODDEELL[i]]
DEVICE = f"cuda:{i}" if torch.cuda.is_available() else 'cpu'

# Path of files
PATH_KEYS = "training1"
HOME_PATH = '/scratch/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/training/training1'

# Hyperparameters

TRAIN = True  # Do new traning or not

BATCH_1_TEST = False
early_stopping_patience = 5
num_epochs = 1000  # Early stopping is used
batch_size = 8  # batch_size vs GPU Memory: 1vs2877; 2vs3485; 4vs4668; 8vs7162
num_workers = 2
learning_rate = 1e-4
in_channels = 1
pin_memory = True


def initial_setting(part_name, home_path, device):
    """
    This function performs the initial setup before training process:
    1. Read the class weights of the UNET model,
    2. Read the saved epoch number,
    3. Generate paths of file that will be loaded to UNET model.
    """

    weight_path = os.path.join(home_path, "TEST", "train_val_data",
                               "class_weight_for_training.txt")

    try:
        # Load the weight for training
        with open(weight_path, "r") as file:
            weight_all = json.load(file)

        if part_name not in weight_all.keys():
            print(f"Weight for {part_name} is not found.")
            exit()
        else:
            c_w = weight_all[part_name]
            class_weights = torch.FloatTensor(c_w).to(device)

            # Calculate the number of output channels based on the number of tags in part_name
            out_channels = len(c_w)

    except FileNotFoundError:
        print("Weight file does not exist.")
        exit()

    load_epoch_num_path = os.path.join(home_path, "TEST", "train_val_data",
                                       "epoch_num_path.txt")

    # Load the load_epoch_num data or initialize it if the file doesn't exist
    if not os.path.exists(load_epoch_num_path):
        load_epoch_num_all = {part_name: 0}
    else:
        with open(load_epoch_num_path, "r") as file:
            load_epoch_num_all = json.load(file)
            if part_name not in load_epoch_num_all.keys():
                load_epoch_num_all[part_name] = 0

    load_epoch_num = load_epoch_num_all[part_name]

    # Save the updated load_epoch_num_all data
    with open(load_epoch_num_path, "w") as file:
        json.dump(load_epoch_num_all, file)

    # Set file paths and make some directories when not exist
    train_map_dir = os.path.join(HOME_PATH, 'TEST', 'train_val_data', 'train',
                                 'map_sample')
    val_map_dir = os.path.join(HOME_PATH, 'TEST', 'train_val_data', 'val',
                               'map_sample')
    train_model_dir = os.path.join(HOME_PATH, 'TEST', 'train_val_data',
                                   'train', part_name)
    val_model_dir = os.path.join(HOME_PATH, 'TEST', 'train_val_data', 'val',
                                 part_name)

    load_model_path = os.path.join(
        HOME_PATH, 'TEST', 'checkpoint', part_name,
        f'{part_name}.epoch{load_epoch_num}.pth.tar')
    evl_folder_path = os.path.join(HOME_PATH, 'TEST', 'evaluation', part_name)
    os.makedirs(os.path.dirname(load_model_path), exist_ok=True)
    os.makedirs(evl_folder_path, exist_ok=True)

    return class_weights, out_channels, load_epoch_num, train_map_dir, val_map_dir, train_model_dir, val_model_dir, load_model_path, evl_folder_path


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    total_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data, targets = data.to(device=DEVICE), targets.long().to(
            device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
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
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():

    for part_name in MODEL_PARTS:

        class_weights, out_channels, load_epoch_num, \
            train_map_dir, val_map_dir, train_model_dir, val_model_dir, \
                load_model_path, evl_folder_path \
                    = initial_setting(part_name, HOME_PATH, DEVICE)

        model = UNET(in_channels, out_channels).to(DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scaler = torch.cuda.amp.GradScaler()

        # Generate data loaders
        train_loader, val_loader = get_loaders(part_name, train_map_dir,
                                               train_model_dir, val_map_dir,
                                               val_model_dir, batch_size,
                                               num_workers, pin_memory)

        # Do batch 1 test to evaluate the U-Net model first
        if BATCH_1_TEST:
            batch_1_test(train_loader, model, optimizer, loss_fn, scaler)
            exit()

        if TRAIN:

            # Early stopping settings
            best_loss, no_improvement_count = 10, 0

            if not load_epoch_num == 0:
                # Load previous check point
                load_checkpoint(load_model_path, model, map_location=DEVICE)

            for epoch in range(load_epoch_num + 1,
                               load_epoch_num + 1 + num_epochs):
                # Train U-net model and get the average loss
                average_loss = train_fn(train_loader, model, optimizer,
                                        loss_fn, scaler)
                print_and_write(
                    f"{evl_folder_path}/check_accuracy.log",
                    f"EPOCH {epoch} of {part_name}: Average Loss = {average_loss}.\n"
                )
                # with open(f"{evl_folder_path}/check_accuracy.log", "a") as file:
                #     print("EPOCH:", epoch, "Average Validation Loss:", average_loss)
                #     file.write(f"EPOCH: {epoch}, Average Validation Loss: {average_loss}\n\n")

                # Check if the current loss is better than the previous best loss
                if average_loss < best_loss:
                    best_loss = average_loss
                    no_improvement_count = 0

                    # Save model and update the loading epoch number file
                    checkpoint = {
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict()
                    }
                    save_model_name = f"{part_name}.epoch{epoch}.pth.tar"
                    save_checkpoint(checkpoint,
                                    os.path.dirname(load_model_path),
                                    save_model_name)

                    # Update the epoch number in file
                    load_epoch_num_path = os.path.join(HOME_PATH, "TEST",
                                                       "train_val_data",
                                                       "epoch_num_path.txt")
                    with open(load_epoch_num_path, "r") as file:
                        load_epoch_num_all = json.load(file)
                        load_epoch_num_all[part_name] = epoch
                    with open(load_epoch_num_path, "w") as file:
                        json.dump(load_epoch_num_all, file)

                else:
                    no_improvement_count += 1

                # Check if early stopping criteria are met
                if no_improvement_count >= early_stopping_patience:
                    print("Early stopping! No improvement for",
                          early_stopping_patience, "epochs.")
                    break

                # Check accuracy
                average_loss_test = check_accuracy(val_loader, model,
                                                   out_channels, loss_fn,
                                                   evl_folder_path, DEVICE,
                                                   epoch)
                save_predictions_as_imgs(val_loader, model, evl_folder_path,
                                         DEVICE, epoch, part_name, batch_size)

        # else:
        #     device_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        #     check_accuracy(val_loader, model, out_channels, evl_folder_path, DEVICE)
        #     # print some examples to a folder
        #     save_predictions_as_imgs(
        #         val_loader, model, folder=evl_folder_path, device=device_
        #     )


if __name__ == "__main__":
    main()
