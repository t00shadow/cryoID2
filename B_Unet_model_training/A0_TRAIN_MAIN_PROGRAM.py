from B0_MODEL import UNET
from B2_UTILS import (load_checkpoint, save_checkpoint, get_loaders,
                      check_accuracy, save_predictions_as_imgs)
import torch
from tqdm import tqdm  # makes your loops show a smart progress meter
import torch.nn as nn
import torch.optim as optim
import os
import json

# Set training epoch munbers
TRAIN = True
training_epoch_num = 1000

# Test training?
BATCH_1_TEST = True

# ====================== Hyperparameters etc. ========================
MAIN_PATH = '/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/Filtered_Dateset/Training/TEST'
MODEL_TYPES = ['secondary_strctures', 'key_atoms', 'residue_types']
MAP_DIRS_TYPES = [['map_sample'], ['map_sample', 'secondary_strctures'],
                  ['map_sample', 'secondary_strctures', 'key_atoms']]

type_index = 0
model_type = MODEL_TYPES[type_index]
map_dirs = MAP_DIRS_TYPES[type_index]

# ====================== Hyperparameters etc. ========================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8  # batch_size vs GPU Memory: 1vs2877; 2vs3485; 4vs4668; 8vs7162

num_workers = 2
learning_rate = 1e-4
pin_memory = True
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH = [64, 64, 64]

# Set input data
data_path = os.path.join(MAIN_PATH, 'train_val_data')
train_map_dirs = [
    os.path.join(data_path, 'train', map_dir) for map_dir in map_dirs
]
val_map_dirs = [
    os.path.join(data_path, 'val', map_dir) for map_dir in map_dirs
]
train_model_dir = os.path.join(data_path, 'train', model_type)
val_model_dir = os.path.join(data_path, 'val', model_type)

# Set for loading model weights
LOAD_MODEL = False
load_epoch_num = 0
load_model_dir = os.path.join(data_path, 'checkpoint')
load_model_name = os.path.join(data_path, 'checkpoint',
                               f'{model_type}.epoch{load_epoch_num}.pth.tar')
evl_folder_name = os.path.join(data_path, 'evaluation', model_type)

# Load the class weights of all model types
weight_path = os.path.join(data_path, 'class_weight_for_training.txt')
with open(weight_path, "r") as file:
    weight_all = json.load(file)
class_weight = weight_all[model_type]
CLASS_WEIGHTS = torch.FloatTensor(class_weight).to(device=DEVICE)

# Set model channels
in_channels = len(map_dirs)
out_channels = len(class_weight)

# import pandas as pd
# import glob
# import numpy as np
# columns = ['0', '1', '2', '3']
# df = pd.DataFrame(columns=columns)

# train_mask_list = sorted(glob.glob(TRAIN_MODEL_DIR+"*.npy"))
# for model in range(len(train_mask_list)):
#     print(model)
#     temp_model = np.load(train_mask_list[model])
#     temp_model = np.argmax(temp_model, axis=3)
#     val, counts = np.unique(temp_model, return_counts=True)
#     zipped = zip(columns, counts)
#     conts_dict = dict(zipped)
#     df = df.append(conts_dict, ignore_index=True)

# label_0 = df['0'].sum()
# label_1 = df['1'].sum()
# label_2 = df['2'].sum()
# label_3 = df['3'].sum()
# total_labels = label_0+label_1+label_2+label_3
# n_classes = 4

# print(label_0, label_1, label_2, label_3)

# exit()


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    if BATCH_1_TEST:
        data, targets = next(iter(loader))
        for epoch in range(1000):
            data = data.to(device=DEVICE)
            targets = targets.long().to(device=DEVICE)
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

    else:
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.long().to(device=DEVICE)
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
    model = UNET(in_channels=in_channels, out_channels=out_channels).to(DEVICE)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader, val_loader = get_loaders(train_map_dirs=train_map_dirs,
                                           train_model_dir=train_model_dir,
                                           val_map_dirs=val_map_dirs,
                                           val_model_dir=val_model_dir,
                                           batch_size=batch_size,
                                           num_workers=num_workers,
                                           pin_memory=pin_memory)

    if LOAD_MODEL:
        load_checkpoint(torch.load(load_model_name, map_location=DEVICE),
                        model)

    if TRAIN:
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(load_epoch_num + 1,
                           load_epoch_num + 1 + training_epoch_num):
            train_fn(train_loader, model, optimizer, loss_fn, scaler)

            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_model_name = f'{model_type}.epoch{epoch}.pth.tar"'
            save_checkpoint(state=checkpoint,
                            folder=load_model_dir,
                            filename=save_model_name)
            # check accuracy
            print("EPOCH" + str(epoch) + ": ")
            check_accuracy(val_loader,
                           model,
                           num_class=out_channels,
                           device=DEVICE)

            # # write to file
            # with open("evaluation/"+MODEL_TYPE+"_0.5A.txt", 'a') as ff:
            #     ff.write("EPOCH="+str(epoch)+":\n")
            #     ff.write("acc:"+str(acc)+":\n")
            #     ff.write("num_pred_y:\n"+str(num_pred_y)+":\n")
            #     ff.write("y_pred_matrix:\n"+str(y_pred_matrix)+":\n")
    else:
        # DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        device_ = 'cuda' if torch.cuda.is_available() else 'cpu'
        check_accuracy(val_loader,
                       model,
                       num_class=out_channels,
                       device=device_)
        # print some examples to a folder
        save_predictions_as_imgs(val_loader,
                                 model,
                                 folder=evl_folder_name,
                                 device=device_)


if __name__ == "__main__":
    main()
