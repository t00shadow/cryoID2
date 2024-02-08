from B0_MODEL import UNET
from B2_UTILS import (load_checkpoint, save_checkpoint, get_loaders,
                      check_accuracy, save_predictions_as_imgs)
import torch
from tqdm import tqdm  # makes your loops show a smart progress meter
import torch.nn as nn
import torch.optim as optim

MODEL_SETTINGS = {
    "phosphorus_C1": [
        70,
        'cuda:1',
        "phosphorus_C1",
        1,
        3,
        [
            1.0,
            315.0,
            315.0,
        ],
    ],
    "sugar_center": [
        60,
        'cuda:2',
        "sugar_center",
        1,
        2,
        [
            1.0,
            316.0,
        ],
    ],
    "nitrogenous_bases_in_sugar_new": [
        83,
        'cuda:1',
        "nitrogenous_bases_in_sugar_new",
        1,
        5,
        [1.0, 344.0, 279.0, 366.0, 419.0],
    ],
    # "nitrogenous_bases_in_sugar": [63, 'cuda:1',
    # "nitrogenous_bases_in_sugar", 1, 5, [1.0, 255.0, 206.0, 270.0, 309.0],],
    "nitrogenous_bases": [
        27,
        'cuda:0',
        "nitrogenous_bases",
        1,
        5,
        [1.0, 200.0, 149.0, 247.0, 282.0],
    ],
    "RNA_backbone": [
        74,
        'cuda:2',
        "RNA_backbone",
        1,
        3,
        [1., 40., 51.],
    ],
    "sugar_ring_new": [
        82,
        'cuda:3',
        "sugar_ring_new",
        1,
        2,
        [1., 85.],
    ],
    # "sugar_ring": [70, 'cuda:3',
    # "sugar_ring", 1, 2, [1.,63.],],
    "all_atoms": [
        21,
        'cuda:2',
        "all_atoms",
        1,
        2,
        [
            1.,
            135.,
        ],
    ],
    # "key_atoms": [4, 'cuda:0',
    # "key_atoms", 1, 4, [, , , ],],
    "key_atoms_0.5A": [
        9,
        ' cuda:0',
        "key_atoms",
        0.5,
        4,
        [1., 2000., 2000., 2000.],
    ],
    "amino_acids_0.5A": [
        29,
        'cuda:0',
        "amino_acids",
        0.5,
        21,
        [
            1.03, 528.67, 445.32, 1157.79, 1093.89, 4696.03, 672.31, 1262.34,
            537.56, 1732.31, 670.01, 531.9, 385.17, 2142.48, 1329.23, 939.09,
            892.0, 793.22, 4664.79, 1395.3, 486.55
        ],
    ],
    # amino_acids = [["ALA"], ["ARG"], ["ASN"], ["ASP"], ["CYS"], ["GLU"], ["GLN"], ["GLY"], ["HIS"], ["ILE"],
    #                 ["LEU"], ["LYS"], ["MET"], ["PHE"], ["PRO"], ["SER"], ["THR"], ["TRP"], ["TYR"], ["VAL"],]
}
MODEL_KEYS = [
    "phosphorus_C1",
    "sugar_center",
    "nitrogenous_bases_in_sugar_new",
    "nitrogenous_bases",
    "RNA_backbone",
    "sugar_ring_new",
    "all_atoms",
    "key_atoms_0.5A",
    "amino_acids_0.5A",
    "sugar_ring",
]

# ====================== Hyperparameters etc. ========================
MODEL_CODE = MODEL_KEYS[0]
TRAIN = False

LOAD_MODEL = True
BATCH_1_TEST = False

HOME_PATH = "/scratch/"
# HOME_PATH = "/amber/electron/"

NUM_EPOCHS = 10
BATCH_SIZE = 8  # batch_size vs GPU Memory: 1vs2877; 2vs3485; 4vs4668; 8vs7162

# ====================== Hyperparameters etc. ========================
LOAD_EPOCH_NUM, DEVICE, MODEL_TYPE, GRID_STRID, OUT_CHANNELS, class_weight = MODEL_SETTINGS[
    MODEL_CODE]
DEVICE = DEVICE if torch.cuda.is_available() else 'cpu'
CLASS_WEIGHTS = torch.FloatTensor(class_weight).to(device=DEVICE)

NUM_WORKERS = 2
LEARNING_RATE = 1e-4
IN_CHANNELS = 1
PIN_MEMORY = True
IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH = [64, 64, 64]
TRAIN_MAP_DIR = HOME_PATH + "qiboxu/Database/U_NET/TEST/train_val_data/train/map_sample/"
VAL_MAP_DIR = HOME_PATH + "qiboxu/Database/U_NET/TEST/train_val_data/val/map_sample/"
TRAIN_MODEL_DIR = HOME_PATH + "qiboxu/Database/U_NET/TEST/train_val_data/train/" + MODEL_TYPE + "/"
VAL_MODEL_DIR = HOME_PATH + "qiboxu/Database/U_NET/TEST/train_val_data/val/" + MODEL_TYPE + "/"
LOAD_MODEL_NAME = "/home/qiboxu/VSCode_code/code_self/3D_U_NET/checkpoint/" + MODEL_TYPE + ".epoch" + str(
    LOAD_EPOCH_NUM) + ".pth.tar"
EVL_FOLD_NAME = "/home/qiboxu/VSCode_code/code_self/3D_U_NET/evaluation/" + MODEL_TYPE + "/"

if GRID_STRID == 0.5:
    TRAIN_MODEL_DIR = HOME_PATH + "qiboxu/Database/U_NET/TEST/train_val_data_0.5A_10samples/train/" + MODEL_TYPE + "/"
    VAL_MODEL_DIR = HOME_PATH + "qiboxu/Database/U_NET/TEST/train_val_data_0.5A_10samples/val/" + MODEL_TYPE + "/"
    LOAD_MODEL_NAME = "/home/qiboxu/VSCode_code/code_self/3D_U_NET/checkpoint/" + MODEL_TYPE + "_0.5A.epoch" + str(
        LOAD_EPOCH_NUM) + ".pth.tar"
    EVL_FOLD_NAME = "/home/qiboxu/VSCode_code/code_self/3D_U_NET/evaluation/" + MODEL_TYPE + "_0.5A/"

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
    model = UNET(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(model_type=MODEL_TYPE,
                                           train_map_dir=TRAIN_MAP_DIR,
                                           train_model_dir=TRAIN_MODEL_DIR,
                                           val_map_dir=VAL_MAP_DIR,
                                           val_model_dir=VAL_MODEL_DIR,
                                           batch_size=BATCH_SIZE,
                                           num_workers=NUM_WORKERS,
                                           pin_memory=PIN_MEMORY)

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_NAME, map_location=DEVICE),
                        model)

    if TRAIN:
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(LOAD_EPOCH_NUM + 1,
                           LOAD_EPOCH_NUM + 1 + NUM_EPOCHS):
            train_fn(train_loader, model, optimizer, loss_fn, scaler)

            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            SAVE_MODEL_NAME = MODEL_TYPE + ".epoch" + str(epoch) + ".pth.tar"
            if GRID_STRID == 0.5:
                SAVE_MODEL_NAME = MODEL_TYPE + "_0.5A.epoch" + str(
                    epoch) + ".pth.tar"
            save_checkpoint(state=checkpoint, filename=SAVE_MODEL_NAME)
            # check accuracy
            print("EPOCH" + str(epoch) + ": ")
            check_accuracy(val_loader,
                           model,
                           num_class=OUT_CHANNELS,
                           device=DEVICE)

            # # write to file
            # with open("evaluation/"+MODEL_TYPE+"_0.5A.txt", 'a') as ff:
            #     ff.write("EPOCH="+str(epoch)+":\n")
            #     ff.write("acc:"+str(acc)+":\n")
            #     ff.write("num_pred_y:\n"+str(num_pred_y)+":\n")
            #     ff.write("y_pred_matrix:\n"+str(y_pred_matrix)+":\n")
    else:
        # DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        device_ = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        check_accuracy(val_loader,
                       model,
                       num_class=OUT_CHANNELS,
                       device=device_)
        # print some examples to a folder
        save_predictions_as_imgs(val_loader,
                                 model,
                                 folder=EVL_FOLD_NAME,
                                 device=device_)


if __name__ == "__main__":
    main()
