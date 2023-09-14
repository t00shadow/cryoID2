import torch
from B1_DATASET import MapModelDataset
from torch.utils.data import DataLoader
import numpy as np
import os

def save_checkpoint(state, folder, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, os.path.join(folder, filename))
    print(f"Checkpoint {filename} is saved.")


def load_checkpoint(load_model_path, model, map_location):
    print(f"=> Loading checkpoint: {os.path.basename(load_model_path)}")
    checkpoint = torch.load(load_model_path, map_location)
    model.load_state_dict(checkpoint["state_dict"])
    print("Checkpoint is loaded.")


def get_loaders(
    model_type,
    train_map_dir,
    train_model_dir,
    val_map_dir,
    val_model_dir,
    batch_size,
    num_workers=4,
    pin_memory=True,
):
    train_ds = MapModelDataset(map_dir=train_map_dir, model_dir=train_model_dir, model_type=model_type)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = MapModelDataset(map_dir=val_map_dir, model_dir=val_model_dir, model_type=model_type)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader



# # ============ TEST ====================
# MODEL_TYPE = "all_atoms"
# TRAIN_MAP_DIR = "/amber/electron/qiboxu/Database/U_NET/TEST/train_val_data/train/map_sample/"
# TRAIN_MODEL_DIR = "/amber/electron/qiboxu/Database/U_NET/TEST/train_val_data/train/"+MODEL_TYPE+"/"
# VAL_MAP_DIR = "/amber/electron/qiboxu/Database/U_NET/TEST/train_val_data/val/map_sample/"
# VAL_MODEL_DIR = "/amber/electron/qiboxu/Database/U_NET/TEST/train_val_data/val/"+MODEL_TYPE+"/"
# NUM_EPOCHS = 3
# NUM_WORKERS = 4
# OUT_CHANNELS = 1
# BATCH_SIZE = 32
# PIN_MEMORY = True
# train_loader, val_loader = get_loaders(
#     model_type=MODEL_TYPE,
#     train_map_dir=TRAIN_MAP_DIR,
#     train_model_dir=TRAIN_MODEL_DIR, 
#     val_map_dir=VAL_MAP_DIR,
#     val_model_dir=VAL_MODEL_DIR,
#     batch_size=BATCH_SIZE,
#     num_workers=NUM_WORKERS,
#     pin_memory=PIN_MEMORY
# )
# train_features, train_labels = next(iter(train_loader))
# print(train_features.size())

def print_and_write(filepath, content):
    with open(filepath, "a") as file:
        print(content)
        file.write(f"{content}\n")



def check_accuracy(loader, model, num_class, loss_fn, evl_folder_path, device, epoch):
    print("=> Checking accuracy")
    num_correct = 0
    num_voxels = 0
    dice_score = 0 
    y_pred_matrix = np.zeros((num_class+1, num_class+1))

    model.eval()

    with torch.no_grad():
        total_loss = 0

        for x, y in loader:
            x, y = x.to(device), y.long().to(device)
            y_pred = model(x)

            preds = torch.argmax(torch.softmax(y_pred, 1),1)
            num_correct += (preds == y).sum()
            num_voxels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / ((preds + y).sum() + 1e-8) 

            for ii in range(num_class):
                pp = torch.where(y==ii, preds, 99)
                # print(ii, pp)
                for jj in range(num_class):
                    qq = torch.where(pp==jj, 1, 0)
                    y_pred_matrix[ii, jj] += qq.sum()
            
            total_loss += loss_fn(y_pred, y).item()
        average_loss = total_loss / len(loader)

    filepath = f"{evl_folder_path}/check_accuracy.log"
    print_and_write(filepath, f"EPOCH: {epoch}, Average Validation Loss: {round(average_loss, 4)}\n")

    for ii in range(num_class): 
        y_pred_matrix[ii, num_class] = y_pred_matrix[ii, :].sum()
    for jj in range(num_class):
        y_pred_matrix[num_class, jj] = y_pred_matrix[:, jj].sum()
    print_and_write(filepath, f"Got {num_correct}/{num_voxels} with acc {num_correct/num_voxels*100:.2f}")
    print_and_write(filepath, f"Dice score: {dice_score/len(loader)}")

    bb = np.expand_dims(y_pred_matrix[:,num_class], 1)
    cc = np.expand_dims(y_pred_matrix[num_class,:], 0)
    num_pred_y = y_pred_matrix/bb
    num_y_pred = y_pred_matrix/cc
    np.set_printoptions(linewidth=400, precision=2, suppress=True)
    print_and_write(filepath, f"Got num_pred/y:\n{num_pred_y}")
    print_and_write(filepath, f"Got num_y/pred:\n{num_y_pred}")

    np.set_printoptions(suppress=True)
    print_and_write(filepath, f"Got y_pred_matrix:\n{y_pred_matrix}\n")




    # with open(f"{evl_folder_path}/check_accuracy.log", "a") as file:

    #     print("EPOCH:", epoch, "Average Validation Loss:", average_loss)
    #     file.write(f"EPOCH: {epoch}, Average Validation Loss: {average_loss}\n\n")

    
    # print(f"Got {num_correct}/{num_voxels} with acc {num_correct/num_voxels*100:.2f}")
    # print(f"Dice score: {dice_score/len(loader)}")    
    # file.write(f"Got {num_correct}/{num_voxels} with acc {num_correct/num_voxels*100:.2f}\n")
    # file.write(f"Dice score: {dice_score/len(loader)}\n")


    # print("Got num_pred/y:\n", num_pred_y)
    # print("Got num_y/pred:\n", num_y_pred)
    # file.write(f"Got num_pred/y:\n{num_pred_y}\n")
    # file.write(f"Got num_y/pred:\n{num_y_pred}\n")

    
    # print("Got y_pred_matrix:\n", y_pred_matrix)
    # file.write(f"Got y_pred_matrix:\n{y_pred_matrix}\n\n\n")
        
    model.train()

    return average_loss




# import glob 
# import numpy as np
# num_y = np.zeros(21)
# train_mask_list = sorted(glob.glob(TRAIN_MODEL_DIR+"*.npy")) 
# for model in range(len(train_mask_list)):
#     print(f"{model}/{len(train_mask_list)}")
#     temp_model = np.load(train_mask_list[model])
#     for ii in range(21):
#         yy = np.where(temp_model==ii, 1, 0)
#         num_y[ii] += yy.sum()
# total_y = num_y.sum()
# print(num_y)
# Weights = np.divide(total_y, num_y)
# print(Weights)
# exit()


import matplotlib.pyplot as plt

def save_predictions_as_imgs(loader, model, folder, device, epoch, part_name, batch_size):

    model.eval()
    # for idx, (x, y) in enumerate(loader):
    x, y = next(iter(loader))
    x = x.to(device=device)
    with torch.no_grad():
        preds = torch.argmax(torch.softmax(model(x), 1),1)
        plt.figure(figsize=(12, 12))
        # print(x.size(), y.size(), preds.size())
        xx = x.cpu().numpy()
        yy = y.cpu().numpy()
        predspreds = preds.cpu().numpy()

        for jj in range(batch_size):
            for ii in range(18, 58, 18):
                pos = ii//18
                plt.subplot(3, 3, 3*pos-2, label=f'subplot{ii+jj}')
                plt.imshow(xx[jj,0,ii,:,:], cmap='gray', origin="lower")
                plt.title(f'map (z={ii})')

                plt.subplot(3, 3, 3*pos-1, label=f'subplot{ii+jj}')
                plt.imshow(yy[jj,ii,:,:], origin="lower")
                plt.title(f'model (z={ii})')

                plt.subplot(3, 3, 3*pos, label=f'subplot{ii+jj}')
                plt.imshow(predspreds[jj,ii,:,:], origin="lower")
                plt.title(f'model_pred (z={ii})')

            plt.suptitle(f"Epoch {epoch} for {part_name}")
            map_name = f"eval.{jj}.epoch{epoch}.png"
            plt.savefig(os.path.join(folder, map_name))
            # plt.close('all')

    model.train()
