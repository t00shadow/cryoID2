import torch
from B1_DATASET import MapModelDataset
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt


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
    train_map_dirs,
    train_model_dir,
    val_map_dirs,
    val_model_dir,
    batch_size,
    num_workers=4,
    pin_memory=True,
):
    train_ds = MapModelDataset(map_dirs=train_map_dirs,
                               model_dir=train_model_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = MapModelDataset(map_dirs=val_map_dirs, model_dir=val_model_dir)

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


# # ============ TEST ====================
# TRAIN_DIR = "/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/Filtered_Dateset/Training/TEST/train_val_data/train"
# train_map_dirs = [
#     os.path.join(TRAIN_DIR, 'map_sample'),
#     os.path.join(TRAIN_DIR, 'secondary_strctures_pred_prob')
# ]
# train_model_dir = os.path.join(TRAIN_DIR, 'key_atoms')

# val_map_dirs = train_map_dirs
# val_model_dir = train_model_dir

# BATCH_SIZE = 8
# NUM_WORKERS = 4
# PIN_MEMORY = True
# train_loader, val_loader = get_loaders(train_map_dirs,
#                                        train_model_dir,
#                                        val_map_dirs,
#                                        val_model_dir,
#                                        batch_size=BATCH_SIZE,
#                                        num_workers=NUM_WORKERS,
#                                        pin_memory=PIN_MEMORY)
# for i in range(100):
#     train_features, train_labels = next(iter(train_loader))
#     print(train_features.size(), train_labels.size())


def print_and_write(filepath, content):
    with open(filepath, "a") as file:
        print(content)
        file.write(f"{content}\n")


def weighted_score(score, weights):
    w = weights[1:]
    w = w.to('cpu')
    a = torch.tensor(score[1:])
    return torch.sum(a * w) / torch.sum(w)


def check_accuracy(loader, model, num_class, loss_fn, evl_folder_path, device,
                   epoch, class_weights):
    print("=> Checking accuracy")
    num_correct = 0
    num_voxels = 0
    # dice_score = 0
    y_pred_matrix = np.zeros((num_class, num_class))

    model.eval()

    with torch.no_grad():
        total_loss = 0

        for x, y in loader:
            x, y = x.to(device), y.long().to(device)
            y_pred = model(x)

            preds = torch.argmax(torch.softmax(y_pred, 1), 1)
            num_correct += (preds == y).sum()
            num_voxels += torch.numel(preds)

            # Calculate the num of (GroundTruth==ii, Prediction==jj), cumulate all
            for ii in range(num_class):
                pp = torch.where(y == ii, preds, 99)
                for jj in range(num_class):
                    qq = torch.where(pp == jj, 1, 0)
                    y_pred_matrix[ii, jj] += qq.sum()

            total_loss += loss_fn(y_pred, y).item()
        average_loss = total_loss / len(loader)

    filepath = f"{evl_folder_path}/check_accuracy.log"
    print_and_write(filepath,
                    f"Average Validation Loss = {round(average_loss, 4)}")
    print_and_write(
        filepath,
        f"Accuracy = {num_correct/num_voxels*100:.2f}, getting {num_correct}/{num_voxels}."
    )

    # Calculate Precision, Recall, F1 score
    Recall, Precision, F1 = [], [], []
    for ii in range(num_class):
        recall = y_pred_matrix[ii][ii] / y_pred_matrix[ii, :].sum()
        Recall.append(round(recall, 4))
    for jj in range(num_class):
        precision = y_pred_matrix[jj][jj] / y_pred_matrix[:, jj].sum()
        Precision.append(round(precision, 4))
    for i in range(len(Recall)):
        f1 = 2 * Precision[i] * Recall[i] / (Precision[i] + Recall[i])
        F1.append(round(f1, 4))

    # Calculated weighted scores of all labels except 0
    weights = 1 / class_weights
    Recall_ = weighted_score(Recall, weights)
    Precision_ = weighted_score(Precision, weights)
    F1_ = weighted_score(F1, weights)

    print_and_write(
        filepath,
        f"Overall Precision, Recall, F1 score: {Precision_:.4f}, {Recall_:.4f}, {F1_:.4f}"
    )
    np.set_printoptions(linewidth=400, precision=2, suppress=True)
    print_and_write(filepath, f"Precision: {Precision}")
    print_and_write(filepath, f"Recall: {Recall}")
    print_and_write(filepath, f"F1 score: {F1}")

    np.set_printoptions(suppress=True)
    print_and_write(filepath,
                    f"Conf_matrix (y_pred_matrix):\n{y_pred_matrix}\n")

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


def save_predictions_as_imgs(loader, model, folder, device, epoch, part_name,
                             batch_size):

    model.eval()
    # for idx, (x, y) in enumerate(loader):
    x, y = next(iter(loader))
    x = x.to(device=device)
    with torch.no_grad():
        preds = torch.argmax(torch.softmax(model(x), 1), 1)
        # print(x.size(), y.size(), preds.size())
        xx = x.cpu().numpy()
        yy = y.cpu().numpy()
        predspreds = preds.cpu().numpy()

        for jj in range(batch_size):
            plt.figure(figsize=(12, 12))
            for ii in range(18, 58, 18):
                pos = ii // 18
                plt.subplot(3, 3, 3 * pos - 2, label=f'subplot{ii+jj}')
                plt.imshow(xx[jj, 0, ii, :, :], cmap='gray', origin="lower")
                plt.title(f'map (z_slice={ii})')

                plt.subplot(3, 3, 3 * pos - 1, label=f'subplot{ii+jj}')
                plt.imshow(yy[jj, ii, :, :], origin="lower")
                plt.title(f'model (z_slice={ii})')

                plt.subplot(3, 3, 3 * pos, label=f'subplot{ii+jj}')
                plt.imshow(predspreds[jj, ii, :, :], origin="lower")
                plt.title(f'model_pred (z_slice={ii})')

            plt.suptitle(f"Epoch {epoch} for {part_name}")
            map_name = f"eval.{jj}.epoch{epoch}.png"
            plt.savefig(os.path.join(folder, map_name))
            plt.close()

    model.train()
