import os

import numpy as np
from torch.utils.data import Dataset


class MapModelDataset(Dataset):

    def __init__(self, map_dirs, model_dir):
        # map_dirs: a list of map directories
        self.map_dirs = map_dirs
        self.model_dir = model_dir
        # self.maps: a list of lists of map_names
        self.maps = [sorted(os.listdir(map_dir)) for map_dir in map_dirs]
        self.models = sorted(os.listdir(model_dir))

    def __len__(self):
        return len(self.models)

    def __getitem__(self, index):
        for i in range(len(self.map_dirs)):
            map_dir = self.map_dirs[i]
            map_name = self.maps[i][index]
            map_path = os.path.join(map_dir, map_name)
            # print(map_path)
            map_part = np.load(map_path)

            # concatenate maps in different features
            if i == 0:
                # add a dimension of feature to original map
                map = np.expand_dims(map_part, axis=0)
                # print(map.shape)
            else:
                map = np.concatenate((map, map_part), axis=0)
                # print(map.shape)

        model_path = os.path.join(self.model_dir, self.models[index])
        model = np.load(model_path)
        # model = np.expand_dims(np.load(model_path), axis=0)
        # print(model.shape)
        return map, model


# # ============ TEST ====================

TRAIN_DIR = "/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/Filtered_Dateset/Training/TEST/train_val_data/train"

map_dirs = [
    os.path.join(TRAIN_DIR, 'map_sample'),
    os.path.join(TRAIN_DIR, 'secondary_strctures_pred_prob'),
    # os.path.join(TRAIN_DIR, 'residue_types_pred_prob')
]
model_dir = os.path.join(TRAIN_DIR, 'key_atoms')

train_ds = MapModelDataset(map_dirs, model_dir)
for i in range(10):
    train_ds.__getitem__(i)
