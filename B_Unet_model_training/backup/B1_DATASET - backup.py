import os

import numpy as np
from torch.utils.data import Dataset


class MapModelDataset(Dataset):

    def __init__(self, map_dir, model_dir, model_type):
        self.map_dir = map_dir
        self.model_dir = model_dir
        self.maps = os.listdir(map_dir)
        self.model_type = model_type

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, index):
        map_path = os.path.join(self.map_dir, self.maps[index])
        model_path = os.path.join(
            self.model_dir,
            self.maps[index].replace('map', "model_" + str(self.model_type)))
        map = np.expand_dims(np.load(map_path),
                             axis=0)  # add feature dimension
        # print(map.shape)
        model = np.load(model_path)
        # model = np.expand_dims(np.load(model_path), axis=0)         # add feature dimension
        # print(model.shape)
        return map, model


# # ============ TEST ====================
MODEL_TYPE = 'secondary_strctures'
TRAIN_DIR = "/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/Filtered_Dateset/Training/TEST/train_val_data/train"
TRAIN_MAP_DIR = os.path.join(TRAIN_DIR, 'map_sample')
TRAIN_MODEL_DIR = os.path.join(TRAIN_DIR, MODEL_TYPE)
# VAL_MAP_DIR = "/amber/electron/qiboxu/Database/U_NET/TEST/train_val_data/val/map_sample/"
# VAL_MODEL_DIR = "/amber/electron/qiboxu/Database/U_NET/TEST/train_val_data/val/"+MODEL_TYPE+"/"

train_ds = MapModelDataset(map_dir=TRAIN_MAP_DIR,
                           model_dir=TRAIN_MODEL_DIR,
                           model_type=MODEL_TYPE)

for i in range(10):
    train_ds.__getitem__(i)
