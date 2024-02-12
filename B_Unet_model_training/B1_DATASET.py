import os

import numpy as np
from torch.utils.data import Dataset


class MapModelDataset(Dataset):

    def __init__(self, map_dirs, model_dir, model_type):
        # map_dirs: a list of map directories
        self.map_dirs = map_dirs
        self.model_dir = model_dir
        # self.maps: a list of lists of map_names
        self.maps = [sorted(os.listdir(map_dir)) for map_dir in map_dirs]
        self.model_type = model_type

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, index):
        map = None
        for i in range(len(self.map_dirs)):
            map_dir = self.map_dirs[i]
            map_name = self.maps[i][index]
            map_path = os.path.join(map_dir, map_name)
            # add a dimension for feature
            map_part = np.expand_dims(np.load(map_path), axis=0)
            # concatenate maps in different features
            if map is None:
                map = map_part
            else:
                map = np.concatenate((map, map_part), axis=0)
            # print(map.shape)

        model_path = os.path.join(
            self.model_dir,
            self.maps[index].replace('map', "model_" + str(self.model_type)))
        model = np.load(model_path)
        # model = np.expand_dims(np.load(model_path), axis=0)
        # print(model.shape)
        return map, model


# # ============ TEST ====================
# MODEL_TYPE = "all_atoms"
# TRAIN_MAP_DIR = "/amber/electron/qiboxu/Database/U_NET/TEST/train_val_data/train/map_sample/"
# TRAIN_MODEL_DIR = "/amber/electron/qiboxu/Database/U_NET/TEST/train_val_data/train/"+MODEL_TYPE+"/"
# VAL_MAP_DIR = "/amber/electron/qiboxu/Database/U_NET/TEST/train_val_data/val/map_sample/"
# VAL_MODEL_DIR = "/amber/electron/qiboxu/Database/U_NET/TEST/train_val_data/val/"+MODEL_TYPE+"/"

# train_ds = MapModelDataset(map_dir=TRAIN_MAP_DIR, model_dir=TRAIN_MODEL_DIR, model_type=MODEL_TYPE)
# for i in range(10):
#     train_ds.__getitem__(i)
