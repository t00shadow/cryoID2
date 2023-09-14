import numpy as np
import torch
from B0_MODEL import UNET 
# from tqdm import tqdm                                   # makes your loops show a smart progress meter
import torch.nn as nn 
import torch.optim as optim 
from B1_UTILS import (MapReading, MapWriting, MapReshape, load_checkpoint,
                      identify_chain)

import numpy as np 
import pandas as pd 
import mrcfile 
import subprocess
import os

import argparse


HOME_PATH = "/amber/electron/"
# MAP_PATH = HOME_PATH+"qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/raw_training_samples/EMD-0243_re_2.65/emd_0243_resampled_1_EXAMPLE_RNA_backbone.mrc"
MAP_PATH = HOME_PATH+"qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/raw_training_samples/EMD-0243_re_2.65/emd_0243_resampled_1.mrc"

OUT_MAP = MAP_PATH.split('.mrc')[0]+'_sugar_center.1.mrc'   # type: ignore


with mrcfile.mmap(MAP_PATH) as mrc:
    map_data = np.array(mrc.data)
    model_data = map_data.astype(np.int_)
    model_data = np.where(model_data==1, 1, 0)
    MapWriting(MAP_PATH, model_data, OUT_MAP)




# pos_tag = []
# ModelData, longest_segments, segments_data = identify_chain(model_data, pos_tag = pos_tag)
# OUT_MAP = MAP_PATH.split('.mrc')[0]+'_segments.mrc'   # type: ignore
# MapWriting(MAP_PATH, segments_data, OUT_MAP)


# for kk in range(1, 13):
#     lks = np.where(segments_data==kk, 1, 0)
#     print(lks.sum())
