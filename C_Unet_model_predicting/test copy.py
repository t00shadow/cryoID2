import numpy as np 
import pandas as pd 
import mrcfile 
import subprocess
import os
from scipy.ndimage import zoom
import torch

import argparse


folder_path = "/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/test_for_dirrerentiaiton_RNA/re8_11"
map_name = "EMD-2624_re_8.9 - UPSIDEDOWN - should be good/emd_2624.map"

# map_name = "EMD-1343_re_9.7 - UPSIDEDOWN/emd_1343.map"




# map_name = "EMD-5643_re_10.7 - GOOD - UPSIDEDOWN/emd_5643.map"
map_name = "EMD-5643_re_10.7/emd_5643.map_RNA_backbone_1_refined.mrc"

# map_name = "EMD-32150_re_8.8 - GOOD/emd_32150.map"


map_path = os.path.join(folder_path, map_name)


with mrcfile.mmap(map_path) as mrc:
    # Load map data
    mrc.print_header()


    # map_data = np.array(mrc.data, dtype=np.float32)
    
    # # Resample map to 1.0A*1.0A*1.0A grid size
    # zoom_factors = [mrc.voxel_size.z, mrc.voxel_size.y, mrc.voxel_size.x]
    # map_data = zoom(map_data, zoom_factors)

    # # Normalize map values to the range (0.0, 1.0)
    # data_99_9 = np.percentile(map_data, 99.9)
    # if data_99_9 == 0.:
    #     print('data_99_9 == 0!!')
    #     raise ValueError('99.9th percentile of map data is zero')
    # map_data /= data_99_9 
    # map_data = np.clip(map_data, 0., 1.)


