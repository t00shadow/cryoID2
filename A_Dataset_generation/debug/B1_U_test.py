import os
import shutil
import subprocess

import mrcfile
import numpy as np
from scipy.ndimage import zoom
import json
import os
import shutil
import numpy as np

import pandas as pd


def download_map_model(emdb, pdb, resolution, directory):
    emdb_id = emdb.split("-")[1]
    directory1 = f"{directory}/{emdb}_re_{resolution}"
    os.makedirs(directory1, exist_ok=True)
    emdb_fetch_link = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/{emdb}/map/emd_{emdb_id}.map.gz"
    pdb_fetch_link = f"http://files.rcsb.org/download/{pdb}.cif"
    if os.path.exists(f"{directory1}/emd_{emdb_id}.map"):
        pass
    else:
        subprocess.run(["wget", "-P", directory1, emdb_fetch_link])
        subprocess.run(["gzip", "-d", f"{directory1}/emd_{emdb_id}.map.gz"])
    if os.path.exists(f"{directory1}/{pdb}.cif"):
        pass
    else:
        subprocess.run(["wget", "-P", directory1, pdb_fetch_link])
        print(f"=> {emdb} and pdb-{pdb} files are downloaded.")


def map_normalizing(map_path):
    with mrcfile.mmap(map_path) as mrc:
        # Load map data
        map_data = np.array(mrc.data, dtype=np.float32)

        # Resample map to 1.0A*1.0A*1.0A grid size
        zoom_factors = [mrc.voxel_size.z, mrc.voxel_size.y, mrc.voxel_size.x]
        map_data = zoom(map_data, zoom_factors)

        # Normalize map values to the range (0.0, 1.0)
        data_99_9 = np.percentile(map_data, 99.9)
        if data_99_9 == 0.:
            print('data_99_9 == 0!!')
            raise ValueError('99.9th percentile of map data is zero')
        map_data /= data_99_9
        map_data = np.clip(map_data, 0., 1.)

    return map_data


def map_output(input_map, map_data, output_map, is_model=False):
    if os.path.exists(output_map):
        os.remove(output_map)

    print(f"=> Writing new map to {output_map}")
    shutil.copyfile(input_map, output_map)
    with mrcfile.open(output_map, mode='r+') as mrc:
        if is_model:
            map_data = map_data.astype(np.int8)
        else:
            map_data = map_data.astype(np.float32)

        mrc.set_data(map_data)
        mrc.header.mz = map_data.shape[0]
        mrc.header.ispg = 1  #401
        mrc.print_header()
        print("=> New map written successfully.")


DOWNLOAD = True
NORMALIZATION = True
MAP2NPY = False

MODEL_PARTS = ['secondary_strctures', 'key_atoms', 'residue_types']

MAIN_HOME_PATH = '/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net'
PATH_SETTINGS = {
    "Filtered_Dateset": [
        os.path.join(MAIN_HOME_PATH, 'Filtered_Dateset', 'Raw'),
        os.path.join(MAIN_HOME_PATH, 'Filtered_Dateset', 'Training'),
        os.path.join(MAIN_HOME_PATH, 'Filtered_Dateset', 'Raw',
                     'final-20240212.csv'),
    ],
}

PATH_KEYS = "Filtered_Dateset"
DATA_PATH, HOME_PATH, csv_path = PATH_SETTINGS[PATH_KEYS]
temp_sample_path = os.path.join(HOME_PATH, "ready_to_train_and_val")
os.makedirs(temp_sample_path, exist_ok=True)

# Read map list and generate raw_map and model downloading paths
df = pd.read_csv(csv_path)
emdbs, pdbs = df["emdb_id"], df["fitted_pdbs"]
resolutions = df["resolution"].astype(str)
emdb_ids = [emdb.split("-")[1] for emdb in emdbs]
folders = [
    f"{emdb}_re_{resolution}" for emdb, resolution in zip(emdbs, resolutions)
]
raw_maps = [f"emd_{emdb_id}.map" for emdb_id in emdb_ids]
models = [f"{pdb}.cif" for pdb in pdbs]
raw_map_paths = [
    f"{DATA_PATH}/{folder}/{raw_map}"
    for folder, raw_map in zip(folders, raw_maps)
]
model_paths = [
    f"{DATA_PATH}/{folder}/{model}" for folder, model in zip(folders, models)
]

map_paths = [
    f"{raw_map_path.split('.map')[0]}_normalized.mrc"
    for raw_map_path in raw_map_paths
]
for idx, emdb_id in enumerate(emdb_ids):
    if DOWNLOAD:
        download_map_model(emdb=emdbs[idx],
                           pdb=pdbs[idx],
                           resolution=resolutions[idx],
                           directory=DATA_PATH)
        # if idx == 20:   # for download limited number of testing data
        #     exit()
    if NORMALIZATION and not os.path.exists(map_paths[idx]):
        map_data = map_normalizing(raw_map_paths[idx])
        map_output(raw_map_paths[idx],
                   map_data,
                   map_paths[idx],
                   is_model=False)
