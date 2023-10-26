import json
import os
import shutil

import pandas as pd
from B1_UTILS import download_map_model, map_normalizing, map_output
from B2_UTILS_data2npy import data_to_npy
from scipy.ndimage import zoom

re_range = "polymerase_re2_3_erase_similar"  # "helicase_re2_3_now_all" #"spliceosome_re2_3" #"5_8_now_all" #

# HOME_PATH = "/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/test_for_dirrerentiaiton_RNA/"
# csv_path = f"/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/test_for_dirrerentiaiton_RNA/EMDB_complex_re{re_range}_with_pdb.csv"
HOME_PATH = "/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/test_for_dirrerentiaiton_ssRNA/"
csv_path = f"{HOME_PATH}EMDB_{re_range}_with_pdb.csv"
DATA_PATH = os.path.join(HOME_PATH,  f"{re_range}")
os.makedirs(DATA_PATH, exist_ok=True)


df = pd.read_csv(csv_path)
emdbs, pdbs, resolutions = df["emdb_id"], df["fitted_pdbs"], df["resolution"].astype(str)

print(emdbs)

emdb_ids = [emdb.split("-")[1] for emdb in emdbs]
folders = [f"{emdb}_re_{resolution}" for emdb, resolution in zip(emdbs, resolutions)]
raw_maps = [f"emd_{emdb_id}.map" for emdb_id in emdb_ids]
models = [f"{pdb}.cif" for pdb in pdbs]
raw_map_paths = [f"{DATA_PATH}/{folder}/{raw_map}" for folder, raw_map in zip(folders, raw_maps)]
model_paths = [f"{DATA_PATH}/{folder}/{model}" for folder, model in zip(folders, models)]


for idx, emdb_id in enumerate(emdb_ids):
    if "now_all" in re_range and idx % 4 != 0:
        continue
    download_map_model(emdb=emdbs[idx], pdb=pdbs[idx], resolution=resolutions[idx], directory=DATA_PATH)
