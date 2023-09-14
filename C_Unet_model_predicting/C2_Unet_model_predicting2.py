import pandas as pd
import numpy as np
import torch
from B0_MODEL import UNET 
import os, json
from B1_UTILS import (map_normalizing, map_writing, load_checkpoint, predict_location,
                      identify_chain, specify_location)


# =======NNEEWW============5/4/2023
# MAP_PATH = "/amber/electron/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/to_be_tested/Yuntaos/01/post_J383_4A.mrc"
# MAP_PATH = "/amber/electron/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/RawData_testing/EMD-26817_re_2.5/emd_26817.map"
# MAP_PATH = "/amber/electron/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/RawData_testing/EMD-34905_re_2.9/emd_34905.map"
# MAP_PATH = "/amber/electron/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/RawData_testing/EMD-26822_re_2.21/emd_26822.map"


PATH_SETTINGS = { 
    "EXAMPLE_KEYS": ["DATA_PATH", "HOME_PATH", "CSV_PATH", ],
    "training2": ['/amber/electron/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/RawData', '/amber/electron/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/training/training2', '/amber/electron/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/training/training2/TEST_EMDB_download_resolution_1-4_link_refined_unique.csv',],
    "training3": ['/amber/electron/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/RawData', '/amber/electron/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/training/training3', '/amber/electron/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/training/training3/TEST_EMDB_download_resolution_1-4_link_refined_unique.csv',],
    "testing1": ['/amber/electron/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/RawData_testing', '/amber/electron/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/to_be_tested/Internet/testing1', '/amber/electron/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/to_be_tested/Internet/testing1/test_cvs_1-3A_20230518-refine-2.csv',],
         }

PATH_KEYS = "testing1" #"training3" # Possible values: "training_data", "test_ribosome_cryoET", "test_ribosome_cryoEM"
DATA_PATH, HOME_PATH, csv_path = PATH_SETTINGS[PATH_KEYS]

# Read map list 
df = pd.read_csv(csv_path)
emdbs, pdbs, resolutions = df["emdb_id"], df["fitted_pdbs"], df["resolution"].astype(str)
emdb_ids = [emdb.split("-")[1] for emdb in emdbs]
folders = [f"{emdb}_re_{resolution}" for emdb, resolution in zip(emdbs, resolutions)]
raw_maps = [f"emd_{emdb_id}.map" for emdb_id in emdb_ids] 
models = [f"{pdb}.cif" for pdb in pdbs]
raw_map_paths = [f"{DATA_PATH}/{folder}/{raw_map}" for folder, raw_map in zip(folders, raw_maps)]
model_paths = [f"{DATA_PATH}/{folder}/{model}" for folder, model in zip(folders, models)]



MODEL_PARTS = ["phosphorus_C1", "sugar_ring_new", "nitrogenous_bases_in_sugar_new", "RNA_backbone", "sugar_center"]
UNET_PATH = '/scratch/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/training/training1'


# Load epoch numbers and the class weights of all parts
load_epoch_num_path = os.path.join(UNET_PATH, "TEST", "train_val_data", "epoch_num_path.txt")
with open(load_epoch_num_path, "r") as file:
    load_epoch_num_all = json.load(file)
weight_path = os.path.join(UNET_PATH, "TEST", "train_val_data", "class_weight_for_training.txt")
with open(weight_path, "r") as file:
    weight_all = json.load(file)


def main(MAP_PATH):

    
    for idx, part_name in enumerate(MODEL_PARTS):
        # Step1: Load U-net model 
        # Load model settings 
        load_epoch_num = load_epoch_num_all[part_name]
        load_model_path = os.path.join(UNET_PATH, 'TEST', 'checkpoint', part_name, f'{part_name}.epoch{load_epoch_num}.pth.tar')
        out_channels = len(weight_all[part_name])
        DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'
        in_channels = 1

        # Load model 
        model = UNET(in_channels, out_channels).to(DEVICE)
        load_checkpoint(load_model_path, model, map_location=DEVICE)

        # Step2: Read map file and normalize it
        map_data = map_normalizing(MAP_PATH)

        # Step3: Predict atom group's location using U-Nets (64*64*64 per sample cube)
        part_possibility = predict_location(map_data, model, out_channels,DEVICE)
        model_data = np.argmax(part_possibility, 0)
        
        # Step4: Save the results 
        if part_name == "RNA_backbone":
            pos_tag = []
            ModelData, longest_segments, segments_data = identify_chain(model_data, pos_tag = pos_tag)
            # out_map = MAP_PATH.split('.mrc')[0]+'_segments.mrc'   
            # map_writing(MAP_PATH, segments_data, out_map)
            out_map = MAP_PATH.split('.mrc')[0]+'_segments_refine.mrc'   
            map_writing(MAP_PATH, specify_location(segments_data, 1), out_map)
        
        for channel_num in range(1, out_channels):
            out_map = f"{MAP_PATH.split('.mrc')[0]}_{part_name}_{channel_num}_possibility.mrc"    
            write_data = part_possibility[channel_num, :, :, :]
            map_writing(MAP_PATH, write_data, out_map)

            out_map = f"{MAP_PATH.split('.mrc')[0]}_{part_name}_{channel_num}.mrc"    
            write_data = np.where(model_data==channel_num, channel_num, 0)
            map_writing(MAP_PATH, write_data, out_map, is_int=True)

            out_map = f"{MAP_PATH.split('.mrc')[0]}_{part_name}_{channel_num}_refine.mrc"    
            map_writing(MAP_PATH, specify_location(write_data, channel_num), out_map, is_int=True)

        # else:
        #     out_map = MAP_PATH.split('.mrc')[0]+"_"+part_name+'.mrc'     
        #     write_data = model_data
        #     map_writing(MAP_PATH, write_data, out_map)

    

if __name__ == "__main__":
    for MAP_PATH in raw_map_paths:
        main(MAP_PATH)    



    
