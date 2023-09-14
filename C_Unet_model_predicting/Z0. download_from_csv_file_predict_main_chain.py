import pandas as pd 
import numpy as np
import torch
from B0_MODEL import UNET 
import os, json, shutil
from B1_UTILS import (download_map_model, map_normalizing, map_writing, load_checkpoint, predict_location,
                      identify_chain, smooth_location)
from B2_UTILS_modeling import (refine_mass_centers, write_pdb, track_5_3)

DOWNLOAD = False



# re_range = "5_8_now_all" #"8_11"
# csv_path = f"/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/test_for_dirrerentiaiton_RNA/EMDB_complex_re{re_range}_with_pdb.csv"
# HOME_PATH = "/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/test_for_dirrerentiaiton_RNA/"
# DATA_PATH = os.path.join(HOME_PATH,  f"re{re_range}")

re_range = "polymerase_re2_3_erase_similar" #"helicase_re2_3_now_all" #"spliceosome_re2_3" #"5_8_now_all" #
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

if DOWNLOAD:
    for idx, emdb_id in enumerate(emdb_ids):
        download_map_model(emdb=emdbs[idx], pdb=pdbs[idx], resolution=resolutions[idx], directory=DATA_PATH)



# MODEL_PARTS = ["RNA_backbone", "phosphorus_C1", "sugar_ring_new", "nitrogenous_bases_in_sugar_new"]
# OUT_CHANNELS = [3, 3, 2, 5]

MODEL_PARTS = ["RNA_backbone"]
OUT_CHANNELS = [3]


CHECKPOINT_PATH = "/home/qiboxu/VSCode_code/3d_u_net/3D_U_NET/C_Unet_model_predicting/checkpoint"

# Load epoch numbers and the class weights of all parts
load_epoch_num_path = os.path.join(CHECKPOINT_PATH, "epoch_num_path.txt")
with open(load_epoch_num_path, "r") as file:
    load_epoch_num_all = json.load(file)




for jj, MAP_PATH in enumerate(raw_map_paths):

    if jj <= 30 :
        continue

    # if jj > 30 :
    #     break 
    
    folder_path = os.path.dirname(MAP_PATH)
    map_name = os.path.basename(MAP_PATH)

    if "now_all" in re_range and jj % 4 != 0:
        continue

    # if map_name not in ["emd_1849.map", "emd_1524.map", "emd_1854.map", "emd_1343.map", "emd_1345.map", "emd_2623.map"]:
    #     continue

    print(map_name, jj)

    os.makedirs(os.path.join(folder_path, "possibility"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "unrefined"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "refined"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "chains"), exist_ok=True)


    # Step1: Model setting
    for idx, part_name in enumerate(MODEL_PARTS):
        # Load U-net model settings 
        load_epoch_num = load_epoch_num_all[part_name]
        load_model_path = os.path.join(CHECKPOINT_PATH, part_name, f'{part_name}.epoch{load_epoch_num}.pth.tar')
        out_channels = OUT_CHANNELS[idx]
        DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        in_channels = 1

        # Step2: read map file and normalize it
        map_data, zoom_factors = map_normalizing(MAP_PATH)

        # Step3: predict grids of 64*64*64 using U-Nets
        # Load model 
        model = UNET(in_channels, out_channels).to(DEVICE)
        load_checkpoint(load_model_path, model, map_location=DEVICE)

        # Predict samples one by one (64*64*64 per sample cube)
        part_possibility = predict_location(map_data, model, out_channels,DEVICE)
        model_data = np.argmax(part_possibility, 0)
        
        # Step4: Save the results 
        # Store backbone position data
        if part_name == "RNA_backbone":
            RNA_backbone_data = np.where(model_data == 1, 1, 0)
        if part_name == "phosphorus_C1":
            RNA_backbone_data = np.where(model_data != 0, 1, RNA_backbone_data)
        if part_name == "sugar_ring_new":
            RNA_backbone_data = np.where(model_data == 1, 1, RNA_backbone_data)

        # Store position posibility data to generate pdb file
        if part_name == "nitrogenous_bases_in_sugar_new":
            nitrogenous_bases_in_sugar_new_possibility = part_possibility[:, :, :, :]
        if part_name == "sugar_ring_new":
            sugar_ring_new_1_possibility = part_possibility[1, :, :, :]
        if part_name == "phosphorus_C1":
           phosphorus_C1_1_possibility = part_possibility[1, :, :, :]
           phosphorus_C1_2_possibility = part_possibility[2, :, :, :]
        if part_name == "sugar_ring_new":
            sugar_ring_new_1_possibility = part_possibility[1, :, :, :]
        
        # # Creat mrc files
        # if part_name == "RNA_backbone":
        #     pos_tag = []
        #     ModelData, longest_segments, segments_data = identify_chain(model_data, pos_tag = pos_tag)
        #     out_map = MAP_PATH.split('.mrc')[0]+'_segments_refine.mrc'   
        #     map_writing(MAP_PATH, smooth_location(segments_data, 1), out_map)



        for channel_num in range(1, out_channels):
            out_map = os.path.join(folder_path, "possibility", f"{map_name.split('.mrc')[0]}_{part_name}_{channel_num}_possibility.mrc")    
            write_data = part_possibility[channel_num, :, :, :]
            map_writing(MAP_PATH, write_data, out_map, zoom_factors)

            out_map = os.path.join(folder_path, "unrefined", f"{map_name.split('.mrc')[0]}_{part_name}_{channel_num}.mrc")    
            write_data = np.where(model_data==channel_num, channel_num, 0)
            map_writing(MAP_PATH, write_data, out_map, zoom_factors, is_int=True)

            # out_map = os.path.join(folder_path, "refined", f"{map_name.split('.mrc')[0]}_{part_name}_{channel_num}_refined.mrc")    
            out_map = os.path.join(folder_path, f"{map_name.split('.mrc')[0]}_{part_name}_{channel_num}_refined.mrc") 
            map_writing(MAP_PATH, smooth_location(write_data, channel_num), out_map, zoom_factors, is_int=True)

                        # else:
                        #     out_map = MAP_PATH.split('.mrc')[0]+"_"+part_name+'.mrc'     
                        #     write_data = model_data
                        #     map_writing(MAP_PATH, write_data, out_map)          


    # pos_tag = []
    # ModelData, longest_segments, segments_data = identify_chain(RNA_backbone_data, pos_tag = pos_tag)
    # out_map = MAP_PATH.split('.mrc')[0]+'_segments_refine.mrc'   
    # map_writing(MAP_PATH, smooth_location(segments_data, 1), out_map)

 



    
