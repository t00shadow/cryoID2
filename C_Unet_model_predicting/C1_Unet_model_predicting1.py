import pandas as pd
import numpy as np
import torch
from B0_MODEL import UNET 
import os, json
from B1_UTILS import (map_normalizing, map_writing, load_checkpoint, predict_location,
                      identify_chain, specify_location)


# =======NNEEWW============5/4/2023
# MAP_PATH = "/amber/electron/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/to_be_tested/Yuntaos/01/post_J383_4A.mrc"
MAP_PATH = "/amber/electron/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/to_be_tested/Yao/emd_26865/emd_26865.map"



MODEL_PARTS = ["phosphorus_C1", "sugar_ring_new", "nitrogenous_bases_in_sugar_new", "RNA_backbone", "sugar_center"]
UNET_PATH = '/scratch/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/training/training1'


# Load epoch numbers and the class weights of all parts
load_epoch_num_path = os.path.join(UNET_PATH, "TEST", "train_val_data", "epoch_num_path.txt")
with open(load_epoch_num_path, "r") as file:
    load_epoch_num_all = json.load(file)
weight_path = os.path.join(UNET_PATH, "TEST", "train_val_data", "class_weight_for_training.txt")
with open(weight_path, "r") as file:
    weight_all = json.load(file)


def main():

    
    for idx, part_name in enumerate(MODEL_PARTS):
        # Step1: Load U-net model 
        # Load model settings 
        load_epoch_num = load_epoch_num_all[part_name]
        load_model_path = os.path.join(UNET_PATH, 'TEST', 'checkpoint', part_name, f'{part_name}.epoch{load_epoch_num}.pth.tar')
        out_channels = len(weight_all[part_name])
        DEVICE = 'cuda:3' if torch.cuda.is_available() else 'cpu'
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

        main()    



    
