import numpy as np
import torch
from B0_MODEL import UNET 
# from tqdm import tqdm                                   # makes your loops show a smart progress meter
import torch.nn as nn 
import torch.optim as optim 
from B1_UTILS import (MapReading, MapWriting, MapReshape, load_checkpoint, identify_chain)

# ==============================================================================================================================
# ==============     step1: resample to 1 -> put it in a empty folder -> input the location of the file 
# ==============================================================================================================================

# "volume resample #1 spacing 1; "save Z:\qiboxu\Database\U_NET\EMDB_PDB_for_U_Net\to_be_tested\..._resampled_1.mrc model #2; close"

GRID_SIZE = 64
IN_CHANNELS = 1
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
MODEL_SETTINGS = [
    # [63, 'cuda:1', "nitrogenous_bases_in_sugar", 1, 5, [1.0, 255.0, 206.0, 270.0, 309.0],],
    # [75, 'cuda:2', "RNA_backbone", 1, 3, [1., 40., 51.],],
    # [70, 'cuda:3', "sugar_ring", 1, 2, [1.,63.],],
    [83, 'cuda:1', "nitrogenous_bases_in_sugar_new", 1, 5, [1.0, 344.0, 279.0, 366.0, 419.0],],
    [75, 'cuda:2', "RNA_backbone", 1, 3, [1., 40., 51.],],
    [82, 'cuda:3', "sugar_ring_new", 1, 2, [1.,85.],],   
                ]
             
# ==============================================================================================================================
# ==============     step2: run U-net model one by one to predict different element we want
# ==============================================================================================================================
def main():    
    MAP_PATH = input(path)
    for element in range(3):
        LOAD_EPOCH_NUM = MODEL_SETTINGS[element][0]
        MODEL_TYPE = MODEL_SETTINGS[element][2]
        GRID_STRID = MODEL_SETTINGS[element][3]
        OUT_CHANNELS = MODEL_SETTINGS[element][4]

        LOAD_MODEL_NAME = "/home/qiboxu/VSCode_code/code_self/3D_U_NET/checkpoint/"+MODEL_TYPE+".epoch"+str(LOAD_EPOCH_NUM)+".pth.tar"
        if GRID_STRID == 0.5:
            LOAD_MODEL_NAME = "/home/qiboxu/VSCode_code/code_self/3D_U_NET/checkpoint/"+MODEL_TYPE+"_0.5A.epoch"+str(LOAD_EPOCH_NUM)+".pth.tar"

        # =========================================================================================================================
        # =======   step2.1: read map file, wrap it with 0s making its size ready to divide into smaller data of 64*64*64
        map_data = MapReading(MAP_PATH)
        new_shape_map, insert_idx = MapReshape(map_data)
        new_shape = new_shape_map.shape
        new_shape_model = np.zeros(new_shape, int)

        # =========================================================================================================================
        # =======   step2.2: load U-Net model, predict data segments of 64*64*64 one by one
        model = UNET(in_channels=IN_CHANNELS, out_channels=OUT_CHANNELS).to(DEVICE)
        load_checkpoint(torch.load(LOAD_MODEL_NAME, map_location=DEVICE), model)
        model.eval()
        new_shape_map = torch.from_numpy(new_shape_map)
        new_shape_model = torch.from_numpy(new_shape_model)

        NUM_predicted = 1
        with torch.no_grad():
            NUM_cube = (new_shape[0]//50)*(new_shape[1]//50)*(new_shape[2]//50)
            for zz in range(new_shape[0]//50):
                for yy in range(new_shape[1]//50):
                    for xx in range(new_shape[2]//50): 
                        map_sample = new_shape_map[None, None, 50*zz:64+50*zz, 50*yy:64+50*yy, 50*xx:64+50*xx, ]
                        # predicting ..............
                        map_sample = map_sample.to(DEVICE)
                        pred_data = torch.argmax(torch.softmax(model(map_sample), 1),1)   
                        # write in new_shape_model
                        new_shape_model[7+50*zz:57+50*zz, 7+50*yy:57+50*yy, 7+50*xx:57+50*xx] = pred_data[0, 7:57, 7:57, 7:57]
                        print(f"{NUM_predicted}/{NUM_cube} of prediction is done.", end='\r')
                        if NUM_predicted == NUM_cube:
                            print("\n")
                        NUM_predicted += 1
        
        new_shape_model = new_shape_model.numpy()
        model_data = new_shape_model[insert_idx[0, 0]:insert_idx[0, 1], insert_idx[1, 0]:insert_idx[1, 1], insert_idx[2, 0]:insert_idx[2, 1]]
        
        # =========================================================================================================================
        # =======   step2.3: save the results into .mrc file

        if element == 1:
            pos_tag = []
            ModelData, longest_segments, segments_data = identify_chain(model_data, pos_tag = pos_tag)
            OUT_MAP = MAP_PATH.split('.mrc')[0]+'_segments.mrc'   
            MapWriting(MAP_PATH, segments_data, OUT_MAP)

            for iii in range(1, OUT_CHANNELS):
                OUT_MAP = MAP_PATH.split('.mrc')[0]+"_"+MODEL_TYPE+"_"+str(iii)+'.mrc'      
                write_data = np.where(model_data==iii, iii, 0)
                MapWriting(MAP_PATH, write_data, OUT_MAP)

        else:
            OUT_MAP = MAP_PATH.split('.mrc')[0]+"_"+MODEL_TYPE+'.mrc'      
            write_data = model_data
            MapWriting(MAP_PATH, write_data, OUT_MAP)
        











if __name__ == "__main__":
    main()