import json
import os
import pickle

import numpy as np
# import pandas as pd
import torch
from B0_MODEL import UNET
from B1_UTILS import (identify_chain, load_checkpoint, map_normalizing,
                      map_writing, predict_location, smooth_location)
from B2_UTILS_modeling import refine_mass_centers, write_pdb  # track_5_3,

# =======NNEEWW============5/4/2023
HOME_PATH = "/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/"
LAB_DATA = HOME_PATH + "LabData_testing/"
RAW_RATA = HOME_PATH + "RawData_testing/"
# MAP_PATH = LAB_DATA + "Alex/segement/postprocess.mrc"
# MAP_PATH = LAB_DATA + "Yuntao/02/after-ISN/post_isonet_J498.mrc"
# MAP_PATH = LAB_DATA + "Jane/J498/cryosparc_P2_J498_010_volume_map_sharp.mrc"
# MAP_PATH = LAB_DATA + "Jensen/RECS-B/treated/post_run2ar1_1sharpen.mrc"
# MAP_PATH = LAB_DATA + "Fumi/emd_27875/emd_27875.map"
# MAP_PATH = LAB_DATA + "Fumi/Nsp15-RNA_complex/cryosparc_P53_cryosparc_P53_J358_010_volume_map_sharp.mrc"
MAP_PATH = LAB_DATA + "Fumi/Nsp15-RNA_complex-unsharpen/cryosparc_P53_cryosparc_P53_J358_010_volume_map.mrc"
# MAP_PATH = LAB_DATA + "Yao/emd_26865/emd_26865.map"

# MAP_PATH = RAW_RATA + "EMD-16016_re_2.8-GOOD/emd_16016.map"
# MAP_PATH = RAW_RATA + "EMD-14956_re_2.7-GOOD/emd_14956.map"
# MAP_PATH = RAW_RATA + "EMD-34905_re_2.9-GOOD/emd_34905.map"

# MAP_PATH = HOME_PATH + "test_for_dirrerentiaiton_ssRNA/spliceosome_re2_3/ribosome/EMD-13112_re_2.4 - GOOD - ribosome/emd_13112.map"


# model parts and out chennels
MODEL_PARTS = {
    "RNA_backbone": 3, "phosphorus_C1": 3, "sugar_ring_new": 2,
    "nitrogenous_bases_in_sugar_new": 5, "sugar_center": 2
}
# OUT_CHANNELS = [3, 3, 2, 5, 2]

# MODEL_PARTS = ["RNA_backbone", "phosphorus_C1", "sugar_ring_new", "nitrogenous_bases_in_sugar_new"]
# OUT_CHANNELS = [3, 3, 2, 5]

# MODEL_PARTS = ["sugar_ring_new", "nitrogenous_bases_in_sugar_new"]
# OUT_CHANNELS = [2, 5]

CHECKPOINT_PATH = "/home/qiboxu/VSCode_code/3d_u_net/3D_U_NET_checkpoint"

# Load epoch numbers and the class weights of all parts
load_epoch_num_path = os.path.join(CHECKPOINT_PATH, "epoch_num_path.txt")
with open(load_epoch_num_path, "r") as file:
    load_epoch_num_all = json.load(file)


def main():

    folder_path = os.path.dirname(MAP_PATH)
    map_name = os.path.basename(MAP_PATH)
    os.makedirs(os.path.join(folder_path, "possibility"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "unrefined"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "refined"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "chains"), exist_ok=True)
    os.makedirs(os.path.join(folder_path, "temp_in_chain"), exist_ok=True)

    # Step1: Model setting
    for part_name in MODEL_PARTS:

        print(f"\n==> Start {part_name} prediction: ")

        # Load U-net model settings
        load_epoch_num = load_epoch_num_all[part_name]
        load_model_path = os.path.join(
            CHECKPOINT_PATH,
            part_name,
            f'{part_name}.epoch{load_epoch_num}.pth.tar'
            )
        out_channels = MODEL_PARTS[part_name]
        DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        in_channels = 1

        # Step2: read map file and normalize it
        map_data, zoom_factors = map_normalizing(MAP_PATH)

        # Step3: predict grids of 64*64*64 using U-Nets
        # Load model
        model = UNET(in_channels, out_channels).to(DEVICE)
        load_checkpoint(load_model_path, model, map_location=DEVICE)

        # Predict samples one by one (64*64*64 per sample cube)
        part_possibility = predict_location(map_data, model, out_channels, DEVICE)
        model_data = np.argmax(part_possibility, 0)

        # Step4: Save the results
        # 4.1 backbone = RNA_backbone + phosphorus + C1 + sugar_ring_new
        if part_name == "RNA_backbone":
            RNA_backbone_data = np.where(model_data == 1, 1, 0)
        if part_name == "phosphorus_C1":
            RNA_backbone_data = np.where(model_data != 0, 1, RNA_backbone_data)
        if part_name == "sugar_ring_new":
            RNA_backbone_data = np.where(model_data == 1, 1, RNA_backbone_data)

        # 4.2 position posibility data
        if part_name == "nitrogenous_bases_in_sugar_new":
            nitrogenous_bases_in_sugar_new_possibility = part_possibility[:, :, :, :]
        # if part_name == "sugar_ring_new":
        #     sugar_ring_new_1_possibility = part_possibility[1, :, :, :]
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
        #     map_writing(MAP_PATH, smooth_location(segments_data, 1), out_map, zoom_factors)

        print("\n==> Start map writing: ")

        for channel_num in range(1, out_channels):
            out_map_file = f"{map_name.split('.mrc')[0]}_{part_name}_{channel_num}_possibility.mrc"
            out_map = os.path.join(folder_path, "possibility", out_map_file)
            write_data = part_possibility[channel_num, :, :, :]
            map_writing(MAP_PATH, write_data, out_map, zoom_factors)

            # if part_name == "RNA_backbone" and channel_num == 1:
            out_file = f"{part_name}_{channel_num}_probability"
            with open(os.path.join(folder_path, "temp_in_chain", out_file),
                      "wb") as file:
                pickle.dump(write_data, file)

            # if part_name == "RNA_backbone" and channel_num == 1:
            #     with open(os.path.join(
            #         folder_path,
            #         "temp_in_chain",
            #         "possibility_in_chain"
            #     ), "wb") as file:
            #         pickle.dump(write_data, file)
            #         exit()

            out_map = os.path.join(
                folder_path,
                "unrefined",
                f"{map_name.split('.mrc')[0]}_{part_name}_{channel_num}.mrc"
            )
            write_data = np.where(model_data == channel_num, channel_num, 0)
            map_writing(MAP_PATH, write_data, out_map, zoom_factors, is_int=True)

            out_map = os.path.join(
                folder_path,
                "refined",
                f"{map_name.split('.mrc')[0]}_{part_name}_{channel_num}_refined.mrc"
            )
            smooth_loc = smooth_location(write_data, channel_num)
            map_writing(MAP_PATH, smooth_loc, out_map, zoom_factors, is_int=True)

            # else:
            #     out_map = MAP_PATH.split('.mrc')[0]+"_"+part_name+'.mrc'
            #     write_data = model_data
            #     map_writing(MAP_PATH, write_data, out_map)

    print("\n==> Start segments identification: ")
    pos_tag = []
    ModelData, longest_segments, segments_data = identify_chain(RNA_backbone_data, pos_tag = pos_tag)

    tag = 1
    out_map = MAP_PATH.split('.mrc')[0]+f'_segments_{tag}_refine.mrc'
    map_writing(MAP_PATH, smooth_location(segments_data, tag), out_map, zoom_factors)

    # TEST PDB FILE
    print("\n==> Start model file writing: ")

    # out_pdb = os.path.join(folder_path, f"{map_name.split('.mrc')[0]}_bases_in_riboses.pdb")
    threshold = 0.9
    radius = 2

    refined_mass_centers_sugar_ring_new, local_maxima_indices_sugar_ring_new = refine_mass_centers(sugar_ring_new_1_possibility, radius=radius, threshold=threshold)
    out_pdb = os.path.join(
        folder_path,
        f"{map_name.split('.mrc')[0]}_bases_in_riboses_radius{radius}_threshold{threshold}.pdb"
    )
    base_types_indices = []
    for local_maxima_index in local_maxima_indices_sugar_ring_new:
        type_possibility = np.zeros(5, dtype='float32')
        z, y, x = local_maxima_index
        dd = 1
        for dz in range(-dd, dd+1):
            for dy in range(-dd, dd+1):
                for dx in range(-dd, dd+1):
                    if dz**2 + dy**2 + dx**2 <= dd:
                        indices = (z+dz, y+dy, x+dx)
                        if all(0 <= index < size for index, size in zip(indices, sugar_ring_new_1_possibility.shape)):
                            # print(nitrogenous_bases_in_sugar_new_possibility[:, z+dz, y+dy, x+dx])
                            type_possibility += nitrogenous_bases_in_sugar_new_possibility[:, z+dz, y+dy, x+dx]
        base_types_indices.append(np.argmax(type_possibility))

    residue_lists = ['O' if base_types_indice == 0
                     else 'A' if base_types_indice == 1
                     else 'G' if base_types_indice == 2
                     else 'C' if base_types_indice == 3
                     else 'U' if base_types_indice == 4
                     else 'X'
                     for base_types_indice in base_types_indices]
    write_pdb("C1'", "C", residue_lists, refined_mass_centers_sugar_ring_new, out_pdb)

    residue_single = "A"
    refined_mass_centers_P, local_maxima_indices_P = refine_mass_centers(phosphorus_C1_1_possibility, radius=radius, threshold=threshold)
    out_pdb = os.path.join(folder_path, f"{map_name.split('.mrc')[0]}_P_radius{radius}_threshold{threshold}.pdb")
    write_pdb("P", "P", residue_single, refined_mass_centers_P, out_pdb)

    refined_mass_centers_C1, local_maxima_indices_C1 = refine_mass_centers(phosphorus_C1_2_possibility, radius=radius, threshold=threshold)
    out_pdb = os.path.join(folder_path, f"{map_name.split('.mrc')[0]}_C1_radius{radius}_threshold{threshold}.pdb")
    write_pdb("C1'", "C", residue_single, refined_mass_centers_C1, out_pdb)



    print("==> Start matching to segments (could be a long time): ")

    for chain_id in np.unique(segments_data):
        if chain_id == 0:
            continue

        write_data = np.where(segments_data == chain_id, chain_id, 0)

        # out_map = MAP_PATH.split('.mrc')[0]+f'_segments_refine_{chain_id}.mrc'
        # map_writing(MAP_PATH, smooth_location(write_data, chain_id), out_map, zoom_factors)

        indices_chain = np.squeeze(np.argwhere(write_data == chain_id)).tolist()

        indices_sugar_ring_new = [i for i, coord in enumerate(local_maxima_indices_sugar_ring_new) if coord in indices_chain]
        refined_mass_centers_sugar_ring_new_in_chain = [refined_mass_centers_sugar_ring_new[i] for i in indices_sugar_ring_new]
        residue_lists_in_chain = [residue_lists[i] for i in indices_sugar_ring_new]
        out_pdb = os.path.join(folder_path, "chains", f"{map_name.split('.mrc')[0]}_chain{chain_id}_ribose.pdb")
        write_pdb("C", "C", residue_lists_in_chain, refined_mass_centers_sugar_ring_new_in_chain, out_pdb)

        indices_P = [i for i, coord in enumerate(local_maxima_indices_P) if coord in indices_chain]
        refined_mass_centers_P_in_chain = [refined_mass_centers_P[i] for i in indices_P]
        out_pdb = os.path.join(folder_path, "chains", f"{map_name.split('.mrc')[0]}_chain{chain_id}_P.pdb")
        write_pdb("P", "P", residue_single, refined_mass_centers_P_in_chain, out_pdb)

        indices_C1 = [i for i, coord in enumerate(local_maxima_indices_C1) if coord in indices_chain]
        refined_mass_centers_C1_in_chain = [refined_mass_centers_C1[i] for i in indices_C1]
        out_pdb = os.path.join(folder_path, "chains", f"{map_name.split('.mrc')[0]}_chain{chain_id}_C1.pdb")
        write_pdb("C1'", "C", residue_single, refined_mass_centers_C1_in_chain, out_pdb)

        # import pickle
        # os.makedirs(os.path.join(folder_path, "temp_in_chain"), exist_ok=True)
        with open(os.path.join(folder_path, "temp_in_chain", f"R_in_chain_{chain_id}"), "wb") as file:
            pickle.dump(refined_mass_centers_sugar_ring_new_in_chain, file)
        with open(os.path.join(folder_path, "temp_in_chain", f"residue_in_chain_{chain_id}"), "wb") as file:
            pickle.dump(residue_lists_in_chain, file)
        with open(os.path.join(folder_path, "temp_in_chain", f"P_in_chain_{chain_id}"), "wb") as file:
            pickle.dump(refined_mass_centers_P_in_chain, file)
        with open(os.path.join(folder_path, "temp_in_chain", f"C1_in_chain_{chain_id}"), "wb") as file:
            pickle.dump(refined_mass_centers_C1_in_chain, file)

        # # connection
        # P5s, Rs, C1s, P3s, residues, scores = [], [], [], [], [], []
        # for P5 in refined_mass_centers_P_in_chain:
        #     score, R, P3, C1 = track_5_3(P5, refined_mass_centers_P_in_chain, refined_mass_centers_sugar_ring_new_in_chain, refined_mass_centers_C1_in_chain)
        #     residue = residue_lists_in_chain[refined_mass_centers_sugar_ring_new_in_chain.index(R)] if R != None else None
        #     P5s.append(P5)
        #     Rs.append(R)
        #     C1s.append(C1)
        #     P3s.append(P3)
        #     residues.append(residue)
        #     scores.append(score)

        # unique_indexes = []
        # for i, P3 in enumerate(P3s):
        #     if P3s.count(P3) == 1:
        #         unique_indexes.append(i)

        # P5s_U = [P5s[i] for i in unique_indexes]

        # connection_indexes = []
        # for i, index in enumerate(unique_indexes):
        #     if index in connection_indexes:
        #         continue
        #     while P3s[index] in P5s_U:
        #         segement = []
        #         segement.append(index)
        #         index = P5s.index(P3s[index])
        #     segement.append(index)
        #     connection_indexes.append(segement)

    print("\n==> All tasks completed.")


if __name__ == "__main__":

    main()
