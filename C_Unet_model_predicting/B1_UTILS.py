import argparse
import os
import subprocess

import mrcfile
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import zoom

GRID_SIZE = 64
SAMPLE_NUMBER = 1


def predict_location(map_data_,
                     model,
                     out_channels,
                     device,
                     cube_size=50,
                     pad_size=7):
    map_shape = map_data_.shape

    cube_nums = [
        map_shape[i] // cube_size + 1 for i in range(1, len(map_shape))
    ]
    num_cubes = cube_nums[0] * cube_nums[1] * cube_nums[2]
    larger_shape = [
        cube_size * cube_nums[i] + pad_size * 2 for i in range(len(cube_nums))
    ]
    larger_shape.insert(0, map_shape[0])

    reshaped_map_data = np.zeros(tuple(larger_shape), np.float32)
    reshaped_map_data[:, pad_size:pad_size + map_shape[1],
                      pad_size:pad_size + map_shape[2],
                      pad_size:pad_size + map_shape[3]] = map_data_

    reshaped_map_data = torch.from_numpy(reshaped_map_data)
    model_larger_shape = larger_shape[-3:]
    model_larger_shape = np.insert(model_larger_shape, 0, out_channels)
    reshaped_model = torch.zeros(tuple(model_larger_shape))
    reshaped_model_ = reshaped_model

    model.eval()
    # Iterate over 3D cubes of the map data
    with torch.no_grad():
        num_completed = 0

        for zz in range(cube_nums[0]):
            zstrt = cube_size * zz
            for yy in range(cube_nums[1]):
                ystrt = cube_size * yy
                for xx in range(cube_nums[2]):
                    # Extract a 3D cube from the map data
                    xstrt = cube_size * xx
                    cube = reshaped_map_data[None, :, zstrt:64 + zstrt,
                                             ystrt:64 + ystrt,
                                             xstrt:64 + xstrt]

                    # Move the cube to the specified device and perform prediction
                    cube = cube.to(device)
                    pred_data_ = torch.softmax(model(cube), 1)

                    # Write predicted data to reshaped_model
                    reshaped_model_[:, 7 + zstrt:57 + zstrt,
                                    7 + ystrt:57 + ystrt,
                                    7 + xstrt:57 + xstrt] = pred_data_[0, :,
                                                                       7:57,
                                                                       7:57,
                                                                       7:57]
                    # # Also output predicted data without softmax
                    # pred_data = model(cube)
                    # reshaped_model[:, 7 + zstrt:57 + zstrt,
                    #                7 + ystrt:57 + ystrt,
                    #                7 + xstrt:57 + xstrt] = pred_data[0, :,
                    #                                                  7:57,
                    #                                                  7:57,
                    #                                                  7:57]
                    # Update progress
                    num_completed += 1
                    print(
                        f"{num_completed}/{num_cubes} of prediction is done.",
                        end='\r')

    # Extract model data
    part_possibility = reshaped_model_[:, pad_size:pad_size + map_shape[1],
                                       pad_size:pad_size + map_shape[2],
                                       pad_size:pad_size + map_shape[3]].cpu().numpy()
    # prediction = reshaped_model[:, pad_size:pad_size + map_shape[1],
    #                             pad_size:pad_size + map_shape[2],
    #                             pad_size:pad_size + map_shape[3]].cpu().numpy()

    return part_possibility  # , prediction,


# map normalization on recommended contour level
def map_normalizing(map_path, recl=0.0):
    if recl == '':
        recl = 0.0
    with mrcfile.mmap(map_path, 'r+') as mrc:
        # Load map data
        map_data = np.array(mrc.data, dtype=np.float32)
        map_origin = np.array([mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart], dtype=np.int8)
        map_orientation = np.array([mrc.header.mapc, mrc.header.mapr, mrc.header.maps], dtype=np.float32)


        # Resample map to 1.0A*1.0A*1.0A grid size
        zoom_factors = [mrc.voxel_size.z, mrc.voxel_size.y, mrc.voxel_size.x]
        map_data = zoom(map_data, zoom_factors)

        # remove noisy values that are too small
        count_good = np.sum(map_data > max(0, recl))
        count_total = map_data.size
        if recl>0.0:
            # set some small values less than 0 to make the recommended contour level 15th percentile among positive values
            bottom_value_percentile = (1-(count_good/0.85)/count_total)*100
            if bottom_value_percentile > 100:
                value_bottom = 0
            else:
                value_bottom = np.percentile(map_data, bottom_value_percentile)
        else:
            # set 30% lowest positive values less than 0
            bottom_value_percentile = (1-(count_good*0.7)/count_total)*100
            value_bottom = np.percentile(map_data, bottom_value_percentile)
        map_data -= max(value_bottom, 0)

        # # Normalize map values to the range (0.0, 1.0)
        # data_99_9 = np.percentile(map_data, 99.9)
        # if data_99_9 == 0.:
        #     print('data_99_9 == 0!!')
        #     raise ValueError('99.9th percentile of map data is zero')
        # map_data /= data_99_9
        # map_data = np.clip(map_data, 0., 1.)

        positive_data = map_data
        positive_data = positive_data[positive_data >= 0]
        positive_data_99 = np.percentile(positive_data, 99)
        if positive_data_99 == 0.:
            raise ValueError('Empty map (99.9th percentile of map data is zero)')
        map_data = map_data / positive_data_99
        map_data = np.clip(map_data, 0., 1.)

        map_orientation = np.array([mrc.header.mapc-1, mrc.header.mapr-1, mrc.header.maps-1], dtype=np.int32)
        map_data = np.transpose(map_data, map_orientation)

    return map_data, zoom_factors



def MapReading(MapPath):
    with mrcfile.mmap(MapPath) as mrc:

        # ========================== scaler map.data to (0.0, 1.0) ==========================
        map_data = mrc.data
        map_data = np.where(map_data > 0, map_data, 0)
        data_99 = np.percentile(map_data, 99)
        if data_99 == 0.:
            print('data_99 == 0!!')
            print(mrc.header.dmin, mrc.header.dmax, data_99)
        # print(mrc.header.dmax, mrc.header.dmin, data_99)
        # dada = np.where(map_data > data_99, 1, 0)
        # print(dada.sum()/(mrc.header.nz*mrc.header.ny*mrc.header.nx))
        # exit()
        map_data = np.where(map_data > data_99, data_99, map_data)
        map_data = map_data / data_99
        map_data = map_data.astype(np.float32)

    return map_data




def map_writing(InputMap, ModelData, OutputMap, zoom_factors, is_int=False):
    if os.path.exists(OutputMap):
        status = subprocess.call(['rm', OutputMap])
    print(f"Writing map: {os.path.basename(OutputMap)}")
    status = subprocess.call(['cp', InputMap, OutputMap])
    mrc = mrcfile.open(OutputMap, mode='r+')
    ModelData = ModelData.astype(np.float32)
    if is_int:
        ModelData = ModelData.astype(np.int8)
    mrc.set_data(ModelData)
    # AA = mrc.header.origin
    # if True:
    # print(mrc.header.nzstart)
    # incase (nz, ny, nx)_start is not (0, 0, 0)
    mrc.header.nzstart = int(mrc.header.nzstart * zoom_factors[0])
    mrc.header.nystart = int(mrc.header.nystart * zoom_factors[1])
    mrc.header.nxstart = int(mrc.header.nxstart * zoom_factors[2])
    # print(mrc.header.nzstart)
    mrc.header.mz = ModelData.shape[0]
    mrc.header.ispg = 401
    # mrc.print_header()
    mrc.close()
    # print("New map is writen.")


# def load_checkpoint(checkpoint, model):
#     print("=> Loading checkpoint")
#     model.load_state_dict(checkpoint["state_dict"])
#     print("Checkpoint is loaded.")


def load_checkpoint(load_model_path, model, map_location):
    # print(f"=> Loading checkpoint: {os.path.basename(load_model_path)}")
    checkpoint = torch.load(load_model_path, map_location)
    model.load_state_dict(checkpoint["state_dict"])
    # print("Checkpoint is loaded.")


def identify_chain(model_data, tag=11, pos_tag=[]):
    VoxelThreshold = 1000.
    # print("=> Identify different chains:")
    [a, b, c] = model_data.shape
    ModelData = np.zeros([a + 2, b + 2, c + 2])
    ModelData[1:-1, 1:-1, 1:-1] = model_data
    shape = ModelData.shape
    # num = np.zeros(tag)
    num = np.zeros([tag, 2])
    longest_segments = np.zeros([1, 2])
    # print(shape)
    for z in range(1, ModelData.shape[0] - 1):
        for y in range(1, ModelData.shape[1] - 1):
            for x in range(1, ModelData.shape[2] - 1):
                if ModelData[z, y, x] == 1:
                    pos_tag = []
                    print(f"\rChain No.{tag-10}: ", end="\t")
                    ModelData[z, y, x] = tag
                    pos_tag.append((z, y, x))
                    pos_around = search_around(ModelData, [z, y, x], tag)
                    pos_tag.extend(pos_around)

                    num = np.append(num, [[tag, 1]], axis=0)
                    while num[tag, 1] < len(pos_tag):
                        pos_around = search_around(
                            ModelData, list(pos_tag[int(num[tag, 1])]), tag)
                        pos_tag.extend(pos_around)
                        print(f"num = {num[tag, 1]}          ", end="\r\t\t")
                        num[tag, 1] += 1
                    if num[tag, 1] > VoxelThreshold:
                        longest_segments = np.append(longest_segments,
                                                     [[tag, num[tag, 1]]], 0)
                    # print(f"{num[tag, 1]} voxels.")

                    # num = np.append(num, 1)
                    # while num[tag] < len(pos_tag):
                    #     pos_around = search_around(ModelData, list(pos_tag[int(num[tag])]), tag)
                    #     pos_tag.extend(pos_around)
                    #     print(f"num = {num[tag]}", end="\r")
                    #     num[tag] += 1
                    # if num[tag] > 1000:
                    #     longest_segments.append([tag, num[tag]])
                    # print(f"{num[tag]} voxels.", end="\r")

                    tag += 1

    print(f"\rThere are {len(longest_segments[:,0])-1} long segments:")
    segments_data = np.zeros([a, b, c]).astype(np.int_)
    Model_Data = (ModelData[1:-1, 1:-1, 1:-1]).astype(np.int_)

    idx_sort = np.argsort(longest_segments[:, 1])[::-1]

    for i in range(len(longest_segments) - 1):
        tagg = longest_segments[idx_sort[i], 0]
        numm = longest_segments[idx_sort[i], 1]
        lks = np.where(Model_Data == int(tagg), 1, 0)
        segments_data = np.where(Model_Data == int(tagg), i + 1, segments_data)
        print(i + 1, "\t", int(tagg), "\t", numm, lks.sum())

    return Model_Data, longest_segments, segments_data


def search_around(ModelData, idx, tag):
    z, y, x = idx
    pos_temp = []
    for around in [(z, y, x - 1), (z, y, x + 1), (z, y - 1, x), (z, y + 1, x),
                   (z - 1, y, x), (z + 1, y, x)]:
        if ModelData[around] == 1:
            ModelData[around] = tag
            pos_temp.append(around)
    return pos_temp


def smooth_location(model_data, tag):
    lenz, leny, lenx = model_data.shape
    location_data = np.zeros(model_data.shape)
    location_data[1:-1, 1:-1, 1:-1] = (  model_data[1:-1, 1:-1, 1:-1]\
                                       + model_data[0:-2, 1:-1, 1:-1] + model_data[1:-1, 0:-2, 1:-1] + model_data[1:-1, 1:-1, 0:-2]\
                                       + model_data[2:lenz, 1:-1, 1:-1]+model_data[1:-1, 2:leny, 1:-1]+model_data[1:-1, 1:-1, 2:lenx])/7
    location_data = np.where(location_data < tag / 2, 0, tag)
    return (location_data)





# def search_around(ModelData, idx, tag):
#     z, y, x = idx
#     num = 0
#     ModelData[z, y, x] = tag
#     num += 1
#     print(f"num = {num}.", end="\r")
#     if ModelData[z, y, x-1] == 1:
#         search_around(ModelData, [z, y, x-1], tag)
#     if ModelData[z, y, x+1] == 1:
#         search_around(ModelData, [z, y, x+1], tag)
#     if ModelData[z, y-1, x] == 1:
#         search_around(ModelData, [z, y-1, x], tag)
#     if ModelData[z, y+1, x] == 1:
#         search_around(ModelData, [z, y+1, x], tag)
#     if ModelData[z-1, y, x] == 1:
#         search_around(ModelData, [z-1, y, x], tag)
#     if ModelData[z+1, y, x] == 1:
#         search_around(ModelData, [z+1, y, x], tag)

# def identify_chain(ModelData, tag = 11):
#     print("Identify different chains:???? ")
#     num = np.zeros([11, 2])
#     # position = np.zeros(ModelData.shape)
#     position = np.array(ModelData)
#     chains = np.zeros(ModelData.shape)
#     # print(num)
#     # for z in range(1, ModelData.shape[0]-1):
#     #     for y in range(1, ModelData.shape[1]-1):
#     #         for x in range(1, ModelData.shape[2]-1):
#     #             NN = [ModelData[z, y, x+1], ModelData[z, y, x-1],ModelData[z, y+1, x],ModelData[z, y-1, x],ModelData[z+1, y, x],ModelData[z-1, y, x]]
#     #             if 1 in NN:
#     #                 position[z, y, x] = 1

#     for z in range(1, ModelData.shape[0]-1):
#         for y in range(1, ModelData.shape[1]-1):
#             for x in range(1, ModelData.shape[2]-1):
#                 if position[z, y, x] == 1:
#                     MM = int(max(position[z, y, x+1], position[z, y, x-1],position[z, y+1, x],position[z, y-1, x],position[z+1, y, x],position[z-1, y, x]))
#                     if MM > 10:
#                         position[z, y, x] = MM
#                         print(MM, end='\r')
#                         num[MM, 1] += 1
#                     else:
#                         position[z, y, x] = tag
#                         num = np.append(num, [[tag, 1]], axis=0)
#                         # print(num)
#                         # print(num.shape)
#                         tag += 1
#     print('\n')

#     idx_sort = np.argsort(num[:,1])
#     idx_sort = idx_sort[::-1]
#     numbers = np.zeros([5, 2])
#     numbers[:, 0] = range(10, 10+5)
#     numbers[:, 1] = num[idx_sort[0:5],1]
#     print(numbers)

#     for iii in range(5):
#         chains[np.where(position==idx_sort[iii])] = 10+iii
#     # numbers = np.zeros([2, tag - 11])
#     # numbers[0, :] = range(11, tag)
#     # for ii in range(11, tag):
#     #     print(ii, end='\r')
#     #     aaa = np.where(ModelData==ii, 1, 0)
#     #     numbers[1, ii-11] = aaa.sum()
#     return chains, numbers


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
