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


def map_reshape(map_data):
    new_shape = np.zeros(3, int)
    insert_idx = np.zeros((3, 2), int)
    for i in range(3):
        new_shape[i] = 50 * ((map_data.shape[i]) // 50 + 1) + 14
        insert_idx[i, 0] = 7
        insert_idx[i, 1] = 7 + map_data.shape[i]
    new_shape_map = np.zeros(tuple(new_shape), np.float32)
    new_shape_map[
        7:insert_idx[0, 1],
        7:insert_idx[1, 1],
        7:insert_idx[2, 1],
    ] = map_data

    return new_shape_map, insert_idx


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


def predict_location(map_data, model, out_channels, device):

    new_shape_map, insert_idx = map_reshape(map_data)
    new_shape = new_shape_map.shape
    new_shape_map = torch.from_numpy(new_shape_map)
    new_shape_model = torch.zeros((out_channels, ) + new_shape)

    model.eval()
    # Iterate over 3D cubes of the map data
    with torch.no_grad():
        cube_size = 50
        num_cubes = (new_shape[0] // cube_size) * (
            new_shape[1] // cube_size) * (new_shape[2] // cube_size)
        num_completed = 0
        for zz in range(new_shape[0] // cube_size):
            for yy in range(new_shape[1] // cube_size):
                for xx in range(new_shape[2] // cube_size):
                    # Extract a 3D cube from the map data
                    cube = new_shape_map[None, None,
                                         cube_size * zz:64 + cube_size * zz,
                                         cube_size * yy:64 + cube_size * yy,
                                         cube_size * xx:64 + cube_size * xx]

                    # Move the cube to the specified device and perform prediction
                    cube = cube.to(device)
                    pred_data = torch.softmax(model(cube), 1)

                    # Write predicted data to new_shape_model
                    new_shape_model[:, 7 + cube_size * zz:57 + cube_size * zz,
                                    7 + cube_size * yy:57 + cube_size * yy,
                                    7 + cube_size * xx:57 +
                                    cube_size * xx] = pred_data[0, :, 7:57,
                                                                7:57, 7:57]

                    # Update progress
                    num_completed += 1
                    print(
                        f"{num_completed}/{num_cubes} of prediction is done.",
                        end='\r')
        # print("\n")

    # Extract model data
    part_possibility = new_shape_model[:, insert_idx[0, 0]:insert_idx[0, 1],
                                       insert_idx[1, 0]:insert_idx[1, 1],
                                       insert_idx[2, 0]:insert_idx[2,
                                                                   1]].numpy()

    return part_possibility


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
