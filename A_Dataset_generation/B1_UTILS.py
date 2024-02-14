import os
import shutil
import subprocess

import mrcfile
import numpy as np
from scipy.ndimage import zoom


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

        if mrc.header.nzstart != 0 or mrc.header.nystart != 0 or mrc.header.nxstart != 0:
            print('The start of axis is not 0!!')
            raise ValueError('The start of axis is not zero!')

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


# def download_map_data_csv():

#     web_api = "https://www.ebi.ac.uk/emdb/api/"
#     search_filter = "search/" + 'rna AND sample_type:"complex" AND resolution:[1 TO 3} AND release_date:[2023-01-01T00:00:00Z TO 2023-12-31T23:59:59Z] AND xref_links:"pdb"'
#     download_filter = "?wt=csv&download=true&fl=emdb_id,title,resolution,fitted_pdbs,map_release_date,overall_molecular_weight,xref_ALPHAFOLD"


def search_emdb(
        query,
        file_names,
        fl='emdb_id,title,resolution,fitted_pdbs,xref_UNIPROTKB,xref_ALPHAFOLD'
):

    # Inputs:
    # query: a string list of search queries
    # Example: ['structure_determination_method:"singleParticle"', 'Human Albumin']
    # The query can also be composed by multiple search terms concatened by AND or OR terms
    # Example: ['sample_type:"virus" and resolution [* TO 3]']

    # file_names: a string list of desired file names
    # Example: 'Ribosome'

    # fl: list of fields to be shown in the csv file; each item is separated by ','
    # Example: 'emdb_id,resolution,fitted_pdbs'
    # Default: 'emdb_id,title,resolution,fitted_pdbs,xref_UNIPROTKB,xref_ALPHAFOLD'

    # Output(s):
    # csv file(s) with the user provided file names

    import requests
    url = 'https://www.ebi.ac.uk/emdb/api/search/'
    payload = {'fl': fl}

    if len(query) == len(file_names):
        for i in range(len(query)):
            r = requests.get(url + query[i],
                             params=payload,
                             headers={'accept': 'text/csv'})
            df = str(r.text)

            file_name = file_names[i] + '.csv'
            with open(file_name, 'w') as out:
                out.write(df)
    else:
        print('The length of query and file_names must match!')

    pass
