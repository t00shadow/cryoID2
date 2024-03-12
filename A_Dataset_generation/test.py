import os

# split data into training and testing dataset
import splitfolders

import shutil




MAIN_HOME_PATH = '/home/qiboxu/Database/U_NET/EMDB_PDB_for_U_Net/Filtered_Dateset'
PATH_SETTINGS = {
    "Filtered_Dateset": [
        os.path.join(MAIN_HOME_PATH, 'Raw'),
        os.path.join(MAIN_HOME_PATH, 'Training'),
        os.path.join(MAIN_HOME_PATH, 'Raw', 'final-20240212.csv'),
    ],
}

PATH_KEYS = "Filtered_Dateset"
DATA_PATH, HOME_PATH, csv_path = PATH_SETTINGS[PATH_KEYS]
temp_sample_path = os.path.join(HOME_PATH, "ready_to_train_and_val")
os.makedirs(temp_sample_path, exist_ok=True)



sample_path = os.path.join(HOME_PATH, "TEST_with_blank", "train_val_data")

print(temp_sample_path)
os.makedirs(sample_path, exist_ok=True)
splitfolders.ratio(input=temp_sample_path,
                    output=sample_path,
                    seed=44,
                    ratio=(.8, .2),
                    group_prefix=None,
                    move=True)
# shutil.rmtree(temp_sample_path)
