from B0_MODEL import UNET
from B2_UTILS import (load_checkpoint, save_checkpoint, get_loaders,
                      check_accuracy, save_predictions_as_imgs)
import torch
from tqdm import tqdm  # makes your loops show a smart progress meter
import torch.nn as nn
import torch.optim as optim
import os, json



for i in range(10000000000000):
    if i != 9999999999999:
        print(f'Finished: {i}', end='\r')
    else:
        print(f'Finished: {i}')

