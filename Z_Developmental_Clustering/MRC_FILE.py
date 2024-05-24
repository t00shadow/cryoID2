import numpy as np
import mrcfile
from tqdm import tqdm
from scipy.spatial import KDTree
import os

class MRC_FILE(object):
    def __init__(self):
        self.xdim, self.ydim, self.zdim = 0, 0, 0
        self.angstromX, self.angstromY, self.angstromZ = 0, 0, 0
        self.origin = 0
        self.orientation = 0
    
    def read_mrc(self, mrc_file_path):
        with mrcfile.open(mrc_file_path) as mrc:
            self.angstromX, self.angstromY, self.angstromZ = float(mrc.header.cella.x.item()), float(mrc.header.cella.y.item()), float(mrc.header.cella.z.item())  # cell dimensions
            self.xdim, self.ydim, self.zdim = int(mrc.header.nx.item()), int(mrc.header.ny.item()), int(mrc.header.nz.item()) #Determine Shape of Array
            self.orientation = [mrc.header.mapc.item(), mrc.header.mapr.item(), mrc.header.maps.item()] #Determine Orientation of Array
            self.origin = [mrc.header.nxstart.item(), mrc.header.nystart.item(), mrc.header.nzstart.item()]
            self.originial_density = np.array(mrc.data)
            mrc.close()
        
        print("=== MRC File Read ===")
        print(f"EM_map Shape: {self.originial_density.shape}")
        print(f"Cell Dimensions (angstroms): {self.angstromX, self.angstromY, self.angstromZ}")
        print(f"Array Dimensions: {self.xdim, self.ydim, self.zdim}")
        print(f"Orientation (MAPC, MAPR, MAPS): {self.orientation}")
        print(f"Origin: {self.origin}")
        print("=======================")
        
    def init_uNet_Structures(self, uNet_map_path, flag):
        if(flag == 0):
            print("Secondary Structure, Initializing")
            with mrcfile.open(uNet_map_path) as mrc:
                self.uNet_map_SS = np.squeeze(mrc.data)
                mrc.close()            
        elif(flag == 1):
            print("Key Atoms: Phosphates")
            with mrcfile.open(uNet_map_path) as mrc:
                self.uNet_map_KA = np.squeeze(mrc.data)
                mrc.close()
        elif(flag == 2):
            print("Key Atoms: Sugar")
            with mrcfile.open(uNet_map_path) as mrc:
                self.uNet_map_SA = np.squeeze(mrc.data)
                mrc.close()