import math

import gemmi
import mrcfile
import numpy as np

# Read mrc file

# Read cif file


class CIF():

    def __init__(self, cif_path):
        self.structure = gemmi.read_structure(cif_path)
        self.box = self.structure.calculate_box()
