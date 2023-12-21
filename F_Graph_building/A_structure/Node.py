import numpy as np


class Node():
    def __init__(self):
        self.N = 0  # num of the connected edges of the node
        self.e = []  # edge list
        self.cid = []  # chain id of the node

        self.node_id = 0  # atoms like ["O", "C", "N", "P", "others"]
        self.atom_prob = np.zeros(5)
        self.log_atom = np.zeros(5)

        self.count_atom = 0  # ["sugar", "phosphorus", "A", "UT", "C", "G", "protein"]
        self.nuc_prob = np.zeros(7)
        self.count_nuc = 0
        self.log_nuc = np.zeros(7)

        self.base_prob = np.zeros(5)
        self.count_base = 0
        self.log_base = np.zeros(5)
