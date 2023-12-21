import math
import os

import numpy as np

from F_Graph_building.A_structure.Edge import Edge
from F_Graph_building.A_structure.Node import Node


class Main_Graph():

    def __init__(self, params):
        self.params = params
        self.edge = []  # Edge list
        self.Nedge = 0  # num of Edges
        self.node = []  # Node list
        self.Nnode = 0  # num of Nodes
        self.Ntree = 0  # num of Trees

    def copy_graph(self, out_graph):
        out_graph.params = self.params
        out_graph.edge = self.edge
        out_graph.Nedge = self.Nedge
        out_graph.node = self.node
        out_graph.Nnode = self.Nnode
        out_graph.Ntree = self.Ntree
        out_graph.cid = self.cid
        out_graph.adj = self.adj


