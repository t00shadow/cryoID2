class Edge():

    def __init__(self):
        self.d = 0
        self.dens = 0
        self.id1 = 0  # Node id1
        self.id2 = 0  # Node id2
        self.eid = 0  # Edge id
        self.mst_lable = False  # edge in mst or not
        self.local_lable = False  # if there exists a node which is in acceptable range to this edge's two nodes, then local_lable=True
        self.keep_lable = False  # ? the 100% will be on the tree will be marked as keep_label=True ?

    def copy_edge(self, in_edge):
        self.d = in_edge.d
        self.dens = in_edge.dens
        self.id1 = in_edge.id1
        self.id2 = in_edge.id2
        self.eid = in_edge.eid
        self.mst_label = in_edge.mst_label
        self.local_label = in_edge.local_label
        self.keep_label = in_edge.keep_label
