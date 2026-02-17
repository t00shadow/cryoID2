# distance based superset graph

import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict


def build_distance_graph(
    coords,
    max_dist,
    return_edge_lengths=False,
):
    """
    Build a distance-based superset graph.

    Nodes: indices into coords
    Edges: undirected edges between nodes within max_dist

    Parameters
    ----------
    coords : (N, 3) ndarray
        Cartesian coordinates (Å)
    max_dist : float
        Distance cutoff (Å)
    return_edge_lengths : bool
        If True, store distances on edges

    Returns
    -------
    adjacency : dict[int, list[int]] OR dict[int, list[tuple[int, float]]]
        Adjacency list
    edges : list[tuple[int, int]] OR list[tuple[int, int, float]]
        Edge list
    """

    n = len(coords)
    if n == 0:
        return {}, []

    nn = NearestNeighbors(radius=max_dist)
    nn.fit(coords)

    # Sparse radius graph
    graph = nn.radius_neighbors_graph(coords, mode="distance")
    graph = graph.tocoo()

    adjacency = defaultdict(list)
    edges = []

    for i, j, d in zip(graph.row, graph.col, graph.data):
        if i >= j:
            continue  # avoid duplicates + self edges

        if return_edge_lengths:
            adjacency[i].append((j, d))
            adjacency[j].append((i, d))
            edges.append((i, j, d))
        else:
            adjacency[i].append(j)
            adjacency[j].append(i)
            edges.append((i, j))

    return dict(adjacency), edges


def save_graph_pdb(coords, edges, filename):
    """
    Save a graph as a PDB with CONECT records.

    Parameters
    ----------
    coords : (N, 3) ndarray
        Node coordinates (Å), ordered by node index
        Expected order: (z, y, x) to match your pipeline
    edges : list of (i, j) or (i, j, d)
        Graph edges (0-based indices)
    filename : str
        Output PDB path
    """
    with open(filename, "w") as f:
        # Write atoms
        for i, (z, y, x) in enumerate(coords, start=1):
            f.write(
                f"HETATM{i:5d}  C   UNK     1    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )

        # Write connectivity
        for edge in edges:
            if len(edge) == 3:
                i, j, _ = edge
            else:
                i, j = edge

            # PDB atom indices are 1-based
            f.write(f"CONECT{i+1:5d}{j+1:5d}\n")


def main():
    from R_A1_clustering import main as run_clustering
    # from R_A1_clustering_2stages_v2 import main as run_clustering
    from datetime import datetime
    import os

    # Run clustering
    result = run_clustering()

    centroids = result["centroids"]
    connect_dist = result["connect_dist"]
    connect_dist = 4.6
    connect_dist = 4.4
    # connect_dist = 4.8  # bad, start creating shortcuts
    # connect_dist = 5.0  # lol looks like a ladder

    print(f"Clustering produced {len(centroids)} centroids")

    # Build superset distance graph
    adjacency, edges = build_distance_graph(
        centroids,
        max_dist=connect_dist,
        return_edge_lengths=True,
    )

    print(f"Superset graph: {len(centroids)} nodes, {len(edges)} edges")

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    graph_pdb_path = os.path.join(
        result["output_dir"],
        # f"superset_graph_{timestamp}.pdb",
        f"superset_graph__dist_{connect_dist}.pdb",
    )

    save_graph_pdb(centroids, edges, graph_pdb_path)

    print(f"Saved superset graph PDB to {graph_pdb_path}")


    return {
        "centroids": centroids,
        "adjacency": adjacency,
        "edges": edges,
    }


if __name__ == "__main__":
    main()
