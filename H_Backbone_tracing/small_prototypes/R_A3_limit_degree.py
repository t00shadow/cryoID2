"""
Wrapper module that imports the distance-based superset graph
and adds a degree-limiting post-processing step.

Only new functionality: limit_node_degree
No changes to the original graph construction logic.
"""

import numpy as np
from collections import defaultdict


def limit_node_degree(
    coords,
    edges,
    max_degree=3,
):
    """
    Limit node degree by keeping only the shortest edges per node.

    Parameters
    ----------
    coords : (N, 3) ndarray
        Cα coordinates (Å)
    edges : list of (i, j) or (i, j, d)
        Undirected edges from superset graph
    max_degree : int
        Maximum allowed degree per node

    Returns
    -------
    new_edges : list of (i, j) or (i, j, d)
        Degree-limited edge list
    """

    # Build neighbor lists with distances
    nbrs = defaultdict(list)

    for edge in edges:
        if len(edge) == 3:
            i, j, d = edge
        else:
            i, j = edge
            d = np.linalg.norm(coords[i] - coords[j])

        nbrs[i].append((j, d))
        nbrs[j].append((i, d))

    # Keep shortest edges per node
    kept_edges = set()

    for i, lst in nbrs.items():
        lst = sorted(lst, key=lambda x: x[1])[:max_degree]
        for j, d in lst:
            if i < j:
                kept_edges.add((i, j, d))
            else:
                kept_edges.add((j, i, d))

    # Return edges in original format (with distances)
    return sorted(kept_edges)


# Example usage (optional)
if __name__ == "__main__":
    from R_A1_clustering import main as run_clustering
    # from R_A1_clustering_2stages_v2 import main as run_clustering
    # Import your existing superset graph code
    from R_A2_superset_graph import build_distance_graph, save_graph_pdb
    import os

    result = run_clustering()
    centroids = result["centroids"]

    adjacency, edges = build_distance_graph(
        centroids,
        max_dist=4.4,
        return_edge_lengths=True,
    )

    limited_edges = limit_node_degree(
        centroids,
        edges,
        max_degree=2,
    )

    out_pdb = os.path.join(
        result["output_dir"],
        "superset_graph_degree_limited.pdb",
    )

    save_graph_pdb(centroids, limited_edges, out_pdb)

    print(f"Saved degree-limited graph to {out_pdb}")
