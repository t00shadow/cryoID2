"""
Wrapper module that imports the distance-based superset graph
and adds a degree-limiting post-processing step.

Supports skipping clustering by reading coordinates directly
from an already-clustered PDB file.
"""

import numpy as np
from collections import defaultdict


def limit_node_degree(coords, edges, max_degree=3):
    """
    Limit node degree by keeping only the shortest edges per node.
    """

    nbrs = defaultdict(list)

    for edge in edges:
        if len(edge) == 3:
            i, j, d = edge
        else:
            i, j = edge
            d = np.linalg.norm(coords[i] - coords[j])

        nbrs[i].append((j, d))
        nbrs[j].append((i, d))

    kept_edges = set()

    for i, lst in nbrs.items():
        lst = sorted(lst, key=lambda x: x[1])[:max_degree]
        for j, d in lst:
            if i < j:
                kept_edges.add((i, j, d))
            else:
                kept_edges.add((j, i, d))

    return sorted(kept_edges)


def load_coords_from_pdb(pdb_path, atom_name="CA"):
    """
    Read coordinates from a PDB file.

    Parameters
    ----------
    pdb_path : str
        Path to clustered PDB file
    atom_name : str or None
        Atom name to read (e.g. "CA").
        If None, read all ATOM/HETATM records.

    Returns
    -------
    coords : (N, 3) ndarray
    """

    coords = []

    with open(pdb_path) as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            if atom_name is not None:
                name = line[12:16].strip()
                if name != atom_name:
                    continue

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            coords.append((z, y, x))   # the rest of the code uses z,y,x. Trivially easy to change to x,y,z; just change this line

    if not coords:
        raise ValueError(f"No atoms read from {pdb_path}")

    return np.asarray(coords, dtype=np.float32)


def get_connected_components(n_nodes, edges):
    """
    Returns list of components, each as a list of node indices.
    """

    adj = [[] for _ in range(n_nodes)]

    for e in edges:
        i, j = e[:2]
        adj[i].append(j)
        adj[j].append(i)

    visited = [False] * n_nodes
    components = []

    for start in range(n_nodes):
        if visited[start]:
            continue

        stack = [start]
        comp = []

        while stack:
            node = stack.pop()
            if visited[node]:
                continue

            visited[node] = True
            comp.append(node)

            for nei in adj[node]:
                if not visited[nei]:
                    stack.append(nei)

        components.append(comp)

    return components


def save_components_as_pdbs(coords, edges, components, output_dir, prefix="component"):
    for i, comp in enumerate(components):
        comp_set = set(comp)

        comp_coords = coords[comp]

        # remap indices (IMPORTANT)
        index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(comp)}

        comp_edges = []
        for e in edges:
            a, b = e[:2]
            if a in comp_set and b in comp_set:
                comp_edges.append((index_map[a], index_map[b]))

        out_path = os.path.join(output_dir, f"{prefix}_{i}.pdb")
        save_graph_pdb(comp_coords, comp_edges, out_path)

# -------------------------------
# Example usage
# -------------------------------
if __name__ == "__main__":

    from R_A2_superset_graph import build_distance_graph, save_graph_pdb
    import os

    # -------------------------------
    # CONFIG
    # -------------------------------
    USE_CLUSTERED_PDB = True   # <-- toggle this
    # CLUSTERED_PDB_PATH = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\compare_dif_versions\real_malaria_1_agglomerative_cd_4.0_data_alone_20260106_204359.pdb"
    CLUSTERED_PDB_PATH = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\E1_D3_meanshift_bw_1.9_CA_only.pdb"
    # NOTE: TEMP FIX: circular, run this file first with above path, then masking script, then this script again with new path below
    # also have to go into R_A2_superset_graph.py and toggle the save_pdb fxn to use other write. needa update my util fxns
    #! EDIT: i think can just directly run masking script on clustered nodes, so just run each script once, first helix masking, then bb tracing, then residue sequence, then convert to fasta
    CLUSTERED_PDB_PATH = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\E1_D3_calpha_clustered_then_masked.pdb"
    # MAX_DIST = 4.4
    MAX_DIST = 5.0
    MAX_DEGREE = 2

    # -------------------------------
    # INPUT COORDS
    # -------------------------------
    if USE_CLUSTERED_PDB:
        centroids = load_coords_from_pdb(
            CLUSTERED_PDB_PATH,
            # atom_name="C",   # CA, C, or None if not CA-only
            atom_name="CA",   # CA, C, or None if not CA-only
        )
        output_dir = os.path.dirname(CLUSTERED_PDB_PATH)
    else:
        from R_A1_clustering_2stages_v2 import main as run_clustering
        result = run_clustering()
        centroids = result["centroids"]
        output_dir = result["output_dir"]

    # -------------------------------
    # Superset graph
    # -------------------------------
    adjacency, edges = build_distance_graph(
        centroids,
        max_dist=MAX_DIST,
        return_edge_lengths=True,
    )

    limited_edges = limit_node_degree(
        centroids,
        edges,
        max_degree=MAX_DEGREE,
    )

    out_pdb = os.path.join(
        output_dir,
        "superset_graph.pdb",
    )

    out_pdb2 = os.path.join(
        output_dir,
        "superset_graph_degree_limited.pdb",
    )

    save_graph_pdb(centroids, edges, out_pdb)
    save_graph_pdb(centroids, limited_edges, out_pdb2)

    print(f"Saved degree-limited graph (one giant connected component) to {out_pdb2}")

    print("Save connected components...")
    components = get_connected_components(len(centroids), limited_edges)

    # add component size filter HERE
    MIN_COMPONENT_SIZE = 18  # tune this
    components = [c for c in components if len(c) >= MIN_COMPONENT_SIZE]
    print(f"Kept {len(components)} components (size >= {MIN_COMPONENT_SIZE})")

    print(f"Found {len(components)} connected components")

    save_components_as_pdbs(
        coords=centroids,
        edges=limited_edges,
        components=components,
        # output_dir=output_dir+"\cc",
        output_dir=output_dir+"\cc_minsize",
        prefix="cc"
    )
