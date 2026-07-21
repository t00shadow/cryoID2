"""
Wrapper module that imports the distance-based superset graph
and adds a degree-limiting post-processing step.

Supports skipping clustering by reading coordinates directly
from an already-clustered PDB file.
"""

import numpy as np
from collections import defaultdict
import os
from H_Backbone_tracing.R_A2_superset_graph import build_distance_graph, save_graph_pdb


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


def main(use_clustered_pdb, clustered_pdb_path, max_dist, max_degree, min_component_size, graph_output_dir, components_output_dir, verbose=False):
    # -------------------------------
    # INPUT COORDS
    # -------------------------------
    if use_clustered_pdb:
        centroids = load_coords_from_pdb(
            clustered_pdb_path,
            # atom_name="C",   # CA, C, or None if not CA-only
            atom_name="CA",   # CA, C, or None if not CA-only
        )
        graph_file_output_dir = os.path.dirname(clustered_pdb_path)
    else:
        from H_Backbone_tracing.archived.still_relevant_but_dif_versions.R_A1_clustering_2stages_v2 import main as run_clustering
        result = run_clustering()
        centroids = result["centroids"]
        graph_file_output_dir = result["output_dir"]

    # Overwrite graph file output_dir if one is given
    if graph_output_dir:
        graph_file_output_dir = graph_output_dir

    # -------------------------------
    # Superset graph
    # -------------------------------
    adjacency, edges = build_distance_graph(
        centroids,
        max_dist=max_dist,
        return_edge_lengths=True,
    )

    limited_edges = limit_node_degree(
        centroids,
        edges,
        max_degree=max_degree,
    )

    out_pdb = os.path.join(
        graph_file_output_dir,
        "superset_graph.pdb",
    )

    out_pdb2 = os.path.join(
        graph_file_output_dir,
        "superset_graph_degree_limited.pdb",
    )

    save_graph_pdb(centroids, edges, out_pdb)
    save_graph_pdb(centroids, limited_edges, out_pdb2)

    if verbose:
        print(f"Saved degree-limited graph (one giant connected component) to {out_pdb2}")
        print("Save connected components...")

    components = get_connected_components(len(centroids), limited_edges)

    if verbose:
        print(f"Found {len(components)} connected components")

    # Component size filter
    components = [c for c in components if len(c) >= min_component_size]

    if verbose:
        print(f"Kept {len(components)} components (size >= {min_component_size})")

    save_components_as_pdbs(
        coords=centroids,
        edges=limited_edges,
        components=components,
        output_dir=components_output_dir,
        prefix="cc"
    )


if __name__ == "__main__":
    # -------------------------------
    # CONFIG
    # -------------------------------
    use_clustered_pdb = True   # <-- toggle this
    # clustered_pdb_path = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\compare_dif_versions\real_malaria_1_agglomerative_cd_4.0_data_alone_20260106_204359.pdb"
    clustered_pdb_path = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\E1_D3_meanshift_bw_1.9_CA_only.pdb"
    # NOTE: TEMP FIX: circular, run this file first with above path, then masking script, then this script again with new path below
    # also have to go into R_A2_superset_graph.py and toggle the save_pdb fxn to use other write. needa update my util fxns
    #! EDIT: i think can just directly run masking script on clustered nodes, so just run each script once, first helix masking, then bb tracing, then residue sequence, then convert to fasta
    clustered_pdb_path = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\E1_D3_calpha_clustered_then_masked.pdb"
    # max_dist = 4.4
    max_dist = 5.0
    max_degree = 2
    min_component_size = 18  # tune this
    ### graph_output_dir is OPTIONAL, if not given, will default to dir of clustered_pdb_path
    graph_output_dir = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs"
    # components_output_dir=graph_output_dir + "\cc"
    components_output_dir = graph_output_dir + "\cc_minsize"
    

    # TODO: might want to 1-index the cc names instead of 0-index

    main(
        use_clustered_pdb=use_clustered_pdb,
        clustered_pdb_path=clustered_pdb_path,
        max_dist=max_dist,
        max_degree=max_degree,
        min_component_size=min_component_size,
        graph_output_dir=graph_output_dir,
        components_output_dir=components_output_dir,
        verbose=True
    )