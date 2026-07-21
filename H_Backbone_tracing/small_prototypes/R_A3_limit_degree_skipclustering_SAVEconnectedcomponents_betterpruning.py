"""
Wrapper module that imports the distance-based superset graph
and adds a degree-limiting post-processing step.

Supports skipping clustering by reading coordinates directly
from an already-clustered PDB file.
"""
# TODO: pasted in from the RNA side, the older version still leaves loops/cycles for some reason?

import numpy as np
from collections import defaultdict
import os
from R_A2_superset_graph import build_distance_graph, save_graph_pdb
from R_A0_utils import load_thresholded_coords
from R_A8_refactor2_A1_clustering import load_volume_metadata


def point_to_segment_distance(p, a, b):
    """
    Distance from point p to segment a-b.
    """
    ab = b - a
    ap = p - a

    denom = np.dot(ab, ab)
    if denom == 0:
        return np.linalg.norm(p - a)

    t = np.dot(ap, ab) / denom
    t = np.clip(t, 0.0, 1.0)

    closest = a + t * ab
    return np.linalg.norm(p - closest)


def cylinder_voxel_count(voxels, a, b, radius):
    """
    Count voxels within a cylinder defined by segment a-b.
    """
    count = 0

    for p in voxels:
        if point_to_segment_distance(p, a, b) <= radius:
            count += 1

    return count


def cylinder_voxel_density(voxels, a, b, radius):
    """
    Returns voxel density inside cylinder:
    count / cylinder volume
    """

    ab = b - a
    L = np.linalg.norm(ab)
    if L == 0:
        return 0

    voxel_count = 0

    for p in voxels:
        # distance to segment axis
        ap = p - a
        ab_unit = ab / L

        t = np.dot(ap, ab_unit)
        if t < 0 or t > L:
            continue  # outside segment bounds

        closest = a + t * ab_unit
        dist = np.linalg.norm(p - closest)

        if dist <= radius:
            voxel_count += 1

    # cylinder volume
    volume = np.pi * radius**2 * L

    return voxel_count / volume


# def limit_node_degree(coords, edges, em_voxels, radius=2.0, max_degree=2):
#     """
#     At each branching node, keep edges that pass through highest EM density.
#     """

#     if max_degree != 2:
#         raise NotImplementedError("This version assumes max_degree=2.")

#     # Normalize edges
#     edge_dict = {}
#     for e in edges:
#         if len(e) == 3:
#             i, j, d = e
#         else:
#             i, j = e
#             d = np.linalg.norm(coords[i] - coords[j])

#         if i > j:
#             i, j = j, i

#         edge_dict[(i, j)] = (i, j, d)

#     edge_list = list(edge_dict.values())

#     def build_adj(edges):
#         adj = defaultdict(list)
#         for i, j, _ in edges:
#             adj[i].append(j)
#             adj[j].append(i)
#         return adj

#     changed = True

#     while changed:
#         changed = False
#         adj = build_adj(edge_list)

#         for node, neighbors in adj.items():
#             if len(neighbors) <= 2:
#                 continue

#             scores = []

#             for nbr in neighbors:
#                 a = coords[node]
#                 b = coords[nbr]

#                 #? seems count does better? should be fine once i combine phosphate and sugar nodes, then the scaffold/guardrail will work better
#                 score = cylinder_voxel_count(em_voxels, a, b, radius)
#                 # score = cylinder_voxel_density(em_voxels, a, b, radius)
#                 scores.append((score, nbr))

#             # keep top-2 highest density edges
#             scores.sort(reverse=True, key=lambda x: x[0])
#             keep = set(nbr for _, nbr in scores[:2])

#             new_edges = []
#             for i, j, d in edge_list:
#                 if i == node and j not in keep:
#                     continue
#                 if j == node and i not in keep:
#                     continue
#                 new_edges.append((i, j, d))

#             if len(new_edges) != len(edge_list):
#                 edge_list = new_edges
#                 changed = True
#                 break

#     return sorted(edge_list)


# helper fxn
# TODO: can rewrite to use cylinder_voxel_count or cylinder_voxel_density. OR just pull out the inner fxn count()
def edge_density_score(coords, em_voxels, i, j, radius, metric="raw_count"):
    a, b = coords[i], coords[j]

    def count(p):
        ab = b - a
        L = np.linalg.norm(ab)
        if L == 0:
            return 0

        ab_unit = ab / L
        ap = p - a

        t = np.dot(ap, ab_unit)
        if t < 0 or t > L:
            return 0

        closest = a + t * ab_unit
        return np.linalg.norm(p - closest) <= radius

    if metric == "raw_count":
        return sum(count(p) for p in em_voxels)
    elif metric == "normalized_by_length":
        L = np.linalg.norm(b - a)
        return sum(count(p) for p in em_voxels) / L
    ### redundant, normalizing by volume is the same as length since radius stays the same (set once when u first run the script, but same all throughout the same run), becomes a constant factor
    # elif metric == "normalized_by_volume":
    #     L = np.linalg.norm(b - a)
    #     volume_proxy = radius**2 * L   # dont need pi since comparing relatively, so pi will cancel out
    #     return sum(count(p) for p in em_voxels) / volume_proxy
    else:
        raise ValueError("metric to be 'raw_count' or 'normalized_by_length'")


# this version is better logically since the prev version can remove too many edges if u have a triangle/small cycle,
# since each point in the triangle is a "branch point" so theyre competing with each other to remove the same edges.
# in the triangle scenario, the optimal solution is to remove just 1 edge, but sometimes 2 are removed
def limit_edges_global(coords, edges, em_voxels, radius=2.0, max_degree=2):
    edge_list = []
    
    for e in edges:
        if len(e) == 3:
            i, j, d = e
        else:
            i, j = e
        edge_list.append((i, j))

    # score all edges using BOTH endpoints implicitly
    scored = []

    for i, j in edge_list:
        score = edge_density_score(coords, em_voxels, i, j, radius, metric="normalized_by_length")
        scored.append((score, i, j))

    # sort by density support
    scored.sort(reverse=True)

    degree = defaultdict(int)
    kept = []

    for score, i, j in scored:
        if degree[i] < max_degree and degree[j] < max_degree:
            kept.append((i, j))
            degree[i] += 1
            degree[j] += 1

    return kept


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


def main(use_clustered_pdb, ca_clustered_pdb_path, guardrail_scaffold_file, guardrail_threshold, guardrail_cylinder_radius, max_dist, max_degree, min_component_size, graph_output_dir, components_output_dir, verbose=False):
    # -------------------------------
    # INPUT COORDS
    # -------------------------------
    if use_clustered_pdb:
        centroids = load_coords_from_pdb(
            ca_clustered_pdb_path,
            # atom_name="C",   # CA, C, or None if not CA-only
            atom_name="CA",   # CA, C, or None if not CA-only
        )
        # print(centroids); assert(False)
        # print(np.shape(centroids)); assert(False)
        graph_file_output_dir = os.path.dirname(ca_clustered_pdb_path)
    else:
        raise Exception("This is not safe to run on the new RNA version of this script")
        # from R_A1_clustering_2stages_v2 import main as run_clustering
        # result = run_clustering()
        # centroids = result["centroids"]
        # graph_file_output_dir = result["output_dir"]

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

    # read in cryoEM map mrc (normalized mrc or secondary_structure_4.mrc) for guardrail
    em_voxels = load_thresholded_coords(path=guardrail_scaffold_file, threshold=guardrail_threshold)

    voxel_size, _ = load_volume_metadata(guardrail_scaffold_file)

    em_voxels_real = np.empty_like(em_voxels, dtype=float)
    em_voxels_real[:, 0] = em_voxels[:, 2] * voxel_size
    em_voxels_real[:, 1] = em_voxels[:, 1] * voxel_size
    em_voxels_real[:, 2] = em_voxels[:, 0] * voxel_size

    # limited_edges = limit_node_degree(
    limited_edges = limit_edges_global(
        centroids,
        edges,
        em_voxels=em_voxels,
        radius=guardrail_cylinder_radius,
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
    # no masking needed prior to this tracer btw, b/c this one does a soft mask (more like a soft edge-priority scoring map) using the guardrail file (hard mask would be too limiting since Unet helix prediction is far from perfect)
    ca_clustered_pdb_path = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\clustering_figure\emd_52422_meanshift_bw_1.9_CA_only.pdb"
    guardrail_scaffold_file=r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\TestSet\Raw\EMD-52422_re_3.07\possibility\emd_52422.map_secondary_strctures_1_possibility.mrc"
    guardrail_threshold=0.97
    guardrail_cylinder_radius=1.0    # dont need a radius as large as 2
    # TODO create dictionaries or classes so can toggle between dif versions just by toggling a single thing  (i think this is referering to using same tracer for RNA and protein? would need to change more things tho since RNA both sugar and phos vs just CA for protein, tho ends up combining sugar and phos immediately, so could pull that logic outside of the main fxn, and then could truly make them use the same fxn)
    max_dist = 5.0   # CA
    max_degree = 2
    min_component_size = 18  # tune this
    ### graph_output_dir is OPTIONAL, if not given, will default to dir of clustered_pdb_path
    graph_output_dir = r"C:\Users\noelu\Downloads\paper_figure_assets\fig1 emd_52422 chimerax images\data"
    components_output_dir = graph_output_dir + "\cc_minsize"
    os.mkdir(components_output_dir)
    

    # TODO: might want to 1-index the cc names instead of 0-index

    main(
        use_clustered_pdb=use_clustered_pdb,
        ca_clustered_pdb_path=ca_clustered_pdb_path,
        guardrail_scaffold_file=guardrail_scaffold_file,
        guardrail_threshold=guardrail_threshold,
        guardrail_cylinder_radius=guardrail_cylinder_radius,
        max_dist=max_dist,
        max_degree=max_degree,
        min_component_size=min_component_size,
        graph_output_dir=graph_output_dir,
        components_output_dir=components_output_dir,
        verbose=True
    )