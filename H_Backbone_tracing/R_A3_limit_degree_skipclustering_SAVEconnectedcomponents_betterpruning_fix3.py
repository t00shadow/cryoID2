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
    superset_edges = set()

    for e in edges:
        if len(e) == 3:
            i, j, d = e
        else:
            i, j = e
        edge_list.append((i, j))
        superset_edges.add((min(i, j), max(i, j)))    # avoids dupes, assuming each node has a unique id

    # score all edges using BOTH endpoints implicitly
    scored = []
    edge_scores = {}

    for i, j in edge_list:
        score = edge_density_score(coords, em_voxels, i, j, radius, metric="normalized_by_length")
        scored.append((score, i, j))
        edge_scores[(i, j)] = score

    # sort by density support
    scored.sort(reverse=True)

    degree = defaultdict(int)
    kept = []

    for score, i, j in scored:
        if degree[i] < max_degree and degree[j] < max_degree:
            kept.append((i, j))
            degree[i] += 1
            degree[j] += 1

    # print(len(kept), len(superset_edges), len(edge_scores))
    return kept, superset_edges, edge_scores


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



from collections import defaultdict, deque
import copy


def repair_cycles(
    n_nodes,
    edges,
    superset_edges,
    edge_scores,
    max_degree=2,
    continuity_weight=0.25,  # 0.25
    continuity_depth=8,   # 4 might be too small, changed to 8, maybe increase continuity_weight too. make figure first, then crank this up to like 40, tried 20 and same result, might need to loosen the density improvement constraint
):

    def norm(e):
        a, b = e
        return (a, b) if a < b else (b, a)

    current_edges = {norm(e) for e in edges}
    superset_edges = {norm(e) for e in superset_edges}

    def build_adj(edge_set):
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def degrees(edge_set):
        deg = defaultdict(int)
        for u, v in edge_set:
            deg[u] += 1
            deg[v] += 1
        return deg

    def graph_density_score(edge_set):
        return sum(edge_scores.get(e, 0.0) for e in edge_set)

    def path_exists(adj, start, goal):
        if start == goal:
            return True

        q = deque([start])
        visited = {start}

        while q:
            node = q.popleft()

            for nbr in adj[node]:
                if nbr == goal:
                    return True

                if nbr not in visited:
                    visited.add(nbr)
                    q.append(nbr)

        return False

    def local_chain_length(adj, start, max_depth=4):
        """
        Count reachable nodes within a few hops.

        We deliberately cap this because:
        - 10 vs 12 residues isn't very important
        - distinguishes tiny fragments from long traces
        """

        q = deque([(start, 0)])
        visited = {start}
        count = 0

        while q:
            node, depth = q.popleft()

            if depth >= max_depth:
                continue

            for nbr in adj[node]:
                if nbr not in visited:
                    visited.add(nbr)
                    count += 1
                    q.append((nbr, depth + 1))

        return count

    def continuity_bonus(adj, added_edges):
        """
        Reward repairs that attach to longer chains.
        """

        bonus = 0

        for u, v in added_edges:
            bonus += local_chain_length(adj, u, max_depth=continuity_depth)
            bonus += local_chain_length(adj, v, max_depth=continuity_depth)

            # bonus += min(local_chain_length(adj, u, max_depth=continuity_depth), continuity_depth)
            # bonus += min(local_chain_length(adj, v, max_depth=continuity_depth), continuity_depth)
            print("u bonus:", local_chain_length(adj, u, max_depth=continuity_depth)); print("v bonus:", local_chain_length(adj, v, max_depth=continuity_depth))

        return bonus

    def find_cycle(edge_set):
        adj = build_adj(edge_set)
        deg = degrees(edge_set)

        visited = set()

        for start in range(n_nodes):

            if start in visited:
                continue

            stack = [start]
            comp_nodes = []

            while stack:
                node = stack.pop()

                if node in visited:
                    continue

                visited.add(node)
                comp_nodes.append(node)

                for nbr in adj[node]:
                    if nbr not in visited:
                        stack.append(nbr)

            if len(comp_nodes) > 2 and all(deg[n] == 2 for n in comp_nodes):

                cycle_edges = set()

                for n in comp_nodes:
                    for nbr in adj[n]:
                        cycle_edges.add(norm((n, nbr)))

                return list(cycle_edges)

        return None

    # # --------------------------------------------------
    # # MAIN LOOP
    # # --------------------------------------------------

    # while True:

    #     cycle_edges = find_cycle(current_edges)

    #     if cycle_edges is None:
    #         print("NO CYCLE FOUND")
    #         break
    #     print("FOUND CYCLE:", len(cycle_edges))

    #     base_adj = build_adj(current_edges)
    #     base_deg = degrees(current_edges)

    #     current_density = graph_density_score(current_edges)

    #     best_candidate = None
    #     best_repair_score = -np.inf

    #     # --------------------------------------------------
    #     # Try breaking EVERY edge in the cycle
    #     # --------------------------------------------------

    #     print("Trying cycle:", cycle_edges)
    #     for removed_edge in cycle_edges:

    #         trial = set(current_edges)
    #         trial.remove(removed_edge)

    #         deg = base_deg.copy()
    #         adj = copy.deepcopy(base_adj)

    #         u, v = removed_edge

    #         adj[u].remove(v)
    #         adj[v].remove(u)

    #         deg[u] -= 1
    #         deg[v] -= 1

    #         freed_nodes = (u, v)

    #         added_edges = []

    #         # ----------------------------------------------
    #         # Candidate replacement edges
    #         # ----------------------------------------------

    #         candidates = []

    #         for e in superset_edges:

    #             if e in trial:
    #                 continue

    #             x, y = e

    #             if x in freed_nodes or y in freed_nodes:
    #                 candidates.append(e)

    #         candidates.sort(
    #             key=lambda e: edge_scores.get(e, 0.0),
    #             reverse=True,
    #         )

    #         # ----------------------------------------------
    #         # Greedy refill
    #         # ----------------------------------------------

    #         for x, y in candidates:

    #             if deg[x] >= max_degree:
    #                 continue

    #             if deg[y] >= max_degree:
    #                 continue

    #             if path_exists(adj, x, y):
    #                 continue

    #             trial.add((x, y))

    #             adj[x].add(y)
    #             adj[y].add(x)

    #             deg[x] += 1
    #             deg[y] += 1

    #             added_edges.append((x, y))

    #         if not added_edges:
    #             print("no replacement edges")
    #             continue

    #         density_score = graph_density_score(trial)

    #         continuity_score = continuity_bonus(
    #             adj,
    #             added_edges,
    #         )

    #         repair_score = (
    #             density_score
    #             + continuity_weight * continuity_score
    #         )

    #         # Require at least no density regression
    #         if density_score < current_density:
    #             print("density regression", density_score - current_density)
    #             continue

    #         if repair_score > best_repair_score:
    #             best_repair_score = repair_score
    #             best_candidate = trial

    #         print("  break", removed_edge,
    #             "added", len(added_edges),
    #             "density", density_score,
    #             "density regression", density_score - current_density,
    #             "repair", repair_score)

    #     # --------------------------------------------------
    #     # No valid repair found
    #     # --------------------------------------------------

    #     if best_candidate is None:
    #         print("FAILED TO REPAIR CYCLE")
    #         break

    #     current_edges = best_candidate


    # --------------------------------------------------
    # NEW MAIN LOOP
    # --------------------------------------------------
    from itertools import combinations
    from math import inf

    while True:

        cycle_edges = find_cycle(current_edges)

        if cycle_edges is None:
            break

        base_adj = build_adj(current_edges)
        base_deg = degrees(current_edges)

        current_density = graph_density_score(current_edges)

        best_candidate = None
        best_repair_score = -inf

        cycle_edges = list(cycle_edges)

        # --------------------------------------------------
        # progressively try removing MORE edges from cycle
        # --------------------------------------------------
        max_remove = 3  # you can tune this (2 is usually enough)

        found_any_valid = False

        for k in range(1, max_remove + 1):

            if found_any_valid:
                break

            for removed_edges in combinations(cycle_edges, k):

                trial = set(current_edges)

                # adj = build_adj(current_edges)
                # deg = degrees(current_edges)
                deg = base_deg.copy()
                adj = copy.deepcopy(base_adj)

                freed_nodes = set()

                # -------------------------
                # remove k edges
                # -------------------------
                for u, v in removed_edges:
                    e = (u, v) if u < v else (v, u)

                    if e in trial:
                        trial.remove(e)

                    adj[u].discard(v)
                    adj[v].discard(u)

                    deg[u] -= 1
                    deg[v] -= 1

                    freed_nodes.add(u)
                    freed_nodes.add(v)

                # --------------------------------------------------
                # candidate edges (same logic as before)
                # --------------------------------------------------
                candidates = []
                for x, y in superset_edges:
                    if (x, y) in trial or (y, x) in trial:
                        continue
                    if x in freed_nodes or y in freed_nodes:
                        candidates.append((x, y))

                candidates.sort(
                    key=lambda e: edge_scores.get(e, 0.0),
                    reverse=True,
                )

                added_edges = []

                # --------------------------------------------------
                # greedy refill
                # --------------------------------------------------
                for x, y in candidates:

                    if deg[x] >= max_degree or deg[y] >= max_degree:
                        continue

                    if path_exists(adj, x, y):
                        continue

                    trial.add((x, y))
                    adj[x].add(y)
                    adj[y].add(x)

                    deg[x] += 1
                    deg[y] += 1

                    added_edges.append((x, y))

                # --------------------------------------------------
                # allow "pure deletion repair"
                # --------------------------------------------------
                density_score = graph_density_score(trial)

                continuity_score = continuity_bonus(adj, added_edges)

                repair_score = (
                    density_score
                    + continuity_weight * continuity_score
                )

                # accept even if no refill edges exist (IMPORTANT CHANGE)
                if density_score >= current_density or k > 1:

                    if repair_score > best_repair_score:
                        best_repair_score = repair_score
                        best_candidate = trial
                        found_any_valid = True

            # end combinations loop

        # --------------------------------------------------
        # no repair found at all
        # --------------------------------------------------
        if best_candidate is None:
            break

        current_edges = best_candidate

    return sorted(current_edges)

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
    limited_edges, superset_edges, edge_scores = limit_edges_global(
        centroids,
        edges,
        em_voxels=em_voxels,
        radius=guardrail_cylinder_radius,
        max_degree=max_degree,
    )

    print(f"Degree-limited edges: {len(limited_edges)}")
    print(f"Superset edges: {len(superset_edges)}")
    print(f"Stored edge scores: {len(edge_scores)}")
    # the last two should satisfy len(edge_scores) == len(superset_edges) btw

    limited_edges = repair_cycles(
        n_nodes=len(centroids),
        edges=limited_edges,
        superset_edges=superset_edges,
        edge_scores=edge_scores,
        max_degree=2,
    )

    # Check for cycles
    # If NetworkX reports a cycle but code prints NO CYCLE FOUND, then bug is: find_cycle() is too restrictive
    # But if NetWorkX reports a cycle and code prints there is a cycle found, then the bug if not cycle detection but rather somewhere in the repairing
    import networkx as nx    # TODO: move this import up if keep this
    G = nx.Graph()
    G.add_edges_from(limited_edges)
    print(nx.cycle_basis(G))

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
    # no masking needed for RNA prior to tracing btw
    # ca_clustered_pdb_path = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\clustering_figure\emd_52422_meanshift_bw_1.9_CA_only.pdb"
    # guardrail_scaffold_file=r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\TestSet\Raw\EMD-52422_re_3.07\possibility\emd_52422.map_secondary_strctures_1_possibility.mrc"
    ca_clustered_pdb_path = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\files_for_grayson\clustering\E1_D3_meanshift_bw_1.9_CA_only.pdb"
    guardrail_scaffold_file=r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\real_malaria\3-E1_D3\possibility\E1_D3_2_7A_cryosparc_P8_J562_008_volume_map_sharp_secondary_strctures_1_possibility.mrc"
    guardrail_threshold=0.97
    guardrail_cylinder_radius=1.0    # dont need a radius as large as 2
    # TODO create dictionaries or classes so can toggle between dif versions just by toggling a single thing  (i think this is referering to using same tracer for RNA and protein?)
    max_dist = 5.0   # CA
    max_degree = 2
    min_component_size = 18  # tune this
    ### graph_output_dir is OPTIONAL, if not given, will default to dir of clustered_pdb_path
    # graph_output_dir = r"C:\Users\noelu\Downloads\paper_figure_assets\fig1 emd_52422 chimerax images\data"
    graph_output_dir = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\files_for_grayson\2026_06_29_malaria_rerun"
    components_output_dir = graph_output_dir + "\cc_minsize"
    os.makedirs(components_output_dir, exist_ok=True)
    

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