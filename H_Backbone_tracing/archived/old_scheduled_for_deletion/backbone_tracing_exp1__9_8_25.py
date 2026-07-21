import numpy as np
import math
import random

def load_pdb_coordinates(pdb_file, gerrorTesting=0, std_dev=0.0):
    """
    Reads a PDB file and returns an Nx3 numpy array of coordinates (in Å).
    
    Optionally, Gaussian noise can be added to the coordinates if gerrorTesting is 1.
    
    Parameters:
      pdb_file      : Path to the PDB file.
      gerrorTesting : (Optional) If set to 1, Gaussian noise is added. Default is 0.
      std_dev       : (Optional) Standard deviation for the Gaussian noise (in Å). Default is 0.0.
    
    Returns:
      coords : Nx3 numpy array of coordinates (with noise added if requested).
    """
    x_list, y_list, z_list = [], [], []

    with open(pdb_file, 'r') as f:
        for line in f:
            record = line[0:6].strip()
            if record in ("ATOM", "HETATM"):
                x_val = float(line[30:38])
                y_val = float(line[38:46])
                z_val = float(line[46:54])
                x_list.append(x_val)
                y_list.append(y_val)
                z_list.append(z_val)

    coords = np.column_stack((x_list, y_list, z_list))  # shape (N, 3)
    
    if gerrorTesting == 1:
        coords = gerror(coords, std_dev)
    
    return coords

##########################
# 1) Distance matrix
##########################

def build_distance_matrix(coords):
    """
    coords: Nx3 numpy array
    returns: NxN distance matrix (Å)
    """
    diff = coords[:, None, :] - coords[None, :, :]
    dist_matrix = np.linalg.norm(diff, axis=-1)
    return dist_matrix

##########################
# 2) Angle computation
##########################

def angle_degrees(v1, v2):
    """
    Computes angle in degrees between vectors v1 and v2
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))

def local_angle_sum(coords, route, idx):
    """
    Sum angles at idx-1, idx, idx+1 if valid
    Each angle is formed by (k-1, k, k+1)
    """
    total = 0.0
    n = len(route)
    for k in [idx-1, idx, idx+1]:
        if 0 < k < n-1:
            a, b, c = route[k-1], route[k], route[k+1]
            v1 = coords[a] - coords[b]
            v2 = coords[c] - coords[b]
            total += angle_degrees(v1, v2)
    return total

##########################
# 3) Route cost
##########################

def route_cost(route, dist_matrix):
    """
    Simple cyclic sum of distances
    """
    cost = np.sum([dist_matrix[route[k], route[k+1]] for k in range(len(route)-1)])
    cost += dist_matrix[route[-1], route[0]]
    return cost

##########################
# 4) Fast Monte Carlo (in-place swaps)
##########################

def mc_refine(route, coords, dist_matrix, n_iters=500, temperature=300.0, skip_threshold=10.0, ml_model=None):
    """
    Fast MC refinement optimized for ~20 points
    """
    kB = 0.001987
    beta = 1.0 / (kB * temperature)
    route = route[:]  # copy once
    best_route = route[:]
    best_cost = route_cost(route, dist_matrix)
    current_cost = best_cost
    n = len(route)

    for _ in range(n_iters):
        i, j = random.sample(range(n), 2)

        # Optionally use ML prediction
        if ml_model is not None:
            feats = extract_move_features(route, i, j, dist_matrix, coords)
            predicted_delta = ml_model.predict([feats])[0]
            if predicted_delta > skip_threshold:
                continue

        # Compute delta cost for swapped route (in-place swap)
        route[i], route[j] = route[j], route[i]

        # Only recompute distances affected by swap
        # Affected indices: i-1, i, i+1, j-1, j, j+1
        affected = set([i-1,i,i+1,j-1,j,j+1])
        new_cost = current_cost
        for k in affected:
            if 0 <= k < n-1:
                new_cost -= dist_matrix[best_route[k], best_route[k+1]]
                new_cost += dist_matrix[route[k], route[k+1]]
        # cyclic edges
        new_cost -= dist_matrix[best_route[-1], best_route[0]]
        new_cost += dist_matrix[route[-1], route[0]]

        delta = new_cost - current_cost
        if delta < 0 or random.random() < math.exp(-beta*delta):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_route = route[:]
        else:
            # undo swap
            route[i], route[j] = route[j], route[i]

    return best_route, best_cost

##########################
# 5) Simple greedy TSP initial route
##########################

def greedy_tsp(dist_matrix):
    """
    Start at node 0 and always go to nearest unvisited neighbor
    """
    n = len(dist_matrix)
    visited = [0]
    remaining = set(range(1,n))
    while remaining:
        last = visited[-1]
        next_node = min(remaining, key=lambda x: dist_matrix[last,x])
        visited.append(next_node)
        remaining.remove(next_node)
    return visited




def save_path_as_pdb(coords, path, pairs, output_file="Large_Error_PDBOUT.pdb"):
    with open(output_file, "w") as pdb_file:
        pdb_file.write("REMARK  TSP Path Saved as PDB\n")
        pdb_file.write("REMARK  Generated by RNA TSP Pipeline\n")

        atom_mapping = {idx: i + 1 for i, idx in enumerate(path)}

        for atom_number, idx in enumerate(path, start=1):
            x, y, z = coords[idx]
            pdb_file.write(
                f"ATOM  {atom_number:5d}  P   TSP A{atom_number:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           P\n"
            )

        pdb_file.write("REMARK  Connectivity\n")
        for A, B in pairs:
            if A in atom_mapping and B in atom_mapping:
                pdb_file.write(f"CONECT{atom_mapping[A]:5d}{atom_mapping[B]:5d}\n")

        pdb_file.write("TER\nEND\n")
    print(f"TSP path saved as {output_file}")


##########################
# Example usage
##########################

# if __name__ == "__main__":
#     coords = np.random.rand(20,3)*10  # 20 random C-alpha positions in Å
#     dist_matrix = build_distance_matrix(coords)
#     initial_route = greedy_tsp(dist_matrix)
#     refined_route, cost = mc_refine(initial_route, coords, dist_matrix, n_iters=500)
#     print("Initial route:", initial_route)
#     print("Refined route:", refined_route)
#     print("Cost:", cost)


if __name__ == "__main__":
    # 1) Load coordinates from PDB
    input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\emd_21887_agglomerative_cd_4.0_combined_largestCC_20250908_190055.pdb"
    coords = load_pdb_coordinates(input_file)  # Nx3 numpy array

    # 2) Build distance matrix
    dist_matrix = build_distance_matrix(coords)

    # 3) Compute initial greedy TSP route
    initial_route = greedy_tsp(dist_matrix)

    # 4) Refine route using Monte Carlo
    refined_route, cost = mc_refine(
        initial_route,
        coords,
        dist_matrix,
        n_iters=1000,         # increase if needed
        temperature=300.0
    )

    print("Refined route:", refined_route)
    print("Total cost:", cost)

    # 5) Optional: filter unrealistic bond lengths
    allowed_pairs = [(refined_route[i], refined_route[i+1]) 
                     for i in range(len(refined_route)-1) 
                     if np.linalg.norm(coords[refined_route[i]] - coords[refined_route[i+1]]) <= 6.0]

    # 6) Save traced backbone as PDB
    save_path_as_pdb(coords, refined_route, allowed_pairs, output_file="traced_backbone1.pdb")
