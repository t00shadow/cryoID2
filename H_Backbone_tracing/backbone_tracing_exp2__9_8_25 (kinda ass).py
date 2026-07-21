import numpy as np
import math

###############################################################################
# 1) Utility functions
###############################################################################

def load_pdb_coordinates(pdb_file):
    """Load all ATOM coordinates as Nx3 array (C-alpha only)."""
    x_list, y_list, z_list = [], [], []

    with open(pdb_file, 'r') as f:
        for line in f:
            record = line[0:6].strip()
            if record == "ATOM" or "HETATM":
                atom_name = line[12:16].strip()
                if atom_name == "C":
                    x_list.append(float(line[30:38]))
                    y_list.append(float(line[38:46]))
                    z_list.append(float(line[46:54]))

    coords = np.column_stack((x_list, y_list, z_list))
    return coords

def angle_between(v1, v2):
    """Returns the angle in degrees between vectors v1 and v2"""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 180.0  # treat degenerate as straight
    cos_theta = np.dot(v1, v2) / (norm1 * norm2)
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    return math.degrees(math.acos(cos_theta))

###############################################################################
# 2) Helical / linear biased backbone tracer
###############################################################################

def helical_backbone_tracer(coords, start_idx=0, max_angle_deviation=120.0, helix_bias_strength=0.5):
    """
    Trace a mostly linear or helical backbone from C-alpha coordinates.

    Parameters
    ----------
    coords : (N,3) numpy array
    start_idx : int, starting point index
    max_angle_deviation : float, max allowed deviation from previous vector (deg)
    helix_bias_strength : float [0,1], how strongly to favor a small spiral

    Returns
    -------
    route : list of indices
    """
    N = len(coords)
    remaining = set(range(N))
    remaining.remove(start_idx)
    route = [start_idx]

    # initial vector: random direction if first point
    prev_vec = np.array([1.0, 0.0, 0.0])

    while remaining:
        last_idx = route[-1]
        last_pos = coords[last_idx]

        best_candidate = None
        best_score = float('inf')

        for candidate_idx in remaining:
            candidate_pos = coords[candidate_idx]
            vec = candidate_pos - last_pos
            dist = np.linalg.norm(vec)
            angle = angle_between(prev_vec, vec)

            # helical bias: slightly prefer vectors rotated ~90 deg from prev_vec
            # small rotation bias to create gentle spiral
            helical_penalty = helix_bias_strength * abs(angle - 100.0)  # 100 deg ~ typical alpha-helix
            # total score: distance + angle penalty
            score = dist + helical_penalty

            # enforce max angle deviation strictly
            if angle <= max_angle_deviation and score < best_score:
                best_score = score
                best_candidate = candidate_idx

        # fallback: no candidate satisfies angle -> pick closest
        if best_candidate is None:
            best_candidate = min(remaining, key=lambda idx: np.linalg.norm(coords[idx] - last_pos))

        route.append(best_candidate)
        remaining.remove(best_candidate)
        prev_vec = coords[best_candidate] - last_pos

    return route

###############################################################################
# 3) Filter pairs by bond length
###############################################################################

def filter_route_by_bond_length(route, coords, max_bond_length=6.0):
    allowed_pairs = []
    for i in range(len(route)-1):
        A, B = route[i], route[i+1]
        dist = np.linalg.norm(coords[A] - coords[B])
        if dist <= max_bond_length:
            allowed_pairs.append((A, B))
    return allowed_pairs

###############################################################################
# 4) Save route as PDB
###############################################################################

def save_path_as_pdb(coords, route, pairs, output_file="backbone.pdb"):
    with open(output_file, "w") as pdb_file:
        pdb_file.write("REMARK  Helical/Linear backbone tracing\n")
        for atom_number, idx in enumerate(route, start=1):
            x, y, z = coords[idx]
            pdb_file.write(
                f"ATOM  {atom_number:5d}  CA  RES A{atom_number:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )

        pdb_file.write("REMARK  Connectivity\n")
        atom_mapping = {idx: i+1 for i, idx in enumerate(route)}
        for A, B in pairs:
            if A in atom_mapping and B in atom_mapping:
                pdb_file.write(f"CONECT{atom_mapping[A]:5d}{atom_mapping[B]:5d}\n")

        pdb_file.write("TER\nEND\n")
    print(f"Saved backbone to {output_file}")

###############################################################################
# 5) Example usage
###############################################################################

if __name__ == "__main__":
    input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\emd_21887_agglomerative_cd_4.0_combined_largestCC_20250908_190055.pdb"
    coords = load_pdb_coordinates(input_file)

    route = helical_backbone_tracer(
        coords,
        start_idx=0,
        max_angle_deviation=130.0,  # slightly relaxed
        helix_bias_strength=0.3     # gentle helix
    )

    allowed_pairs = filter_route_by_bond_length(route, coords, max_bond_length=6.0)
    save_path_as_pdb(coords, route, allowed_pairs, output_file="traced_backbone2.pdb")
