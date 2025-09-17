import numpy as np
import math

def load_ca_coordinates(pdb_file):
    """Load only C-alpha atoms from a PDB."""
    coords = []
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith("HETATM") and line[13:15].strip() == "C":
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append([x, y, z])
    return np.array(coords)

def distance(a, b):
    """Euclidean distance."""
    return np.linalg.norm(a - b)

def helical_backbone_tracer(coords, min_bond=3.0, max_bond=4.5, beam_width=10):
    """
    Trace a continuous backbone including all points using beam search.

    Returns:
        route: list of indices in order.
    """
    if len(coords) == 0:
        raise ValueError("No coordinates provided.")

    N = len(coords)
    remaining_points = set(range(N))

    # Precompute neighbors within bond range
    neighbors = [set() for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                d = distance(coords[i], coords[j])
                if min_bond <= d <= max_bond:
                    neighbors[i].add(j)

    # Beam search paths: each item is (path_list, used_points_set)
    beams = []
    for start in range(N):
        beams.append(([start], {start}))

    final_paths = []

    while beams:
        new_beams = []
        for path, used in beams:
            last = path[-1]
            available_neighbors = neighbors[last] - used
            if not available_neighbors:
                # if len(path) == N:
                # if len(path) >= N-1:
                if len(path) >= min(20, round(N * 0.95)):
                    final_paths.append(path)
                continue
            # Extend to each neighbor
            for n in available_neighbors:
                new_path = path + [n]
                new_used = used | {n}
                new_beams.append((new_path, new_used))

        # Keep only top beam_width paths
        beams = sorted(new_beams, key=lambda x: len(x[0]), reverse=True)[:beam_width]

    if not final_paths:
        raise RuntimeError("No valid continuous backbone found.")

    # Choose the longest path (all points)
    best_path = max(final_paths, key=lambda x: len(x))
    if len(best_path) != N:
        print(f"Warning: only traced {len(best_path)}/{N} points")
    return best_path

def save_path_as_pdb(coords, route, output_file="traced_backbone.pdb"):
    """Save traced backbone as PDB using CA atoms only."""
    with open(output_file, "w") as f:
        f.write("REMARK Traced backbone\n")
        for idx, i in enumerate(route, start=1):
            x, y, z = coords[i]
            f.write(
                f"ATOM  {idx:5d}  CA  ALA A{idx:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
        f.write("TER\nEND\n")
    print(f"Saved traced backbone to {output_file}")

if __name__ == "__main__":
    # input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\emd_21887_agglomerative_cd_4.0_combined_largestCC_20250908_190055.pdb"
    input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\emd_34843_agglomerative_cd_4.0_combined_largestCC_20250916_202639.pdb"
    output_file = r"traced_backbone3.pdb"

    coords = load_ca_coordinates(input_file)
    print(f"Loaded {len(coords)} C-alpha atoms from {input_file}")

    route = helical_backbone_tracer(coords, min_bond=3.0, max_bond=4.5, beam_width=20)
    print("Traced route:", route)

    save_path_as_pdb(coords, route, output_file=output_file)
