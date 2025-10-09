# same as A2_backbone_tracing_ANGLES_TOO.py except modfied helical_backbone_tracer() so it tracks multiple best paths per starting point instead of global beam pruning -> helical_backbone_tracer_per_start()
# without looking at the details, this shoullddddd be better? since it explores more. Can also compare this approach to simply increasing beamwidth in original code
# EDIT: results are mostly the same, traced exact same amt of regions (22 for emd_38402), and most of them were the same. 1 is noticeabily dif but that region is just bad since even the newer smoother one is wrong (connected where theres a gap). another 2 are slightly dif, one terminates early in the old one, another bends in the wrong direction at the end. a few you have to zoom in to even see the dif
# so in short since most results are the same (so nothing got worse... need to check each one by one prob to verify this, but it is intuitive- therefore keep each version of this code in my archive) but some are improved sooo use THIS file (<- tldr)
# TODO: rename this file to smth better

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

def bond_angle(a, b, c):
    """Return the angle (in degrees) between vectors AB and BC."""
    v1 = a - b
    v2 = c - b
    dot = np.dot(v1, v2)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)
    if denom == 0:
        return 180.0
    cos_theta = np.clip(dot / denom, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def is_too_close(coords, path, candidate_idx, cutoff=3.0):
    """
    Check if the candidate coordinate is too close to any non-adjacent atom.
    Prevents circular self-intersection.
    """
    cand = coords[candidate_idx]
    for i in path[:-2]:  # ignore immediate neighbors
        if distance(coords[i], cand) < cutoff:
            return True
    return False

def helical_backbone_tracer_per_start(coords, 
                                      min_bond=3.0, max_bond=4.5,
                                      min_angle=80.0, max_angle=160.0,
                                      self_clash_cutoff=3.0,
                                      beam_width=10):
    """
    Trace a continuous backbone using beam search **per starting point**.
    """
    if len(coords) == 0:
        raise ValueError("No coordinates provided.")

    N = len(coords)
    
    # Precompute neighbors within bond range
    neighbors = [set() for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i != j:
                d = distance(coords[i], coords[j])
                if min_bond <= d <= max_bond:
                    neighbors[i].add(j)

    all_best_paths = []

    for start in range(N):
        beams = [([start], {start})]
        final_paths = []

        while beams:
            new_beams = []
            for path, used in beams:
                last = path[-1]
                available_neighbors = neighbors[last] - used
                if not available_neighbors:
                    if len(path) >= min(20, round(N * 0.95)):
                        final_paths.append(path)
                    continue

                for n in available_neighbors:
                    if is_too_close(coords, path, n, cutoff=self_clash_cutoff):
                        continue

                    if len(path) >= 2:
                        prev = path[-2]
                        angle = bond_angle(coords[prev], coords[last], coords[n])
                        if not (min_angle <= angle <= max_angle):
                            continue

                        # Prevent near-180° reversals
                        v1 = coords[last] - coords[prev]
                        v2 = coords[n] - coords[last]
                        dot = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        if dot < -0.8:
                            continue

                    new_path = path + [n]
                    new_used = used | {n}
                    new_beams.append((new_path, new_used))

            # Prune beams only within this starting point
            beams = sorted(new_beams, key=lambda x: len(x[0]), reverse=True)[:beam_width]

        # Store best path for this starting point
        if final_paths:
            best_path_for_start = max(final_paths, key=lambda x: len(x))
            all_best_paths.append(best_path_for_start)

    if not all_best_paths:
        raise RuntimeError("No valid backbone found (constraints too strict).")

    # Overall best path (longest)
    best_path = max(all_best_paths, key=lambda x: len(x))
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



def main(input_file_path = None):
    # input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\emd_21887_agglomerative_cd_4.0_combined_largestCC_20250908_190055.pdb"
    # input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\emd_34843_agglomerative_cd_4.0_combined_largestCC_20250916_202639.pdb"
    # emd_34843 helix
    input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\emd_34843_agglomerative_cd_4.0_combined_largestCC_20250923_200759.pdb"
    input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\emd_34843_agglomerative_cd_4.0_combined_largestCC_20250923_200917.pdb"
    # emd_34843 sheet
    input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing/outputs/clustering_v2/emd_34843_agglomerative_cd_4.0_combined_largestCC_20250923_204446.pdb"
    input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing/outputs/clustering_v2/emd_34843_agglomerative_cd_4.0_combined_largestCC_20250923_204749.pdb"
    # actually using SS (helix)
    input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\emd_34843_agglomerative_cd_4.0_combined_largestCC_20250923_212018.pdb"
    # full structure (way more coords)
    # input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing/outputs/clustering_v2\emd_34843_agglomerative_cd_4.0_data_alone_20250925_184041.pdb"
    
    # 10/3/25 sending more results to grayson, can delete these lines later, ALSO AUTOMATE this part already jeez
    # z: (0, 999), y: (0, 999), x: (0, 104)
    input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\emd_34843_agglomerative_cd_4.0_combined_largestCC_20251004_010559.pdb"
    # z: (0, 999), y: (0, 999), x: (104, 999)
    input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\emd_34843_agglomerative_cd_4.0_combined_largestCC_20251004_015012.pdb"
    
    # 10/4/25 about 5 more samples for 8hk3 #1/B for grayson
    # input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\emd_34843_agglomerative_cd_4.0_component_008_label34_size30_20251005_001334.pdb"
    # 1)
    input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\emd_34843_agglomerative_cd_4.0_component_001_label13_size43_20251005_001334.pdb"
    # 2)
    input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\emd_34843_agglomerative_cd_4.0_component_005_label26_size21_20251005_001334.pdb"
    # 3)
    input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\emd_34843_agglomerative_cd_4.0_component_006_label29_size32_20251005_001334.pdb"
    # 4)   #! fall back to old script for this one, better result
    # input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\emd_34843_agglomerative_cd_4.0_component_007_label31_size56_20251005_001334.pdb"
    # 5)
    input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\emd_34843_agglomerative_cd_4.0_component_009_label35_size20_20251005_001334.pdb"

    if input_file_path:
        input_file = input_file_path
    output_file = r"traced_backbone3.pdb"

    coords = load_ca_coordinates(input_file)
    print(f"Loaded {len(coords)} C-alpha atoms from {input_file}")

    # route = helical_backbone_tracer(coords, min_bond=3.0, max_bond=4.5, beam_width=20)
    route = helical_backbone_tracer_per_start(coords,
                                    min_bond=3.0, max_bond=4.5,
                                    min_angle=80.0, max_angle=160.0,
                                    self_clash_cutoff=3.0,
                                    beam_width=20)
    print("Traced route:", route)

    save_path_as_pdb(coords, route, output_file=output_file)

    return output_file


if __name__ == "__main__":
    main()