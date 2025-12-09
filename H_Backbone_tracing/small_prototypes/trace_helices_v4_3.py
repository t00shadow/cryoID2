# making bigger changes to tracing logic

import numpy as np
from scipy.spatial import cKDTree

# -----------------------------
# Graph and penalties
# -----------------------------
def build_full_ca_graph(coords, hard_cutoff=5.0):
    tree = cKDTree(coords)
    neighbors = tree.query_ball_tree(tree, r=hard_cutoff)
    graph = {i: [j for j in nbrs if j != i] for i, nbrs in enumerate(neighbors)}
    return graph

def compute_penalty(coords, prev_idx, current_idx, cand_idx, soft_dist=3.8, soft_angle=40.0):
    # distance
    dist = np.linalg.norm(coords[cand_idx] - coords[current_idx])
    penalty_dist = max(0.0, (dist - soft_dist)/soft_dist)**2

    # angle
    if prev_idx is None:
        penalty_angle = 0.0
    else:
        v1 = coords[current_idx] - coords[prev_idx]
        v1 /= np.linalg.norm(v1)
        v2 = coords[cand_idx] - coords[current_idx]
        v2 /= np.linalg.norm(v2)
        cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
        angle = np.arccos(cos_theta) * 180/np.pi
        penalty_angle = max(0.0, (angle - soft_angle)/soft_angle)**2

    return penalty_dist + penalty_angle

# -----------------------------
# Helix tracing
# -----------------------------
def trace_helices(coords, ss, soft_dist=3.8, soft_angle=40.0, hard_cutoff=5.0, min_len=4):
    helix_indices = np.where(ss == 'H')[0]
    graph = build_full_ca_graph(coords, hard_cutoff=hard_cutoff)

    visited = set()
    helices = []

    for seed in helix_indices:
        if seed in visited:
            continue
        path = [seed]
        visited.add(seed)
        current, prev = seed, None

        while True:
            neighbors = [n for n in graph[current] if n in helix_indices and n not in visited]
            if not neighbors:
                break
            # compute penalties
            penalties = [(n, compute_penalty(coords, prev, current, n, soft_dist, soft_angle)) for n in neighbors]
            penalties.sort(key=lambda x: x[1])
            next_idx, next_penalty = penalties[0]

            # stop if distance exceeds hard cutoff
            if np.linalg.norm(coords[next_idx] - coords[current]) > hard_cutoff:
                break

            # optional soft adjustment to ideal distance
            vec = coords[next_idx] - coords[current]
            d = np.linalg.norm(vec)
            if d != soft_dist and d>0:
                coords[next_idx] = coords[current] + vec*(soft_dist/d)

            path.append(next_idx)
            visited.add(next_idx)
            prev, current = current, next_idx

        if len(path) >= min_len:
            helices.append(sorted(path))

    return helices, coords  # coords may be updated for helix adjustment

# -----------------------------
# PDB writing
# -----------------------------
def write_pdb(path, coords, atom_indices=None, helices=None):
    """
    Write PDB. If atom_indices is None, write all coords.
    """
    atom_counter = 1
    with open(path, "w") as f:
        if helices:
            for h_id, helix in enumerate(helices, start=1):
                f.write(f"HELIX  {h_id:3d}{h_id:4d} UNK ? {helix[0]:4d} UNK ? {helix[-1]:4d} 1\n")
        indices = atom_indices if atom_indices is not None else range(len(coords))
        for idx in indices:
            x, y, z = coords[idx]
            f.write(f"ATOM  {atom_counter:5d}  CA  UNK ?{idx:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          C\n")
            atom_counter +=1
        f.write("END\n")

# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    from parse_PDB_file__FINAL import extract_ca_and_ss

    pdb_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\8HK3.pdb"
    coords, ss = extract_ca_and_ss(pdb_file)

    from trace_helices_v3__perturbations_for_testing import shake_coords            # just import it instead of copy pasting, saves vertical space
    coords_noisy = shake_coords(coords, fraction=0.5, min_shift=0.2, max_shift=1.0, seed=123)
    coords_noisy = shake_coords(coords, fraction=0.5, min_shift=0.2, max_shift=0.5, seed=123)

    # helices, corrected_coords = trace_helices(coords, ss)
    helices, corrected_coords = trace_helices(coords_noisy, ss)

    # --- Step 4.5: check if any bonds are > 4.0 (if none, it's just chimerax) ---
    import numpy as np

    # coords: np.ndarray of shape (N,3), all CA positions
    # helices_traced: list of helices, each a list of indices into coords

    max_bond_dist = 4.0  # Å

    # for h_idx, helix in enumerate(helices_traced, start=1):
    for h_idx, helix in enumerate(helices, start=1):
        coords_helix = coords[helix]
        # distances between consecutive CA atoms
        dists = np.linalg.norm(np.diff(coords_helix, axis=0), axis=1)
        long_bonds = np.where(dists > max_bond_dist)[0]
        if len(long_bonds) > 0:
            for i in long_bonds:
                print(
                    f"Helix {h_idx}, bond {i}-{i+1} exceeds {max_bond_dist} Å: "
                    f"{dists[i]:.2f} Å"
                )

    # Helix-only PDB
    helix_atoms = [idx for h in helices for idx in h]
    write_pdb("H_Backbone_tracing/small_prototypes/traced_helices_v4_3.pdb", corrected_coords, atom_indices=helix_atoms, helices=helices)

    # Full CA PDB with corrected helix positions
    write_pdb("H_Backbone_tracing/small_prototypes/traced_helices_PLUS_other_CA_v4_3.pdb", corrected_coords, helices=helices)
