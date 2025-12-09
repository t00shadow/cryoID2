# READ:
#! make copies of this file when testing better tracing algos

import numpy as np
from scipy.spatial import cKDTree
from parse_PDB_file__FINAL import extract_ca_and_ss, write_ca_pdb  # your previous script
from Bio.PDB import PDBParser

def trace_helices_from_coords(coords, ss,
                              min_helix_len=4,
                              soft_dist=4.0,
                              soft_angle=40.0,
                              hard_dist=5.0):
    """
    Trace helices from CA positions using SS info, handling noisy coordinates.
    Uses a soft cutoff with relative squared penalty and a hard cutoff to terminate paths.

    Args:
        coords (np.ndarray): (N,3) CA atom positions
        ss (np.ndarray): (N,) secondary structure type ('H','S','C')
        min_helix_len (int): minimum number of residues to consider a helix
        soft_dist (float): preferred CA-CA distance (~3.5-4.0 Å)
        soft_angle (float): preferred angle between consecutive CA vectors (deg)
        hard_dist (float): absolute maximum distance to consider a neighbor (Å)

    Returns:
        helices (list of list): traced helices, each as a list of indices into coords/ss
    """
    helix_indices = np.where(ss == 'H')[0]
    coords_h = coords[helix_indices]
    tree = cKDTree(coords_h)
    neighbors = tree.query_ball_tree(tree, r=hard_dist)

    visited = set()
    helices = []

    for idx in helix_indices:
        if idx in visited:
            continue

        path = [idx]
        visited.add(idx)
        current = idx
        prev = None

        while True:
            neighbor_idxs = [helix_indices[j] for j in neighbors[np.where(helix_indices==current)[0][0]]]
            neighbor_idxs = [n for n in neighbor_idxs if n not in visited]

            if not neighbor_idxs:
                break

            penalties = []
            for n in neighbor_idxs:
                dist = np.linalg.norm(coords[n] - coords[current])

                # distance penalty relative to soft cutoff
                penalty_dist = max(0.0, (dist - soft_dist) / soft_dist) ** 2

                # angle penalty relative to previous vector
                if prev is None:
                    penalty_angle = 0.0
                else:
                    prev_vec = coords[current] - coords[prev]
                    prev_vec /= np.linalg.norm(prev_vec)
                    cand_vec = coords[n] - coords[current]
                    cand_vec /= np.linalg.norm(cand_vec)
                    cos_theta = np.clip(np.dot(prev_vec, cand_vec), -1.0, 1.0)
                    angle = np.arccos(cos_theta) * 180 / np.pi
                    penalty_angle = max(0.0, (angle - soft_angle) / soft_angle) ** 2

                total_penalty = penalty_dist + penalty_angle
                penalties.append((n, total_penalty))

            # pick neighbor with lowest penalty
            penalties.sort(key=lambda x: x[1])
            next_idx, next_penalty = penalties[0]

            # terminate if distance exceeds hard_dist
            if np.linalg.norm(coords[next_idx] - coords[current]) > hard_dist:
                break

            path.append(next_idx)
            visited.add(next_idx)
            prev, current = current, next_idx

        if len(path) >= min_helix_len:
            helices.append(sorted(path))

    return helices


def write_traced_helices_pdb(path, coords, helices, edges=None):
    """
    Write a CA-only PDB containing only traced helices.

    Args:
        path (str): output file path
        coords (np.ndarray): (N,3) CA coordinates for all atoms
        helices (list of lists): each helix is a list of indices into coords
        edges (list of tuples): optional connectivity (pairs of indices within each helix)
    """
    edges = edges or []

    with open(path, "w") as f:
        atom_serial = 1  # PDB atom numbering
        helix_id = 1

        for helix in helices:
            # --- HELIX record ---
            start_resnum = 1
            end_resnum = len(helix)
            f.write(
                f"HELIX  {helix_id:3d}  {helix_id:3d} UNK A {start_resnum:4d}  UNK A {end_resnum:4d}  1\n"
            )
            helix_id += 1

            # --- ATOM records ---
            for i, idx in enumerate(helix):
                x, y, z = coords[idx]
                atom_name = "CA"
                resname = "UNK"
                chain = "A"  # placeholder
                resnum = i + 1  # sequential numbering within helix
                element = "C"
                f.write(
                    f"ATOM  {atom_serial:5d}  {atom_name:<4s}{resname:>3s} {chain}"
                    f"{resnum:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}"
                    f"  1.00  0.00          {element:>2s}\n"
                )
                atom_serial += 1

            # --- CONECT records (optional) ---
            for i in range(len(helix)-1):
                f.write(f"CONECT{atom_serial - len(helix) + i:5d}{atom_serial - len(helix) + i + 1:5d}\n")

        f.write("END\n")








# v3: adding pertubations to simulate real world data
def shake_coords(coords, fraction=0.05, min_shift=0.2, max_shift=0.5, seed=42):
    """
    Randomly perturb a fraction of CA coordinates to simulate noise.

    Args:
        coords (np.ndarray): (N,3) array of CA positions
        fraction (float): fraction of atoms to perturb (0-1)
        min_shift (float): minimum shift in Å
        max_shift (float): maximum shift in Å
        seed (int): random seed for reproducibility

    Returns:
        np.ndarray: new array with shaken coordinates
    """
    np.random.seed(seed)
    coords_shaken = coords.copy()
    N = coords.shape[0]
    n_perturb = int(N * fraction)
    
    # choose random indices to perturb
    indices = np.random.choice(N, n_perturb, replace=False)
    
    # generate random shift vectors
    directions = np.random.randn(n_perturb, 3)  # random directions
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]  # normalize
    magnitudes = np.random.uniform(min_shift, max_shift, n_perturb)[:, np.newaxis]
    shifts = directions * magnitudes
    
    # apply shifts
    coords_shaken[indices] += shifts
    return coords_shaken



if __name__ == "__main__":
    pdb_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\8HK3.pdb"
    
    # --- Step 1: Extract CA positions and SS ---
    coords, ss = extract_ca_and_ss(pdb_file)
    
    # --- Step 1.5: Perturb (some) CA positions to simulate real world data ---
    # coords: your original (N,3) CA positions
    # print(coords.shape)
    # coords_noisy = shake_coords(coords, fraction=0.05, min_shift=0.2, max_shift=0.5, seed=123)
    # coords_noisy = shake_coords(coords, fraction=0.15, min_shift=0.2, max_shift=0.5, seed=123)
    coords_noisy = shake_coords(coords, fraction=0.15, min_shift=0.2, max_shift=1.0, seed=123)
    coords_noisy = shake_coords(coords, fraction=0.5, min_shift=0.2, max_shift=0.5, seed=123)
    # coords_noisy = shake_coords(coords, fraction=0.5, min_shift=0.2, max_shift=1.0, seed=123)

    # --- Step 2: Prepare info list (chain, resseq) ---
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("S", pdb_file)
    info = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    info.append((chain.id, residue.id[1]))

    # --- Step 3: Trace helices ---
    # helices_traced = trace_helices_from_coords(coords_noisy, ss, info)
    helices_traced = trace_helices_from_coords(coords_noisy, ss)   # no longer need to pass in info variable
    
    # --- Step 4: Convert traced helices to (chain, start_res, end_res) tuples ---
    helices_pdb = []
    for helix in helices_traced:
        chain = info[helix[0]][0]
        start_res = info[helix[0]][1]
        end_res = info[helix[-1]][1]
        helices_pdb.append((chain, start_res, end_res))

    # --- Step 4.5: check if any bonds are > 4.0 (if none, it's just chimerax) ---
    import numpy as np

    # coords: np.ndarray of shape (N,3), all CA positions
    # helices_traced: list of helices, each a list of indices into coords

    max_bond_dist = 4.0  # Å

    for h_idx, helix in enumerate(helices_traced, start=1):
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


    # --- Step 5: Write CA-only PDB with traced HELIX records ---
    output_path = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\traced_helices_perturbed.pdb"
    write_traced_helices_pdb(output_path, coords, helices_traced)
    
    print(f"Wrote traced helices to: {output_path}")
    print(f"Traced {len(helices_traced)} helices.")
