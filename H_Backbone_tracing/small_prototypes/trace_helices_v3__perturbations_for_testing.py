# READ:
#! make copies of this file when testing better tracing algos
#TODO rename perturbations to noise

import numpy as np
from scipy.spatial import cKDTree
from parse_PDB_file__FINAL import extract_ca_and_ss, write_ca_pdb  # your previous script
from Bio.PDB import PDBParser

def trace_helices_from_coords(coords, ss, info, min_helix_len=4, max_dist=4.0):
    """
    Trace helices from CA positions and secondary structure labels.
    Uses a KD-tree and greedy neighbor growth.
    
    Args:
        coords (np.ndarray): (N,3) CA positions
        ss (np.ndarray): (N,) secondary structure labels ('H','S','C')
        info (list of tuples): (chain, resseq) aligned with coords
        min_helix_len (int): minimum residues to consider a helix
        max_dist (float): max distance between consecutive CA atoms (~3.5–4 Å)
    
    Returns:
        helices: list of helices, each as a list of indices into coords/info
    """
    helix_indices = np.where(ss == 'H')[0]
    if len(helix_indices) == 0:
        return []

    coords_h = coords[helix_indices]
    tree = cKDTree(coords_h)
    neighbors = tree.query_ball_tree(tree, r=max_dist)

    visited = set()
    helices = []

    for i, idx in enumerate(helix_indices):
        if idx in visited:
            continue

        path = [idx]
        visited.add(idx)
        current = idx

        while True:
            # Find neighbors of current CA in coords_h
            h_idx = np.where(helix_indices == current)[0][0]
            neighbor_idxs = [helix_indices[j] for j in neighbors[h_idx]]
            neighbor_idxs = [n for n in neighbor_idxs if n not in visited]
            if not neighbor_idxs:
                break
            # Pick the closest neighbor
            dists = np.linalg.norm(coords[neighbor_idxs] - coords[current], axis=1)
            next_idx = neighbor_idxs[np.argmin(dists)]
            path.append(next_idx)
            visited.add(next_idx)
            current = next_idx

        if len(path) >= min_helix_len:
            # Sort by residue number for consistency
            helices.append(sorted(path, key=lambda x: info[x][1]))

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
    coords_noisy = shake_coords(coords, fraction=0.5, min_shift=0.2, max_shift=1.0, seed=123)

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
    helices_traced = trace_helices_from_coords(coords_noisy, ss, info)
    
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
