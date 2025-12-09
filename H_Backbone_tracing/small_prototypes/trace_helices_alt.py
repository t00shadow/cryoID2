import numpy as np
from scipy.spatial import cKDTree

def trace_helices_from_coords(coords, ss, min_helix_len=4, max_dist=4.0):
    """
    Trace helices from CA positions and their secondary structure types
    without using PDB HELIX records.

    Args:
        coords (np.ndarray): (N,3) CA atom positions
        ss (np.ndarray): (N,) secondary structure type for each CA ('H','S','C')
        min_helix_len (int): minimum number of residues to consider a helix
        max_dist (float): maximum distance between consecutive CA atoms in a helix (~3.5-4 Å)

    Returns:
        helices: list of helices, each as a list of indices into coords/ss
    """
    helix_indices = np.where(ss == 'H')[0]
    coords_h = coords[helix_indices]

    # Use a KD-tree for fast neighbor search
    tree = cKDTree(coords_h)
    neighbors = tree.query_ball_tree(tree, r=max_dist)

    visited = set()
    helices = []

    for i, idx in enumerate(helix_indices):
        if idx in visited:
            continue

        # Start new helix path
        path = [idx]
        visited.add(idx)
        current = idx

        while True:
            # Get neighbor indices in coords_h space
            neighbor_idxs = [helix_indices[j] for j in neighbors[np.where(helix_indices==current)[0][0]]]
            # Keep only unvisited neighbors
            neighbor_idxs = [n for n in neighbor_idxs if n not in visited]
            if not neighbor_idxs:
                break
            # Pick the closest neighbor (greedy)
            dists = np.linalg.norm(coords[neighbor_idxs] - coords[current], axis=1)
            next_idx = neighbor_idxs[np.argmin(dists)]
            path.append(next_idx)
            visited.add(next_idx)
            current = next_idx

        if len(path) >= min_helix_len:
            helices.append(sorted(path, key=lambda x: x))  # sort indices for consistency

    return helices


if __name__ == "__main__":
    from parse_PDB_file__FINAL import extract_ca_and_ss  # your previous script

    pdb_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\8HK3.pdb"
    coords, ss = extract_ca_and_ss(pdb_file)

    helices_traced = trace_helices_from_coords(coords, ss)

    print(f"Traced {len(helices_traced)} helices:")
    np.set_printoptions(legacy='1.25')    # suppresses the np ints being printed with the np.int64() prefix
    for i, helix in enumerate(helices_traced, 1):
        # print(f"Helix {i}: indices {helix}, length {len(helix)}")
        print(f"Helix {i}: length {len(helix)}")
    

    # --- Step 4.5: check if any bonds are > 4.0 (if none, it's just chimerax) ---   #! copied from trace_helices_v2.py, everything above is an alter version of trace_helices.py. V2 just added outputting to a pdb and checking for bond lengths exceeding 4.0 AFTER tracing.
    #                                                                                #& UPDATE: ok the results are the exact same, so the trace_helices.py is valid and not cheating by using information that wouldnt be known in real data

    import numpy as np

    # coords: np.ndarray of shape (N,3)
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
    
    from trace_helices_v2 import write_traced_helices_pdb

    # --- Step 5: Write CA-only PDB with traced HELIX records ---
    output_path = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\traced_helices_alt.pdb"
    write_traced_helices_pdb(output_path, coords, helices_traced)
    
    print(f"Wrote traced helices to: {output_path}")
    print(f"Traced {len(helices_traced)} helices.")

