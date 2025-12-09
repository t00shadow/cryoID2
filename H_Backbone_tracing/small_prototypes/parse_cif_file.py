# small bug: just connects all consecutive atoms, without checking if a bond rly exists (should check if peptide bond exists between residues in cif file, etc). It added one bond between two areas that are rly far (have a dotted line in chimerax thats prob a pseudobond to show interactions or smth)
# EDIT: do a dif on the results and there is a slight yet meaningful difference. also can see from the print out that there's 1127 bonds for v2 vs 1130 for this version of the script
# KEEP both versions since this first version lists what chimerax shows

import numpy as np
from pathlib import Path


def parse_cif_for_calpha(cif_path):
    """
    Extract C-alpha atoms and minimal connectivity from a CIF file.
    
    Returns:
        ca_coords: (N, 3) numpy array of CA coordinates
        ca_info: list of (chain_id, resseq, resname, atom_id)
        edges: (M, 2) numpy array of index pairs representing CA–CA connections
    """

    cif_text = Path(cif_path).read_text().splitlines()

    # mmCIF atom loop begins at:
    # loop_
    # _atom_site.group_PDB
    # _atom_site.id
    # ...
    # Then all atoms follow as space- or tab-separated values.
    
    atom_headers = []
    atom_rows = []
    in_atom_loop = False

    for line in cif_text:
        if line.startswith("loop_"):
            atom_headers = []
            in_atom_loop = False

        # capture header lines
        if line.startswith("_atom_site."):
            atom_headers.append(line.strip())
            in_atom_loop = True
            continue

        # once headers collected and we see non-header data, parse rows
        if in_atom_loop and atom_headers and not line.startswith("_"):
            # split respecting mmCIF quoting rules (simplified)
            parts = line.split()
            if len(parts) < len(atom_headers):
                continue  # malformed

            row = dict(zip(atom_headers, parts))
            atom_rows.append(row)

    # Extract CA atoms
    ca_atoms = []
    for r in atom_rows:
        # These standard column names usually exist in PDB mmCIFs:
        # _atom_site.label_atom_id   → "CA"
        # _atom_site.label_comp_id   → "ALA", "GLY", ...
        # _atom_site.label_seq_id    → residue index
        atom_name = r.get("_atom_site.label_atom_id", "")
        if atom_name.upper() != "CA":
            continue

        try:
            x = float(r["_atom_site.Cartn_x"])
            y = float(r["_atom_site.Cartn_y"])
            z = float(r["_atom_site.Cartn_z"])
        except Exception:
            continue

        chain_id = r.get("_atom_site.label_asym_id", "A")
        resseq = int(float(r.get("_atom_site.label_seq_id", 0)))
        resname = r.get("_atom_site.label_comp_id", "UNK")
        atom_id = r.get("_atom_site.id", None)

        ca_atoms.append((chain_id, resseq, resname, atom_id, np.array([x, y, z])))

    # Sort by chain + residue number to build edges correctly
    ca_atoms.sort(key=lambda a: (a[0], a[1]))

    ca_coords = np.array([a[4] for a in ca_atoms])
    ca_info = [(a[0], a[1], a[2], a[3]) for a in ca_atoms]

    # Build edges: connect consecutive residues within each chain
    edges = []
    for i in range(len(ca_atoms) - 1):
        c1, r1, _, _, _ = ca_atoms[i]
        c2, r2, _, _, _ = ca_atoms[i + 1]
        if c1 == c2:  # only connect within same chain
            edges.append((i, i + 1))

    edges = np.array(edges, dtype=np.int32)

    return ca_coords, ca_info, edges


# ----------------------------------------------------------------------

def write_ca_pdb(output_path, ca_coords, ca_info, edges):
    """
    Write a CA-only PDB file from numpy coords + info + CA-CA edges.
    """

    with open(output_path, "w") as f:
        for i, ((chain, resi, resname, atom_id), (x, y, z)) in enumerate(zip(ca_info, ca_coords)):
            f.write(
                f"ATOM  {i+1:5d}  CA  {resname:3s} {chain}{resi:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00\n"
            )

        # Write CONECT (optional)
        for a, b in edges:
            f.write(f"CONECT{a+1:5d}{b+1:5d}\n")

        f.write("END\n")


# ----------------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    cif_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\8hk3.cif"
    print(cif_file[-8:-4])

    coords, info, edges = parse_cif_for_calpha(cif_file)

    print("Extracted CA atoms:", coords.shape[0])
    print("Edges:", edges.shape)

    output_pdb = f"H_Backbone_tracing/small_prototypes/{cif_file[-8:-4]}_ca.pdb"
    write_ca_pdb(output_pdb, coords, info, edges)
    print("Wrote", output_pdb)
