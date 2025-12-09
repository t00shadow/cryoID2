# use this version, p similar to first one but has a distance limit to keep out some of those weirdly long pseudobonds (chimerax still shows it, but my tracing script will avoid reading it. the first version actyually include them, check the dif between the outputs)

import numpy as np
from pathlib import Path

def parse_cif_for_calpha(cif_path, max_ca_distance=4.0):
    """
    Extract C-alpha atoms and connectivity from a CIF file.
    Connect consecutive residues in the same chain if their CAs are within max_ca_distance.
    Returns:
        ca_coords: (N,3) numpy array of CA coordinates
        ca_info: list of (chain_id, resseq, resname, atom_id)
        edges: (M,2) numpy array of index pairs representing CA–CA connections
    """
    cif_lines = Path(cif_path).read_text().splitlines()

    # ---------- Parse atoms ----------
    atom_headers = []
    atom_rows = []
    in_atom_loop = False

    for line in cif_lines:
        if line.startswith("loop_"):
            atom_headers = []
            in_atom_loop = False
        if line.startswith("_atom_site."):
            atom_headers.append(line.strip())
            in_atom_loop = True
            continue
        if in_atom_loop and atom_headers and not line.startswith("_") and line.strip():
            parts = line.split()
            if len(parts) < len(atom_headers):
                continue
            row = dict(zip(atom_headers, parts))
            atom_rows.append(row)

    # Extract C-alpha atoms
    ca_atoms = []
    for idx, r in enumerate(atom_rows):
        if r.get("_atom_site.label_atom_id","").upper() != "CA":
            continue
        try:
            x = float(r["_atom_site.Cartn_x"])
            y = float(r["_atom_site.Cartn_y"])
            z = float(r["_atom_site.Cartn_z"])
        except:
            continue
        chain = r.get("_atom_site.label_asym_id","A")
        resseq = int(float(r.get("_atom_site.label_seq_id",0)))
        resname = r.get("_atom_site.label_comp_id","UNK")
        atom_id = r.get("_atom_site.id",None)
        ca_atoms.append((chain, resseq, resname, atom_id, np.array([x,y,z])))

    # Sort by chain + residue number
    ca_atoms.sort(key=lambda a: (a[0], a[1]))
    ca_coords = np.array([a[4] for a in ca_atoms])
    ca_info = [(a[0], a[1], a[2], a[3]) for a in ca_atoms]

    # ---------- Build edges ----------
    edges = []
    for i in range(len(ca_atoms)-1):
        c1, r1, _, _, pos1 = ca_atoms[i]
        c2, r2, _, _, pos2 = ca_atoms[i+1]
        if c1 == c2:
            distance = np.linalg.norm(pos2 - pos1)
            if distance <= max_ca_distance:
                edges.append((i, i+1))

    edges = np.array(edges, dtype=np.int32)
    return ca_coords, ca_info, edges

# ----------------------------------------------------------------------

def write_ca_pdb(output_path, ca_coords, ca_info, edges):
    """
    Write a CA-only PDB file from numpy coords + info + CA-CA edges.
    """
    with open(output_path,"w") as f:
        for i, ((chain,resi,resname,atom_id),(x,y,z)) in enumerate(zip(ca_info,ca_coords)):
            f.write(
                f"ATOM  {i+1:5d}  CA  {resname:3s} {chain}{resi:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00 20.00\n"
            )
        # Write edges as CONECT (only verified by distance)
        for a,b in edges:
            f.write(f"CONECT{a+1:5d}{b+1:5d}\n")
        f.write("END\n")

# ----------------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    cif_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\8hk3.cif"
    coords, info, edges = parse_cif_for_calpha(cif_file)
    print("Extracted CA atoms:", coords.shape[0])
    print("Edges:", edges.shape)

    output_pdb = f"H_Backbone_tracing/small_prototypes/{cif_file[-8:-4]}_ca_v2.pdb"
    write_ca_pdb(output_pdb, coords, info, edges)
    print("Wrote", output_pdb)
