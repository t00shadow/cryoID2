# extract secondary struct too now

import numpy as np
from pathlib import Path

def parse_cif_for_calpha(cif_path, max_ca_distance=4.0):
    """
    Extract C-alpha atoms, connectivity, and secondary structure from a CIF file.
    Connect consecutive residues in the same chain if their CAs are within max_ca_distance.
    Returns:
        ca_coords: (N,3) numpy array of CA coordinates
        ca_info: list of (chain_id, resseq, resname, atom_id)
        edges: (M,2) numpy array of index pairs representing CA–CA connections
        ss_list: list of length N, secondary structure code: 'H', 'E', or 'C'
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
    for r in atom_rows:
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

    # ---------- Parse secondary structure ----------
    # Helices from _struct_conf
    helices = []
    in_helix_loop = False
    helix_headers = []
    for line in cif_lines:
        if line.startswith("loop_"):
            helix_headers = []
            in_helix_loop = False
        if line.startswith("_struct_conf."):
            helix_headers.append(line.strip())
            in_helix_loop = True
            continue
        if in_helix_loop and helix_headers and not line.startswith("_") and line.strip():
            parts = line.split()
            if len(parts) < len(helix_headers):
                continue
            row = dict(zip(helix_headers, parts))
            # get chain, start/end residue number
            chain = row.get("_struct_conf.pdbx_strand_id") or row.get("_struct_conf.id")  # fallback
            start_chain = row.get("_struct_conf.beg_auth_asym_id") or row.get("_struct_conf.beg_label_asym_id")
            start_res = row.get("_struct_conf.beg_auth_seq_num") or row.get("_struct_conf.beg_label_seq_id")
            end_chain = row.get("_struct_conf.end_auth_asym_id") or row.get("_struct_conf.end_label_asym_id")
            end_res = row.get("_struct_conf.end_auth_seq_num") or row.get("_struct_conf.end_label_seq_id")
            if start_chain and start_res and end_chain and end_res:
                helices.append((start_chain, int(start_res), end_chain, int(end_res)))

    # Sheets from _struct_sheet_range
    sheets = []
    in_sheet_loop = False
    sheet_headers = []
    for line in cif_lines:
        if line.startswith("loop_"):
            sheet_headers = []
            in_sheet_loop = False
        if line.startswith("_struct_sheet_range."):
            sheet_headers.append(line.strip())
            in_sheet_loop = True
            continue
        if in_sheet_loop and sheet_headers and not line.startswith("_") and line.strip():
            parts = line.split()
            if len(parts) < len(sheet_headers):
                continue
            row = dict(zip(sheet_headers, parts))
            chain = row.get("_struct_sheet_range.sheet_id")  # fallback
            start_chain = row.get("_struct_sheet_range.beg_auth_asym_id") or row.get("_struct_sheet_range.beg_label_asym_id")
            start_res = row.get("_struct_sheet_range.beg_auth_seq_id") or row.get("_struct_sheet_range.beg_label_seq_id")
            end_chain = row.get("_struct_sheet_range.end_auth_asym_id") or row.get("_struct_sheet_range.end_label_asym_id")
            end_res = row.get("_struct_sheet_range.end_auth_seq_id") or row.get("_struct_sheet_range.end_label_seq_id")
            if start_chain and start_res and end_chain and end_res:
                sheets.append((start_chain, int(start_res), end_chain, int(end_res)))

    # Assign secondary structure
    ss_list = []
    for chain, resseq, _, _ in ca_info:
        ss = 'C'  # default coil
        for s_start_chain, s_start_res, s_end_chain, s_end_res in helices:
            if chain == s_start_chain and s_start_res <= resseq <= s_end_res:
                ss = 'H'
                break
        for s_start_chain, s_start_res, s_end_chain, s_end_res in sheets:
            if chain == s_start_chain and s_start_res <= resseq <= s_end_res:
                ss = 'E'
                break
        ss_list.append(ss)

    return ca_coords, ca_info, edges, ss_list

# ----------------------------------------------------------------------

def write_ca_pdb(output_path, ca_coords, ca_info, edges, ss_list=None):
    """
    Write a CA-only PDB file from numpy coords + info + CA-CA edges.
    Optionally include secondary structure in the B-factor column for visualization.
    """
    with open(output_path,"w") as f:
        for i, ((chain,resi,resname,atom_id),(x,y,z)) in enumerate(zip(ca_info,ca_coords)):
            b_factor = 0.0
            if ss_list:
                # Encode secondary structure as numeric B-factor for quick visualization
                # H->50, E->100, C->20
                ss = ss_list[i]
                if ss=='H': 
                    b_factor=50.0
                elif ss=='E': 
                    b_factor=100.0
                else: 
                    b_factor=20.0
            f.write(
                f"ATOM  {i+1:5d}  CA  {resname:3s} {chain}{resi:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b_factor:6.2f}\n"
            )
        # Write edges as CONECT (only verified by distance)
        for a,b in edges:
            f.write(f"CONECT{a+1:5d}{b+1:5d}\n")
        f.write("END\n")

# ----------------------------------------------------------------------
# Example usage
if __name__ == "__main__":
    cif_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\8hk3.cif"
    coords, info, edges, ss_list = parse_cif_for_calpha(cif_file)
    print("Extracted CA atoms:", coords.shape[0])
    print("Edges:", edges.shape)
    print("Secondary structure:", ss_list[0:100], "...")
    print("H = helix, E = strand, C = coil")

    output_pdb = f"H_Backbone_tracing/small_prototypes/{cif_file[-8:-4]}_ca_v2_2.pdb"
    write_ca_pdb(output_pdb, coords, info, edges, ss_list)
    print("Wrote", output_pdb)
