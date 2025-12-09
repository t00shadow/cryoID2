import numpy as np
import pandas as pd
from Bio.PDB import PDBParser

def extract_ca_and_ss(pdb_file):
    # --- Step 1: Parse the structure ---
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('structure', pdb_file)

    # --- Step 2: Initialize lists to store results ---
    ca_positions = []
    ca_ss = []

    # --- Step 3: Parse HELIX and SHEET records manually ---
    helix_ranges = []  # list of tuples: (chain, start_resseq, end_resseq)
    sheet_ranges = []

    with open(pdb_file, 'r') as f:
        for line in f:
            record_type = line[0:6].strip()
            if record_type == 'HELIX':
                chain = line[19]
                start_resseq = int(line[21:25])
                end_resseq = int(line[33:37])
                helix_ranges.append((chain, start_resseq, end_resseq))
            elif record_type == 'SHEET':
                chain = line[21]
                start_resseq = int(line[22:26])
                end_resseq = int(line[33:37])
                sheet_ranges.append((chain, start_resseq, end_resseq))

    # --- Step 4: Iterate over CA atoms ---
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    ca_atom = residue['CA']
                    ca_positions.append(ca_atom.coord)

                    # Determine secondary structure
                    resseq = residue.id[1]
                    ss_type = 'C'  # default coil
                    # Check HELIX
                    for h_chain, h_start, h_end in helix_ranges:
                        if chain.id == h_chain and h_start <= resseq <= h_end:
                            ss_type = 'H'
                            break
                    # Check SHEET
                    if ss_type == 'C':
                        for s_chain, s_start, s_end in sheet_ranges:
                            if chain.id == s_chain and s_start <= resseq <= s_end:
                                ss_type = 'S'
                                break
                    ca_ss.append(ss_type)

    # --- Step 5: Convert to numpy arrays ---
    ca_positions = np.array(ca_positions)  # shape (N, 3)
    ca_ss = np.array(ca_ss)  # shape (N,)

    return ca_positions, ca_ss









# writing out results just to check (not actually needed for extract data from input files)
def write_ca_pdb(path, coords, info, helices=None, sheets=None, edges=None):
    helices = helices or []  # default empty list if None
    sheets = sheets or []
    edges = edges or []

    with open(path, "w") as f:

        # -----------------------------
        # HELIX records
        # -----------------------------
        for helix_id, (chain, start, end) in enumerate(helices, start=1):
            f.write(
                f"HELIX  {helix_id:3d}{helix_id:4d} "
                f"UNK {chain} {start:4d}  UNK {chain} {end:4d}  1\n"
            )

        # -----------------------------
        # SHEET records
        # -----------------------------
        for sheet_id, (chain, start, end) in enumerate(sheets, start=1):
            strand_no = 1  # could adjust if multiple strands
            f.write(
                f"SHEET  {sheet_id:3d} A  {strand_no:2d} "
                f"UNK {chain}{start:4d}  UNK {chain}{end:4d}  0\n"
            )

        # -----------------------------
        # ATOM records (CA only)
        # -----------------------------
        for i, ((chain, res), (x, y, z)) in enumerate(zip(info, coords)):
            atom_name = "CA"
            resname = "UNK"
            element = "C"
            f.write(
                f"ATOM  {i+1:5d}  {atom_name:<4s}{resname:>3s} {chain}"
                f"{res:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}"
                f"  1.00  0.00          {element:>2s}\n"
            )

        # -----------------------------
        # CONECT records
        # -----------------------------
        for i, j in edges:
            f.write(f"CONECT{i+1:5d}{j+1:5d}\n")

        f.write("END\n")


def test_fxn():
    # --- Step 1: Extract CA positions and secondary structure ---
    pdb_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\8HK3.pdb"
    coords, ss = extract_ca_and_ss(pdb_file)  # use the function we wrote before

    # --- Step 2: Re-parse helices and sheets from the PDB header for writing ---
    helix_ranges = []
    sheet_ranges = []

    with open(pdb_file, 'r') as f:
        for line in f:
            record_type = line[0:6].strip()
            if record_type == 'HELIX':
                chain = line[19]
                start_resseq = int(line[21:25])
                end_resseq = int(line[33:37])
                helix_ranges.append((chain, start_resseq, end_resseq))
            elif record_type == 'SHEET':
                chain = line[21]
                start_resseq = int(line[22:26])
                end_resseq = int(line[33:37])
                sheet_ranges.append((chain, start_resseq, end_resseq))

    # --- Step 3: Prepare the info list (chain, resseq) aligned with coords ---
    # from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("S", pdb_file)

    info = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    info.append((chain.id, residue.id[1]))

    # --- Step 4: Write the CA-only PDB file ---
    write_ca_pdb(
        path="H_Backbone_tracing/small_prototypes/verify_parse_pdb_script.pdb",
        coords=coords,
        info=info,
        helices=helix_ranges,
        sheets=sheet_ranges,
        edges=[]  # empty if you don't have connectivity
    )





# if __name__ == "__main__":
#     pdb_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\8HK3.pdb"
#     coords, ss = extract_ca_and_ss(pdb_file)
#     print(coords.shape, ss.shape)
#     print(coords[:5])
#     print(ss[:50])

if __name__ == "__main__":
    test_fxn()

