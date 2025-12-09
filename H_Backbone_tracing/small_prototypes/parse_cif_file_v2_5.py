# extract secondary struct too now, EVEN MORE improved SS numpy storage format

'''
ca_coords      → (N,3) float array
ca_info        → list of (chain, resseq, resname, atom_id)
edges          → (M,2) int32 array
ss_array       → (N,) int array (0=C,1=H,2=E)
ss_segments    → (K,4) object array (chain, start, end, type)
ss_full        → (N,) structured array with per-residue SS info
'''


import numpy as np
from pathlib import Path

# ======================================================================
#  Parse mmCIF → C-alpha coords, edges, secondary structure
# ======================================================================

def parse_cif_for_calpha(cif_path, max_ca_distance=4.0):
    """
    Extracts:
      - CA coordinates
      - Metadata
      - CA–CA edges (distance-based)
      - ss_array: numeric SS per CA (0=coil,1=helix,2=sheet)
      - ss_segments: secondary structure segments (helix/sheet)
      - ss_full: structured array containing SS per-residue
    """

    cif_lines = Path(cif_path).read_text().splitlines()

    # ------------------------------------------------------------------
    #  Parse ATOMS
    # ------------------------------------------------------------------
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
            atom_rows.append(dict(zip(atom_headers, parts)))

    # ---- Extract CA atoms only ----
    ca_atoms = []
    for r in atom_rows:
        if r.get("_atom_site.label_atom_id", "").upper() != "CA":
            continue

        try:
            x = float(r["_atom_site.Cartn_x"])
            y = float(r["_atom_site.Cartn_y"])
            z = float(r["_atom_site.Cartn_z"])
        except:
            continue

        chain   = r.get("_atom_site.label_asym_id", "A")
        resseq  = int(float(r.get("_atom_site.label_seq_id", 0)))
        resname = r.get("_atom_site.label_comp_id", "UNK")
        atom_id = r.get("_atom_site.id", None)

        ca_atoms.append((chain, resseq, resname, atom_id, np.array([x, y, z])))

    # Sort by chain then residue
    ca_atoms.sort(key=lambda x: (x[0], x[1]))

    ca_coords = np.array([a[4] for a in ca_atoms], dtype=float)
    ca_info   = [(a[0], a[1], a[2], a[3]) for a in ca_atoms]

    # ------------------------------------------------------------------
    #  CA–CA EDGES
    # ------------------------------------------------------------------
    edges = []
    for i in range(len(ca_atoms) - 1):
        c1, r1, _, _, p1 = ca_atoms[i]
        c2, r2, _, _, p2 = ca_atoms[i + 1]

        if c1 != c2:
            continue
        if np.linalg.norm(p2 - p1) <= max_ca_distance:
            edges.append((i, i + 1))

    edges = np.array(edges, dtype=np.int32)

    # ------------------------------------------------------------------
    #  SECONDARY STRUCTURE — HELICES & SHEETS/STRANDS
    # ------------------------------------------------------------------
    helices, sheets = parse_struct_conf_and_sheets(cif_lines)

    ss_segments = []
    for chain, start_res, end_res in helices:
        ss_segments.append((chain, start_res, end_res, 1))
    for chain, start_res, end_res in sheets:
        ss_segments.append((chain, start_res, end_res, 2))

    ss_segments = np.array(ss_segments, dtype=object)

    # ------------------------------------------------------------------
    #  ASSIGN SS PER RESIDUE
    # ------------------------------------------------------------------
    ss_array = np.zeros(len(ca_info), dtype=np.int8)   # 0=C,1=H,2=E
    ss_list  = ["C"] * len(ca_info)

    for i, (chain, resseq, _, _) in enumerate(ca_info):
        for seg_chain, seg_start, seg_end, seg_type in ss_segments:
            if seg_chain == chain and seg_start <= resseq <= seg_end:
                ss_array[i] = seg_type
                ss_list[i]  = "H" if seg_type == 1 else "E"
                break

    # ------------------------------------------------------------------
    #  NEW: ss_full structured array
    # ------------------------------------------------------------------
    dtype_ss = np.dtype([
        ("chain",   "U4"),
        ("resseq",  np.int32),
        ("resname", "U4"),
        ("atom_id", "U16"),
        ("ss",      "U1"),      # 'H', 'E', 'C'
        ("ss_code", np.int8),   # 0,1,2
    ])

    ss_full = np.zeros(len(ca_info), dtype=dtype_ss)

    for i, ((chain, resseq, resname, atom_id), ss_char, ss_code) in enumerate(
        zip(ca_info, ss_list, ss_array)
    ):
        ss_full[i] = (chain, resseq, resname, atom_id, ss_char, ss_code)

    # ------------------------------------------------------------------
    return ca_coords, ca_info, edges, ss_array, ss_segments, ss_full


def parse_struct_conf_and_sheets(cif_lines):
    """
    Extract all helix and sheet segments.
    Works for all CIF variants and all helix types.
    """
    helices = []
    sheets = []

    # Helper: parse loops starting with prefix
    def parse_all_loops(prefix):
        loops = []
        headers = []
        rows = []
        in_loop = False

        for line in cif_lines:
            if line.startswith("loop_"):
                if headers:
                    loops.append((headers, rows))
                headers = []
                rows = []
                in_loop = False
                continue

            if line.startswith(prefix):
                headers.append(line.strip())
                in_loop = True
                continue

            if in_loop and headers and not line.startswith("_") and line.strip():
                parts = line.split()
                if len(parts) >= len(headers):
                    rows.append(dict(zip(headers, parts)))

        if headers:
            loops.append((headers, rows))
        return loops

    # ---------------------------------------------------------------
    # HELICES from struct_conf
    # ---------------------------------------------------------------
    helix_loops = parse_all_loops("_struct_conf.")

    for headers, rows in helix_loops:
        header_set = set(headers)
        required = {
            "_struct_conf.conf_type_id",
            "_struct_conf.beg_auth_asym_id",
            "_struct_conf.beg_auth_seq_id",
            "_struct_conf.end_auth_asym_id",
            "_struct_conf.end_auth_seq_id",
        }
        if not required.issubset(header_set):
            continue

        for r in rows:
            ctype = r["_struct_conf.conf_type_id"].upper()
            # MATCH ANY HELIX TYPE (not only 'helix')
            if not ctype.startswith("HELX"):
                continue

            chain = r["_struct_conf.beg_auth_asym_id"]
            start = int(float(r["_struct_conf.beg_auth_seq_id"]))
            end   = int(float(r["_struct_conf.end_auth_seq_id"]))
            helices.append((chain, start, end))

    # ---------------------------------------------------------------
    # SHEETS from struct_sheet_range
    # ---------------------------------------------------------------
    sheet_loops = parse_all_loops("_struct_sheet_range.")

    for headers, rows in sheet_loops:
        header_set = set(headers)

        # Look for either label_* or auth_* variants
        chain_key = None
        start_key = None
        end_key   = None

        for variant in [
            "_struct_sheet_range.beg_auth_asym_id",
            "_struct_sheet_range.beg_label_asym_id"
        ]:
            if variant in header_set:
                chain_key = variant
                break

        for variant in [
            "_struct_sheet_range.beg_auth_seq_id",
            "_struct_sheet_range.beg_label_seq_id"
        ]:
            if variant in header_set:
                start_key = variant
                break

        for variant in [
            "_struct_sheet_range.end_auth_seq_id",
            "_struct_sheet_range.end_label_seq_id"
        ]:
            if variant in header_set:
                end_key = variant
                break

        # If any missing → skip this loop
        if not chain_key or not start_key or not end_key:
            continue

        # Parse
        for r in rows:
            chain = r[chain_key]
            start = int(float(r[start_key]))
            end   = int(float(r[end_key]))
            sheets.append((chain, start, end))

    return helices, sheets


# ======================================================================
#  Write CA-only PDB
# ======================================================================

def write_ca_pdb(output_path, ca_coords, ca_info, edges, ss_array=None):
    """
    Writes CA-only PDB.
    Optionally encode SS in B-factor:
        coil=20, helix=50, sheet=100
    """
    with open(output_path, "w") as f:

        for i, ((chain, resi, resname, atom_id), (x, y, z)) in enumerate(zip(ca_info, ca_coords)):

            if ss_array is None:
                b = 0.0
            else:
                b = 20.0 if ss_array[i] == 0 else (50.0 if ss_array[i] == 1 else 100.0)

            f.write(
                f"ATOM  {i+1:5d}  CA  {resname:3s} {chain}{resi:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b:6.2f}\n"
            )

        # CONECT bonds (CA graph)
        for (a, b) in edges:
            f.write(f"CONECT{a+1:5d}{b+1:5d}\n")

        f.write("END\n")


# ======================================================================
#  Example Usage
# ======================================================================
if __name__ == "__main__":
    cif_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\8hk3.cif"

    coords, info, edges, ss_array, ss_segments, ss_full = parse_cif_for_calpha(cif_file)

    print("Num CA:", len(coords))
    print("Num edges:", edges.shape)
    print("Sample SS array:", ss_array[:20])
    print("frequeny counts of SS array", np.unique(ss_array, return_counts=True))
    # checked in chimerax and there are 444 helix ca (sel #1 & helix & @ca)
    # 304 strand ca (sel #1 & strand & @ca)
    # 387 coil ca (sel #1 & coil & @ca)
    # that adds up to 1135
    print(ss_array.shape)
    print("Segments:", ss_segments[:4])
    print("Full SS row:", ss_full[0])

    out = f"{cif_file[:-4]}_CA_only.pdb"
    write_ca_pdb(out, coords, info, edges, ss_array)
    print("Wrote", out)
