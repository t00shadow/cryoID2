import numpy as np
import re


# =====================================================================
# 1. CIF LOADING
# =====================================================================

def load_cif_lines(path):
    with open(path, "r") as f:
        return f.readlines()


# =====================================================================
# 2. PARSE ATOMS → EXTRACT C-ALPHA ONLY
# =====================================================================

def parse_ca_atoms(cif_lines):
    """
    Parse _atom_site entries and extract only CA atoms.
    Returns:
      coords : (N,3) float32
      info   : list of (chain, resseq)
    """
    headers = []
    rows = []
    in_loop = False

    # collect atom_site loop blocks
    for line in cif_lines:
        if line.startswith("loop_"):
            headers = []
            rows = []
            in_loop = False
            continue

        if line.startswith("_atom_site."):
            headers.append(line.strip())
            in_loop = True
            continue

        if in_loop and headers and not line.startswith("_") and line.strip():
            parts = line.split()
            if len(parts) >= len(headers):
                rows.append(dict(zip(headers, parts)))

    # keys
    atom_key = next((h for h in headers if h.endswith("label_atom_id")), None)
    x_key = next((h for h in headers if h.endswith("Cartn_x")), None)
    y_key = next((h for h in headers if h.endswith("Cartn_y")), None)
    z_key = next((h for h in headers if h.endswith("Cartn_z")), None)
    chain_key = next((h for h in headers if h.endswith("auth_asym_id")), None)
    if not chain_key:
        chain_key = next((h for h in headers if h.endswith("label_asym_id")), None)

    res_key = next((h for h in headers if h.endswith("auth_seq_id")), None)
    if not res_key:
        res_key = next((h for h in headers if h.endswith("label_seq_id")), None)

    ca_coords = []
    ca_info = []

    for r in rows:
        if r[atom_key] != "CA":
            continue
        chain = r[chain_key]
        resseq = int(float(r[res_key]))
        x = float(r[x_key])
        y = float(r[y_key])
        z = float(r[z_key])
        ca_coords.append((x, y, z))
        ca_info.append((chain, resseq))

    return np.array(ca_coords, dtype=np.float32), ca_info


# =====================================================================
# 3. PARSE SS (ONLY WHAT IS IN THE CIF)
# =====================================================================

def parse_secondary_structure(cif_lines):
    """
    Returns two lists:
      helices: list of (chain, start, end)
      sheets:  list of (chain, start, end)
    Only extracts segments actually stored in the CIF.
    """

    def parse_loops(prefix):
        all_loops = []
        headers = []
        rows = []
        in_loop = False

        for line in cif_lines:
            if line.startswith("loop_"):
                if headers:
                    all_loops.append((headers, rows))
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
            all_loops.append((headers, rows))
        return all_loops

    helices = []
    sheets = []

    # ---------------- HELICES ----------------
    helix_loops = parse_loops("_struct_conf.")

    for headers, rows in helix_loops:
        header_set = set(headers)
        req = {
            "_struct_conf.conf_type_id",
            "_struct_conf.beg_auth_asym_id",
            "_struct_conf.beg_auth_seq_id",
            "_struct_conf.end_auth_asym_id",
            "_struct_conf.end_auth_seq_id",
        }
        if not req.issubset(header_set):
            continue

        for r in rows:
            ctype = r["_struct_conf.conf_type_id"].upper()
            if not ctype.startswith("HELX"):
                continue
            chain = r["_struct_conf.beg_auth_asym_id"]
            start = int(float(r["_struct_conf.beg_auth_seq_id"]))
            end = int(float(r["_struct_conf.end_auth_seq_id"]))
            helices.append((chain, start, end))

    # ---------------- SHEETS ----------------
    sheet_loops = parse_loops("_struct_sheet_range.")

    for headers, rows in sheet_loops:
        header_set = set(headers)

        chain_key = None
        start_key = None
        end_key = None

        for k in ["_struct_sheet_range.beg_auth_asym_id",
                  "_struct_sheet_range.beg_label_asym_id"]:
            if k in header_set:
                chain_key = k
                break

        for k in ["_struct_sheet_range.beg_auth_seq_id",
                  "_struct_sheet_range.beg_label_seq_id"]:
            if k in header_set:
                start_key = k
                break

        for k in ["_struct_sheet_range.end_auth_seq_id",
                  "_struct_sheet_range.end_label_seq_id"]:
            if k in header_set:
                end_key = k
                break

        if not chain_key or not start_key or not end_key:
            continue

        for r in rows:
            chain = r[chain_key]
            start = int(float(r[start_key]))
            end = int(float(r[end_key]))
            sheets.append((chain, start, end))

    return helices, sheets


# =====================================================================
# 4. BUILD SS ARRAY PER CA
# =====================================================================

def assign_ss_array(ca_info, helices, sheets):
    """
    Returns array of shape (N,) with:
      0 = coil
      1 = helix
      2 = sheet
    """
    ss = np.zeros(len(ca_info), dtype=np.int8)

    for i, (chain, res) in enumerate(ca_info):
        for (h_chain, h_start, h_end) in helices:
            if chain == h_chain and h_start <= res <= h_end:
                ss[i] = 1
                break
        else:
            for (s_chain, s_start, s_end) in sheets:
                if chain == s_chain and s_start <= res <= s_end:
                    ss[i] = 2
                    break

    return ss


# =====================================================================
# 5. BUILD EDGES (CA–CA PEPTIDE BONDS)
# =====================================================================

def build_edges(ca_info, coords, cutoff=4.0):
    """
    Connects CA atoms that belong to 
      same chain AND |i-j|=1 (sequence neighbors) AND distance < cutoff.
    """
    edges = []
    info = np.array(ca_info, dtype=object)

    for i in range(len(ca_info) - 1):
        c1, r1 = info[i]
        c2, r2 = info[i + 1]
        if c1 == c2 and r2 == r1 + 1:
            d = np.linalg.norm(coords[i + 1] - coords[i])
            if d <= cutoff:
                edges.append((i, i + 1))

    return np.array(edges, dtype=np.int32)


# =====================================================================
# 6. WRITE C-ALPHA ONLY PDB
# =====================================================================

def write_ca_pdb(path, coords, info, edges, helices, sheets):
    with open(path, "w") as f:

        # -----------------------------
        # HELIX records  (proper PDB)
        # -----------------------------
        helix_id = 1
        for (chain, start, end) in helices:
            f.write(
                f"HELIX  {helix_id:3d}{helix_id:4d} "
                f"UNK {chain} {start:4d}  UNK {chain} {end:4d}  1\n"           # btw u dont even need to specify the length of the helices to select them in chimerax
            )
            helix_id += 1

        # -----------------------------
        # SHEET records  (proper PDB)
        # -----------------------------
        sheet_id = 1
        strand_no = 1
        for (chain, start, end) in sheets:
            f.write(
                f"SHEET  {sheet_id:3d} A  {strand_no:2d} "         # added 2 spaces after A since that is sheet identifier which spans columns 12-14, so 3 character widths
                f"UNK {chain}{start:4d}  UNK {chain}{end:4d}  0\n"
            )
            strand_no += 1
            sheet_id += 1

        # -----------------------------
        # ATOM records (CA only)
        # -----------------------------
        for i, ((chain, res), (x, y, z)) in enumerate(zip(info, coords)):
            # residue name: unknown since we only have CA → use "UNK"
            atom_name = "CA"
            resname = "UNK"
            element = "C"  # CA atom is carbon

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


# =====================================================================
# 7. MAIN PIPELINE
# =====================================================================

def parse_cif_for_calpha(path):
    cif = load_cif_lines(path)
    coords, info = parse_ca_atoms(cif)
    helices, sheets = parse_secondary_structure(cif)
    ss_array = assign_ss_array(info, helices, sheets)
    edges = build_edges(info, coords)
    return coords, info, edges, ss_array, helices, sheets


# =====================================================================
# 8. EXAMPLE USAGE
# =====================================================================

if __name__ == "__main__":
    cif_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\8hk3.cif"
    pdb_out = f"{cif_file[:-4]}_ca_v3_4.pdb"

    coords, info, edges, ss_array, helices, sheets = parse_cif_for_calpha(cif_file)
    write_ca_pdb(pdb_out, coords, info, edges, helices, sheets)


    print("SS counts:", np.unique(ss_array, return_counts=True))
    print("(0=coil, 1=helix, 2=strand)")
    print("Done.")
