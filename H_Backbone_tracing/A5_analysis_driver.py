from A4_analysis_ALT import main as run_analysis

# ----------------------------- Parrameters that get shared across all runs  -----------------------------

# (Optional) Intermediate outputs (dont need these rn, let them get overwritten)
SAVE_GT_PDB = "zz_calpha_gt_residues.pdb"              # Optional intermediate CA PDB
SAVE_EXP_PDB = "zz_calpha_exp_residues.pdb"     # Optional predicted residues PDB

# Parameters (for determining the experimental residue type)
VOXEL_NEIGHBOR_RADIUS = 1  # Å

# Degenerate groupings
DEGENERATE_GROUPS = {
    # Group 1 (G)
    "GLY": "G", "ALA": "G", "SER": "G", "CYS": "G",
    "VAL": "G", "THR": "G", "ILE": "G",
    # Group 2 (P)
    "PRO": "P",
    # Group 3 (L)
    "LEU": "L", "ASP": "L", "ASN": "L",
    "GLU": "L", "GLN": "L", "MET": "L",
    # Group 4 (K)
    "LYS": "K", "ARG": "K",
    # Group 5 (Y)
    "HIS": "Y", "PHE": "Y", "TYR": "Y",
    # Group 6 (W)
    "TRP": "W",
}
# --------------------------------------------------------------------------------------------------------


# for loop
#! pause for now, realize the test set only has 6letter degen group possibilities, saves some file i/o :shrug:

# Input files
CA_INPUT_FILE = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\8HK3.pdb"
RESIDUE_MRC_DIR = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\EMD-34843_re_3.2\possibility"
MAIN_MRC_MAP = RESIDUE_MRC_DIR + r"\emd_34843.map_key_atoms_1_possibility.mrc"

print(type(RESIDUE_MRC_DIR))
assert(False)

run_analysis(CA_INPUT_FILE, RESIDUE_MRC_DIR, MAIN_MRC_MAP, SAVE_GT_PDB, SAVE_EXP_PDB, VOXEL_NEIGHBOR_RADIUS, DEGENERATE_GROUPS)

