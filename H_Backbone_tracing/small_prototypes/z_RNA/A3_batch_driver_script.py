import os
from A3_backbone_to_residue_6degen_groups import main


# =========================================================
# USER SETTINGS
# =========================================================

pdb_folder = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\rna_cc\cc_minsize"
# from inside the imported fxn: main_map_path is only for getting the dimensions (all the mrc files for a specific sample should be the same size).
main_map_path = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\RNAs\Yao\EMD-47655\possibility\emd_47655.map_key_atoms_2_possibility.mrc"
residue_dir_path = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\RNAs\Yao\EMD-47655\possibility"
output_dir_path = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\rna_cc\cc_minsize_w_residues"


# =========================================================
# GET ALL PDB FILES
# =========================================================

pdb_files = sorted([
    f for f in os.listdir(pdb_folder)
    if f.endswith(".pdb")
])

print(f"Found {len(pdb_files)} pdb files\n")


# =========================================================
# LOOP
# =========================================================

#! DISABLE PRINTS TO RUN FASTER, PRINTING IS P SLOW
#! or even just redirect stdout to a file, still much faster than to console

for pdb_name in pdb_files:
    pdb_file = os.path.join(pdb_folder, pdb_name)

    print("=" * 80)
    print(f"Running: {pdb_name}")
    print("=" * 80)

    output_pdb = main(
        input_file_path=pdb_file,
        main_map_path=main_map_path,
        residue_dir_path=residue_dir_path,
        output_dir_path=output_dir_path
    )

    print(f"Finished -> {output_pdb}\n")