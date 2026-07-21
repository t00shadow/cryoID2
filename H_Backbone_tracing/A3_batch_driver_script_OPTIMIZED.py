import os
from A3_backbone_to_residue_6degen_groups import main


# =========================================================
# USER SETTINGS
# =========================================================

# pdb_folder = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\results_3_26_26\clustered_and_traced"
# main_map_path = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\real_malaria\3-E1_D3\possibility\E1_D3_2_7A_cryosparc_P8_J562_008_volume_map_sharp_key_atoms_1_possibility.mrc"
# residue_dir_path = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\real_malaria\3-E1_D3\possibility"
# output_dir_path = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\results_3_26_26\TEST_STDOUT_to_file"

pdb_folder = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\cc_minsize"
main_map_path = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\real_malaria\3-E1_D3\possibility\E1_D3_2_7A_cryosparc_P8_J562_008_volume_map_sharp_key_atoms_1_possibility.mrc"
residue_dir_path = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\real_malaria\3-E1_D3\possibility"
output_dir_path = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\cc_minsize_w_residues"


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