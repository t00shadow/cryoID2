import os
from R_A8_zNEW_A1_5_masking_CA import mask_and_save_pdb
from R_A3_limit_degree_skipclustering_SAVEconnectedcomponents import main as bb_trace
import sys
# scuffed patch to import from a higher directory level
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from A3_backbone_to_residue_6degen_groups import main as residue_sequence
from A3_convert_queries_to_fasta import process_directory

# TODO: to loop this for all samples, only have to modify output_dir_base. Also use os.path.join or Pathlib
output_dir_base = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\batch_bb_tracing"
output_dir_masked_ca = output_dir_base + r"\masked_ca"
output_dir_bb_trace = output_dir_base + r"\bb_trace"
output_dir_residue_seq = output_dir_base + r"\residue_seq"
output_dir_fasta = output_dir_base + r"\fasta"
# print(output_dir_fasta)
# assert(False)

'''
This script assumes clustering has already been done (there's another script to batch cluster)
Lowk should have a batch script for each step separately? Or a wrapper to let it run on individual files or folders/lists of files

1. masking helix regions:
    - H_Backbone_tracing\small_prototypes\R_A8_zNEW_A1.5_masking_CA.py
2. bb tracing:
    - H_Backbone_tracing\small_prototypes\R_A3_limit_degree_skipclustering_SAVEconnectedcomponents.py
        - H_Backbone_tracing\small_prototypes\R_A2_superset_graph.py
3. residue sequence:
    - H_Backbone_tracing\A3_batch_driver_script.py
        - H_Backbone_tracing\A3_backbone_to_residue_6degen_groups.py
4. convert to fasta:
    - H_Backbone_tracing\A3_convert_queries_to_fasta
'''


#* Step 1: masking CAs (helix regions)
#* note the CAs are the clustered nodes from the clustering step
pdb_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\E1_D3_meanshift_bw_1.9_CA_only.pdb"
mrc_file = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\real_malaria\3-E1_D3\possibility\E1_D3_2_7A_cryosparc_P8_J562_008_volume_map_sharp_secondary_strctures_1_possibility.mrc"
# TODO: this name cannot be hardcoded
masked_ca_name = "E1_D3_calpha_clustered_then_masked"
mask_threshold = 0.7
radius = 1.5
percent = 0.5

# # This commented out fxn (mask_pdb_with_mrc) gets called internally by the mask_and_save_pdb fxn, 
# # but note this commented out fxn returns coordinates in x,y,z order and does not save any files, 
# # whereas mask_and_save_pdb flips it to z,y,x and saves it.
# result = mask_pdb_with_mrc(pdb_file=pdb_file, mrc_file=mrc_file, mask_threshold=mask_threshold, radius=radius, percent=percent)

output_path_masked_ca = mask_and_save_pdb(
    pdb_file=pdb_file,
    mrc_file=mrc_file,
    output_dir=output_dir_masked_ca,
    output_name=masked_ca_name,
    mask_threshold=mask_threshold,
    radius=radius,
    percent=percent,
    timestamp=False,
    verbose=False,
)
print("Intermediate file: masked CAs")
print(output_path_masked_ca)



#* Step 2: bb tracing
clustered_pdb_path = output_path_masked_ca
max_dist = 5.0
max_degree = 2
# min_component_size = 18  # tune this, 18 was an arbitrary number
min_component_size = 10  # tune this
### graph_output_dir is OPTIONAL, if not given, will default to dir of clustered_pdb_path
graph_output_dir = output_dir_bb_trace + "\graph"
components_output_dir = output_dir_bb_trace + "\components"

# TODO: might want to 1-index the cc names instead of 0-index
bb_trace(
    use_clustered_pdb=True,
    clustered_pdb_path=clustered_pdb_path,
    max_dist=max_dist,
    max_degree=max_degree,
    min_component_size=min_component_size,
    graph_output_dir=graph_output_dir,
    components_output_dir=components_output_dir,
    verbose=True
)



#* Step 3: residue sequence
# note this NOT the same as mrcfile above. mrcfile holds a secondary structure mrc file's path, This is key atoms
main_mrc_path = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\real_malaria\3-E1_D3\possibility\E1_D3_2_7A_cryosparc_P8_J562_008_volume_map_sharp_key_atoms_1_possibility.mrc"
residue_dir_path = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\real_malaria\3-E1_D3\possibility"

# get all traced bb components (.pdb files)
bb_component_files = sorted([
    f for f in os.listdir(components_output_dir)
    if f.endswith(".pdb")
])
file_count = len(bb_component_files)
print(f"Found {file_count} component pdb files")

for i, bb_component_name in enumerate(bb_component_files, start=1):
    pdb_file = os.path.join(components_output_dir, bb_component_name)

    # print(f"Running: {bb_component_name}")
    print(f"\r[{i}/{file_count}] Running: {bb_component_name}", end="", flush=True)

    output_pdb = residue_sequence(
        input_file_path=pdb_file,
        main_map_path=main_mrc_path,    # TODO: prolly should change the name of the argument in the actual script too (mrc instead of map)
        residue_dir_path=residue_dir_path,
        output_dir_path=output_dir_residue_seq
    )

    # print(f"Finished -> {output_pdb}\n")
print(f"\nFinished processing {file_count} files: {output_dir_residue_seq}")



#* Step 4: convert to FASTA
process_directory(
    directory=output_dir_residue_seq,
    fasta_output=output_dir_fasta+f"\queries_cutoff_length_{min_component_size}.fasta",
    verbose=False
)