from R_A8_zNEW_A1_5_masking_CA import mask_and_save_pdb

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


#! Step 1: masking CAs (helix regions)
#^ note the CAs are the clustered nodes from the clustering step
pdb_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\E1_D3_meanshift_bw_1.9_CA_only.pdb"
mrc_file = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\real_malaria\3-E1_D3\possibility\E1_D3_2_7A_cryosparc_P8_J562_008_volume_map_sharp_secondary_strctures_1_possibility.mrc"
output_name = "E1_D3_calpha_clustered_then_masked"
mask_threshold = 0.7
radius = 1.5
percent = 0.5

# # This commented out fxn (mask_pdb_with_mrc) gets called internally by the mask_and_save_pdb fxn, 
# # but note this commented out fxn returns coordinates in x,y,z order and does not save any files, 
# # whereas mask_and_save_pdb flips it to z,y,x and saves it.
# result = mask_pdb_with_mrc(pdb_file=pdb_file, mrc_file=mrc_file, mask_threshold=mask_threshold, radius=radius, percent=percent)

# TODO: is output_dir_masked_ca or masked_ca_path a better name?
output_path_masked_ca = mask_and_save_pdb(
    pdb_file=pdb_file,
    mrc_file=mrc_file,
    output_dir=output_dir_masked_ca,
    output_name=output_name,
    mask_threshold=mask_threshold,
    radius=radius,
    percent=percent,
    timestamp=False,
)

# TODO: add an option to turn off the print msgs from above fxn (modify the inside of that fxn)
print("Intermediate file: masked CAs")
print(output_path_masked_ca)



#! Step 2: bb tracing
# TODO: output_path_masked_ca is the input path to this step