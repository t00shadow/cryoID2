from pathlib import Path
from R_B8_sample import Sample
from R_A8_zNEW_A1_5_masking_CA import mask_and_save_pdb
from R_A3_limit_degree_skipclustering_SAVEconnectedcomponents import main as backbone_trace
import sys
# scuffed patch to import from a higher directory level
sys.path.append(str(Path(__file__).resolve().parent.parent))
from A3_backbone_to_residue_6degen_groups import main as residue_sequence
from A3_convert_queries_to_fasta import process_directory

'''
This script assumes clustering has already been done (there's another script to batch cluster)
Lowk should have a batch script for each step separately? Or a wrapper to let it run on individual files or folders/lists of files

1. masking helix regions:     # NOTE: in the RNA version, this is baked into step 2
    - H_Backbone_tracing\small_prototypes\R_A8_zNEW_A1.5_masking_CA.py
2. bb tracing:
    - H_Backbone_tracing\small_prototypes\R_A3_limit_degree_skipclustering_SAVEconnectedcomponents.py
        - H_Backbone_tracing\small_prototypes\R_A2_superset_graph.py
3. residue sequence:
    - H_Backbone_tracing\A3_batch_driver_script.py
        - H_Backbone_tracing\A3_backbone_to_residue_6degen_groups.py
4. convert to fasta (these are the queries):
    - H_Backbone_tracing\A3_convert_queries_to_fasta
'''

# TODO: to implement start_step, stop_step, and any number of samples (single or many), can do EITHER loop entire pipeline or step by step
# for loop entire pipeline, just need to make a wrapper
# for step by step, might need a SampleList class to manage Samples. A regular list might work too?

test_sample = Sample(
    name="E1_D3",
    unet_possibility_mrc_dir=r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\real_malaria\3-E1_D3\possibility",
    key_atom_ca_mrc=r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\real_malaria\3-E1_D3\possibility\E1_D3_2_7A_cryosparc_P8_J562_008_volume_map_sharp_key_atoms_1_possibility.mrc",
    secondary_struct_helix_mrc=r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\real_malaria\3-E1_D3\possibility\E1_D3_2_7A_cryosparc_P8_J562_008_volume_map_sharp_secondary_strctures_1_possibility.mrc",
    clustered_ca_pdb=r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\E1_D3_meanshift_bw_1.9_CA_only.pdb",
)

output_dir_base = Path(r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\batch_backbone_tracing")
output_dir_base = output_dir_base / test_sample.name
output_dirs = {
    "masked_ca": output_dir_base / "masked_ca",
    "backbone_graph": output_dir_base / "backbone_trace" / "graph",
    "backbone_components": output_dir_base / "backbone_trace" / "components",
    "residue_seq": output_dir_base / "residue_seq",
    "queries": output_dir_base / "queries",
}

# TODO: consider determining test_sample.masked_ca_pdb AND test_sample.queries_fasta_file ahead of time too, so can delete the above dictionary and have all paths in the same data structure. OR store their dirs instead of the filenames. And then could also move this logic into the Sample class, but then makes the Sample class less flexible
# store in Sample object
# test_sample.masked_ca_pdb AND test_sample.queries_fasta_file will be generated and stored during the pipeline
test_sample.backbone_graph_dir = output_dirs["backbone_graph"]
test_sample.backbone_components_dir = output_dirs["backbone_components"]
test_sample.residue_sequences_dir = output_dirs["residue_seq"]

for path in output_dirs.values():
    path.mkdir(parents=True, exist_ok=True)



### Pipeline

#* Step 1: masking CAs (helix regions)
#TODO: Move this to the top and make a config class
mask_threshold = 0.7
mask_radius = 1.5
mask_required_percent = 0.5

test_sample.masked_ca_pdb = mask_and_save_pdb(
    pdb_file=test_sample.clustered_ca_pdb,
    mrc_file=test_sample.secondary_struct_helix_mrc,
    output_dir=output_dirs["masked_ca"],
    output_name=test_sample.name+"_ca_clustered_then_masked",
    mask_threshold=mask_threshold,
    radius=mask_radius,
    percent=mask_required_percent,
    timestamp=False,
    verbose=False,
)
print("Intermediate file: masked CAs")
print(test_sample.masked_ca_pdb)



#* Step 2: bb tracing
tracing_max_dist = 5.0
tracing_max_degree = 2
# tracing_min_component_size = 18  # tune this, 18 was an arbitrary number
tracing_min_component_size = 10  # tune this

# TODO: might want to 1-index the cc names instead of 0-index
backbone_trace(
    use_clustered_pdb=True,
    clustered_pdb_path=test_sample.masked_ca_pdb,
    max_dist=tracing_max_dist,
    max_degree=tracing_max_degree,
    min_component_size=tracing_min_component_size,
    graph_output_dir=test_sample.backbone_graph_dir,   # OPTIONAL, defaults to dir of clustered_pdb_path
    components_output_dir=test_sample.backbone_components_dir,
    verbose=True
)



#* Step 3: residue sequence
# TODO: edit A3_batch_driver.py and replace that with this and wrap it in a fxn def
bb_component_files = sorted(Path(test_sample.backbone_components_dir).glob("*.pdb"))
file_count = len(bb_component_files)
print(f"Found {file_count} component pdb files")

for i, pdb_file in enumerate(bb_component_files, start=1):

    print(f"\r[{i}/{file_count}] Running: {pdb_file.name}", end="", flush=True)

    output_pdb = residue_sequence(
        input_file_path=pdb_file,
        main_map_path=test_sample.key_atom_ca_mrc,    # TODO: prolly should change the name of the argument in the actual script too (mrc instead of map)
        residue_dir_path=test_sample.unet_possibility_mrc_dir,
        output_dir_path=test_sample.residue_sequences_dir
    )

    # print(f"Finished -> {output_pdb}\n")
print(f"\nFinished processing {file_count} files: {test_sample.residue_sequences_dir}")



#* Step 4: convert to FASTA
# process_directory(
#     directory=test_sample.residue_sequences_dir,
#     fasta_output=output_dirs["queries"] / f"queries_cutoff_length_{tracing_min_component_size}.fasta",   # TODO: move this to the top, after moving the numeric parameters to the top too. then can condense my dictionary of output paths with the instance of the Samples class, so all relevant file structure is observable in one place
#     verbose=False
# )

process_directory(
    directory=split_output_dir,
    fasta_output=output_dirs["queries"] / f"queries_cutoff_length_{tracing_min_component_size}_optimized_method.fasta",
    verbose=False
)