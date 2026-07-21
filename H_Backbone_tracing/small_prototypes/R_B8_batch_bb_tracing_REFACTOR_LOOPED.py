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
def run_pipeline(sample: Sample, output_dir_root):
    # To test logging, uncomment the line below so it simulates a failed run for every sample.
    # assert False

    output_dir_base = output_dir_root / sample.name
    output_dirs = {
        "masked_ca": output_dir_base / "masked_ca",
        "backbone_graph": output_dir_base / "backbone_trace" / "graph",
        "backbone_components": output_dir_base / "backbone_trace" / "components",
        "residue_seq": output_dir_base / "residue_seq",
        "queries": output_dir_base / "queries",
    }

    # TODO: consider determining sample.masked_ca_pdb AND sample.queries_fasta_file ahead of time too, so can delete the above dictionary and have all paths in the same data structure. OR store their dirs instead of the filenames. And then could also move this logic into the Sample class, but then makes the Sample class less flexible
    # store in Sample object
    # sample.masked_ca_pdb AND sample.queries_fasta_file will be generated and stored during the pipeline
    sample.backbone_graph_dir = output_dirs["backbone_graph"]
    sample.backbone_components_dir = output_dirs["backbone_components"]
    sample.residue_sequences_dir = output_dirs["residue_seq"]

    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)



    ### Pipeline

    #* Step 1: masking CAs (helix regions)
    #TODO: Move this to the top and make a config class
    mask_threshold = 0.7
    mask_radius = 1.5
    mask_required_percent = 0.5

    sample.masked_ca_pdb = mask_and_save_pdb(
        pdb_file=sample.clustered_ca_pdb,
        mrc_file=sample.secondary_struct_helix_mrc,
        output_dir=output_dirs["masked_ca"],
        output_name=sample.name+"_ca_clustered_then_masked",
        mask_threshold=mask_threshold,
        radius=mask_radius,
        percent=mask_required_percent,
        timestamp=False,
        verbose=False,
    )
    print("Step 1: Masking CAs done.")
    print(f"Intermediate file (masked CAs): {sample.masked_ca_pdb}")



    #* Step 2: bb tracing
    tracing_max_dist = 5.0
    tracing_max_degree = 2
    # tracing_min_component_size = 18  # tune this, 18 was an arbitrary number
    tracing_min_component_size = 10  # tune this

    # TODO: might want to 1-index the cc names instead of 0-index
    backbone_trace(
        use_clustered_pdb=True,
        clustered_pdb_path=sample.masked_ca_pdb,
        max_dist=tracing_max_dist,
        max_degree=tracing_max_degree,
        min_component_size=tracing_min_component_size,
        graph_output_dir=sample.backbone_graph_dir,   # OPTIONAL, defaults to dir of clustered_pdb_path
        components_output_dir=sample.backbone_components_dir,
        verbose=True
    )
    print("Step 2: Backbone tracing done.")



    # #* Step 3: residue sequence
    # # TODO: edit A3_batch_driver.py and replace that with this and wrap it in a fxn def
    # bb_component_files = sorted(Path(sample.backbone_components_dir).glob("*.pdb"))
    # file_count = len(bb_component_files)
    # print(f"Found {file_count} component pdb files")

    # for i, pdb_file in enumerate(bb_component_files, start=1):

    #     print(f"\r[{i}/{file_count}] Running: {pdb_file.name}", end="", flush=True)

    #     output_pdb = residue_sequence(
    #         input_file_path=pdb_file,
    #         main_map_path=sample.key_atom_ca_mrc,    # TODO: prolly should change the name of the argument in the actual script too (mrc instead of map)
    #         residue_dir_path=sample.unet_possibility_mrc_dir,
    #         output_dir_path=sample.residue_sequences_dir
    #     )

    #     # print(f"Finished -> {output_pdb}\n")
    # print(f"\nFinished processing {file_count} files: {sample.residue_sequences_dir}")



    #& Step 3 (OPTIMIZED): residue sequence
    # TODO: move ts into its own file
    from R_A3_limit_degree_skipclustering_SAVEconnectedcomponents import load_coords_from_pdb
    import numpy as np

    # ---------------------------------------------------------
    # Step 3.0: Natural sorting, not rly essential
    # ---------------------------------------------------------
    import re
    def natural_key(path):
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r"(\d+)", path.name)]

    # ---------------------------------------------------------
    # Step 3.1: Load and combine all input PDB coordinate arrays
    # ---------------------------------------------------------
    all_coords = []
    metadata = []
    start = 0
    bb_component_files = sorted(Path(sample.backbone_components_dir).glob("*.pdb"), key=natural_key)
    for pdb_path in bb_component_files:
        coords = load_coords_from_pdb(
            pdb_path,
            atom_name="C",
        )
        end = start + len(coords)
        metadata.append({
            "input_path": pdb_path,
            "slice": slice(start, end),
            "length": len(coords),
        })
        all_coords.append(coords)
        start = end
    combined_coords = np.vstack(all_coords)
    # print(np.shape(combined_coords)); assert False

    # ---------------------------------------------------------
    # Step 3.2: Save combined coords as ONE temporary PDB
    # ---------------------------------------------------------
    # TODO: better var name(s), input isnt a good name here since it's an intermediate input, not input to the whole pipeline
    combined_input_dir = output_dir_base / "optimized"
    combined_input_dir.mkdir(exist_ok=True)
    combined_input_pdb = combined_input_dir / "temp_combined_components.pdb"
    def save_pdb(coords, filename):
        with open(filename, "w") as f:
            # Write atoms
            for i, (z, y, x) in enumerate(coords, start=1):
                f.write(
                    f"HETATM{i:5d}  C   UNK     1    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                )
    save_pdb(
        coords=combined_coords,
        filename=combined_input_pdb,
    )
    # assert False

    # ---------------------------------------------------------
    # Step 3.3: Run actual pipeline step
    # ---------------------------------------------------------
    combined_output_pdb = residue_sequence(
        input_file_path=combined_input_pdb,
        main_map_path=sample.key_atom_ca_mrc,
        residue_dir_path=sample.unet_possibility_mrc_dir,
        output_dir_path=output_dir_base / "optimized"
    )

    # ---------------------------------------------------------
    # Step 3.4: Read modified combined output
    # (EDIT: fixed) BUG: need to read in the residue column too, or else can't save it to the output properly
    # ---------------------------------------------------------
    # modified = load_coords_from_pdb(
    #     combined_output_pdb,
    #     atom_name="C",
    # )

    def load_from_pdb_modified(pdb_path, atom_name="CA"):
        """
        Returns structured array with:
        - coordinates (z, y, x)
        - residue name
        - residue sequence number
        - atom name
        """
        data = []
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith(("ATOM", "HETATM")):
                    continue
                if atom_name is not None:
                    name = line[12:16].strip()
                    if name != atom_name:
                        continue
                atom_name_col = line[12:16].strip()
                res_name = line[17:21].strip()
                res_seq = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                data.append((z, y, x, res_name, res_seq, atom_name_col))
        if not data:
            raise ValueError(f"No atoms read from {pdb_path}")
        dtype = [
            ("coord", float, (3,)),
            ("resname", "U4"),
            ("resseq", int),
            ("atom", "U4"),
        ]
        arr = np.zeros(len(data), dtype=dtype)
        for i, (z, y, x, res, seq, atom) in enumerate(data):
            arr["coord"][i] = (z, y, x)
            arr["resname"][i] = res
            arr["resseq"][i] = seq
            arr["atom"][i] = atom
        return arr

    modified = load_from_pdb_modified(
        combined_output_pdb,
        atom_name="C",
    )
    modified_coords = modified["coord"]
    resnames = modified["resname"]
    resseq = modified["resseq"]

    def save_pdb_modified(arr, filename):
        with open(filename, "w") as f:

            for i, atom in enumerate(arr, start=1):

                z, y, x = atom["coord"]
                resname = atom["resname"]
                resseq = atom["resseq"]
                atom_name = atom["atom"]

                # TODO: gradually standardize all pdb write fxns to smth closer to official pdb standard
                # f.write(
                #     f"HETATM{i:5d} {atom_name:>4s} {resname:>3s} "
                #     f"A{resseq:4d}    "
                #     f"{x:8.3f}{y:8.3f}{z:8.3f}  "
                #     f"1.00  0.00           {atom_name[0]:>1s}\n"
                # )
                f.write(
                    f"HETATM{i:5d}  C   {resname:>3s}     1    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                )

    # ---------------------------------------------------------
    # Optional safety check
    # ---------------------------------------------------------
    if len(modified_coords) != len(combined_coords):
        raise ValueError(
            "Pipeline changed atom count/order; "
            "cannot safely split back into constituents."
        )
    # assert False

    # ---------------------------------------------------------
    # Step 3.5: Split back into original constituents
    # ---------------------------------------------------------
    split_output_dir = output_dir_base / "residue_seq_optimized_method"
    split_output_dir.mkdir(exist_ok=True)

    # for info in metadata:
    #     split_coords = modified_coords[info["slice"]]
    #     input_path = Path(info["input_path"])
    #     output_name = split_output_dir / f"{input_path.stem}_updated_residues.pdb"
    #     save_pdb(
    #         coords=split_coords,
    #         filename=output_name,
    #     )
    start = 0
    for info in metadata:
        end = start + info["length"]
        chunk = modified[start:end]   # structured slice
        save_pdb_modified(
            chunk,
            split_output_dir / f"{info['input_path'].stem}_updated_residues.pdb"
        )
        start = end
    # assert False

    # ---------------------------------------------------------
    # Step 3.6: Copy over the CONECT records from the inputs to step 3
    # (not necessary for generating queries, but nice so still have connections when viewing in chimerax/taking pics)
    # ---------------------------------------------------------
    def extract_conect_block(pdb_path):
        conect_lines = []
        recording = False
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("CONECT"):
                    recording = True
                    conect_lines.append(line)
                elif recording:
                    # once CONECT starts, it's typically all at end
                    conect_lines.append(line)
        return conect_lines

    def append_conect(output_pdb_path, conect_lines):
        with open(output_pdb_path, "a") as f:
            if not conect_lines:
                return
            # if conect_lines[0].startswith("CONECT"):
            #     f.write("\n")  # ensure separation    #! dont need this b/c the input doesnt have it, tho lowk might be smth to change in the input. Yeah just checked, R_A2's save_graph_pdb doesnt have a newline inbtwn.
            for line in conect_lines:
                f.write(line if line.endswith("\n") else line + "\n")

    for original_path in bb_component_files:
        processed_path = Path(split_output_dir) / f"{original_path.stem}_updated_residues.pdb"
        # print(original_path.stem, processed_path.stem)
        conect = extract_conect_block(original_path)
        append_conect(processed_path, conect)

    print("Step 3: Residue sequence done.")



    #* Step 4: convert to FASTA
    # process_directory(
    #     directory=sample.residue_sequences_dir,
    #     fasta_output=output_dirs["queries"] / f"queries_cutoff_length_{tracing_min_component_size}.fasta",   # TODO: move this to the top, after moving the numeric parameters to the top too. then can condense my dictionary of output paths with the instance of the Samples class, so all relevant file structure is observable in one place
    #     verbose=False
    # )

    # TODO: move this to the top, after moving the numeric parameters to the top too.
    fasta_file = output_dirs["queries"] / f"queries_cutoff_length_{tracing_min_component_size}_optimized_method.fasta"
    process_directory(
        directory=split_output_dir,
        fasta_output=fasta_file,
        verbose=False
    )
    sample.queries_fasta_file = fasta_file

    print("Step 4: Fasta done.")




if __name__ == "__main__":
    MODE = "single"   # "single" or "batch" (actually any string besides "single" will work)

    # ========== Single sample mode ==========
    # NOTE: Use this is if you need to manually set input file paths (ie stored in scattered locations), the batch mode supports inputting a single sample id (list of 1 id), but the paths have to be organized somewhat specifically
    if MODE == "single":
        # =================================
        # single run (manual file locations)
        output_dir_root = Path(r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\single_backbone_tracing")
        test_sample = Sample(
            name="E1_D3",
            unet_possibility_mrc_dir=r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\real_malaria\3-E1_D3\possibility",
            key_atom_ca_mrc=r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\real_malaria\3-E1_D3\possibility\E1_D3_2_7A_cryosparc_P8_J562_008_volume_map_sharp_key_atoms_1_possibility.mrc",
            secondary_struct_helix_mrc=r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\real_malaria\3-E1_D3\possibility\E1_D3_2_7A_cryosparc_P8_J562_008_volume_map_sharp_secondary_strctures_1_possibility.mrc",
            clustered_ca_pdb=r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\E1_D3_meanshift_bw_1.9_CA_only.pdb",
        )
        run_pipeline(sample=test_sample, output_dir_root=output_dir_root)
        sys.exit("Finished single sample run")

    # ========== Batch mode ==========
    from R_B8_clustering_eval_driver import get_column_values, find_emdb_dirs, clustering_result_exists
    import re
    import json

    csv_file = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\TestSet\New\ready_to_test_new.csv"
    sample_list = csv_file
    # sample_list = [28584]    # TODO: run this individually to see why tf it took so long. Is it the sample itself? or just bad internet connection? or gdrive rate limiting?
    #! WTF THIS FUCKASS SHIT STOPPED AGAIN, lowk just run the remaining ones manually fuck it. only have to change the id in line 361.
    #! also pipe the output into a txt file next time. there were some other msgs
    base_input_dir = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\TestSet\Raw"
    # TODO: convert this to an absolute path, just use pathlib
    clustering_cache_dir = "H_Backbone_tracing/misc_outputs/clustering_figure"
    output_dir_root = Path(r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\misc_outputs\batch_backbone_tracing")
    log_file = output_dir_root / "samples.json"
    fail_log = output_dir_root / "failed_samples.txt"
    # Clear log file every run. Change to append if want to save old logs
    with open(log_file, "w") as f:
        f.write("")
    with open(fail_log, "w") as f:
        f.write("Failed samples:\n")

    if type(sample_list) is str:
        emdb_ids = get_column_values(sample_list, "emdb_id")
        # print(emdb_ids)
        emdb_ids = [x.removeprefix('EMD-') for x in emdb_ids]
        # print(emdb_ids)
    elif type(sample_list) is list:
        if not all(type(x) is int for x in sample_list):
            raise TypeError("sample_list list must only contain ints")
        emdb_ids = sample_list
    else:
        raise TypeError("sample_list needs to be a string OR list of ints")

    # print(emdb_ids)
    def natural_sort_key(s):
        # Splits the string into numbers and text; converts numbers to int
        return [int(text) if text.isdigit() else text.lower() 
                for text in re.split('([0-9]+)', str(s))]
    emdb_ids = sorted(emdb_ids, key=natural_sort_key)
    # print(emdb_ids); assert False

    emdb_folders = find_emdb_dirs(base_input_dir, emdb_ids)
    # for key, value in emdb_folders.items():     # dict[str, str]
    #     print(f"{key}: {value}")
    # assert False

    # TODO (cosmetic/good practice): just realized sample_list is used above as a dif variable too, does not matter since the above one is no longer needed, but would be good practice to use dif names
    sample_list = []
    failed_samples = []
    for key, value in emdb_folders.items():
        # TODO: make some fxns/use some fxns i already have for discovering these files, esp if there are multiple clustering ones with dif settings. Hardcoding it in these string patterns makes it easier to debug but more brittle to future changes. But ehh would rather it break if im using the wrong files. Just have user (me) specify the exact folders at the top. Just throw an error msg if the folder structure inside is wrong. OR, just add a pattern variable at the top too with the base dirs, like base_dir + pattern. Could also just create/find the list of files outside of the loop
        # clustered_ca_pdb = clustering_result_exists(clustering_cache_dir, key)
        # print(clustered_ca_pdb); assert False
        unet_poss_dir = Path(value) / "possibility"
        sample = Sample(
            name="EMD_" + key,
            unet_possibility_mrc_dir=unet_poss_dir,
            key_atom_ca_mrc=unet_poss_dir / f"emd_{key}.map_key_atoms_1_possibility.mrc",
            secondary_struct_helix_mrc=unet_poss_dir / f"emd_{key}.map_secondary_strctures_1_possibility.mrc",
            clustered_ca_pdb=Path(clustering_cache_dir) / f"emd_{key}_meanshift_bw_1.9_CA_only.pdb",
        )
        # print(json.dumps(sample.to_dict(), indent=4)); assert False
        try:
            run_pipeline(sample, output_dir_root)
        except Exception as e:
            failed_samples.append(key)   # NOTE: dont rly need this anymore, since moved the logging into the loop, but can output all the failed samples at the end of the run
            print(f"[ERROR] Sample {key} failed: {e}")
            with open(fail_log, "a") as f:
                f.write(str(key) + "\n")
            continue
        sample_list.append(sample)
        with open(log_file, "w") as f:
            # rewrites the log file every iteration (so it has proper json format), but adds negligible overhead with the number of iterations/samples
            json.dump([s.to_dict() for s in sample_list], f, indent=4)
        # print(json.dumps(sample.to_dict(), indent=4)); assert False

    # for sample in sample_list:
    #     print(json.dumps(sample.to_dict(), indent=4))
    #     print()
    # assert False

    print(f"Dumped sample jsons to: {log_file}")
    if not failed_samples:
        print(f"All runs succeeded: {len(emdb_ids)}/{len(emdb_ids)}")
    else:
        print(f"{len(emdb_ids) - len(failed_samples)}/{len(emdb_ids)} runs succeeded")
        print(f"Failed these runs: {failed_samples}")
        print(f"Failed samples are also logged to: {fail_log}")
    print("="*50); assert False
