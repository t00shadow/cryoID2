import A1_clustering
import A1_clustering_KEEP_MORE_COMPONENTS

import A2_backbone_tracing
import A2_backbone_tracing_ANGLES_TOO
import A2_backbone_tracing_ANGLES_TOO_copy    # same thing but tracks multiple best paths per starting point instead of global beam pruning

import A3_backbone_to_residue

# TODO: one/some of these imports have code that executes globally (gets executed on import), so either import specific functions or change that in the scripts

from contextlib import redirect_stdout, redirect_stderr
import os

import shutil

MODE = "single_output"
MODE = "multi_output"
# MODE = "one_off"    # one time use cases, hotfixes, etc



if MODE == "single_output":
    step1_output_path = A1_clustering.main()
    print("=" * 20, "\n")
    print(step1_output_path)

    try:
        step2_output_path = A2_backbone_tracing_ANGLES_TOO.main(input_file_path=step1_output_path)
        # step2_output_path = A2_backbone_tracing_ANGLES_TOO_copy.main(input_file_path=step1_output_path)
    except Exception as e:
        print("========== New code too strict, default to old code ==========")
        print(e)    
        try:
            step2_output_path = A2_backbone_tracing.main(input_file_path=step1_output_path)
        except Exception as e:
            print("========== Ok this specific prediction is just shit, old code also cant build valid backbone ==========")
            print(e)
    # step2 output path always uses "traced_backbone3.pdb", but it's fine to constantly overwrite this file since this is an intermediate step, buttt for future, should prob have a toggle that lets me produce a new file. Just overwrite it the step3's output file name, instead of this "globally" shared file name. Also make test cases/files to test each script. I already verified them, but just for future cases where i change things majorly
    print(step2_output_path)

    main_map = r"D:\NEW_TEST_DATA\UnetPrediction\EMD-38402_re_3.06\possibility\emd_38402.map_key_atoms_1_possibility.mrc"    # only need this for the size of the original file, so TODO: could rewrite this code a bit better, ie have A3....py find the filepath using the config file and config loaer script.
    # with open("zzz_stdout_dump.txt", "w") as f:   # send to a junk file if wanna actually read them
    with open(os.devnull, "w") as f:   # silence outputs, os.devnull is basically a garbage bin / blackhole, discards data so avoids the need to write to a file and then needing to delete the file later
        with redirect_stdout(f):
            step3_output_path = A3_backbone_to_residue.main(input_file_path=step2_output_path)
    print(step3_output_path)
    # Copy file.txt → copy_of_file.txt
    shutil.copy(step3_output_path, "H_Backbone_tracing/outputs/clustering_v2/10_7_25/step3_output.pdb")   #! for dev purposes it is easier to keep the file name constant in the A3....py script. So this MIGHT be better than modifying A3....py. Tho another approach is to add a parameter with a default argument, and leaving it blank will just have the same behavior as rn, where it uses the same file every time.


### =======================


elif MODE == "multi_output":
    step1_output_path_list = A1_clustering_KEEP_MORE_COMPONENTS.main()
    print("=" * 20, "\n")
    for item in step1_output_path_list:
        print(item)
    for i, item in enumerate(step1_output_path_list, start=1):
        try:
            # step2_output_path = A2_backbone_tracing_ANGLES_TOO.main(input_file_path=item)     #& run this again for old outputs
            step2_output_path = A2_backbone_tracing_ANGLES_TOO_copy.main(input_file_path=item)    #& run this again for new outputs
        except Exception as e:
            print("========== tracing code too strict or region just sucks ==========")
            # print("========== New code too strict, default to old code ==========")
            print(e)
            continue
            # try:    
            #     step2_output_path = A2_backbone_tracing.main(input_file_path=item)
            # except Exception as e:
            #     print("========== Ok this specific prediction is just shit, old code also cant build valid backbone ==========")
            #     print(e)
            #     continue
        # step2 output path always uses "traced_backbone3.pdb", but it's fine to constantly overwrite this file since this is an intermediate step, buttt for future, should prob have a toggle that lets me produce a new file. Just overwrite it the step3's output file name, instead of this "globally" shared file name. Also make test cases/files to test each script. I already verified them, but just for future cases where i change things majorly
        print(step2_output_path)

        #!!! [BUG] oh bruh i think this is the reason... need to change this file, build it from config file. EDIT: the real problem was the residue_dir in A3.py. This main_map which needs to be renamed... is just for the dimensions.
        # TODO: fix this. need to rerun emd_42791 too now
        # TODO: rename main_map in A3,py since this is NOT a map
        # main_map = r"D:\NEW_TEST_DATA\UnetPrediction\EMD-38402_re_3.06\possibility\emd_38402.map_key_atoms_1_possibility.mrc"    # only need this for the size of the original file, so TODO: could rewrite this code a bit better, ie have A3....py find the filepath using the config file and config loaer script.
        main_map = r"D:\NEW_TEST_DATA\UnetPrediction\real_malaria\1\possibility\cryosparc_P164_J166_006_volume_map_key_atoms_1_possibility.mrc"    # only need this for the size of the original file, so TODO: could rewrite this code a bit better, ie have A3....py find the filepath using the config file and config loaer script.
        residue_possibility_dir_for_A3_script = r"D:\NEW_TEST_DATA\UnetPrediction\real_malaria\1\possibility"
        # with open("zzz_stdout_dump.txt", "w") as f:   # send to a junk file if wanna actually read them
        with open(os.devnull, "w") as f:   # silence outputs, os.devnull is basically a garbage bin / blackhole, discards data so avoids the need to write to a file and then needing to delete the file later
            with redirect_stdout(f):
                step3_output_path = A3_backbone_to_residue.main(input_file_path=step2_output_path, main_map_path=main_map, residue_dir_path=residue_possibility_dir_for_A3_script)   #! bruh i didnt even set main_map_path=main_map earlier, but since it worked, 34843 and 38402 prob have same dimensions for unet possibilities?
        print(step3_output_path)
        # Copy file.txt → copy_of_file.txt
        shutil.copy(step3_output_path, f"H_Backbone_tracing/outputs/clustering_v2/10_7_25/10_10_25_rerun2/step3_output_{i}.pdb")   #! for dev purposes it is easier to keep the file name constant in the A3....py script. So this MIGHT be better than modifying A3....py. Tho another approach is to add a parameter with a default argument, and leaving it blank will just have the same behavior as rn, where it uses the same file every time.
    
    # put results in fasta file
    from A0_util_pdbtofasta import find_pdb_files, sort_pdb_files_by_natural_number, write_fasta_file

    output_file = "malaria_predictions_new.fasta"
    folder = f"H_Backbone_tracing/outputs/clustering_v2/10_7_25/10_10_25_rerun2"
    pattern = "step3_output_*.pdb"
    pdb_files = find_pdb_files(folder, pattern)
    pdb_files = sort_pdb_files_by_natural_number(pdb_files)
    for file in pdb_files:
        print(file)    # file name
    print("=" * 50)

    print(f"Found {len(pdb_files)} files.")
    write_fasta_file(pdb_files, output_file=output_file, verbose=True)
    print(f"Converted {len(pdb_files)} PDB files to FASTA format.")



### =======================


elif MODE == "one_off":    # input: folder of pdb files (path to that folder), pattern of files to match. output: single fasta file
    from A0_util_pdbtofasta import find_pdb_files, sort_pdb_files_by_natural_number, write_fasta_file
    folder = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\10_7_25\overwritten_here_but_recovered_from_chimerax_session\script_safe_names"
    pattern = "step3_output_*.pdb"
    output_file = "malaria_predictions_FIXED.fasta"
    residue_possibility_dir_for_A3_script = r"D:\NEW_TEST_DATA\UnetPrediction\real_malaria\1\possibility"
    
    pdb_files = find_pdb_files(folder, pattern)
    pdb_files = sort_pdb_files_by_natural_number(pdb_files)
    for file in pdb_files:
        print(file)    # file name
    print("=" * 50)
    # assert(False)

    # only need to run A3.py here
    # TODO do see what happens when you feed A2 through itself (it only uses nodes btw, just ignores edges. in short test just now, basically no dif, but if u have another script that first removes some obviously junk nodes, then it might help. Draw this pipeline out on a whiteboard to see where should target for improvement)
    for i, item in enumerate(pdb_files, start=1):
        # programmatically keeps only the good entries, so ONLY using the stricter tracer
        try:
            # step2_output_path = A2_backbone_tracing_ANGLES_TOO.main(input_file_path=item)     #& run this again for old outputs
            step2_output_path = A2_backbone_tracing_ANGLES_TOO_copy.main(input_file_path=item)    #& run this again for new outputs
        except Exception as e:
            print("========== tracing code too strict or region just sucks ==========")
            # print("========== New code too strict, default to old code ==========")
            print(e)
            continue
            # try:    
            #     step2_output_path = A2_backbone_tracing.main(input_file_path=item)
            # except Exception as e:
            #     print("========== Ok this specific prediction is just shit, old code also cant build valid backbone ==========")
            #     print(e)
            #     continue
        # step2 output path always uses "traced_backbone3.pdb", but it's fine to constantly overwrite this file since this is an intermediate step, buttt for future, should prob have a toggle that lets me produce a new file. Just overwrite it the step3's output file name, instead of this "globally" shared file name. Also make test cases/files to test each script. I already verified them, but just for future cases where i change things majorly
        print(step2_output_path)

        # TODO: see above in the multi_output mode for comments, like found main bug was in A3.py, and should rename main_map in A3.py. And also yeah still need to rerun for emd_42791
        main_map = r"D:\NEW_TEST_DATA\UnetPrediction\real_malaria\1\possibility\cryosparc_P164_J166_006_volume_map_key_atoms_1_possibility.mrc"    # only need this for the size of the original file, so TODO: could rewrite this code a bit better, ie have A3....py find the filepath using the config file and config loaer script.
        with open("zzz_stdout_dump.txt", "w") as f:   # send to a junk file if wanna actually read them
        # with open(os.devnull, "w") as f:   # silence outputs, os.devnull is basically a garbage bin / blackhole, discards data so avoids the need to write to a file and then needing to delete the file later
            with redirect_stdout(f):
                # step3_output_path = A3_backbone_to_residue.main(input_file_path=step2_output_path)
                step3_output_path = A3_backbone_to_residue.main(input_file_path=step2_output_path, main_map_path=main_map, residue_dir_path=residue_possibility_dir_for_A3_script)
                # step3_output_path = A3_backbone_to_residue.main(input_file_path=item, main_map_path=main_map, residue_dir_path=residue_possibility_dir_for_A3_script)
        print(f"{step3_output_path} progress = {i} / {len(pdb_files)}")
        # Copy file.txt → copy_of_file.txt
        # shutil.copy(step3_output_path, f"H_Backbone_tracing/outputs/clustering_v2/10_7_25/step3_output_{i}.pdb")   #! for dev purposes it is easier to keep the file name constant in the A3....py script. So this MIGHT be better than modifying A3....py. Tho another approach is to add a parameter with a default argument, and leaving it blank will just have the same behavior as rn, where it uses the same file every time.
        shutil.copy(step3_output_path, f"H_Backbone_tracing/outputs/clustering_v2/10_7_25/10_10_25_rerun/step3_output_{i}.pdb")   #! for dev purposes it is easier to keep the file name constant in the A3....py script. So this MIGHT be better than modifying A3....py. Tho another approach is to add a parameter with a default argument, and leaving it blank will just have the same behavior as rn, where it uses the same file every time.

    # now need to glob again for the output files
    # folder = f"H_Backbone_tracing/outputs/clustering_v2/10_7_25"
    folder = f"H_Backbone_tracing/outputs/clustering_v2/10_7_25/10_10_25_rerun"
    pattern = "step3_output_*.pdb"
    pdb_files = find_pdb_files(folder, pattern)
    pdb_files = sort_pdb_files_by_natural_number(pdb_files)
    for file in pdb_files:
        print(file)    # file name
    print("=" * 50)

    print(f"Found {len(pdb_files)} files.")
    write_fasta_file(pdb_files, output_file=output_file, verbose=True)
    print(f"Converted {len(pdb_files)} PDB files to FASTA format.")