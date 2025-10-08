import A1_clustering
import A1_clustering_KEEP_MORE_COMPONENTS

import A2_backbone_tracing
import A2_backbone_tracing_ANGLES_TOO
import A2_backbone_tracing_ANGLES_TOO_copy    # same thing but tracks multiple best paths per starting point instead of global beam pruning

import A3_backbone_to_residue

from contextlib import redirect_stdout, redirect_stderr
import os

import shutil

MODE = "single_output"
MODE = "multi_output"



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


if MODE == "multi_output":
    step1_output_path_list = A1_clustering_KEEP_MORE_COMPONENTS.main()
    print("=" * 20, "\n")
    for item in step1_output_path_list:
        print(item)
    for i, item in enumerate(step1_output_path_list, start=1):
        try:
            # step2_output_path = A2_backbone_tracing_ANGLES_TOO.main(input_file_path=item)     #& run this again for old outputs
            step2_output_path = A2_backbone_tracing_ANGLES_TOO_copy.main(input_file_path=item)    #& run this again for new outputs
        except Exception as e:
            print("========== New code too strict, default to old code ==========")
            print(e)
            try:    
                step2_output_path = A2_backbone_tracing.main(input_file_path=item)
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
        shutil.copy(step3_output_path, f"H_Backbone_tracing/outputs/clustering_v2/10_7_25/step3_output_{i}.pdb")   #! for dev purposes it is easier to keep the file name constant in the A3....py script. So this MIGHT be better than modifying A3....py. Tho another approach is to add a parameter with a default argument, and leaving it blank will just have the same behavior as rn, where it uses the same file every time.

