# Here's some example usage

import os
from clustering import clustering_fxn, write_as_PDB
from backbone_tracing import trace_backbone


######################
# Clustering
######################
print("=" * 50)
print("Clustering")
print("=" * 50)

foo = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\test_files\emd_0065_Unet\emd_0065.map_key_atoms_2_possibility.mrc"    # nucleic acid phosphoruses (P)
foo = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\test_files\emd_0065_Unet\emd_0065.map_key_atoms_1_possibility.mrc"    # protein (alpha carbons, CA)
bar = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\test_files\emd_0065_Unet\emd_0065_normalized.mrc"

THRESHOLD = 0.25
output = clustering_fxn(UNET_FILE=foo, OG_MRC_FILE=bar, unet_threshold=THRESHOLD)
dirname = os.path.dirname(__file__)
outputfilename = os.path.join(dirname, 'outputs/clustering/meanshift_clusters' + "_threshold_" + str(THRESHOLD).replace(".", "_") + '.pdb')
write_as_PDB(output, outputfilename)
print("Wrote to", outputfilename)



######################
# Backbone Tracing
######################
print("=" * 50)
print("Backbone Tracing")
print("=" * 50)

input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering\meanshift_clusters_threshold_0_25.pdb"
# joblib_file = "my_trained_model.joblib"      #* used to refine the backbone after TSP
output_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\tracing\protein_backbone.pdb"
trace_backbone("protein", input_file=input_file, output_file=output_file)