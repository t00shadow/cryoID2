import mrcfile
import numpy as np
from sklearn.ensemble import IsolationForest    # unneeded now
from sklearn.cluster import MeanShift
from timeit import default_timer as timer    # for measuring execution time
import math
import os
#print(os.getcwd())
#print(__file__)



def clustering_fxn(UNET_FILE, OG_MRC_FILE, unet_threshold=0.5, og_mrc_threshold=0.1, meanshift_bandwith=3, FILTEROUTLIERSFLAG=False): 
    '''
        input: original mrc densities, unet probabilites
        output: numpy array of representative coordinates

        TODO: maybe rename the threshold arguments above to "probability_threshold" and "density_threshold"
    '''

    # Toggle whether to filter outliers or not, uses sklearn IsolationForest
    FILTEROUTLIERSFLAG = FILTEROUTLIERSFLAG

    # Threshold
    threshold = unet_threshold     # unet threshold

    # Read MRC data
    FILE = UNET_FILE
    #TODO: calculate num_atoms instead of explicitly setting it
    #num_atoms = 4
    #num_atoms = 170
    #num_atoms = 50

    with mrcfile.open(FILE) as mrc:
        probability = mrc.data
        probability = np.squeeze(probability)

    # Convert 3D densities to 2D array of shape (n, 3)
    coordinates = np.argwhere(probability >= threshold)      # returns the indices of densities that are above threshold, NOTE: should it be > or >= ?
    print("total:", coordinates.shape)

    #num_atoms = round(coordinates.shape[0] / 20)          # TODO: mess around with this calculation
    num_atoms = math.ceil(coordinates.shape[0] / 20)          # round rounds 0.5s to nearest evens (bankers rounding), so can round down. Honestly just use ceiling cuz overestimate is better than underestimate, also why meanshift is prob the way to go
    #num_atoms = 59


    #NOTE: new stuff start
    FILE2 = OG_MRC_FILE

    with mrcfile.open(FILE2) as mrc:
        density = mrc.data
        density = np.squeeze(density)

    # density_threshold = 0.255     # the current level value i have in chimerax
    density_threshold = og_mrc_threshold      # for the original mrc file, different from "threshold" which is for the unet probabilites. Not sure how to automatically set this value, prob should depend on orginal mrc file resolution, soooo read header.
    junkcount = 0
    for i in range(coordinates.shape[0]):
        # print(i)
        if (density[coordinates[i, 0], coordinates[i, 1], coordinates[i, 2]] < density_threshold):   # NOTE: this threshold should not be 0.5 like for unet probabilities
            junkcount += 1
            coordinates[i] = [-1, -1, -1]             # mark for removal

    print("rows to remove from coordinates:", junkcount)
    rows_del = np.squeeze(np.argwhere(np.all(np.isin(coordinates, [-1,-1,-1]), axis=1)))
    print(rows_del.shape)
    print(coordinates.shape, "<-- before")
    coordinates = np.delete(coordinates, rows_del, 0)
    print(coordinates.shape, "<-- after")
    #NOTE: new stuff end


    if FILTEROUTLIERSFLAG == True:
        # Apply outlier detection
        outlier_detector = IsolationForest(contamination='auto', random_state=0)  # Adjust contamination parameter as needed
        outlier_labels = outlier_detector.fit_predict(coordinates)
        outliers = coordinates[outlier_labels == -1]
        inliers = coordinates[outlier_labels == 1]
        print("inliers:", inliers.shape)
        print("outliers:", outliers.shape)
        coordinates = inliers


    # Apply Meanshift
    start = timer()
    meanshift = MeanShift(bandwidth=meanshift_bandwith)  # Adjust bandwidth as needed
    meanshift.fit(coordinates)
    cluster_centers = meanshift.cluster_centers_
    cluster_centers = np.round(cluster_centers, 3)
    end = timer()
    print(f"Meanshift execution time: {end - start}s")
    print("cluster_centers shape:", cluster_centers.shape)
    #print(cluster_centers)
    print("return type:", type(cluster_centers))
    assert(str(type(cluster_centers)) == "<class 'numpy.ndarray'>")       # check return type
    return cluster_centers



def write_as_PDB(final_coords, output_file_name):   # add additional args as needed
    '''
    :final_coords: expects z,y,x
    '''    

    with open(output_file_name, "w") as text_file:
        for i in range(final_coords.shape[0]):
            print(f"ATOM  {str(i+1):>5} {' C':<4}{' '}{'HIS':>3} {'A'}{'1':>4}{' '}   {final_coords[i][2]:>8}{final_coords[i][1]:>8}{final_coords[i][0]:>8}{'1.00':>6}{'16.00':>6}{'':6}{' ':<4}{'C':>2}", file=text_file)





# Example Usage
# foo = r"C:\Users\noelu\Downloads\Testing clustering-20240514T081052Z-001(DELETE LATER)\Testing clustering\emd_0065\emd_0065.map_key_atoms_2_possibility.mrc"
# bar = r"C:\Users\noelu\Downloads\Testing clustering-20240514T081052Z-001(DELETE LATER)\Testing clustering\emd_0065\emd_0065_normalized.mrc"
foo = r"C:\Users\noelu\Downloads\drive-download-20250702T231250Z-1-001\emd_0065.map_key_atoms_2_possibility.mrc"    # nucleic acid phosphoruses (P)
foo = r"C:\Users\noelu\Downloads\drive-download-20250702T231250Z-1-001\emd_0065.map_key_atoms_1_possibility.mrc"    # protein (alpha carbons, CA)
bar = r"C:\Users\noelu\Downloads\drive-download-20250702T231250Z-1-001\emd_0065_normalized.mrc"

THRESHOLD = 0.25
output = clustering_fxn(UNET_FILE=foo, OG_MRC_FILE=bar, unet_threshold=THRESHOLD)
#print(output)

# Write to PDB file
# TODO: make a wrapper fxn
dirname = os.path.dirname(__file__)
#print(dirname)
outputfilename = os.path.join(dirname, 'output/meanshift_clusters' + "_threshold_" + str(THRESHOLD).replace(".", "_") + '.pdb')
#print(outputfilename)
write_as_PDB(output, outputfilename)
print("Wrote to", outputfilename)