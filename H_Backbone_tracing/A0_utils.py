# TODO start migrating some functions here
# TODO: add this to A1....py. Check the shape of the numpy array in current code b4 adding. Use a dif sample (any of the gproteins) to test this change
# check this shit...

import numpy as np

def load_coords_from_pdb(pdb_path):
    """
    Load 3D coordinates from a PDB file previously saved by this pipeline.
    Returns a numpy array of shape (N, 3) in the same real-world coordinates as used in clustering.
    
    Only reads HETATM lines (ignores CONECT lines).
    """
    coords = []
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith(("HETATM", "ATOM")):
                try:
                    # PDB format: columns for x,y,z are at positions 30:38, 38:46, 46:54
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append((x, y, z))
                except ValueError:
                    # skip malformed lines
                    continue
    if coords:
        return np.array(coords, dtype=float)
    else:
        return np.empty((0, 3), dtype=float)


if __name__ == "__main__":
    pdb_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\10_7_25\real_malaria_1_agglomerative_cd_4.0_data_alone_20251008_173308.pdb"
    pdb_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\10_7_25\real_malaria_1_agglomerative_cd_4.0_data_plus_loop_20251008_173316.pdb"
    centroids = load_coords_from_pdb(pdb_file)

    print(f"Loaded {len(centroids)} centroids from {pdb_file}")
