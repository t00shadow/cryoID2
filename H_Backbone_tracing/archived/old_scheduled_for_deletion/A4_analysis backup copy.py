import os
import re
import numpy as np
import mrcfile
from collections import defaultdict
from Bio.PDB import Structure, Model, Chain, Residue, Atom, PDBIO


# ----------------------------- User Config -----------------------------
# Modify these paths / parameters
CA_INPUT_FILE = r"C:\Users\noelu\programming\python_projects\cryoID_v2_DATA\UnetPrediction\EMD-34843_re_3.2\8hk3.cif"
RESIDUE_MRC_DIR = r"C:\Users\noelu\programming\python_projects\cryoID_v2_DATA\UnetPrediction\EMD-34843_re_3.2\possibility_residue_only"
MAIN_MRC_MAP = r"C:\Users\noelu\programming\python_projects\cryoID_v2_DATA\UnetPrediction\EMD-34843_re_3.2\possibility\emd_34843.map_key_atoms_1_possibility.mrc"

SAVE_C_ALPHA_PDB = "zz_calpha.pdb"              # Optional intermediate CA PDB
SAVE_EXP_PDB = "experimental_residues.pdb"     # Optional predicted residues PDB

VOXEL_NEIGHBOR_RADIUS = 1.5  # Å
DEGENERATE_GROUPS = {
    # optional example: group 20 aa into 6 pseudo groups
    "ALA": "A", "VAL": "A", "LEU": "A", "ILE": "A",
    "ASP": "B", "GLU": "B", "ASN": "B", "GLN": "B",
    "LYS": "C", "ARG": "C", "HIS": "C",
    "PHE": "D", "TYR": "D", "TRP": "D",
    "SER": "E", "THR": "E", "CYS": "E", "MET": "E",
    "PRO": "F", "GLY": "F"
}
# -----------------------------------------------------------------------


# ----------------------------- Function 1 -----------------------------
def read_calpha_from_pdb(pdb_file, save_pdb=None):
    """
    Reads C-alpha atoms from a PDB file.
    Returns:
        coords: np.ndarray of shape (N,3) in Å (z,y,x)
        gt_dict: {atom_index: residue_type}
    """
    coords = []
    gt_dict = {}
    with open(pdb_file, 'r') as f:
        for idx, line in enumerate(f):
            if (line.startswith("ATOM") and line[13:15].strip() == "CA") or \
               (line.startswith("HETATM") and line[13:15].strip() == "C"):
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords.append((z, y, x))  # flip to z,y,x for MRC
                resname = line[17:20].strip()
                gt_dict[idx] = resname
    coords = np.array(coords, dtype=float)

    if save_pdb:
        with open(save_pdb, 'w') as f:
            for i, (z, y, x) in enumerate(coords, 1):
                resname = gt_dict[i-1]
                f.write(f"HETATM{i:5d}  C   {resname:<3}     1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        print(f"Saved C-alpha PDB to: {save_pdb}")

    return coords, gt_dict


# ----------------------------- Function 2 -----------------------------
def sample_residue_types(calpha_coords,
                         gt_dict,
                         residue_dir,
                         main_map_path,
                         radius=VOXEL_NEIGHBOR_RADIUS,
                         sampling_fn=None,
                         save_pdb=None):
    """
    Samples residue types at CA atom positions using residue_type MRC maps.
    Returns:
        exp_dict: {atom_index: predicted_residue}
    """
    if sampling_fn is None:
        # Default: weighted average within voxel neighborhood
        def sampling_fn(atom_coord, data, voxel_size, origin):
            zi = int(round((atom_coord[0] - origin[0]) / voxel_size[0]))
            yi = int(round((atom_coord[1] - origin[1]) / voxel_size[1]))
            xi = int(round((atom_coord[2] - origin[2]) / voxel_size[2]))
            zs, ys, xs = np.ogrid[-radius:radius+1, -radius:radius+1, -radius:radius+1]
            coords = (zi+zs, yi+ys, xi+xs)
            mask = (
                (coords[0] >= 0) & (coords[0] < data.shape[0]) &
                (coords[1] >= 0) & (coords[1] < data.shape[1]) &
                (coords[2] >= 0) & (coords[2] < data.shape[2])
            )
            vals = data[coords[0][mask], coords[1][mask], coords[2][mask]]
            return np.mean(vals) if vals.size else 0.0

    # Load main map voxel info
    with mrcfile.open(main_map_path, permissive=True) as mrc:
        nx, ny, nz = mrc.header.nx, mrc.header.ny, mrc.header.nz
        cella = mrc.header.cella
        voxel_size = (cella.z/nz, cella.y/ny, cella.x/nx)  # z,y,x
        origin = (0.0, 0.0, 0.0)
        # shape = mrc.data.shape
        shape = (nx, ny, nz)
        print("shapeasdfasdfasf:", shape)

    # Load residue MRC files
    residue_files = [f for f in os.listdir(residue_dir) if f.endswith(".mrc")]
    residue_data = {}
    for f in residue_files:
        match = re.search(r"_([0-9]+)_possibility\.mrc$", f)
        if not match:
            continue
        idx = int(match.group(1))-1
        residues_protein = ['ALA','ARG','ASN','ASP','CYS','GLU','GLN','GLY','HIS','ILE',
                            'LEU','LYS','MET','PHE','PRO','SER','THR','TRP','TYR','VAL']
        resname = residues_protein[idx] if idx < len(residues_protein) else f"RES{idx+1}"
        with mrcfile.open(os.path.join(residue_dir,f), permissive=True) as mrc:
            data = np.squeeze(mrc.data)
            if data.shape != shape:
                raise ValueError(f"Shape mismatch for {f}: {data.shape} vs {shape}")
            residue_data[resname] = data

    # Sample each atom
    exp_dict = {}
    for idx, atom_coord in gt_dict.items():
        scores = {res: sampling_fn(calpha_coords[idx], data, voxel_size, origin)
                  for res, data in residue_data.items()}
        exp_dict[idx] = max(scores, key=scores.get)

    # Optional: save predicted residues as PDB
    if save_pdb:
        structure = Structure.Structure("Experimental")
        model = Model.Model(0)
        structure.add(model)
        chain = Chain.Chain("A")
        model.add(chain)
        for i, coord in enumerate(calpha_coords):
            resname = exp_dict[i]
            residue = Residue.Residue((' ', i+1, ' '), resname, '')
            chain.add(residue)
            atom = Atom.Atom("CA", coord, 0.0, 1.0, " ", "CA", i, element="C")
            residue.add(atom)
        io = PDBIO()
        io.set_structure(structure)
        io.save(save_pdb)
        print(f"Saved experimental PDB to: {save_pdb}")

    return exp_dict


# ----------------------------- Function 3 -----------------------------
def compare_residues(gt_dict, exp_dict, degenerate_groups=None):
    """
    Compare ground truth vs experimental residues.
    Returns:
        accuracy (float): fraction of matches
        mismatches (list of tuples): [(atom_index, gt, exp), ...]
    """
    if degenerate_groups:
        gt_dict = {k: degenerate_groups.get(v,v) for k,v in gt_dict.items()}
        exp_dict = {k: degenerate_groups.get(v,v) for k,v in exp_dict.items()}

    mismatches = []
    matches = 0
    for k in gt_dict:
        if gt_dict[k] == exp_dict[k]:
            matches += 1
        else:
            mismatches.append((k, gt_dict[k], exp_dict[k]))
    accuracy = matches / len(gt_dict)
    print(f"Accuracy: {accuracy*100:.2f}%")
    print(f"Mismatches: {mismatches}")
    return accuracy, mismatches


# ----------------------------- Main Pipeline -----------------------------
def main():
    print("Reading C-alpha atoms...")
    calpha_coords, gt_dict = read_calpha_from_pdb(CA_INPUT_FILE, save_pdb=SAVE_C_ALPHA_PDB)
    calpha_ids = list(gt_dict.keys())
    print(f"Found {len(calpha_coords)} C-alpha atoms.")

    print("Sampling residue types from MRC maps...")
    exp_dict = sample_residue_types(calpha_coords,
                                    gt_dict,
                                    RESIDUE_MRC_DIR,
                                    MAIN_MRC_MAP,
                                    radius=VOXEL_NEIGHBOR_RADIUS,
                                    save_pdb=SAVE_EXP_PDB)

    print("Comparing experimental vs ground truth residues...")
    compare_residues(gt_dict, exp_dict, degenerate_groups=DEGENERATE_GROUPS)


if __name__ == "__main__":
    main()
