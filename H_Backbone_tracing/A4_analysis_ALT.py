import os
import re
from dataclasses import dataclass
import numpy as np
import mrcfile
from collections import defaultdict
from Bio.PDB import Structure, Model, Chain, Residue, Atom, PDBIO



@dataclass
class CAAtom:
    coord: np.ndarray        # (x, y, z) in Å
    gt_res: str              # ground truth residue (e.g. "ALA")
    exp_res: str | None = None


# ----------------------------- Function 1 -----------------------------
def read_calpha_from_pdb(pdb_path, save_pdb=False):
    """
    Read C-alpha atoms from a PDB file.

    Returns
    -------
    atoms : list[CAAtom]
        Ordered list of CA atoms with coordinates (x, y, z) and
        ground-truth residue types.
    """
    atoms = []    # ca atoms only currently, could make this function more flexible by accepting parameters for what atoms to read in

    with open(pdb_path, "r") as f:
        for line in f:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            atom_name = line[12:16].strip()
            # if atom_name != "CA" and atom_name != "C":
            if atom_name != "CA":
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                # Skip malformed coordinate lines
                continue

            resname = line[17:20].strip()
            if not resname:
                resname = "UNK"

            coord = np.array([x, y, z], dtype=float)

            atoms.append(
                CAAtom(
                    coord=coord,
                    gt_res=resname,
                    exp_res=None
                )
            )

    if not atoms:
        raise ValueError(f"No C-alpha atoms found in PDB: {pdb_path}")

    print(f"Loaded {len(atoms)} C-alpha atoms from {pdb_path}")

    # Optional intermediate output
    if save_pdb:
        with open(save_pdb, "w") as f:
            for i, atom in enumerate(atoms, start=1):
                x, y, z = atom.coord
                resname = atom.gt_res
                f.write(
                    f"HETATM{i:5d}  CA  {resname:<3}     1    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                )
        print(f"Saved C-alpha PDB to: {save_pdb}")

    # Some santiy checks
    # assert atoms[0].coord.shape == (3,)
    # print(atoms[0].coord.shape)
    # print(atoms[0].coord)  # should print [x y z]

    return atoms


# ----------------------------- Function 2 -----------------------------
RESIDUE_ORDER = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS',
    'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO',
    'SER', 'THR', 'TRP', 'TYR', 'VAL'
]


def sample_residue_types(
    atoms,
    residue_mrc_dir,
    main_mrc_path,
    radius=1,
    save_pdb=None,
):
    """
    Sample residue-type probability maps at CA atom positions.

    Parameters
    ----------
    atoms : list[CAAtom]
        CA atoms with (x, y, z) coordinates and ground-truth residues.
        This function fills atom.exp_res in place.
    residue_mrc_dir : str
        Directory containing 20 residue-type .mrc files.
    main_mrc_path : str
        Reference MRC map used for voxel size, origin, and shape.
    radius : int
        Sampling radius in voxels.
    save_pdb : str or None
        If provided, write a PDB with experimental residues.

    Returns
    -------
    None
    """

    # ---------------- Load reference map ----------------
    with mrcfile.open(main_mrc_path, permissive=True) as mrc:
        ref_data = np.squeeze(mrc.data)
        shape = ref_data.shape           # (z, y, x)
        voxel_size = float(mrc.voxel_size.x)
        origin = np.array([
            mrc.header.origin.x,
            mrc.header.origin.y,
            mrc.header.origin.z,
        ], dtype=float)

    # ---------------- Collect residue maps ----------------
    residue_files = sorted(
        f for f in os.listdir(residue_mrc_dir) if f.endswith(".mrc")
    )

    if not residue_files:
        raise ValueError(f"No .mrc files found in {residue_mrc_dir}")

    # Preload all residue maps
    residue_maps = []
    for fname in residue_files:
        path = os.path.join(residue_mrc_dir, fname)
        with mrcfile.open(path, permissive=True) as mrc:
            data = np.squeeze(mrc.data)
            if data.shape != shape:
                raise ValueError(
                    f"Shape mismatch for {fname}: {data.shape} vs {shape}"
                )
            residue_maps.append(data)

    # ---------------- Sample residues ----------------
    density_threshold = 0.5  # <-- NEW   (it should make no dif)

    radius_vox = int(np.ceil(radius / voxel_size))

    for atom in atoms:
        x, y, z = atom.coord  # Å, xyz

        # Convert Å → voxel coordinates (float)
        ix = (x - origin[0]) / voxel_size
        iy = (y - origin[1]) / voxel_size
        iz = (z - origin[2]) / voxel_size

        residue_scores = np.zeros(len(residue_maps), dtype=float)

        for dx in range(-radius_vox, radius_vox + 1):
            for dy in range(-radius_vox, radius_vox + 1):
                for dz in range(-radius_vox, radius_vox + 1):

                    vx = int(round(ix + dx))
                    vy = int(round(iy + dy))
                    vz = int(round(iz + dz))

                    if not (
                        0 <= vz < shape[0] and
                        0 <= vy < shape[1] and
                        0 <= vx < shape[2]
                    ):
                        continue

                    # Physical distance (Å)
                    dx_a = (vx - ix) * voxel_size
                    dy_a = (vy - iy) * voxel_size
                    dz_a = (vz - iz) * voxel_size
                    if dx_a*dx_a + dy_a*dy_a + dz_a*dz_a > radius*radius:
                        continue

                    # Accumulate density only if above threshold
                    for r_idx, data in enumerate(residue_maps):
                        val = data[vz, vy, vx]
                        if val >= density_threshold:
                            residue_scores[r_idx] += val

        best_idx = int(np.argmax(residue_scores))
        atom.exp_res = RESIDUE_ORDER[best_idx]


    # ---------------- Optional PDB output ----------------
    if save_pdb:
        with open(save_pdb, "w") as f:
            for i, atom in enumerate(atoms, start=1):
                x, y, z = atom.coord
                resname = atom.exp_res or "UNK"
                f.write(
                    f"HETATM{i:5d}  CA  {resname:<3}     1    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                )
        print(f"Saved experimental residue PDB to: {save_pdb}")

    print("Finished sampling experimental residue types.")



# ----------------------------- Function 3 -----------------------------
def evaluate_degenerate_group_coverage(
    atoms,
    residue_mrc_dir,
    main_mrc_path,
    degenerate_groups,
    radius=5.0,
    density_threshold=0.8,
):
    import os
    import re
    import numpy as np
    import mrcfile
    from collections import defaultdict

    # Canonical residue order (matches UNet output)
    RESIDUE_ORDER = [
        'ALA','ARG','ASN','ASP','CYS',
        'GLU','GLN','GLY','HIS','ILE',
        'LEU','LYS','MET','PHE','PRO',
        'SER','THR','TRP','TYR','VAL'
    ]

    # ---------------- Load reference map ----------------
    with mrcfile.open(main_mrc_path, permissive=True) as mrc:
        shape = np.squeeze(mrc.data).shape   # (z, y, x)
        voxel_size = float(mrc.voxel_size.x)

    radius_vox = int(np.ceil(radius / voxel_size))

    # ---------------- Load ALL residue maps ----------------
    residue_maps = {}  # residue -> data

    for fname in os.listdir(residue_mrc_dir):
        if not fname.endswith(".mrc"):
            continue

        # extract residue index from filename
        m = re.search(r"_([0-9]+)_possibility\.mrc$", fname)
        if not m:
            continue

        idx = int(m.group(1)) - 1
        if not (0 <= idx < len(RESIDUE_ORDER)):
            continue

        residue = RESIDUE_ORDER[idx]
        path = os.path.join(residue_mrc_dir, fname)

        with mrcfile.open(path, permissive=True) as mrc:
            data = np.squeeze(mrc.data)
            if data.shape != shape:
                raise ValueError(f"Shape mismatch for {fname}")
            residue_maps[residue] = data

    print("SANITY CHECK =-=-= Loaded residue maps:", residue_maps.keys())   # sanity check


    # ---------------- Group maps by degenerate group ----------------
    group_maps = defaultdict(list)
    for res, group in degenerate_groups.items():
        if res in residue_maps:
            group_maps[group].append(residue_maps[res])

    # ---------------- Evaluate coverage ----------------
    results = {}

    for group, maps in group_maps.items():
        gt_atoms = [a for a in atoms if degenerate_groups.get(a.gt_res) == group]
        covered = 0

        for atom in gt_atoms:
            x, y, z = atom.coord

            # MATCH WORKING PIPELINE: (z, y, x)
            vz0 = int(round(z / voxel_size))
            vy0 = int(round(y / voxel_size))
            vx0 = int(round(x / voxel_size))

            is_covered = False

            for dz in range(-radius_vox, radius_vox + 1):
                for dy in range(-radius_vox, radius_vox + 1):
                    for dx in range(-radius_vox, radius_vox + 1):

                        vz = vz0 + dz
                        vy = vy0 + dy
                        vx = vx0 + dx

                        if not (
                            0 <= vz < shape[0] and
                            0 <= vy < shape[1] and
                            0 <= vx < shape[2]
                        ):
                            continue

                        # physical radius check
                        if (dx*dx + dy*dy + dz*dz) * voxel_size**2 > radius*radius:
                            continue

                        for data in maps:
                            if data[vz, vy, vx] >= density_threshold:
                                is_covered = True
                                break

                        if is_covered:
                            break
                    if is_covered:
                        break
                if is_covered:
                    break

            if is_covered:
                covered += 1

        total = len(gt_atoms)
        accuracy = covered / total if total > 0 else 0.0

        results[group] = {
            "covered": covered,
            "total": total,
            "accuracy": accuracy,
        }

    # ---------------- Weighted overall accuracy ----------------
    total_atoms = sum(v["total"] for v in results.values())
    overall_accuracy = (
        sum(v["accuracy"] * v["total"] for v in results.values()) / total_atoms
        if total_atoms > 0 else 0.0
    )

    return results, overall_accuracy




# ----------------------------- Main Pipeline -----------------------------
def main():
    # ----------------------------- User Config -----------------------------
    # Input files
    CA_INPUT_FILE = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\8HK3.pdb"    # i actually have a cif file parsing function, so can imtegrate that too later
    RESIDUE_MRC_DIR = r"C:\Users\noelu\programming\python_projects\cryoID_v2_DATA\UnetPrediction\EMD-34843_re_3.2\possibility_residue_only"
    MAIN_MRC_MAP = r"C:\Users\noelu\programming\python_projects\cryoID_v2_DATA\UnetPrediction\EMD-34843_re_3.2\possibility\emd_34843.map_key_atoms_1_possibility.mrc"

    # (Optional) Intermediate outputs (file names up to you)
    SAVE_GT_PDB = "zz_calpha_gt_residues.pdb"              # Optional intermediate CA PDB
    SAVE_EXP_PDB = "zz_calpha_exp_residues.pdb"     # Optional predicted residues PDB

    # Parameters (for determining the experimental residue type)
    VOXEL_NEIGHBOR_RADIUS = 1  # Å

    # Degenerate groupings
    DEGENERATE_GROUPS = {
        # Group 1 (G)
        "GLY": "G", "ALA": "G", "SER": "G", "CYS": "G",
        "VAL": "G", "THR": "G", "ILE": "G",
        # Group 2 (P)
        "PRO": "P",
        # Group 3 (L)
        "LEU": "L", "ASP": "L", "ASN": "L",
        "GLU": "L", "GLN": "L", "MET": "L",
        # Group 4 (K)
        "LYS": "K", "ARG": "K",
        # Group 5 (Y)
        "HIS": "Y", "PHE": "Y", "TYR": "Y",
        # Group 6 (W)
        "TRP": "W",
    }
    # -----------------------------------------------------------------------



    print("Reading C-alpha atoms...")
    calpha_atoms = read_calpha_from_pdb(
        CA_INPUT_FILE, 
        save_pdb=SAVE_GT_PDB
    )
    print(f"Found {len(calpha_atoms)} C-alpha atoms.")
    # assert(False)
    print("1) ✅")

    print("Sampling residue types from MRC maps...")
    sample_residue_types(    # modifies calpha_atoms in place
        calpha_atoms,
        residue_mrc_dir=RESIDUE_MRC_DIR,
        main_mrc_path=MAIN_MRC_MAP,
        radius=VOXEL_NEIGHBOR_RADIUS,
        save_pdb=SAVE_EXP_PDB
    )
    print(len(calpha_atoms))  # number of atoms sampled
    print("2) ✅")

    # assert(False)

    print("Comparing experimental vs ground truth residues...")
    print("Evaluating degenerate group coverage...")
    results, overall_accuracy = evaluate_degenerate_group_coverage(
        calpha_atoms,
        residue_mrc_dir=RESIDUE_MRC_DIR,
        main_mrc_path=MAIN_MRC_MAP,
        degenerate_groups=DEGENERATE_GROUPS,
        radius=VOXEL_NEIGHBOR_RADIUS,
        density_threshold=0.8,
    )

    print("\nDegenerate Group Results:")
    print(f"(using voxel sampling radius = {VOXEL_NEIGHBOR_RADIUS})")
    # for group, r in results.items():
    for i, (group, r) in enumerate(results.items(), start=1):
        print(
            # f"  Group {group}: "
            f"  Group {i} ({group}): "
            f"recall = {r['accuracy']:.4f}, "    # TODO: should rename the metric in the code too, change 'accuracy' to 'recall' in the code above
            f"count = {r['total']}"
        )

    print(f"\nOverall weighted accuracy: {overall_accuracy:.4f}")

    print("3) ✅")


if __name__ == "__main__":
    main()
