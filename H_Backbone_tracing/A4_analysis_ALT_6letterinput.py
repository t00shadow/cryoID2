# This file is only for the data analysis portion. Input is a single emdb entry/sample's filepaths.
# If you want to feed in a list of entries (auto-find file paths, etc), do that in the driver file.

import os
import re
import numpy as np
import mrcfile
from dataclasses import dataclass


# ============================================================
# CONFIG
# ============================================================

GROUP_ORDER = ["G", "P", "L", "K", "Y", "W"]  # must match CNN channel order

DEGENERATE_GROUPS = {
    # Group G
    "GLY": "G", "ALA": "G", "SER": "G", "CYS": "G",
    "VAL": "G", "THR": "G", "ILE": "G",
    # Group P
    "PRO": "P",
    # Group L
    "LEU": "L", "ASP": "L", "ASN": "L",
    "GLU": "L", "GLN": "L", "MET": "L",
    # Group K
    "LYS": "K", "ARG": "K",
    # Group Y
    "HIS": "Y", "PHE": "Y", "TYR": "Y",
    # Group W
    "TRP": "W",
}

VOXEL_NEIGHBOR_RADIUS = 1.0


# ============================================================
# DATA CLASS
# ============================================================

@dataclass
class CAAtom:
    coord: np.ndarray
    gt_res: str
    exp_res: str | None = None


# ============================================================
# 1) READ CA ATOMS
# ============================================================

def read_calpha_from_pdb(pdb_path):
    atoms = []

    with open(pdb_path, "r") as f:
        for line in f:
            if not line.startswith(("ATOM", "HETATM")):
                continue

            if line[12:16].strip() != "CA":
                continue

            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue

            resname = line[17:20].strip() or "UNK"

            atoms.append(
                CAAtom(
                    coord=np.array([x, y, z], dtype=float),
                    gt_res=resname
                )
            )

    if not atoms:
        raise ValueError("No C-alpha atoms found.")

    print(f"Loaded {len(atoms)} CA atoms.")
    return atoms


def read_calpha_from_cif(cif_path):
    atoms = []

    with open(cif_path, "r") as f:
        lines = f.readlines()

    # Find the atom_site loop
    atom_site_start = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "loop_":
            # Check if next lines define _atom_site fields
            j = i + 1
            fields = []
            while j < len(lines) and lines[j].strip().startswith("_atom_site."):
                fields.append(lines[j].strip())
                j += 1

            if fields:
                atom_site_start = (i, j, fields)
                break

    if atom_site_start is None:
        raise ValueError("No _atom_site loop found in CIF file.")

    _, data_start, fields = atom_site_start

    # Map column names to indices
    field_map = {name: idx for idx, name in enumerate(fields)}

    required_fields = [
        "_atom_site.label_atom_id",
        "_atom_site.label_comp_id",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
    ]

    for rf in required_fields:
        if rf not in field_map:
            raise ValueError(f"Missing required CIF field: {rf}")

    # Read atom records until blank line or next loop
    i = data_start
    while i < len(lines):
        line = lines[i].strip()

        if not line or line.startswith("#") or line.startswith("loop_"):
            break

        parts = line.split()
        if len(parts) < len(fields):
            i += 1
            continue

        atom_name = parts[field_map["_atom_site.label_atom_id"]]
        if atom_name != "CA":
            i += 1
            continue

        try:
            x = float(parts[field_map["_atom_site.Cartn_x"]])
            y = float(parts[field_map["_atom_site.Cartn_y"]])
            z = float(parts[field_map["_atom_site.Cartn_z"]])
        except ValueError:
            i += 1
            continue

        resname = parts[field_map["_atom_site.label_comp_id"]] or "UNK"

        atoms.append(
            CAAtom(
                coord=np.array([x, y, z], dtype=float),
                gt_res=resname
            )
        )

        i += 1

    if not atoms:
        raise ValueError("No C-alpha atoms found in CIF file.")

    print(f"Loaded {len(atoms)} CA atoms.")
    return atoms


# ============================================================
# 2) SAMPLE 6 GROUP MAPS DIRECTLY
# ============================================================

def sample_group_types(atoms, group_mrc_dir, main_mrc_path, radius):

    with mrcfile.open(main_mrc_path, permissive=True) as mrc:
        shape = np.squeeze(mrc.data).shape
        voxel_size = float(mrc.voxel_size.x)
        origin = np.array([
            mrc.header.origin.x,
            mrc.header.origin.y,
            mrc.header.origin.z,
        ], dtype=float)

    group_files = sorted(
        f for f in os.listdir(group_mrc_dir) if f.endswith(".mrc") and re.search(r"residue_types_6letter_([1-6])_possibility\.mrc$", f)
    )

    assert len(group_files) == 6, "Expected exactly 6 group maps."

    group_maps = []
    for fname in group_files:
        with mrcfile.open(os.path.join(group_mrc_dir, fname), permissive=True) as mrc:
            data = np.squeeze(mrc.data)
            if data.shape != shape:
                raise ValueError(f"Shape mismatch: {fname}")
            group_maps.append(data)

    radius_vox = int(np.ceil(radius / voxel_size))

    for atom in atoms:
        x, y, z = atom.coord

        ix = (x - origin[0]) / voxel_size
        iy = (y - origin[1]) / voxel_size
        iz = (z - origin[2]) / voxel_size

        scores = np.zeros(6)

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

                    dx_a = (vx - ix) * voxel_size
                    dy_a = (vy - iy) * voxel_size
                    dz_a = (vz - iz) * voxel_size

                    if dx_a*dx_a + dy_a*dy_a + dz_a*dz_a > radius*radius:
                        continue

                    for g_idx, data in enumerate(group_maps):
                        scores[g_idx] += data[vz, vy, vx]

        atom.exp_res = GROUP_ORDER[int(np.argmax(scores))]

    print("Finished sampling group predictions.")


# ============================================================
# 3) DIRECT GROUP ACCURACY
# ============================================================

def evaluate_group_accuracy(atoms):

    total = 0
    correct = 0
    per_group = {}

    for atom in atoms:
        gt_group = DEGENERATE_GROUPS.get(atom.gt_res, None)
        pred_group = atom.exp_res

        if gt_group is None:
            continue

        total += 1

        if gt_group not in per_group:
            per_group[gt_group] = {"correct": 0, "total": 0}

        per_group[gt_group]["total"] += 1

        if pred_group == gt_group:
            correct += 1
            per_group[gt_group]["correct"] += 1

    for g in per_group:
        per_group[g]["recall"] = (
            per_group[g]["correct"] / per_group[g]["total"]
        )

    overall_accuracy = correct / total if total > 0 else 0.0

    return per_group, overall_accuracy





# ============================================================
# 4) RUN PIPELINE
# ============================================================

def run_analysis(CA_INPUT_FILE, GROUP_MRC_DIR, MAIN_MRC_MAP):

    extension = os.path.splitext(CA_INPUT_FILE)[1]
    if extension == ".pdb":
        atoms = read_calpha_from_pdb(CA_INPUT_FILE)
    elif extension == ".cif":
        atoms = read_calpha_from_cif(CA_INPUT_FILE)
    else:
        raise ValueError(f"Invalid file extension: '{extension}'. Expected '.pdb' or '.cif'.")

    sample_group_types(
        atoms,
        GROUP_MRC_DIR,
        MAIN_MRC_MAP,
        radius=VOXEL_NEIGHBOR_RADIUS
    )

    results, overall_accuracy = evaluate_group_accuracy(atoms)

    print("\nDegenerate Group Recall:")
    # for g, r in results.items():
    #     print(f"  {g}: recall={r['recall']:.4f}, count={r['total']}")
    for g in GROUP_ORDER:     # print in this specific order
        if g in results:
            r = results[g]
            print(f"  {g}: recall={r['recall']:.4f}, count={r['total']}")

    print(f"\nOverall recall: {overall_accuracy:.4f}")

    return results, overall_accuracy



if __name__ == "__main__":
    # CA_INPUT_FILE = r"C:\Users\noelu\Downloads\6LHU.pdb"
    # GROUP_MRC_DIR = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\TestSet\Raw\EMD-0899_re_3.46\possibility"
    # MAIN_MRC_MAP = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\TestSet\Raw\EMD-0899_re_3.46\possibility\emd_0899.map_residue_types_6letter_1_possibility.mrc"

    # use emd_34843 to verify the change in the script didnt break anything (it's also smaller so runs faster)
    CA_INPUT_FILE = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\8HK3.pdb"
    GROUP_MRC_DIR = r"C:\Users\noelu\programming\python_projects\cryoID_v2_DATA\UnetPrediction\EMD-34843_re_3.2\possibility"
    MAIN_MRC_MAP = r"C:\Users\noelu\programming\python_projects\cryoID_v2_DATA\UnetPrediction\EMD-34843_re_3.2\possibility\emd_34843.map_key_atoms_1_possibility.mrc"

    run_analysis(CA_INPUT_FILE, GROUP_MRC_DIR, MAIN_MRC_MAP)