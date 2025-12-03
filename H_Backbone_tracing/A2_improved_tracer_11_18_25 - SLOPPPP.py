# improved_backbone_tracer.py
"""
Improved backbone tracer:
- Detects short helical seeds first and grows them (both directions).
- Also optionally grows from remaining points.
- Uses a beam search (configurable beam_width) for growth.
- Glues segments whose endpoints are geometrically compatible.
- Writes the final (longest) traced chain as a PDB of CA atoms.

Notes:
- Tweak parameters at top-level or pass input file path to main().
- This script focuses on geometric heuristics and is intentionally conservative
  with self-clash checks to avoid loops.
"""

from typing import List, Set, Tuple, Dict, Optional
import numpy as np
import math
import argparse
import os

# ----------------------
# Utility geometry funcs
# ----------------------
def load_ca_coordinates(pdb_file: str) -> np.ndarray:
    """Load only C-alpha (or C) atom coordinates from a PDB-like file."""
    coords = []
    with open(pdb_file, "r") as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")) and line[13:15].strip() in ("CA", "C"):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except Exception:
                    # tolerate slightly different formatting
                    parts = line.split()
                    if len(parts) >= 7:
                        x, y, z = float(parts[-3]), float(parts[-2]), float(parts[-1])
                    else:
                        continue
                coords.append([x, y, z])
    return np.array(coords, dtype=float)

def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def bond_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Angle (degrees) between vectors AB and CB at B.
    """
    v1 = a - b
    v2 = c - b
    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 180.0
    cos_theta = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))

def is_too_close(coords: np.ndarray, path: List[int], candidate_idx: int, cutoff: float = 3.0) -> bool:
    """Prevent circular self-intersection by checking candidate vs non-adjacent residues."""
    cand = coords[candidate_idx]
    # ignore the path's last two positions (adjacent)
    for i in path[:-2]:
        if distance(coords[i], cand) < cutoff:
            return True
    return False

# -----------------------
# Neighbor precomputation
# -----------------------
def precompute_neighbors(coords: np.ndarray, min_bond: float, max_bond: float) -> List[Set[int]]:
    N = len(coords)
    neighbors = [set() for _ in range(N)]
    # naive O(N^2) loop — fine for moderate N; replace with KDTree if needed.
    for i in range(N):
        for j in range(i + 1, N):
            d = distance(coords[i], coords[j])
            if min_bond <= d <= max_bond:
                neighbors[i].add(j)
                neighbors[j].add(i)
    return neighbors

# -------------------------
# Helix seed identification
# -------------------------
def find_helical_seeds(coords: np.ndarray,
                       bond_range: Tuple[float, float] = (3.4, 4.1),
                       angle_range: Tuple[float, float] = (85.0, 105.0),
                       min_len: int = 5) -> List[List[int]]:
    """
    Identify candidate short helical fragments based on local bond distances and angles.
    Simple greedy growth from each starting index.
    """
    N = len(coords)
    seeds: List[List[int]] = []
    for i in range(N):
        seq = [i]
        # greedy: try nearest neighbors (by index order here; geometric order would be better)
        for j in range(i + 1, N):
            d = distance(coords[seq[-1]], coords[j])
            if not (bond_range[0] <= d <= bond_range[1]):
                continue
            if len(seq) >= 2:
                ang = bond_angle(coords[seq[-2]], coords[seq[-1]], coords[j])
                if not (angle_range[0] <= ang <= angle_range[1]):
                    continue
            seq.append(j)
        if len(seq) >= min_len:
            seeds.append(seq)
    # deduplicate overlapping seeds: keep ones that are not subsegments of others
    filtered: List[List[int]] = []
    for s in seeds:
        if not any((len(set(s) & set(t)) == len(s) and len(t) >= len(s)) for t in seeds if t is not s):
            filtered.append(s)
    return filtered

# -----------------------
# Beam-based path extender
# -----------------------
def extend_path_beam(coords: np.ndarray,
                     init_path: List[int],
                     neighbors: List[Set[int]],
                     params: Dict) -> List[int]:
    """
    Beam-search extension starting from init_path; extends at the tail only.
    Returns the best (longest) extension found.
    """
    beam_width = params.get("beam_width", 10)
    min_angle = params.get("min_angle", 60.0)
    max_angle = params.get("max_angle", 170.0)
    self_clash = params.get("self_clash", 3.0)
    max_iters = params.get("max_extend_iters", 5000)  # safety cap to avoid infinite loops

    beams: List[Tuple[List[int], Set[int]]] = [(list(init_path), set(init_path))]
    iters = 0
    best_path = init_path
    while beams and iters < max_iters:
        iters += 1
        new_beams: List[Tuple[List[int], Set[int]]] = []
        progressed = False
        for path, used in beams:
            last = path[-1]
            candidates = neighbors[last] - used
            if not candidates:
                continue
            for c in candidates:
                # geometry checks
                if is_too_close(coords, path, c, cutoff=self_clash):
                    continue
                if len(path) >= 2:
                    prev = path[-2]
                    ang = bond_angle(coords[prev], coords[last], coords[c])
                    if not (min_angle <= ang <= max_angle):
                        continue
                    # prevent near-180° reversal
                    v1 = coords[last] - coords[prev]
                    v2 = coords[c] - coords[last]
                    denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
                    if denom > 0:
                        dot = np.dot(v1, v2) / denom
                        if dot < -0.85:
                            continue
                new_path = path + [c]
                new_used = used | {c}
                new_beams.append((new_path, new_used))
                progressed = True

        if not new_beams:
            break

        # prune to beam_width longest paths
        new_beams = sorted(new_beams, key=lambda x: len(x[0]), reverse=True)[:beam_width]
        beams = new_beams

        # update best path
        if beams:
            cand_best = max(beams, key=lambda x: len(x[0]))[0]
            if len(cand_best) > len(best_path):
                best_path = cand_best

    return best_path

def grow_from_seed_both_dirs(coords: np.ndarray,
                             seed: List[int],
                             neighbors: List[Set[int]],
                             params: Dict) -> List[int]:
    """
    Extend the seed in both directions using beam search.
    We extend the tail, then reverse and extend the reversed seed to extend the other direction.
    Finally combine (avoiding duplicate middle element).
    """
    # forward grow (seed as-is)
    forward = extend_path_beam(coords, seed, neighbors, params)

    # backward grow: reverse the path and extend, then reverse back
    rev_seed = list(reversed(seed))
    backward_extended = extend_path_beam(coords, rev_seed, neighbors, params)
    backward = list(reversed(backward_extended))

    # combine - avoid double-counting the center overlap
    # choose overlap point by finding shared sequence center (middle index)
    # simpler: drop duplicate contiguous joint
    if backward and forward and backward[-1] == forward[0]:
        combined = backward + forward[1:]
    else:
        # If they don't share exactly, attempt to merge at nearest join if endpoints are close
        if backward and forward and distance(coords[backward[-1]], coords[forward[0]]) < params.get("join_gap_tol", 4.6):
            combined = backward + forward[1:]
        else:
            # fallback: just concatenate (should be rare)
            combined = backward + forward
    return combined

# -----------------------
# Segment gluing routines
# -----------------------
def glue_segments(coords: np.ndarray, segments: List[List[int]], max_gap: float = 4.8) -> List[List[int]]:
    """
    Iteratively merge segments whose endpoints are within max_gap.
    Attempts multiple orientation combinations.
    Returns a new list of merged segments.
    """
    segs = [list(s) for s in segments if s]  # shallow copy
    changed = True
    while changed:
        changed = False
        new_segs: List[List[int]] = []
        used = set()
        for i in range(len(segs)):
            if i in used:
                continue
            base = segs[i]
            merged_flag = False
            for j in range(i + 1, len(segs)):
                if j in used:
                    continue
                s2 = segs[j]
                # endpoint combos: (base_end, s2_start), (base_end, s2_end), (base_start, s2_start), (base_start, s2_end)
                combos = [
                    (base[-1], s2[0], "fwd_fwd"),
                    (base[-1], s2[-1], "fwd_rev"),
                    (base[0], s2[0], "rev_fwd"),
                    (base[0], s2[-1], "rev_rev")
                ]
                stitched = None
                for a, b, mode in combos:
                    if distance(coords[a], coords[b]) <= max_gap:
                        if mode == "fwd_fwd":
                            stitched = base + s2
                        elif mode == "fwd_rev":
                            stitched = base + s2[::-1]
                        elif mode == "rev_fwd":
                            stitched = base[::-1] + s2
                        else:  # rev_rev
                            stitched = base[::-1] + s2[::-1]
                        break
                if stitched is not None:
                    new_segs.append(stitched)
                    used.add(i)
                    used.add(j)
                    merged_flag = True
                    changed = True
                    break
            if not merged_flag and i not in used:
                new_segs.append(base)
                used.add(i)
        segs = new_segs
    # optionally remove duplicate segments or subsegments
    unique = []
    seen_sets = []
    for s in segs:
        sset = tuple(s)
        if sset not in seen_sets:
            unique.append(s)
            seen_sets.append(sset)
    return unique

# -----------------------
# Top-level pipeline
# -----------------------
def trace_backbone(coords: np.ndarray,
                   min_bond: float = 3.0,
                   max_bond: float = 4.5,
                   min_angle: float = 80.0,
                   max_angle: float = 160.0,
                   self_clash_cutoff: float = 3.0,
                   beam_width: int = 20,
                   extend_from_all: bool = True,
                   helix_seed_min_len: int = 5,
                   helix_bond_range: Tuple[float, float] = (3.4, 4.1),
                   helix_angle_range: Tuple[float, float] = (85.0, 105.0),
                   max_glue_gap: float = 4.8) -> List[int]:
    """
    Full trace pipeline:
    1) precompute neighbors
    2) find helix seeds, grow them both directions
    3) optionally grow from other starting points (remaining points)
    4) glue segments
    5) return the longest merged segment
    """

    if coords is None or len(coords) == 0:
        raise ValueError("Coordinates are empty.")

    N = len(coords)
    neighbors = precompute_neighbors(coords, min_bond, max_bond)

    # params for extension
    params = {
        "beam_width": beam_width,
        "min_angle": min_angle,
        "max_angle": max_angle,
        "self_clash": self_clash_cutoff,
        "join_gap_tol": max_glue_gap,
        "max_extend_iters": 5000
    }

    # 1) find helical seeds
    helix_seeds = find_helical_seeds(coords,
                                     bond_range=helix_bond_range,
                                     angle_range=helix_angle_range,
                                     min_len=helix_seed_min_len)

    # heuristic: sort seeds by length desc (grow longer seeds first)
    helix_seeds = sorted(helix_seeds, key=lambda s: len(s), reverse=True)

    segments: List[List[int]] = []
    used_indices: Set[int] = set()

    # 2) grow from each helix seed
    for seed in helix_seeds:
        # skip seeds that are fully used already
        if set(seed) <= used_indices:
            continue
        seg = grow_from_seed_both_dirs(coords, seed, neighbors, params)
        segments.append(seg)
        used_indices.update(seg)

    # 3) optionally grow from remaining points using per-start beam search
    if extend_from_all:
        # sample remaining points as starting positions; to limit work, choose
        # points that are not yet used and have at least one neighbor
        remaining = [i for i in range(N) if i not in used_indices and neighbors[i]]
        # to avoid huge runtime, optionally limit attempts by picking top-k seeds
        remaining_to_try = remaining  # can be subsampled if needed
        for start in remaining_to_try:
            seg = extend_path_beam(coords, [start], neighbors, params)
            if len(seg) >= min(8, max(5, N // 20)):  # keep segments of meaningful length
                segments.append(seg)
                used_indices.update(seg)

    if not segments:
        # fallback: single best beam from every start and pick longest
        best = []
        for i in range(N):
            seg = extend_path_beam(coords, [i], neighbors, params)
            if len(seg) > len(best):
                best = seg
        return best

    # 4) glue segments
    merged = glue_segments(coords, segments, max_gap=max_glue_gap)

    # 5) pick longest merged chain
    best = max(merged, key=len)
    if len(best) != N:
        print(f"Warning: traced {len(best)}/{N} points in best chain")

    return best

# -----------------------
# Save results
# -----------------------
def save_path_as_pdb(coords: np.ndarray, route: List[int], output_file: str = "traced_backbone.pdb") -> None:
    with open(output_file, "w") as f:
        f.write("REMARK Traced backbone\n")
        for idx, i in enumerate(route, start=1):
            x, y, z = coords[i]
            f.write(
                f"ATOM  {idx:5d}  CA  ALA A{idx:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
        f.write("TER\nEND\n")
    print(f"Saved traced backbone to {output_file}")

# # -----------------------
# # CLI / main wrapper
# # -----------------------
# def main(input_file: Optional[str] = None, output_file: Optional[str] = None):
#     parser = argparse.ArgumentParser(description="Improved backbone tracer.")
#     parser.add_argument("input", nargs="?", help="Input PDB file", default=input_file)
#     parser.add_argument("--out", "-o", help="Output PDB file", default=output_file or "traced_backbone_improved.pdb")
#     parser.add_argument("--beam", type=int, default=20, help="Beam width for extension")
#     parser.add_argument("--no-extend-all", dest="extend_all", action="store_false",
#                         help="Do not extend from non-helical points")
#     parser.add_argument("--min-helix", type=int, default=5, help="Minimum helix seed length")
#     args = parser.parse_args([] if input_file else None) if input_file is None else parser.parse_args([input_file])
#     inp = args.input if args.input else input_file
#     outp = args.out

#     if not inp or not os.path.isfile(inp):
#         raise FileNotFoundError(f"Input file not found: {inp}")

#     coords = load_ca_coordinates(inp)
#     print(f"Loaded {len(coords)} coordinates from {inp}")

#     route = trace_backbone(coords,
#                            min_bond=3.0,
#                            max_bond=4.5,
#                            min_angle=80.0,
#                            max_angle=160.0,
#                            self_clash_cutoff=3.0,
#                            beam_width=args.beam,
#                            extend_from_all=args.extend_all,
#                            helix_seed_min_len=args.min_helix,
#                            helix_bond_range=(3.4, 4.1),
#                            helix_angle_range=(85.0, 105.0),
#                            max_glue_gap=4.8)

#     print(f"Traced route length: {len(route)}")
#     save_path_as_pdb(coords, route, outp)
#     return outp

# if __name__ == "__main__":
#     # allow running as: python improved_backbone_tracer.py path/to/file.pdb
#     import sys
#     if len(sys.argv) > 1:
#         # pass through real CLI args
#         main()
#     else:
#         # interactive example: modify this path when testing
#         example_path = None
#         if example_path:
#             main(input_file=example_path)
#         else:
#             print("No input file specified. Run with: python improved_backbone_tracer.py <input.pdb>")



# -----------------------
# Simplified main()
# -----------------------
def main(input_file: str,
         output_file: str = "traced_backbone_improved.pdb",
         beam_width: int = 20,
         extend_from_all: bool = True,
         helix_seed_min_len: int = 5) -> str:
    """
    Run the improved backbone tracer.

    Args:
        input_file: Path to input PDB.
        output_file: Path to save traced PDB.
        beam_width: Beam width for beam search.
        extend_from_all: Whether to extend from non-helical points.
        helix_seed_min_len: Minimum length of helical seeds.

    Returns:
        Path to saved PDB file.
    """
    import os

    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    coords = load_ca_coordinates(input_file)
    print(f"Loaded {len(coords)} coordinates from {input_file}")

    route = trace_backbone(coords,
                           min_bond=3.0,
                           max_bond=4.5,
                           min_angle=80.0,
                           max_angle=160.0,
                           self_clash_cutoff=3.0,
                           beam_width=beam_width,
                           extend_from_all=extend_from_all,
                           helix_seed_min_len=helix_seed_min_len,
                           helix_bond_range=(3.4, 4.1),
                           helix_angle_range=(85.0, 105.0),
                           max_glue_gap=4.8)

    print(f"Traced route length: {len(route)}")
    save_path_as_pdb(coords, route, output_file)

    return output_file


if __name__ == "__main__":
    # Example usage
    # input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\example.pdb"
    input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\emd_34843_agglomerative_cd_4.0_data_alone_20250923_210010.pdb"
    output_file = r"traced_backbone_output.pdb"
    main(input_file=input_file, output_file=output_file, beam_width=20)
