from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Callable

import mrcfile
import numpy as np

from cryoid_pipeline.models import PredictionBundle, QueryBundle
ProgressCallback = Callable[[str], None]
PROTEIN_CODES = ("G", "P", "L", "K", "Y", "W")
DEFAULT_MIN_TRACE_LENGTH = 10


def _natural_key(path: Path) -> list[object]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", path.name)]


def _require_file(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


def validate_protein_prediction_bundle(bundle: PredictionBundle) -> tuple[Path, Path, list[Path]]:
    key_map = _require_file(bundle.raw_key_atoms_1_path, "protein C-alpha prediction map")
    if not bundle.secondary_structure_paths:
        raise FileNotFoundError("Missing protein secondary-structure prediction map.")
    scaffold_map = _require_file(bundle.secondary_structure_paths[0], "protein tracing scaffold map")
    if len(bundle.residue_type_paths) < 6:
        raise FileNotFoundError(
            f"Protein query generation requires six residue prediction maps; found {len(bundle.residue_type_paths)}."
        )
    residue_maps = [
        _require_file(path, f"protein residue prediction channel {index}")
        for index, path in enumerate(bundle.residue_type_paths[:6], start=1)
    ]
    return key_map, scaffold_map, residue_maps


def _component_coordinates(path: Path) -> list[tuple[float, float, float]]:
    from H_Backbone_tracing.protein.residue_assignment import read_pdb_coords

    coords = list(read_pdb_coords(path))
    if not coords:
        raise ValueError(f"No traced atoms were found in {path}.")
    return coords


def _sample_protein_residues(
    *,
    component_paths: list[Path],
    key_map: Path,
    residue_maps: list[Path],
    output_dir: Path,
) -> tuple[dict[Path, list[str]], Path]:
    from H_Backbone_tracing.protein.residue_assignment import (
        coords_to_voxels,
        get_voxel_size,
        update_pdb_residues,
    )

    component_coords = {path: _component_coordinates(path) for path in component_paths}
    all_coords = [coord for path in component_paths for coord in component_coords[path]]
    voxel_size, _shape = get_voxel_size(key_map)
    neighborhoods = coords_to_voxels(all_coords, voxel_size, radius=1)
    probabilities = np.zeros((len(all_coords), len(PROTEIN_CODES)), dtype=np.float32)

    for channel, map_path in enumerate(residue_maps):
        with mrcfile.mmap(map_path, mode="r", permissive=True) as mrc:
            data = np.squeeze(mrc.data)
            for atom_index, voxels in neighborhoods.items():
                values = [
                    float(data[z, y, x])
                    for z, y, x in voxels
                    if 0 <= z < data.shape[0] and 0 <= y < data.shape[1] and 0 <= x < data.shape[2]
                ]
                probabilities[atom_index, channel] = float(np.mean(values)) if values else 0.0

    selected_codes = [PROTEIN_CODES[int(index)] for index in np.argmax(probabilities, axis=1)]
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = output_dir / "residue_probabilities.csv"
    sequences: dict[Path, list[str]] = {}
    offset = 0
    with matrix_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["component", "atom_index", *PROTEIN_CODES, "selected"])
        for component_path in component_paths:
            count = len(component_coords[component_path])
            component_codes = selected_codes[offset : offset + count]
            sequences[component_path] = component_codes
            for local_index in range(count):
                row = probabilities[offset + local_index]
                writer.writerow(
                    [
                        component_path.stem,
                        local_index + 1,
                        *[f"{float(value):.8g}" for value in row],
                        component_codes[local_index],
                    ]
                )
            assigned_path = output_dir / f"{component_path.stem}_updated_residues.pdb"
            update_pdb_residues(component_path, assigned_path, component_codes)
            offset += count

    return sequences, matrix_path


def _no_fragment_queries(tracing_root, min_trace_length, reason, outputs, progress):
    """Record a valid empty tracing result for the input-order ranking policy."""
    query_fasta = tracing_root / "generated_queries.fasta"
    query_fasta.write_text("; No protein query fragments were produced.\n", encoding="utf-8")
    filter_path = tracing_root / "graphs" / "trace_length_filter.json"
    filter_path.write_text(json.dumps({
        "requested_min_trace_length": min_trace_length,
        "effective_min_trace_length": 0,
        "fallback_applied": True,
        "component_count_before_filter": 0,
        "longest_component_length": 0,
        "retained_component_count": 0,
        "no_fragments_reason": reason,
    }, indent=2) + "\n", encoding="utf-8")
    progress(f"Protein tracing: {reason}; no fragments are available. Sequence ranking will preserve FASTA input order.")
    files = [p.resolve() for p in [*outputs, filter_path, query_fasta] if p.is_file() and p.stat().st_size > 0]
    return QueryBundle(query_fasta=query_fasta.resolve(), source="protein_backbone_tracing_no_fragments", output_files=files)


def run_protein_query_generation(
    bundle: PredictionBundle,
    run_dir: Path,
    progress: ProgressCallback,
    min_trace_length: int = DEFAULT_MIN_TRACE_LENGTH,
) -> QueryBundle:
    if isinstance(min_trace_length, bool) or int(min_trace_length) != min_trace_length:
        raise ValueError("Protein minimum trace length must be a whole number.")
    min_trace_length = int(min_trace_length)
    if min_trace_length < 1:
        raise ValueError("Protein minimum trace length must be at least 1.")
    key_map, scaffold_map, residue_maps = validate_protein_prediction_bundle(bundle)
    identification_dir = run_dir / "tmp"
    tracing_root = identification_dir / "tracing"
    clustering_dir = tracing_root / "clusters"
    masking_dir = tracing_root / "masking"
    tracing_dir = tracing_root / "graphs"
    components_dir = tracing_root / "components"
    assigned_dir = tracing_root / "residue_assignments"
    for directory in (clustering_dir, masking_dir, tracing_dir, components_dir, assigned_dir):
        directory.mkdir(parents=True, exist_ok=True)

    progress("Protein identification: clustering predicted C-alpha positions.")
    from H_Backbone_tracing.common.clustering import ClusteringConfig, run_clustering

    clustering = run_clustering(
        ClusteringConfig(
            input_mrc=str(key_map),
            clustering_method="meanshift",
            output_dir=str(clustering_dir),
            threshold=0.97,
            min_cluster_size=5,
            min_spacing=3.0,
            bandwidth=1.9,
            manual_file_name="E1_D3",
            timestamp=False,
        )
    )
    clustered_pdb = Path(clustering["output_path"])
    if not clustered_pdb.is_file():
        raise FileNotFoundError(f"Missing clustered C-alpha PDB: {clustered_pdb}")
    if "centroids" in clustering and len(clustering["centroids"]) == 0:
        return _no_fragment_queries(tracing_root, min_trace_length, "clustering produced no C-alpha positions", [], progress)
    clustered_pdb = _require_file(clustered_pdb, "clustered C-alpha PDB")

    progress("Protein identification: masking predicted helix C-alpha positions.")
    from H_Backbone_tracing.protein.masking import mask_and_save_pdb

    masked_pdb = mask_and_save_pdb(
        pdb_file=clustered_pdb,
        mrc_file=scaffold_map,
        output_dir=masking_dir,
        output_name="E1_D3_ca_clustered_then_masked",
        mask_threshold=0.7,
        radius=1.5,
        percent=0.5,
        timestamp=False,
        verbose=False,
    )
    if masked_pdb is None:
        return _no_fragment_queries(tracing_root, min_trace_length, "helix masking removed all C-alpha positions", [clustered_pdb], progress)
    masked_pdb = _require_file(Path(masked_pdb), "helix-masked C-alpha PDB")

    progress("Protein identification: tracing degree-limited backbone components.")
    from H_Backbone_tracing.protein.tracing import trace_backbone_components

    trace_backbone_components(
        clustered_pdb_path=str(masked_pdb),
        scaffold_path=str(scaffold_map),
        scaffold_threshold=0.97,
        scaffold_cylinder_radius=1.0,
        max_distance=5.0,
        max_degree=2,
        min_component_size=min_trace_length,
        graph_output_dir=str(tracing_dir),
        components_output_dir=str(components_dir),
        verbose=True,
    )
    component_paths = sorted(components_dir.glob("cc_*.pdb"), key=_natural_key)
    filter_path = tracing_dir / "trace_length_filter.json"
    length_filter = json.loads(filter_path.read_text(encoding="utf-8"))
    if length_filter["fallback_applied"]:
        progress(
            f"Protein tracing: no component reached the requested minimum length {min_trace_length} "
            f"(longest: {length_filter['longest_component_length']}); using minimum length 0 "
            f"and retaining all {len(component_paths)} available component(s)."
        )
    if not component_paths:
        return _no_fragment_queries(tracing_root, min_trace_length, "tracing produced no components", [clustered_pdb, masked_pdb], progress)

    progress(f"Protein identification: assigning six-class residues to {len(component_paths)} trace(s).")
    sequences, matrix_path = _sample_protein_residues(
        component_paths=component_paths,
        key_map=key_map,
        residue_maps=residue_maps,
        output_dir=assigned_dir,
    )

    query_fasta = tracing_root / "generated_queries.fasta"
    ordered_components = sorted(component_paths, key=lambda path: path.stat().st_size, reverse=True)
    with query_fasta.open("w", encoding="utf-8", newline="\n") as handle:
        for component_path in ordered_components:
            handle.write(f">{component_path.stem}_updated_residues\n")
            handle.write("".join(sequences[component_path]) + "\n\n")

    output_files = [
        clustered_pdb,
        masked_pdb,
        tracing_dir / "superset_graph.pdb",
        tracing_dir / "superset_graph_degree_limited.pdb",
        filter_path,
        *component_paths,
        *sorted(assigned_dir.glob("cc_*_updated_residues.pdb"), key=_natural_key),
        matrix_path,
        query_fasta,
    ]
    output_files = [path.resolve() for path in output_files if path.is_file()]
    progress(f"Protein identification: wrote {len(ordered_components)} query sequence(s) to {query_fasta}.")
    return QueryBundle(query_fasta=query_fasta.resolve(), source="protein_backbone_tracing", output_files=output_files)
