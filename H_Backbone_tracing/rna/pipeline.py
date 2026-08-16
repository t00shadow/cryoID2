from __future__ import annotations

from pathlib import Path
from typing import Callable

import mrcfile
import numpy as np

from cryoid_pipeline.models import PredictionBundle, QueryBundle


ProgressCallback = Callable[[str], None]

CLUSTER_THRESHOLD = 0.97
CLUSTER_BANDWIDTH = 1.9
CLUSTER_MIN_SIZE = 5
CLUSTER_MIN_SPACING = 3.0
GUARDRAIL_THRESHOLD = 0.97
GUARDRAIL_RADIUS = 1.0
TRACE_MAX_DISTANCE = 5.5
TRACE_MAX_DEGREE = 2
TRACE_MIN_COMPONENT_SIZE = 10


def read_component(path: Path) -> list[np.ndarray]:
    """Read trace atoms in the order emitted by the RNA tracer."""

    coordinates: list[np.ndarray] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                coordinates.append(
                    np.array(
                        (float(line[46:54]), float(line[38:46]), float(line[30:38])),
                        dtype=float,
                    )
                )
    if not coordinates:
        raise ValueError(f"No trace atoms found in {path}.")
    return coordinates


def _assign_two_letter_sequence(
    coordinates: list[np.ndarray],
    a_map: np.ndarray,
    u_map: np.ndarray,
    voxel: np.ndarray,
) -> str:
    sequence: list[str] = []
    for coordinate in coordinates:
        center = np.rint(coordinate / voxel).astype(int)
        slices = tuple(
            slice(max(0, index - 1), min(size, index + 2))
            for index, size in zip(center, a_map.shape)
        )
        sequence.append("A" if float(np.mean(a_map[slices])) >= float(np.mean(u_map[slices])) else "U")
    return "".join(sequence)


def run_rna_query_generation(
    prediction_bundle: PredictionBundle,
    run_dir: Path,
    progress: ProgressCallback,
) -> QueryBundle:
    from H_Backbone_tracing.common.clustering import ClusteringConfig, run_clustering
    from H_Backbone_tracing.rna.tracing import trace_rna_components

    output_dir = Path(run_dir) / "tmp"
    tracing_root = output_dir / "tracing"
    cluster_dir = tracing_root / "clusters"
    graph_dir = tracing_root / "graphs"
    component_dir = tracing_root / "components"
    for directory in (cluster_dir, graph_dir, component_dir):
        directory.mkdir(parents=True, exist_ok=True)

    phosphate_map = prediction_bundle.key_atom_path(2)
    sugar_map = prediction_bundle.key_atom_path(3)
    guardrail_map = prediction_bundle.secondary_structure_path(4)
    a_map = prediction_bundle.residue_type_path(7)
    u_map = prediction_bundle.residue_type_path(8)
    required = (phosphate_map, sugar_map, guardrail_map, a_map, u_map)
    missing = [path for path in required if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError("Missing RNA prediction map(s): " + ", ".join(str(path) for path in missing))

    clustered: dict[str, Path] = {}
    for label, prediction_map in (("phosphate", phosphate_map), ("sugar", sugar_map)):
        progress(f"RNA query generation: clustering {label} predictions.")
        result = run_clustering(
            ClusteringConfig(
                input_mrc=str(prediction_map),
                clustering_method="meanshift",
                output_dir=str(cluster_dir),
                threshold=CLUSTER_THRESHOLD,
                min_cluster_size=CLUSTER_MIN_SIZE,
                min_spacing=CLUSTER_MIN_SPACING,
                bandwidth=CLUSTER_BANDWIDTH,
                manual_file_name=label,
            )
        )
        clustered[label] = Path(result["output_path"]).resolve()

    progress("RNA query generation: tracing phosphate/sugar components.")
    trace_rna_components(
        phosphate_pdb_path=str(clustered["phosphate"]),
        sugar_pdb_path=str(clustered["sugar"]),
        scaffold_path=str(guardrail_map),
        scaffold_threshold=GUARDRAIL_THRESHOLD,
        scaffold_cylinder_radius=GUARDRAIL_RADIUS,
        max_distance=TRACE_MAX_DISTANCE,
        max_degree=TRACE_MAX_DEGREE,
        min_component_size=TRACE_MIN_COMPONENT_SIZE,
        graph_output_dir=str(graph_dir),
        components_output_dir=str(component_dir),
        verbose=True,
    )

    rows: list[tuple[str, str]] = []
    with mrcfile.open(a_map, permissive=True) as a_mrc, mrcfile.open(u_map, permissive=True) as u_mrc:
        a_data = np.asarray(np.squeeze(a_mrc.data))
        u_data = np.asarray(np.squeeze(u_mrc.data))
        if a_data.shape != u_data.shape:
            raise ValueError("RNA A/U prediction maps have different shapes.")
        voxel = np.array(
            (float(a_mrc.voxel_size.z), float(a_mrc.voxel_size.y), float(a_mrc.voxel_size.x))
        )
        for component in sorted(component_dir.glob("cc_*.pdb")):
            sequence = _assign_two_letter_sequence(read_component(component), a_data, u_data, voxel)
            rows.append((f"{component.stem}_updated_residues", sequence))
    if not rows:
        raise RuntimeError("RNA tracing produced no components at the configured size cutoff.")
    rows.sort(key=lambda item: (-len(item[1]), item[0]))
    query_path = tracing_root / "generated_queries.fasta"
    with query_path.open("w", encoding="ascii", newline="\n") as handle:
        for name, sequence in rows:
            handle.write(f">{name}\n{sequence}\n\n")
    progress(f"RNA query generation: wrote {len(rows)} traced queries.")
    outputs = [query_path, *clustered.values(), *sorted(component_dir.glob("cc_*.pdb"))]
    return QueryBundle(
        query_fasta=query_path.resolve(),
        source="rna_phosphate_sugar_component_order",
        output_files=outputs,
    )
