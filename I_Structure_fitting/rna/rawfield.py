#!/usr/bin/env python3
"""Fit shortlisted RNA structures with StructFetch's raw-field optimizer.

It reads the selected public-PDB hit table, fits the isolated RNA chains, and
records structural fit diagnostics. These scores never alter the saved RNA
sequence ranking or its selected top sequence.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path


def link_large_artifact(source: Path, destination: Path) -> None:
    """Expose a large prediction map without copying it."""
    source = source.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if os.path.samefile(source, destination):
            return
        raise FileExistsError(
            f"Refusing to reuse a prediction map that is not linked to its source: {destination}"
        )
    try:
        os.link(source, destination)
    except OSError as hardlink_error:
        try:
            destination.symlink_to(source)
        except OSError as symlink_error:
            raise OSError(
                "Could not hard-link or symbolically link the prediction map. "
                "Keep large CryoID artifacts on a linkable filesystem; they are "
                "never copied into a run directory. "
                f"Hard-link error: {hardlink_error}; symlink error: {symlink_error}"
            ) from symlink_error


def prepare_prediction_aliases(
    *, output: Path, emdb_number: str, phosphate_map: Path, scaffold_map: Path
) -> tuple[Path, Path]:
    """Map RNA fields onto the protein-oriented raw-field input names."""
    prediction_dir = output / "prediction_aliases"
    prefix = f"emd_{emdb_number}.map"
    key_path = prediction_dir / f"{prefix}_key_atoms_1_possibility.mrc"
    link_large_artifact(phosphate_map, key_path)
    for channel in (1, 2, 3):
        link_large_artifact(
            scaffold_map,
            prediction_dir / f"{prefix}_secondary_strctures_{channel}_possibility.mrc",
        )
    return prediction_dir, key_path


def download_structure(pdb_id: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return
    temporary = destination.with_suffix(destination.suffix + f".{os.getpid()}.part")
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif.gz"
    with urllib.request.urlopen(url, timeout=60) as response, gzip.GzipFile(fileobj=response) as source, temporary.open("wb") as handle:
        shutil.copyfileobj(source, handle)
    temporary.replace(destination)


def extract_chain(source: Path, chain_id: str, destination: Path) -> None:
    """Write only the SEQRES hit's author chain to a fresh mmCIF."""
    import gemmi  # noqa: PLC0415

    if destination.exists() and destination.stat().st_size > 0:
        return
    structure = gemmi.read_structure(str(source))
    selected = None
    for model in structure:
        for chain in model:
            if chain.name == chain_id:
                selected = chain.clone()
                break
        if selected is not None:
            break
    if selected is None:
        available = sorted({chain.name for model in structure for chain in model})
        raise KeyError(f"Chain {chain_id!r} absent from {source}; available={available}")
    isolated = gemmi.Structure()
    isolated.name = f"{source.stem}_{chain_id}"
    isolated.cell = structure.cell
    isolated.spacegroup_hm = structure.spacegroup_hm
    model = gemmi.Model("1")
    model.add_chain(selected)
    isolated.add_model(model)
    destination.parent.mkdir(parents=True, exist_ok=True)
    isolated.make_mmcif_document().write_file(str(destination))


def fit_rna_structures(
    *,
    hits_path: Path,
    structures_dir: Path,
    density_map: Path,
    phosphate_map: Path,
    scaffold_map: Path,
    output_dir: Path,
    emdb_number: str,
    resolution: float = 3.0,
    progress=lambda _message: None,
    check_cancel=lambda: None,
    keep_tmp: bool = False,
) -> list[dict[str, object]]:
    """Download, isolate, and fit shortlisted RNA chains in the current process."""

    from I_Structure_fitting.protein.core.aligner import AlignmentConfig, align_model_to_map
    from I_Structure_fitting.protein.core.rawfield import configure_local_cache_dir

    output_dir = Path(output_dir).resolve()
    structures_dir = Path(structures_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_dir, key_path = prepare_prediction_aliases(
        output=output_dir,
        emdb_number=emdb_number,
        phosphate_map=Path(phosphate_map),
        scaffold_map=Path(scaffold_map),
    )
    configure_local_cache_dir(output_dir / "local_mrc_cache")

    with Path(hits_path).open(encoding="utf-8", newline="") as handle:
        hits = list(csv.DictReader(handle, delimiter="\t"))
    results: list[dict[str, object]] = []
    fitted_dir = output_dir / "fitted"
    scratch_dir = output_dir / "rawfield_case_runs"
    fitted_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.perf_counter()
    best_row: dict[str, object] | None = None

    for index, hit in enumerate(hits, start=1):
        check_cancel()
        pdb_chain = hit["pdb_chain"]
        pdb_id = hit["pdb_id"].lower()
        started = time.perf_counter()
        try:
            source_model_path = structures_dir / f"{pdb_id}.cif"
            download_structure(pdb_id, source_model_path)
            safe_chain = hit["chain_id"].replace("/", "_").replace("\\", "_")
            model_path = output_dir / "isolated_chains" / f"{pdb_id}_{safe_chain}.cif"
            extract_chain(source_model_path, hit["chain_id"], model_path)
        except Exception as exc:
            row = {
                "status": "error",
                **{key: value for key, value in hit.items() if key != "structure_sequence"},
                "error": f"structure preparation failed: {exc!r}",
                "wall_seconds": time.perf_counter() - started,
            }
            results.append(row)
            (output_dir / "fit_summary.json").write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8"
            )
            progress(f"RNA structure fitting: {pdb_chain} preparation failed: {exc!r}")
            continue
        fitted_path = fitted_dir / f"{pdb_chain}_rawfield.cif"
        case_run = scratch_dir / pdb_chain
        meta = {
            "case_key": f"rna_structure_fit_{pdb_chain}",
            "gene_id": hit.get("query") or pdb_chain,
            "emdb_id": f"EMD-{emdb_number}",
            "resolution": resolution,
            # The shared raw-field API uses protein-oriented field names for
            # the model being fitted. In RNA mode this is simply the
            # shortlisted public PDB chain, not benchmark ground truth.
            "ground_truth_model": hit["pdb_id"],
            "gt_model_path": str(model_path.resolve()),
            "raw_prediction_dir": str(prediction_dir.resolve()),
            "raw_key_atoms_1_path": str(key_path.resolve()),
        }
        config = AlignmentConfig(
            compute_gt_metrics=False,
            randomize_start=False,
        )
        progress(f"RNA structure fitting: fitting {pdb_chain} ({index}/{len(hits)}).")
        try:
            payload = align_model_to_map(
                Path(density_map).resolve(),
                model_path.resolve(),
                fitted_path.resolve(),
                method_run_dir=case_run.resolve(),
                config=config,
                input_meta=meta,
            )
            row: dict[str, object] = {
                "status": "ok",
                **{key: value for key, value in hit.items() if key != "structure_sequence"},
                **payload,
                "wall_seconds": time.perf_counter() - started,
            }
        except Exception as exc:
            row = {
                "status": "error",
                **{key: value for key, value in hit.items() if key != "structure_sequence"},
                "error": repr(exc),
                "wall_seconds": time.perf_counter() - started,
            }
        results.append(row)
        if not keep_tmp and row.get("status") == "ok" and row.get("score") is not None:
            if best_row is None or float(row["score"]) > float(best_row["score"]):
                if best_row is not None:
                    for key in ("fitted_path", "source_model"):
                        old_path = Path(str(best_row.get(key) or ""))
                        if old_path.is_file():
                            old_path.unlink()
                best_row = row
            else:
                for key in ("fitted_path", "source_model"):
                    losing_path = Path(str(row.get(key) or ""))
                    if losing_path.is_file():
                        losing_path.unlink()
        (output_dir / "fit_summary.json").write_text(
            json.dumps(results, indent=2) + "\n", encoding="utf-8"
        )
        progress(f"RNA structure fitting: {pdb_chain} status={row['status']} score={row.get('score')}.")

    successful = [row for row in results if row.get("status") == "ok"]
    successful.sort(key=lambda row: float(row["score"]), reverse=True)
    for rank, row in enumerate(successful, start=1):
        row["rawfield_rank"] = rank
    ranking_fields = [
        "rawfield_rank",
        "pdb_chain",
        "score",
        "map_cc_score",
        "mean_z",
        "final_map_meanz",
        "alignment_seconds",
        "wall_seconds",
    ]
    with (output_dir / "rawfield_structure_ranking.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=ranking_fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(successful)
    summary = {
        "candidate_count": len(results),
        "successful_count": len(successful),
        "best_pdb_chain": successful[0]["pdb_chain"] if successful else None,
        "best_score": successful[0]["score"] if successful else None,
        "total_wall_seconds": time.perf_counter() - run_started,
        "sum_candidate_wall_seconds": sum(float(row["wall_seconds"]) for row in results),
        "randomized_start": False,
    }
    (output_dir / "fit_run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    progress(
        f"RNA structure fitting: completed {len(results)} candidate(s), "
        f"{len(successful)} successful."
    )
    return results
