"""Production API for stage I (structure retrieval and fitting)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

from .protein.core.pipeline import RunConfig, run_csv
from .rna.structures import (
    PublicRnaDatabase,
    RnaStructureResult,
    ensure_public_rna_database,
    run_rna_structure_pipeline,
    select_structure_hits,
)


ProgressCallback = Callable[[str], None]
CancellationCheckpoint = Callable[[], None]


def collect_protein_structure_outputs(run_dir: Path) -> list[Path]:
    """Return the durable outputs owned by one protein structure-fitting run."""

    outputs: list[Path] = []
    for candidate in (
        Path(run_dir) / "data" / "logs" / "summary.csv",
        Path(run_dir) / "data" / "logs" / "summary.json",
    ):
        if candidate.is_file():
            outputs.append(candidate)
    for directory in (Path(run_dir) / "data" / "fitted_models", Path(run_dir) / "data" / "sources"):
        if directory.is_dir():
            outputs.extend(sorted(path for path in directory.rglob("*") if path.is_file()))
    return outputs


def _validate_protein_summary(run_dir: Path) -> None:
    summary_csv = Path(run_dir) / "data" / "logs" / "summary.csv"
    if not summary_csv.is_file():
        raise RuntimeError(f"Structure fitting did not write summary.csv under {summary_csv.parent}")
    with summary_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"Structure-fitting summary has no target rows: {summary_csv}")


def run_protein_structure_pipeline(
    *,
    csv_path: Path,
    run_dir: Path,
    tmp_dir: Path,
    progress: ProgressCallback,
    check_cancel: CancellationCheckpoint,
    keep_tmp: bool = False,
) -> list[Path]:
    """Run protein retrieval/fitting directly, without a script subprocess."""

    run_dir = Path(run_dir).resolve()
    tmp_dir = Path(tmp_dir).resolve()
    data_dir = run_dir / "data"
    config = RunConfig(
        run_dir=run_dir,
        data_dir=data_dir,
        tmp_dir=tmp_dir,
        rawfield_scratch_dir=tmp_dir / "rawfield_case_runs",
        source_dir=data_dir / "sources",
        fitted_dir=data_dir / "fitted_models",
        logs_dir=data_dir / "logs",
        entries=0,
        max_experimental_tries=None,
        include_alphafold=True,
        randomize_candidate_start=False,
        random_start_margin_A=4.0,
        keep_only_best_models=not keep_tmp,
        progress=progress,
        check_cancel=check_cancel,
    )
    progress("Struct-fetch: selecting public structures and fitting them to the predicted fields.")
    check_cancel()
    run_csv(Path(csv_path).resolve(), run_config=config)
    check_cancel()
    _validate_protein_summary(run_dir)
    return collect_protein_structure_outputs(run_dir)


__all__ = [
    "PublicRnaDatabase",
    "RnaStructureResult",
    "collect_protein_structure_outputs",
    "ensure_public_rna_database",
    "run_protein_structure_pipeline",
    "run_rna_structure_pipeline",
    "select_structure_hits",
]
