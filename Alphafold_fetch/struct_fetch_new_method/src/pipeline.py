"""
Production StructFetch pipeline.

For each row of the input CSV this module:

1. Calls :func:`fetch.candidates_for_sequence` to obtain the candidate PDB list
   and the UniProt accessions cross-referenced from those PDBs.
2. (Optional) Downloads AlphaFold predictions for the accessions and aligns
   each one to the cryo-EM map with the rawfield_ss_dominant method.
3. Downloads each candidate PDB model and aligns it to the cryo-EM map with
   the same method.
4. Ranks all candidates by the rawfield score, keeps the best AlphaFold, best
   PDB, and best overall, and (if ``keep_only_best_models`` is set) deletes
   the rest of the on-disk source/fitted files.

The CSV ``gene_id`` column is **label-only** — it is used to name run
directories, output files, and log lines. Candidate selection no longer
depends on it. If it's missing, the first top-hit UniProt accession is used
as a fallback label.

Pool lifecycle: :func:`run_csv_with_alignment_config` wraps its main loop in
``try/finally`` and invokes :func:`fetch.cleanup_pool` to delete the on-disk
pool FASTA at the end of every run, success or failure.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

import fetch
import map_utils
import rawfield
from aligner import AlignmentConfig, align_model_to_map, fitted_output_path
from cases import float_or_none
from config_io import apply_cli_overrides, load_json_config, resolve_method_path
from inputs import DensityInputs, PIPELINE_REQUIRED_CSV_COLUMNS, resolve_density_inputs
from io_utils import format_float, write_json
from paths import DEFAULT_RUNS_DIR, METHOD_ROOT
from run_logging import tee_output


DEFAULT_CONFIG_PATH = METHOD_ROOT / "configs" / "pipeline.json"
FINAL_SCORE_METRIC = AlignmentConfig().score_metric

DEFAULT_SETTINGS: dict[str, Any] = {
    "csv": "test_data/ready_to_test_1_29.csv",
    "entries": 1,
    "run_dir": "",
    "max_experimental_tries": None,
    "include_alphafold": True,
    "recommended_contour_level": None,
    "randomize_candidate_start": False,
    "random_start_margin_A": 4.0,
    "keep_only_best_models": True,
}


def _timestamped_run_dir() -> Path:
    """Generate a fresh ``runs/struct_fetch_rawfield_<YYYYMMDD_HHMMSS>`` path."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RUNS_DIR / f"struct_fetch_rawfield_{stamp}"


@dataclass
class RunConfig:
    """Resolved per-run knobs assembled from defaults + JSON config + CLI overrides."""

    run_dir: Path
    data_dir: Path
    source_dir: Path
    fitted_dir: Path
    logs_dir: Path
    entries: int
    max_experimental_tries: Optional[int]
    include_alphafold: bool
    recommended_contour_level: Optional[float]
    randomize_candidate_start: bool
    random_start_margin_A: float
    keep_only_best_models: bool


def _score_value(row: dict[str, Any]) -> float:
    """Return ``row['score']`` as a float, or ``-inf`` if absent / non-numeric."""
    value = float_or_none(row.get("score"))
    return value if value is not None else float("-inf")


def _elapsed_seconds(started: float) -> float:
    """Return elapsed seconds from a ``time.perf_counter()`` checkpoint."""
    return float(time.perf_counter() - started)


def _format_seconds(value: Any) -> str:
    """Format a float-coercible value as ``"X.XXs"`` for log output (``"n/a"`` on failure)."""
    text = format_float(value, fmt=".2f", fallback="")
    return f"{text}s" if text else "n/a"


def _print_candidate_result(tag: str, kind: str, candidate_id: str, row: dict[str, Any]) -> None:
    """One-line status print after a candidate alignment completes."""
    print(
        f"[{tag}] {kind} {candidate_id}: status={row.get('status')} "
        f"score={row.get('score')} elapsed={_format_seconds(row.get('candidate_total_seconds'))}",
        flush=True,
    )


def _candidate_case_key(gene_id: str, candidate_id: str) -> str:
    """Build a stable case key from a gene label and a candidate identifier."""
    gene_part = fetch.sanitize_name(str(gene_id or "").strip())
    candidate_part = fetch.sanitize_name(str(candidate_id or "").strip())
    return f"{gene_part}_{candidate_part}" if gene_part and candidate_part else gene_part or candidate_part


def _is_better(candidate: dict[str, Any], incumbent: Optional[dict[str, Any]]) -> bool:
    """Return True iff ``candidate`` outranks ``incumbent`` by score then label."""
    if _score_value(candidate) == float("-inf"):
        return False
    if incumbent is None:
        return True
    c_key = (_score_value(candidate), str(candidate.get("label") or ""))
    i_key = (_score_value(incumbent), str(incumbent.get("label") or ""))
    return c_key > i_key


def _fit_candidate(
    *,
    label: str,
    kind: str,
    candidate_id: str,
    map_path: Path,
    model_path: Path,
    recommended_contour_level: Optional[float],
    run_config: RunConfig,
    align_config: AlignmentConfig,
    input_meta: dict[str, Any],
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Run rawfield alignment for one candidate model and return a status row.

    Catches and serializes alignment errors so a single failing candidate does
    not abort the whole run.
    """
    out_model = fitted_output_path(run_config.fitted_dir, label, model_path)
    method_run_dir = out_model.parent.parent / "rawfield_case_runs" / out_model.stem
    started = time.perf_counter()
    row: dict[str, Any] = {
        "label": label,
        "type": kind,
        "id": candidate_id,
        "score_metric": align_config.score_metric,
        "selected_method_id": align_config.method_id,
        "alignment_runner": "local_rawfield.run_raw_field",
        "source_model": str(model_path),
        "fitted_path": str(out_model),
        "rawfield_case_run_dir": str(method_run_dir),
        "case_key": str(input_meta.get("case_key") or ""),
        "kept": False,
    }
    if extra:
        row.update(extra)
    try:
        payload = align_model_to_map(
            map_path,
            model_path,
            out_model,
            recommended_contour_level=recommended_contour_level,
            config=align_config,
            input_meta=input_meta,
        )
        row.update(payload)
        row["status"] = "ok"
    except Exception as exc:
        row.update(
            {
                "status": "error",
                "score": float("nan"),
                "raw_field_score": float("nan"),
                "mean_z": float("nan"),
                "map_cc_score": float("nan"),
                "error": repr(exc),
                "alignment_seconds": float(time.perf_counter() - started),
            }
        )
    return row


def _write_target_log(logs_dir: Path, safe_name: str, rows: list[dict[str, Any]]) -> Path:
    """Write the per-target candidate-score CSV and return its path."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    path = logs_dir / f"{safe_name}__candidate_scores.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _download_error_row(
    *,
    label: str,
    kind: str,
    candidate_id: str,
    align_config: AlignmentConfig,
    candidate_started: float,
    fetch_seconds: float,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Build a per-candidate row marking a failed model download."""
    row: dict[str, Any] = {
        "label": label,
        "type": kind,
        "id": candidate_id,
        "score_metric": align_config.score_metric,
        "selected_method_id": align_config.method_id,
        "alignment_runner": "local_rawfield.run_raw_field",
        "source_model": "",
        "fitted_path": "",
        "rawfield_case_run_dir": "",
        "case_key": str((extra or {}).get("case_key") or ""),
        "kept": False,
        "status": "download_error",
        "score": float("nan"),
        "raw_field_score": float("nan"),
        "mean_z": float("nan"),
        "map_cc_score": float("nan"),
        "source_fetch_seconds": float(fetch_seconds),
        "alignment_seconds": float("nan"),
        "candidate_total_seconds": _elapsed_seconds(candidate_started),
        "error": "model_download_failed",
    }
    if extra:
        row.update(extra)
    return row


def _resolve_candidate_path(raw: Any) -> Optional[Path]:
    """Coerce a path-like value to a resolved absolute path, or ``None`` if empty."""
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()


def _path_is_under(path: Path, root: Path) -> bool:
    """True iff ``path`` is the same as or a descendant of ``root``."""
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _remove_empty_parents(start: Path, stop_at: Path) -> None:
    """Walk up from ``start``, removing empty directories until reaching ``stop_at``."""
    stop = Path(stop_at).resolve()
    current = Path(start).resolve()
    while current != stop and _path_is_under(current, stop):
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def _candidate_model_paths(row: Optional[dict[str, Any]]) -> set[Path]:
    """Resolved paths of every model file referenced by a candidate row."""
    paths: set[Path] = set()
    if not row:
        return paths
    for key in ("source_model", "fitted_path", "alignment_input_model"):
        path = _resolve_candidate_path(row.get(key))
        if path is not None:
            paths.add(path)
    return paths


def _safe_unlink_candidate_file(path: Path, root: Path, protected_paths: set[Path]) -> bool:
    """Delete ``path`` if it lives under ``root`` and isn't in ``protected_paths``."""
    resolved = path.resolve()
    if resolved in protected_paths:
        return False
    root = Path(root).resolve()
    if not _path_is_under(resolved, root):
        return False
    if resolved.exists() and resolved.is_file():
        resolved.unlink()
        _remove_empty_parents(resolved.parent, root)
        return True
    return False


def _safe_rmtree_candidate_dir(path: Path, root: Path) -> bool:
    """Recursively remove ``path`` if it lives under ``root`` and isn't ``root`` itself."""
    resolved = Path(path).resolve()
    root = Path(root).resolve()
    if resolved == root or not _path_is_under(resolved, root):
        return False
    if resolved.exists() and resolved.is_dir():
        shutil.rmtree(resolved, ignore_errors=True)
        _remove_empty_parents(resolved.parent, root.parent)
        return True
    return False


def _method_run_dir_from_row(row: dict[str, Any]) -> Optional[Path]:
    """Pull the rawfield scratch directory path out of a per-candidate row."""
    raw = str(row.get("rawfield_case_run_dir") or "").strip()
    if not raw:
        alignment_meta = row.get("alignment_meta") or {}
        raw = str(alignment_meta.get("method_run_dir") or "").strip()
    return _resolve_candidate_path(raw)


def _cleanup_candidate_scratch(
    row: dict[str, Any],
    *,
    run_config: RunConfig,
    removed_by_kind: dict[str, list[str]],
) -> None:
    """Delete the rawfield scratch directory created for a single candidate."""
    method_run_dir = _method_run_dir_from_row(row)
    if method_run_dir is None:
        return
    scratch_root = run_config.data_dir / "rawfield_case_runs"
    alignment_input = _resolve_candidate_path(row.get("alignment_input_model"))
    source_model = _resolve_candidate_path(row.get("source_model"))
    if _safe_rmtree_candidate_dir(method_run_dir, scratch_root):
        row["rawfield_case_run_dir_removed"] = True
        removed_by_kind.setdefault("scratch", []).append(str(method_run_dir))
        if alignment_input is not None and alignment_input != source_model and _path_is_under(alignment_input, method_run_dir):
            row["alignment_input_model_removed"] = True
            removed_by_kind.setdefault("alignment_input", []).append(str(alignment_input))


def _delete_candidate_model_files(
    row: dict[str, Any],
    *,
    run_config: RunConfig,
    removed_by_kind: dict[str, list[str]],
    protected_rows: list[Optional[dict[str, Any]]] | None = None,
) -> None:
    """Delete source / fitted / alignment-input model files for a losing candidate.

    Rows in ``protected_rows`` shield their referenced paths from deletion so
    we never delete a file that another retained best-of-its-kind row points
    to (e.g. when best_overall == best_pdb).
    """
    protected_paths: set[Path] = set()
    for protected_row in protected_rows or []:
        protected_paths.update(_candidate_model_paths(protected_row))

    specs = [
        ("source_model", run_config.source_dir, "source_model_removed", "source"),
        ("fitted_path", run_config.fitted_dir, "fitted_path_removed", "fitted"),
        ("alignment_input_model", run_config.data_dir, "alignment_input_model_removed", "alignment_input"),
    ]
    source_path = _resolve_candidate_path(row.get("source_model"))
    removed_paths: set[Path] = set()
    for key, root, flag, kind in specs:
        path = _resolve_candidate_path(row.get(key))
        if path is None or path in removed_paths:
            continue
        if key == "alignment_input_model" and source_path is not None and path == source_path:
            continue
        if _safe_unlink_candidate_file(path, root, protected_paths):
            row[flag] = True
            removed_paths.add(path)
            removed_by_kind.setdefault(kind, []).append(str(path))


def _update_best_and_cleanup(
    *,
    row: dict[str, Any],
    incumbent: Optional[dict[str, Any]],
    run_config: RunConfig,
    removed_by_kind: dict[str, list[str]],
) -> Optional[dict[str, Any]]:
    """Pick the better of ``row`` vs ``incumbent`` and delete the loser's files."""
    if _is_better(row, incumbent):
        if bool(run_config.keep_only_best_models) and incumbent is not None:
            _delete_candidate_model_files(
                incumbent,
                run_config=run_config,
                removed_by_kind=removed_by_kind,
                protected_rows=[row],
            )
        return row

    if bool(run_config.keep_only_best_models):
        _delete_candidate_model_files(
            row,
            run_config=run_config,
            removed_by_kind=removed_by_kind,
            protected_rows=[incumbent],
        )
    return incumbent


def _candidate_selection(gene_id: str, sequence: str) -> tuple[str, str, fetch.StructureSequenceSelection]:
    """Run the chain-level candidate search and attach pipeline-level label metadata.

    ``gene_id`` is used purely for human-readable labels and case-key generation;
    it is not used for the sequence search. When ``gene_id`` is empty, the
    first top-hit UniProt accession (if any) is used as a fallback label, and
    if there are also no accessions the label falls back to ``"NO_TARGET"``.
    Returns ``(label, sanitized_label, selection)``.
    """
    original_gene_id = str(gene_id or "").strip()
    sequence = str(sequence or "").strip()
    selection = fetch.candidates_for_sequence(sequence)
    label = original_gene_id or (selection.accessions[0] if selection.accessions else "NO_TARGET")
    selection.metadata.update(
        {
            "original_gene_id": original_gene_id,
            "resolved_gene_id": original_gene_id,
            "label_source": "csv_gene_id" if original_gene_id else ("top_uniprot_accession" if selection.accessions else "none"),
        }
    )
    return label, fetch.sanitize_name(label), selection


def process_target(
    *,
    gene_id: str,
    sequence: str,
    map_path: Path,
    recommended_contour_level: Optional[float],
    input_meta: dict[str, Any],
    run_config: RunConfig,
    align_config: AlignmentConfig,
) -> dict[str, Any]:
    """Fetch + align + rank candidates for one CSV row, returning a target-level result dict.

    Steps:

    1. ``_candidate_selection`` runs the chain-level search and produces a
       :class:`fetch.StructureSequenceSelection`.
    2. If ``include_alphafold`` is true, fetch and align AlphaFold predictions
       for each UniProt accession returned by the search.
    3. Fetch and align each candidate PDB (capped by
       ``max_experimental_tries`` when set).
    4. Pick the best AlphaFold, best PDB, and best overall candidate by rawfield
       score; delete losing files when ``keep_only_best_models`` is true.

    Per-case map caches (``map_utils._MAPINFO_CACHE``, ``rawfield._RAW_FIELD_CACHE``,
    and the on-disk ``_raw_mrc_cache/`` staging directory) are wiped in a
    ``finally`` block at the end of this function so the next case starts
    with no residual map state — the caches are useful within a case (every
    candidate aligned reuses the same prediction maps) but never reused
    across cases in any current workflow.
    """
    entry_started = time.perf_counter()
    if not str(sequence or "").strip():
        raise ValueError("sequence is empty")
    try:
        return _process_target_inner(
            gene_id=gene_id,
            sequence=sequence,
            map_path=map_path,
            recommended_contour_level=recommended_contour_level,
            input_meta=input_meta,
            run_config=run_config,
            align_config=align_config,
            entry_started=entry_started,
        )
    finally:
        map_utils.clear_per_case_caches()
        rawfield.clear_per_case_caches()


def _process_target_inner(
    *,
    gene_id: str,
    sequence: str,
    map_path: Path,
    recommended_contour_level: Optional[float],
    input_meta: dict[str, Any],
    run_config: RunConfig,
    align_config: AlignmentConfig,
    entry_started: float,
) -> dict[str, Any]:
    """Body of :func:`process_target`. Separated so the per-case cleanup wraps cleanly via try/finally."""
    selection_started = time.perf_counter()
    tag, safe_name, selection = _candidate_selection(gene_id, sequence)
    sequence_selection_seconds = _elapsed_seconds(selection_started)
    accessions = list(selection.accessions)
    pdb_ids = list(selection.pdb_ids)
    rows: list[dict[str, Any]] = []
    resolved_gene_id = str(selection.metadata.get("resolved_gene_id") or (tag if tag != "NO_TARGET" else "")).strip()
    if resolved_gene_id:
        input_meta["gene_id"] = resolved_gene_id
    if not pdb_ids:
        entry_total_seconds = _elapsed_seconds(entry_started)
        print(f"[{tag}] entry complete: status=error elapsed={_format_seconds(entry_total_seconds)}", flush=True)
        return {
            "target": tag,
            "safe_name": safe_name,
            "status": "error",
            "error": selection.metadata.get("error") or "no_pdb_candidates",
            "rows": rows,
            "candidate_generation": selection.metadata,
            "sequence_selection_seconds": sequence_selection_seconds,
            "entry_total_seconds": entry_total_seconds,
        }

    tied_count = int(selection.metadata.get("tied_top_sequence_count", 1))
    tied_note = f" ({tied_count} tied)" if tied_count > 1 else ""
    print(
        f"[{tag}] top chain SW score={selection.best_score}{tied_note} "
        f"-> {len(pdb_ids)} candidate PDB(s), {len(accessions)} UniProt accession(s) for AlphaFold path",
        flush=True,
    )
    best_af: Optional[dict[str, Any]] = None
    best_pdb: Optional[dict[str, Any]] = None
    removed_by_kind: dict[str, list[str]] = {"source": [], "fitted": [], "alignment_input": [], "scratch": []}
    if run_config.include_alphafold:
        af_sources: list[tuple[str, dict[str, Any]]] = []
        for accession in accessions:
            metas = fetch.alphafold_prediction_metas(accession)
            af_sources.extend((accession, meta) for meta in metas)
            print(f"[{tag}] AF {accession}: found {len(metas)} model metadata record(s)", flush=True)
        print(f"[{tag}] AF candidates to evaluate: {len(af_sources)} from {len(accessions)} selected UniProt accession(s)", flush=True)
        for accession, meta in af_sources:
            model_version = meta.get("modelVersion")
            model_tag = f"{accession}_v{model_version}" if model_version not in (None, "") else accession
            candidate_started = time.perf_counter()
            fetch_started = time.perf_counter()
            model_path = fetch.download_alphafold_prediction(accession, meta, run_config.source_dir / safe_name / "af" / accession)
            fetch_seconds = _elapsed_seconds(fetch_started)
            af_extra = {
                "case_key": _candidate_case_key(resolved_gene_id or tag, model_tag),
                "uniprot_accession": accession,
                "af_confidence": meta.get("confidence"),
                "af_model_version": model_version,
                "af_file_url": meta.get("fileUrl") or meta.get("pdbUrl") or meta.get("cifUrl"),
                "sequence_selection_score": selection.best_score,
                "candidate_generation_method": selection.metadata.get("candidate_generation_method"),
            }
            if not model_path:
                row = _download_error_row(
                    label=f"{safe_name}__AF_{fetch.sanitize_name(model_tag)}",
                    kind="AF",
                    candidate_id=model_tag,
                    align_config=align_config,
                    candidate_started=candidate_started,
                    fetch_seconds=fetch_seconds,
                    extra=af_extra,
                )
                rows.append(row)
                _print_candidate_result(tag, "AF", model_tag, row)
                continue
            row = _fit_candidate(
                label=f"{safe_name}__AF_{fetch.sanitize_name(model_tag)}",
                kind="AF",
                candidate_id=model_tag,
                map_path=map_path,
                model_path=Path(model_path),
                recommended_contour_level=recommended_contour_level,
                run_config=run_config,
                align_config=align_config,
                input_meta={**input_meta, "case_key": af_extra["case_key"]},
                extra=af_extra,
            )
            row["source_fetch_seconds"] = fetch_seconds
            row["candidate_total_seconds"] = _elapsed_seconds(candidate_started)
            _cleanup_candidate_scratch(row, run_config=run_config, removed_by_kind=removed_by_kind)
            rows.append(row)
            _print_candidate_result(tag, "AF", model_tag, row)
            best_af = _update_best_and_cleanup(
                row=row,
                incumbent=best_af,
                run_config=run_config,
                removed_by_kind=removed_by_kind,
            )
    else:
        print(f"[{tag}] AF candidates disabled; evaluating selected PDB/CIF structures only.", flush=True)

    selected_pdb_ids = list(pdb_ids)
    if run_config.max_experimental_tries is not None:
        selected_pdb_ids = selected_pdb_ids[: max(0, int(run_config.max_experimental_tries))]
    print(f"[{tag}] PDB candidates to evaluate: {len(selected_pdb_ids)}", flush=True)
    for pdb_id in selected_pdb_ids:
        candidate_started = time.perf_counter()
        fetch_started = time.perf_counter()
        model_path = fetch.download_rcsb_model_any(pdb_id, run_config.source_dir / safe_name / "pdb")
        fetch_seconds = _elapsed_seconds(fetch_started)
        pdb_extra = {
            "case_key": _candidate_case_key(resolved_gene_id or tag, pdb_id),
            "resolution": None,
            "assembly_id": None,
            "sequence_selection_score": selection.best_score,
            "candidate_generation_method": selection.metadata.get("candidate_generation_method"),
        }
        if not model_path:
            row = _download_error_row(
                label=f"{safe_name}__PDB_{pdb_id}",
                kind="PDB",
                candidate_id=pdb_id,
                align_config=align_config,
                candidate_started=candidate_started,
                fetch_seconds=fetch_seconds,
                extra=pdb_extra,
            )
            rows.append(row)
            _print_candidate_result(tag, "PDB", pdb_id, row)
            continue
        row = _fit_candidate(
            label=f"{safe_name}__PDB_{pdb_id}",
            kind="PDB",
            candidate_id=pdb_id,
            map_path=map_path,
            model_path=Path(model_path),
            recommended_contour_level=recommended_contour_level,
            run_config=run_config,
            align_config=align_config,
            input_meta={**input_meta, "case_key": pdb_extra["case_key"]},
            extra=pdb_extra,
        )
        row["source_fetch_seconds"] = fetch_seconds
        row["candidate_total_seconds"] = _elapsed_seconds(candidate_started)
        _cleanup_candidate_scratch(row, run_config=run_config, removed_by_kind=removed_by_kind)
        rows.append(row)
        _print_candidate_result(tag, "PDB", pdb_id, row)
        best_pdb = _update_best_and_cleanup(
            row=row,
            incumbent=best_pdb,
            run_config=run_config,
            removed_by_kind=removed_by_kind,
        )

    valid = [row for row in rows if row.get("status") == "ok" and _score_value(row) != float("-inf")]
    valid.sort(key=lambda row: (_score_value(row), str(row.get("label") or "")), reverse=True)
    best_overall = valid[0] if valid else None
    keep_labels = {row.get("label") for row in (best_af, best_pdb, best_overall) if row}
    for row in rows:
        row["kept"] = row.get("label") in keep_labels

    case_key = str((best_overall or {}).get("case_key") or (best_overall or {}).get("alignment_meta", {}).get("case_key") or "")
    entry_total_seconds = _elapsed_seconds(entry_started)
    print(
        f"[{tag}] entry complete: status={'ok' if valid else 'error'} "
        f"candidates={len(rows)} ok={len(valid)} "
        f"elapsed={_format_seconds(entry_total_seconds)}",
        flush=True,
    )
    log_path = _write_target_log(run_config.logs_dir, safe_name, rows)
    return {
        "target": tag,
        "case_key": case_key,
        "safe_name": safe_name,
        "status": "ok" if valid else "error",
        "error": "" if valid else "no_successful_candidates",
        "map_path": map_path,
        "recommended_contour_level": recommended_contour_level,
        "uniprot_accessions": accessions,
        "best_sw_score": selection.best_score,
        "candidate_generation": selection.metadata,
        "selected_pdb_ids": pdb_ids,
        "score_metric": align_config.score_metric,
        "selected_method_id": align_config.method_id,
        "alignment_runner": "local_rawfield.run_raw_field",
        "candidate_count": len(rows),
        "ok_candidate_count": len(valid),
        "sequence_selection_seconds": sequence_selection_seconds,
        "entry_total_seconds": entry_total_seconds,
        "best_af": best_af,
        "best_pdb": best_pdb,
        "best_overall": best_overall,
        "kept_candidate_labels": sorted(label for label in keep_labels if label),
        "pruned_model_count": sum(len(paths) for paths in removed_by_kind.values()),
        "pruned_source_model_count": len(removed_by_kind.get("source", [])),
        "pruned_source_models": removed_by_kind.get("source", []),
        "pruned_fitted_model_count": len(removed_by_kind.get("fitted", [])),
        "pruned_fitted_models": removed_by_kind.get("fitted", []),
        "pruned_alignment_input_model_count": len(removed_by_kind.get("alignment_input", [])),
        "pruned_alignment_input_models": removed_by_kind.get("alignment_input", []),
        "pruned_scratch_dir_count": len(removed_by_kind.get("scratch", [])),
        "pruned_scratch_dirs": removed_by_kind.get("scratch", []),
        "candidate_log": log_path,
    }


def _input_meta_from_row(
    row: dict[str, Any],
    *,
    density_inputs: DensityInputs,
    recommended_contour_level: Optional[float],
) -> dict[str, Any]:
    """Assemble the per-row meta dict passed into the aligner via input_meta."""
    keys = ["gene_id", "density_dir"]
    meta = {key: row.get(key, "") for key in keys if key in row}
    if str(density_inputs.emdb_id or "").strip():
        meta["emdb_id"] = density_inputs.emdb_id
    meta["density_dir"] = str(density_inputs.density_dir)
    meta["map_path"] = str(density_inputs.ground_truth_map_path)
    meta["raw_prediction_dir"] = str(density_inputs.prediction_dir)
    meta["raw_key_atoms_1_path"] = str(density_inputs.raw_key_atoms_1_path)
    meta["recommended_contour_level"] = recommended_contour_level
    return meta


def write_summary_csv(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Write the human-readable run summary CSV to ``run_dir/summary.csv``."""
    fieldnames = [
        "case_key",
        "target",
        "status",
        "candidate_count",
        "ok_candidate_count",
        "sequence_selection_seconds",
        "entry_total_seconds",
        "candidate_generation_method",
        "selected_uniprot_accessions",
        "selected_pdb_candidate_count",
        "best_sequence_score",
        "tied_top_sequence_count",
        "tied_top_chain_count",
        "score_metric",
        "best_type",
        "best_id",
        "best_score",
        "best_source_fetch_seconds",
        "best_alignment_seconds",
        "best_candidate_total_seconds",
        "best_raw_field_score",
        "best_final_map_meanz",
        "best_map_cc_score",
        "best_source_model",
        "best_af_id",
        "best_af_score",
        "best_af_source_model",
        "best_af_fitted_path",
        "best_pdb_id",
        "best_pdb_score",
        "best_pdb_source_model",
        "best_pdb_fitted_path",
        "pruned_source_model_count",
        "pruned_fitted_model_count",
        "pruned_alignment_input_model_count",
        "pruned_scratch_dir_count",
        "selected_method_id",
        "alignment_runner",
        "best_fitted_path",
        "candidate_log",
        "error",
    ]
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            best = row.get("best_overall") or {}
            best_af = row.get("best_af") or {}
            best_pdb = row.get("best_pdb") or {}
            candidate_generation = row.get("candidate_generation") or {}
            writer.writerow(
                {
                    "case_key": row.get("case_key") or best.get("case_key") or best.get("alignment_meta", {}).get("case_key"),
                    "target": row.get("target"),
                    "status": row.get("status"),
                    "candidate_count": row.get("candidate_count"),
                    "ok_candidate_count": row.get("ok_candidate_count"),
                    "sequence_selection_seconds": row.get("sequence_selection_seconds"),
                    "entry_total_seconds": row.get("entry_total_seconds"),
                    "candidate_generation_method": candidate_generation.get("candidate_generation_method"),
                    "selected_uniprot_accessions": ";".join(row.get("uniprot_accessions") or []),
                    "selected_pdb_candidate_count": len(row.get("selected_pdb_ids") or []),
                    "best_sequence_score": row.get("best_sw_score"),
                    "tied_top_sequence_count": candidate_generation.get("tied_top_sequence_count"),
                    "tied_top_chain_count": candidate_generation.get("tied_top_chain_count"),
                    "score_metric": row.get("score_metric"),
                    "best_type": best.get("type"),
                    "best_id": best.get("id"),
                    "best_score": best.get("score"),
                    "best_source_fetch_seconds": best.get("source_fetch_seconds"),
                    "best_alignment_seconds": best.get("alignment_seconds"),
                    "best_candidate_total_seconds": best.get("candidate_total_seconds"),
                    "best_raw_field_score": best.get("raw_field_score"),
                    "best_final_map_meanz": best.get("final_map_meanz"),
                    "best_map_cc_score": best.get("map_cc_score"),
                    "best_source_model": best.get("source_model"),
                    "best_af_id": best_af.get("id"),
                    "best_af_score": best_af.get("score"),
                    "best_af_source_model": best_af.get("source_model"),
                    "best_af_fitted_path": best_af.get("fitted_path"),
                    "best_pdb_id": best_pdb.get("id"),
                    "best_pdb_score": best_pdb.get("score"),
                    "best_pdb_source_model": best_pdb.get("source_model"),
                    "best_pdb_fitted_path": best_pdb.get("fitted_path"),
                    "pruned_source_model_count": row.get("pruned_source_model_count"),
                    "pruned_fitted_model_count": row.get("pruned_fitted_model_count"),
                    "pruned_alignment_input_model_count": row.get("pruned_alignment_input_model_count"),
                    "pruned_scratch_dir_count": row.get("pruned_scratch_dir_count"),
                    "selected_method_id": best.get("selected_method_id") or row.get("selected_method_id"),
                    "alignment_runner": best.get("alignment_runner") or row.get("alignment_runner"),
                    "best_fitted_path": best.get("fitted_path"),
                    "candidate_log": row.get("candidate_log"),
                    "error": row.get("error"),
                }
            )


REQUIRED_CSV_COLUMNS = PIPELINE_REQUIRED_CSV_COLUMNS


def _validate_required_columns(df: pd.DataFrame) -> None:
    """Raise ``ValueError`` listing any missing required CSV columns."""
    missing = [column for column in REQUIRED_CSV_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required column(s): {', '.join(missing)}")


def run_csv(csv_path: Path, *, run_config: RunConfig) -> list[dict[str, Any]]:
    """Production entry point: build the default ``AlignmentConfig`` and delegate to ``run_csv_with_alignment_config``."""
    align_config = AlignmentConfig(
        randomize_start=bool(run_config.randomize_candidate_start),
        random_start_margin_A=float(run_config.random_start_margin_A),
        compute_gt_metrics=False,
    )
    return run_csv_with_alignment_config(csv_path, run_config=run_config, align_config=align_config)


def run_csv_with_alignment_config(csv_path: Path, *, run_config: RunConfig, align_config: AlignmentConfig) -> list[dict[str, Any]]:
    """Iterate through the input CSV, processing each row with the given alignment config.

    Wraps the main loop in ``try/finally`` so :func:`fetch.cleanup_pool` runs
    even when an exception aborts processing. Writes ``summary.csv`` and
    ``summary.json`` incrementally after each row so partial results are
    preserved on crash.
    """
    df = pd.read_csv(csv_path, dtype=str)
    _validate_required_columns(df)
    total_rows = len(df)
    if int(run_config.entries) == 0:
        run_config.entries = total_rows
    if int(run_config.entries) < 1 or int(run_config.entries) > total_rows:
        raise ValueError(f"entries must be 0/all or between 1 and {total_rows}")
    for path in (run_config.run_dir, run_config.data_dir, run_config.source_dir, run_config.fitted_dir, run_config.logs_dir):
        path.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    processed = 0
    try:
        for _, raw_row in df.iterrows():
            if processed >= int(run_config.entries):
                break
            row = raw_row.to_dict()
            sequence = str(row.get("sequence") or "").strip()
            if not sequence:
                raise ValueError("Input CSV row has empty required sequence")
            density_inputs = resolve_density_inputs(row)
            row_recommended = float_or_none(row.get("recommended_contour_level"))
            recommended = run_config.recommended_contour_level if run_config.recommended_contour_level is not None else row_recommended
            input_meta = _input_meta_from_row(
                row,
                density_inputs=density_inputs,
                recommended_contour_level=recommended,
            )
            result = process_target(
                gene_id=str(row.get("gene_id") or "").strip(),
                sequence=sequence,
                map_path=density_inputs.ground_truth_map_path,
                recommended_contour_level=recommended,
                input_meta=input_meta,
                run_config=run_config,
                align_config=align_config,
            )
            result["input_meta"] = input_meta
            results.append(result)
            write_summary_csv(run_config.logs_dir, results)
            write_json(run_config.logs_dir / "summary.json", results)
            processed += 1
        if processed == 0:
            raise ValueError("No rows with non-empty sequence were processed")
        return results
    finally:
        fetch.cleanup_pool()
        map_utils.clear_caches()
        rawfield.clear_caches()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse pipeline CLI args, merging the JSON config with CLI overrides."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Pipeline JSON config path. Defaults to configs/pipeline.json.",
    )
    config_args, _ = config_parser.parse_known_args(argv)
    config_path = resolve_method_path(config_args.config, must_exist=True)

    parser = argparse.ArgumentParser(
        parents=[config_parser],
        description="Production struct fetch with local rawfield_ss_dominant alignment.",
    )
    parser.set_defaults(
        csv=None,
        entries=None,
        run_dir=None,
        max_experimental_tries=None,
        include_alphafold=None,
        recommended_contour_level=None,
        randomize_candidate_start=None,
        random_start_margin_A=None,
        keep_only_best_models=None,
    )
    parser.add_argument("--csv", help="CSV with sequence and density_dir columns. gene_id and recommended_contour_level are optional.")
    parser.add_argument("--entries", type=int, help="Number of non-empty sequence rows to process. Use 0 for all rows.")
    parser.add_argument("--run-dir", help="Run directory. Empty config value creates runs/struct_fetch_rawfield_<timestamp>.")
    parser.add_argument("--max-experimental-tries", type=int, help="Optional cap on experimental PDB candidates.")
    parser.add_argument("--include-alphafold", dest="include_alphafold", action="store_true", default=None)
    parser.add_argument("--skip-alphafold", dest="include_alphafold", action="store_false", default=None)
    parser.add_argument("--recommended-contour-level", type=float, help="Global contour override.")
    parser.add_argument("--randomize-candidate-start", dest="randomize_candidate_start", action="store_true", default=None)
    parser.add_argument("--no-randomize-candidate-start", dest="randomize_candidate_start", action="store_false", default=None)
    parser.add_argument("--random-start-margin-A", dest="random_start_margin_A", type=float)
    parser.add_argument("--keep-only-best-models", dest="keep_only_best_models", action="store_true", default=None)
    parser.add_argument("--keep-all-models", dest="keep_only_best_models", action="store_false", default=None)
    cli_args = parser.parse_args(argv)
    settings = apply_cli_overrides(load_json_config(config_path, DEFAULT_SETTINGS), cli_args)
    return argparse.Namespace(**settings)


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for the production pipeline."""
    args = parse_args(argv)
    csv_path = resolve_method_path(str(args.csv), must_exist=True)
    run_dir = resolve_method_path(str(args.run_dir), must_exist=False) if str(args.run_dir).strip() else _timestamped_run_dir()
    data_dir = run_dir / "data"
    config = RunConfig(
        run_dir=run_dir,
        data_dir=data_dir,
        source_dir=data_dir / "sources",
        fitted_dir=data_dir / "fitted_models",
        logs_dir=data_dir / "logs",
        entries=int(args.entries),
        max_experimental_tries=args.max_experimental_tries,
        include_alphafold=bool(args.include_alphafold),
        recommended_contour_level=args.recommended_contour_level,
        randomize_candidate_start=bool(args.randomize_candidate_start),
        random_start_margin_A=float(args.random_start_margin_A),
        keep_only_best_models=bool(args.keep_only_best_models),
    )
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    with tee_output(config.logs_dir / "terminal.log"):
        print(f"[struct_fetch_new_method] terminal log -> {(config.logs_dir / 'terminal.log').resolve().as_posix()}", flush=True)
        write_json(
            config.logs_dir / "run_config.json",
            {
                "config_file": args.config,
                "resolved_settings": vars(args),
                "csv": csv_path,
                **config.__dict__,
                "score_metric": FINAL_SCORE_METRIC,
                "selected_method_id": AlignmentConfig().method_id,
                "alignment_runner": "local_rawfield.run_raw_field",
                "candidate_generation_method": "pdb_chain_sw",
                "include_alphafold": bool(args.include_alphafold),
            },
        )
        rows = run_csv(csv_path, run_config=config)
        print(f"\nRun data      -> {config.data_dir.resolve().as_posix()}", flush=True)
        print(f"Fitted models -> {config.fitted_dir.resolve().as_posix()}", flush=True)
        print(f"Logs          -> {config.logs_dir.resolve().as_posix()}", flush=True)
        print(f"[struct_fetch_new_method] wrote {config.logs_dir / 'summary.csv'} ({len(rows)} target(s))", flush=True)


if __name__ == "__main__":
    main()
