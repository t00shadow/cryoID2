"""
GT-only StructFetch benchmark (alignment-only control).

Skips candidate fetching entirely. For each CSV row, takes the deposited
ground-truth model, runs the same rawfield_ss_dominant alignment against the
cryo-EM map (with the same randomization protocol as the fetching benchmark),
computes Q-scores in ChimeraX against both the original and the re-aligned
model, and writes a Q-score CSV + plot.

This isolates the contribution of rigid-body pose recovery: any deviation
from the diagonal in the resulting plot reflects alignment behaviour rather
than candidate-selection error.

Supports reuse of prior alignments and prior Q-scores via the
``reuse_run_dirs`` and ``reuse_qscore_csvs`` config knobs.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

from aligner import AlignmentConfig, align_model_to_map
from cases import float_or_none
from config_io import apply_cli_overrides, load_json_config, resolve_method_path
from inputs import ground_truth_model_id, ground_truth_model_path_for_row, resolve_density_inputs
from io_utils import format_float, write_json
from paths import METHOD_ROOT
from run_logging import tee_output


DEFAULT_CONFIG_PATH = METHOD_ROOT / "configs" / "benchmark_gt_only.json"

DEFAULT_SETTINGS: dict[str, Any] = {
    "csv": "test_data/ready_to_test_1_29.csv",
    "entries": 0,
    "run_dir": "runs/gt_only_benchmark_current",
    "run_alignment": True,
    "compute_qscore": True,
    "generate_plot": True,
    "chimerax": "",
    "alignment_workers": 4,
    "qscore_workers": 2,
    "qscore_timeout_sec": 1800,
    "randomize_candidate_start": True,
    "random_start_margin_A": 4.0,
    "reuse_existing": True,
    "reuse_run_dirs": ["runs/gt_only_alignment_qscore_check"],
    "reuse_qscore_csvs": [
        "runs/benchmark_all_20260511_133707/data/qscore/benchmark_qscores.csv",
        "runs/gt_only_alignment_qscore_check/data/gt_only_qscore_check.csv",
    ],
}

SUMMARY_FIELDS = [
    "case_key",
    "gene_id",
    "emdb_id",
    "ground_truth_model",
    "resolution",
    "status",
    "error",
    "alignment_score_metric",
    "alignment_score",
    "alignment_mean_z",
    "alignment_map_cc_score",
    "alignment_elapsed_s",
    "wall_alignment_seconds",
    "reused_alignment",
    "reused_alignment_from",
    "randomized_start",
    "map_path",
    "gt_model_path",
    "fitted_path",
]

QSCORE_FIELDS = [
    "case_key",
    "gene_id",
    "emdb_id",
    "ground_truth_model",
    "resolution",
    "candidate_type",
    "candidate_id",
    "best_is_ground_truth",
    "alignment_score_metric",
    "alignment_score",
    "alignment_mean_z",
    "alignment_map_cc_score",
    "qscore_ground_truth",
    "qscore_identified",
    "qscore_diff_identified_minus_gt",
    "status",
    "error",
    "alignment_status",
    "alignment_elapsed_s",
    "qscore_elapsed_s",
    "reused_alignment",
    "reused_qscore_ground_truth",
    "reused_qscore_identified",
    "map_path",
    "gt_model_path",
    "fitted_path",
]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write ``rows`` to ``path`` with the given ``fields`` as column order; extras ignored."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _csv_rows(csv_path: Path, entries: int) -> list[dict[str, str]]:
    """Read a CSV; cap to the first ``entries`` rows when ``entries > 0``."""
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if int(entries) > 0:
        return rows[: int(entries)]
    return rows


def _case_key_from_density_dir(density_dir: Path) -> str:
    """Derive a case key from the parent directory name of ``density_dir``."""
    return Path(density_dir).parent.name


def _case_key_from_map_path(path: str | Path) -> str:
    """Recover the case key from a deeply-nested ground-truth-map path (best effort)."""
    try:
        return Path(path).parents[2].name
    except Exception:
        return ""


def _resolve_optional_paths(values: list[Any]) -> list[Path]:
    """Resolve each non-empty path string and keep only those that exist on disk."""
    out: list[Path] = []
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        path = resolve_method_path(text, must_exist=False)
        if path.exists():
            out.append(path)
    return out


def _row_meta(row: dict[str, Any]) -> dict[str, Any]:
    """Resolve density inputs and GT model for one CSV row into a meta dict."""
    density_inputs = resolve_density_inputs(row)
    gt_model_path = ground_truth_model_path_for_row(row)
    case_key = _case_key_from_density_dir(density_inputs.density_dir)
    recommended = float_or_none(row.get("recommended_contour_level"))
    gt_model = ground_truth_model_id(row, gt_model_path)
    return {
        "case_key": case_key,
        "gene_id": str(row.get("gene_id") or "").strip(),
        "emdb_id": density_inputs.emdb_id,
        "ground_truth_model": gt_model,
        "resolution": str(row.get("resolution") or ""),
        "recommended_contour_level": recommended,
        "density_dir": str(density_inputs.density_dir),
        "map_path": str(density_inputs.ground_truth_map_path),
        "raw_prediction_dir": str(density_inputs.prediction_dir),
        "raw_key_atoms_1_path": str(density_inputs.raw_key_atoms_1_path),
        "gt_model_path": str(gt_model_path),
    }


def _output_model_path(fitted_dir: Path, meta: dict[str, Any]) -> Path:
    """Compose the per-case fitted-model output path for the GT-only benchmark."""
    gt_path = Path(str(meta["gt_model_path"]))
    suffix = gt_path.suffix.lower()
    if suffix not in {".cif", ".mmcif", ".pdb"}:
        suffix = ".cif"
    emdb_short = str(meta.get("emdb_id") or "").replace("EMD-", "").replace("EMD", "")
    stem = f"{meta['gene_id']}__{meta['ground_truth_model']}__{emdb_short}__GT_rawfield"
    return fitted_dir / f"{stem}{suffix}"


def _load_existing_summary(path: Path) -> dict[str, dict[str, Any]]:
    """Read an existing GT-only summary CSV into a dict keyed by case_key."""
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {str(row.get("case_key") or row.get("case") or ""): row for row in csv.DictReader(handle)}


def _legacy_alignment_row(meta: dict[str, Any], output_path: Path, reuse_run_dirs: list[Path]) -> Optional[dict[str, Any]]:
    """Try to satisfy a case by copying a fitted model from a prior run-dir; ``None`` if not found."""
    legacy_name = f"{meta['case_key']}__GT_only_structfetch_rawfield{output_path.suffix}"
    names = [output_path.name, legacy_name]
    for run_dir in reuse_run_dirs:
        for name in names:
            candidate = run_dir / "data" / "fitted_models" / name
            if candidate.exists() and candidate.stat().st_size > 0:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if candidate.resolve() != output_path.resolve():
                    shutil.copy2(candidate, output_path)
                row: dict[str, Any] = {
                    **meta,
                    "status": "ok",
                    "error": "",
                    "alignment_score_metric": AlignmentConfig().score_metric,
                    "alignment_score": "",
                    "alignment_mean_z": "",
                    "alignment_map_cc_score": "",
                    "alignment_elapsed_s": "",
                    "wall_alignment_seconds": "",
                    "reused_alignment": True,
                    "reused_alignment_from": str(candidate),
                    "randomized_start": True,
                    "map_path": meta["map_path"],
                    "gt_model_path": meta["gt_model_path"],
                    "fitted_path": str(output_path),
                }
                return row
    return None


def _align_one(job: dict[str, Any]) -> dict[str, Any]:
    """Run rawfield alignment for one GT-only job and return a status row."""
    meta = dict(job["meta"])
    output_path = Path(str(job["output_path"]))
    config = AlignmentConfig(
        randomize_start=bool(job["randomize_start"]),
        random_start_margin_A=float(job["random_start_margin_A"]),
    )
    started = time.perf_counter()
    base = {
        **meta,
        "alignment_score_metric": config.score_metric,
        "map_path": meta["map_path"],
        "gt_model_path": meta["gt_model_path"],
        "fitted_path": str(output_path),
        "reused_alignment": False,
        "reused_alignment_from": "",
        "randomized_start": bool(job["randomize_start"]),
    }
    try:
        payload = align_model_to_map(
            Path(str(meta["map_path"])),
            Path(str(meta["gt_model_path"])),
            output_path,
            recommended_contour_level=float_or_none(meta.get("recommended_contour_level")),
            config=config,
            input_meta=meta,
        )
        return {
            **base,
            "status": "ok",
            "error": "",
            "alignment_score": payload.get("score"),
            "alignment_mean_z": payload.get("final_map_meanz") or payload.get("mean_z"),
            "alignment_map_cc_score": payload.get("map_cc_score"),
            "alignment_elapsed_s": payload.get("alignment_seconds"),
            "wall_alignment_seconds": float(time.perf_counter() - started),
        }
    except Exception as exc:
        return {
            **base,
            "status": "error",
            "error": repr(exc),
            "alignment_score": "",
            "alignment_mean_z": "",
            "alignment_map_cc_score": "",
            "alignment_elapsed_s": "",
            "wall_alignment_seconds": float(time.perf_counter() - started),
        }


def run_alignment(
    *,
    csv_rows: list[dict[str, str]],
    fitted_dir: Path,
    summary_csv: Path,
    summary_json: Path,
    reuse_existing: bool,
    reuse_run_dirs: list[Path],
    workers: int,
    randomize_start: bool,
    random_start_margin_A: float,
) -> list[dict[str, Any]]:
    """Align the deposited GT model for every CSV row, reusing previous results when possible.

    Reuse strategy: prefer rows from the current run's summary CSV, then fall
    back to fitted models from any of ``reuse_run_dirs``. Anything not reusable
    is aligned fresh, in parallel when ``workers > 1``.
    """
    existing_by_key = _load_existing_summary(summary_csv) if reuse_existing else {}
    rows: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    for input_row in csv_rows:
        meta = _row_meta(input_row)
        output_path = _output_model_path(fitted_dir, meta)
        existing = existing_by_key.get(str(meta["case_key"]))
        if reuse_existing and existing and str(existing.get("status") or "") == "ok":
            fitted_path = Path(str(existing.get("fitted_path") or output_path))
            if fitted_path.exists() and fitted_path.stat().st_size > 0:
                reused = dict(existing)
                reused["reused_alignment"] = True
                reused["reused_alignment_from"] = str(fitted_path)
                rows.append(reused)
                continue
        if reuse_existing:
            legacy = _legacy_alignment_row(meta, output_path, reuse_run_dirs)
            if legacy is not None:
                rows.append(legacy)
                continue
        jobs.append(
            {
                "meta": meta,
                "output_path": str(output_path),
                "randomize_start": bool(randomize_start),
                "random_start_margin_A": float(random_start_margin_A),
            }
        )

    by_key = {str(row["case_key"]): row for row in rows}
    if jobs:
        print(f"[gt-only] aligning {len(jobs)} case(s) with {max(1, int(workers))} worker(s)", flush=True)
        if int(workers) <= 1:
            for idx, job in enumerate(jobs, start=1):
                row = _align_one(job)
                by_key[str(row["case_key"])] = row
                ordered = [by_key[str(_row_meta(input_row)["case_key"])] for input_row in csv_rows if str(_row_meta(input_row)["case_key"]) in by_key]
                write_csv(summary_csv, ordered, SUMMARY_FIELDS)
                write_json(summary_json, ordered)
                print(f"[align] {idx}/{len(jobs)} {row['case_key']} status={row['status']} score={row.get('alignment_score')}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="gt-only-align") as executor:
                future_to_case = {executor.submit(_align_one, job): job["meta"]["case_key"] for job in jobs}
                for idx, future in enumerate(as_completed(future_to_case), start=1):
                    row = future.result()
                    by_key[str(row["case_key"])] = row
                    ordered = [by_key[str(_row_meta(input_row)["case_key"])] for input_row in csv_rows if str(_row_meta(input_row)["case_key"]) in by_key]
                    write_csv(summary_csv, ordered, SUMMARY_FIELDS)
                    write_json(summary_json, ordered)
                    print(f"[align] {idx}/{len(jobs)} {row['case_key']} status={row['status']} score={row.get('alignment_score')}", flush=True)
    else:
        print("[gt-only] all alignments reused", flush=True)

    ordered_final = [by_key[str(_row_meta(input_row)["case_key"])] for input_row in csv_rows if str(_row_meta(input_row)["case_key"]) in by_key]
    write_csv(summary_csv, ordered_final, SUMMARY_FIELDS)
    write_json(summary_json, ordered_final)
    return ordered_final


def _load_reuse_qscores(paths: list[Path]) -> tuple[dict[str, str], dict[str, str]]:
    """Read GT and identified Q-score values from prior CSVs keyed by case_key."""
    gt_by_case: dict[str, str] = {}
    identified_by_case: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                case_key = _case_key_from_map_path(str(row.get("map_path") or "")) or str(row.get("case_key") or row.get("case") or "").strip()
                if not case_key:
                    continue
                gt_value = row.get("qscore_ground_truth") or row.get("reference_gt_qscore")
                identified_value = ""
                if row.get("structfetch_gt_aligned_qscore") not in (None, ""):
                    identified_value = str(row.get("structfetch_gt_aligned_qscore"))
                elif str(row.get("candidate_type") or "").upper() == "GT":
                    identified_value = str(row.get("qscore_identified") or "")
                elif str(row.get("best_is_ground_truth") or "").lower() == "true":
                    identified_value = str(row.get("qscore_identified") or "")
                if gt_value not in (None, ""):
                    gt_by_case.setdefault(case_key, str(gt_value))
                if identified_value not in (None, ""):
                    identified_by_case.setdefault(case_key, str(identified_value))
    return gt_by_case, identified_by_case


def _qscore_one(
    *,
    row: dict[str, Any],
    chimerax_path: Path,
    work_root: Path,
    timeout_sec: int,
    qscore_gt_reuse: str,
    qscore_identified_reuse: str,
) -> dict[str, Any]:
    """Compute (or reuse) Q-scores for one alignment row and return an output dict."""
    import qscore

    out = {
        "case_key": row.get("case_key"),
        "gene_id": row.get("gene_id"),
        "emdb_id": row.get("emdb_id"),
        "ground_truth_model": row.get("ground_truth_model"),
        "resolution": row.get("resolution"),
        "candidate_type": "GT",
        "candidate_id": row.get("ground_truth_model"),
        "best_is_ground_truth": True,
        "alignment_score_metric": row.get("alignment_score_metric"),
        "alignment_score": row.get("alignment_score"),
        "alignment_mean_z": row.get("alignment_mean_z"),
        "alignment_map_cc_score": row.get("alignment_map_cc_score"),
        "alignment_status": row.get("status"),
        "alignment_elapsed_s": row.get("alignment_elapsed_s"),
        "reused_alignment": row.get("reused_alignment"),
        "map_path": row.get("map_path"),
        "gt_model_path": row.get("gt_model_path"),
        "fitted_path": row.get("fitted_path"),
        "reused_qscore_ground_truth": bool(qscore_gt_reuse),
        "reused_qscore_identified": bool(qscore_identified_reuse),
    }
    if str(row.get("status") or "") != "ok":
        out.update({"status": "error", "error": f"alignment_status={row.get('status')}"})
        return out
    try:
        gt_q = float(qscore_gt_reuse) if qscore_gt_reuse not in (None, "") else None
        identified_q = float(qscore_identified_reuse) if qscore_identified_reuse not in (None, "") else None
        elapsed = 0.0
        if gt_q is None:
            gt_q, error, gt_elapsed = qscore.compute_model_qscore(
                model_path=Path(str(row["gt_model_path"])),
                map_path=Path(str(row["map_path"])),
                chimerax_path=chimerax_path,
                work_root=work_root / str(row.get("case_key") or "unknown") / "gt",
                timeout_sec=timeout_sec,
            )
            elapsed += gt_elapsed
            if error:
                out.update({"status": "error", "error": error, "qscore_elapsed_s": elapsed})
                return out
        if identified_q is None:
            identified_q, error, identified_elapsed = qscore.compute_model_qscore(
                model_path=Path(str(row["fitted_path"])),
                map_path=Path(str(row["map_path"])),
                chimerax_path=chimerax_path,
                work_root=work_root / str(row.get("case_key") or "unknown") / "identified",
                timeout_sec=timeout_sec,
            )
            elapsed += identified_elapsed
            if error:
                out.update({"status": "error", "error": error, "qscore_elapsed_s": elapsed})
                return out
        out["qscore_ground_truth"] = format_float(gt_q)
        out["qscore_identified"] = format_float(identified_q)
        out["qscore_diff_identified_minus_gt"] = format_float(float(identified_q) - float(gt_q))
        out["status"] = "ok"
        out["error"] = ""
        out["qscore_elapsed_s"] = elapsed
        return out
    except Exception as exc:
        out.update({"status": "error", "error": repr(exc)})
        return out


def compute_qscores(
    *,
    alignment_rows: list[dict[str, Any]],
    output_csv: Path,
    output_json: Path,
    chimerax_path: Path,
    work_root: Path,
    timeout_sec: int,
    workers: int,
    reuse_qscore_csvs: list[Path],
    reuse_existing: bool,
) -> list[dict[str, Any]]:
    """Compute Q-scores for every alignment row in parallel, writing CSV/JSON incrementally."""
    gt_reuse, identified_reuse = _load_reuse_qscores(reuse_qscore_csvs)
    if reuse_existing:
        existing_gt, existing_identified = _load_reuse_qscores([output_csv])
        gt_reuse = {**gt_reuse, **existing_gt}
        identified_reuse = {**identified_reuse, **existing_identified}
    results_by_key: dict[str, dict[str, Any]] = {}
    jobs = []
    for row in alignment_rows:
        case_key = str(row.get("case_key") or "")
        jobs.append(
            {
                "row": row,
                "chimerax_path": chimerax_path,
                "work_root": work_root,
                "timeout_sec": int(timeout_sec),
                "qscore_gt_reuse": gt_reuse.get(case_key, ""),
                "qscore_identified_reuse": identified_reuse.get(case_key, ""),
            }
        )

    print(f"[gt-only] qscore jobs: {len(jobs)} with {max(1, int(workers))} worker(s)", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="gt-only-qscore") as executor:
        futures = [executor.submit(_qscore_one, **job) for job in jobs]
        for idx, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            results_by_key[str(row.get("case_key") or "")] = row
            ordered = [results_by_key[str(src.get("case_key") or "")] for src in alignment_rows if str(src.get("case_key") or "") in results_by_key]
            write_csv(output_csv, ordered, QSCORE_FIELDS)
            write_json(output_json, ordered)
            print(
                f"[qscore] {idx}/{len(jobs)} {row.get('case_key')} status={row.get('status')} "
                f"gt={row.get('qscore_ground_truth')} identified={row.get('qscore_identified')}",
                flush=True,
            )
    shutil.rmtree(work_root, ignore_errors=True)
    final_rows = [results_by_key[str(src.get("case_key") or "")] for src in alignment_rows if str(src.get("case_key") or "") in results_by_key]
    write_csv(output_csv, final_rows, QSCORE_FIELDS)
    write_json(output_json, final_rows)
    return final_rows


def _print_summary(qscore_rows: list[dict[str, Any]]) -> None:
    """Print mean / median diff and the within-0.001 count to stdout."""
    ok = [row for row in qscore_rows if str(row.get("status") or "") == "ok"]
    diffs = [float(row["qscore_diff_identified_minus_gt"]) for row in ok if str(row.get("qscore_diff_identified_minus_gt") or "").strip()]
    if not ok:
        print("[summary] no ok qscore rows", flush=True)
        return
    mean_diff = sum(diffs) / len(diffs) if diffs else float("nan")
    sorted_diffs = sorted(diffs)
    median_diff = sorted_diffs[len(sorted_diffs) // 2] if sorted_diffs else float("nan")
    within_001 = sum(1 for value in diffs if abs(value) <= 0.001)
    print(f"[summary] ok={len(ok)} errors={len(qscore_rows) - len(ok)}", flush=True)
    print(f"[summary] mean_diff={mean_diff:.9f} median_diff={median_diff:.9f} within_0.001={within_001}/{len(ok)}", flush=True)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse GT-only benchmark CLI args, merging the JSON config with CLI overrides."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="GT-only benchmark JSON config.")
    config_args, _ = config_parser.parse_known_args(argv)
    config_path = resolve_method_path(config_args.config, must_exist=True)
    parser = argparse.ArgumentParser(parents=[config_parser], description="Run GT-only StructFetch alignment and Q-score benchmark.")
    parser.set_defaults(**{key: None for key in DEFAULT_SETTINGS})
    parser.add_argument("--csv")
    parser.add_argument("--entries", type=int)
    parser.add_argument("--run-dir")
    parser.add_argument("--run-alignment", dest="run_alignment", action="store_true", default=None)
    parser.add_argument("--skip-alignment", dest="run_alignment", action="store_false", default=None)
    parser.add_argument("--compute-qscore", dest="compute_qscore", action="store_true", default=None)
    parser.add_argument("--skip-qscore", dest="compute_qscore", action="store_false", default=None)
    parser.add_argument("--generate-plot", dest="generate_plot", action="store_true", default=None)
    parser.add_argument("--skip-plot", dest="generate_plot", action="store_false", default=None)
    parser.add_argument("--chimerax")
    parser.add_argument("--alignment-workers", type=int)
    parser.add_argument("--qscore-workers", type=int)
    parser.add_argument("--qscore-timeout-sec", type=int)
    parser.add_argument("--randomize-candidate-start", dest="randomize_candidate_start", action="store_true", default=None)
    parser.add_argument("--no-randomize-candidate-start", dest="randomize_candidate_start", action="store_false", default=None)
    parser.add_argument("--random-start-margin-A", dest="random_start_margin_A", type=float)
    parser.add_argument("--reuse-existing", dest="reuse_existing", action="store_true", default=None)
    parser.add_argument("--no-reuse-existing", dest="reuse_existing", action="store_false", default=None)
    cli_args = parser.parse_args(argv)
    settings = load_json_config(config_path, DEFAULT_SETTINGS)
    settings = apply_cli_overrides(settings, cli_args)
    settings["config"] = str(config_path)
    return argparse.Namespace(**settings)


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for the GT-only alignment + Q-score benchmark."""
    args = parse_args(argv)
    run_dir = resolve_method_path(args.run_dir, must_exist=False)
    data_dir = run_dir / "data"
    logs_dir = data_dir / "logs"
    fitted_dir = data_dir / "fitted_models"
    qscore_dir = data_dir / "qscore"
    csv_path = resolve_method_path(args.csv, must_exist=True)
    reuse_run_dirs = _resolve_optional_paths(list(args.reuse_run_dirs or []))
    reuse_qscore_csvs = _resolve_optional_paths(list(args.reuse_qscore_csvs or []))
    logs_dir.mkdir(parents=True, exist_ok=True)
    with tee_output(logs_dir / "terminal.log"):
        print(f"[gt-only] terminal log -> {(logs_dir / 'terminal.log').resolve().as_posix()}", flush=True)
        rows = _csv_rows(csv_path, int(args.entries))
        write_json(
            data_dir / "benchmark_gt_only_config.json",
            {
                "config_file": args.config,
                "resolved_settings": vars(args),
                "csv": csv_path,
                "run_dir": run_dir,
                "row_count": len(rows),
                "reuse_run_dirs": reuse_run_dirs,
                "reuse_qscore_csvs": reuse_qscore_csvs,
            },
        )
        summary_csv = logs_dir / "gt_only_summary.csv"
        summary_json = logs_dir / "gt_only_summary.json"
        qscore_csv = qscore_dir / "gt_only_qscores.csv"
        qscore_json = qscore_dir / "gt_only_qscores.json"
        plot_png = qscore_dir / "gt_only_qscores.png"

        if args.run_alignment:
            alignment_rows = run_alignment(
                csv_rows=rows,
                fitted_dir=fitted_dir,
                summary_csv=summary_csv,
                summary_json=summary_json,
                reuse_existing=bool(args.reuse_existing),
                reuse_run_dirs=reuse_run_dirs,
                workers=max(1, int(args.alignment_workers)),
                randomize_start=bool(args.randomize_candidate_start),
                random_start_margin_A=float(args.random_start_margin_A),
            )
        else:
            if not summary_csv.exists():
                raise FileNotFoundError(f"GT-only summary not found: {summary_csv}")
            with summary_csv.open("r", newline="", encoding="utf-8") as handle:
                alignment_rows = list(csv.DictReader(handle))

        if args.compute_qscore:
            import qscore

            chimerax_path = qscore.find_chimerax(args.chimerax or None)
            qscore_rows = compute_qscores(
                alignment_rows=alignment_rows,
                output_csv=qscore_csv,
                output_json=qscore_json,
                chimerax_path=chimerax_path,
                work_root=qscore_dir / "_tmp",
                timeout_sec=int(args.qscore_timeout_sec),
                workers=max(1, int(args.qscore_workers)),
                reuse_qscore_csvs=reuse_qscore_csvs,
                reuse_existing=bool(args.reuse_existing),
            )
            _print_summary(qscore_rows)
        elif args.generate_plot:
            if not qscore_csv.exists():
                raise FileNotFoundError(f"GT-only Q-score CSV not found: {qscore_csv}")
            with qscore_csv.open("r", newline="", encoding="utf-8") as handle:
                qscore_rows = list(csv.DictReader(handle))
        else:
            qscore_rows = []

        if args.generate_plot:
            from plot_qscore import plot_qscores

            plot_qscores(qscore_csv, plot_png, title="Alignment-only Q-scores")
            print(f"[gt-only] plot -> {plot_png.resolve().as_posix()}", flush=True)

        print(f"[gt-only] summary -> {summary_csv.resolve().as_posix()}", flush=True)
        print(f"[gt-only] qscores -> {qscore_csv.resolve().as_posix()}", flush=True)
        print(f"[gt-only] run data -> {data_dir.resolve().as_posix()}", flush=True)


if __name__ == "__main__":
    main()
