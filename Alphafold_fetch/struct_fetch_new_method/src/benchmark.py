"""
Full StructFetch benchmark workflow.

Wraps the production pipeline with a Q-score evaluation + plot phase:

1. (Optional) Call ``pipeline.run_csv`` to do candidate fetching and rawfield
   alignment for every CSV row, writing per-target results into
   ``run-dir/data/logs/``.
2. (Optional) For each completed target, run ChimeraX Q-score against both the
   deposited GT model and the StructFetch-selected fitted model, writing a
   merged Q-score CSV/JSON into ``run-dir/data/qscore/``.
3. (Optional) Render the GT-vs-identified Q-score scatter plot.

Each phase can be skipped with the matching ``--skip-*`` flag so partial
re-runs (e.g. plot-only after a previous Q-score pass) are cheap.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from config_io import apply_cli_overrides, load_json_config, resolve_method_path
from io_utils import format_float, write_json
from paths import DEFAULT_RUNS_DIR, METHOD_ROOT
from run_logging import tee_output


DEFAULT_CONFIG_PATH = METHOD_ROOT / "configs" / "benchmark_all.json"

DEFAULT_SETTINGS: dict[str, Any] = {
    "csv": "test_data/ready_to_test_1_29.csv",
    "entries": 0,
    "run_dir": "",
    "max_experimental_tries": None,
    "include_alphafold": True,
    "run_pipeline": True,
    "compute_qscore": True,
    "generate_plot": True,
    "chimerax": "",
    "qscore_workers": 1,
    "qscore_timeout_sec": 1800,
    "randomize_candidate_start": True,
    "random_start_margin_A": 4.0,
    "keep_only_best_models": True,
}


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
    "elapsed_s",
    "map_path",
    "gt_model_path",
    "fitted_path",
]


def _timestamped_run_dir() -> Path:
    """Generate a fresh ``runs/benchmark_all_<YYYYMMDD_HHMMSS>`` path."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return DEFAULT_RUNS_DIR / f"benchmark_all_{stamp}"


def _merged_settings(config_path: Path, cli_args: argparse.Namespace) -> dict[str, Any]:
    """Load the JSON config and overlay any non-None CLI overrides."""
    settings = load_json_config(config_path, DEFAULT_SETTINGS)
    return apply_cli_overrides(settings, cli_args)


def write_qscore_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write Q-score rows in the canonical :data:`QSCORE_FIELDS` column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=QSCORE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(summary_json_path: Path) -> list[dict[str, Any]]:
    """Read the pipeline summary.json into a list of per-target result dicts."""
    return json.loads(summary_json_path.read_text(encoding="utf-8"))


def _csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read a CSV into a list of dicts keyed by header."""
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _qscore_job(row: dict[str, Any], csv_row: dict[str, Any]) -> dict[str, Any]:
    """Build one Q-score job dict by joining a pipeline summary row with its source CSV row."""
    from cases import infer_emd_id, normalize_pdb
    from inputs import ground_truth_model_id, ground_truth_model_path_for_row, resolve_density_inputs

    input_meta = dict(row.get("input_meta") or {})
    best = dict(row.get("best_overall") or {})
    density_inputs = resolve_density_inputs(csv_row)
    map_path = density_inputs.ground_truth_map_path
    inferred_emdb_id = density_inputs.emdb_id
    gt_model_path = ground_truth_model_path_for_row(csv_row)
    case_key = str(best.get("case_key") or best.get("alignment_meta", {}).get("case_key") or "")
    gene_id = str(input_meta.get("gene_id") or row.get("target") or "")
    gt_model = ground_truth_model_id(csv_row, gt_model_path)
    best_id = normalize_pdb(best.get("id") or "")
    emdb_id = inferred_emdb_id or infer_emd_id(map_path.name)
    if not case_key and gene_id:
        raw_candidate_id = str(best.get("id") or "").strip()
        candidate_id = normalize_pdb(raw_candidate_id) or raw_candidate_id
        case_key = f"{gene_id}_{candidate_id}" if candidate_id else gene_id
    return {
        "case_key": case_key,
        "gene_id": gene_id,
        "emdb_id": emdb_id,
        "ground_truth_model": gt_model,
        "resolution": str(csv_row.get("resolution") or ""),
        "candidate_type": str(best.get("type") or ""),
        "candidate_id": str(best.get("id") or ""),
        "best_is_ground_truth": bool(str(best.get("type") or "").upper() == "PDB" and best_id == gt_model),
        "alignment_score_metric": str(best.get("score_metric") or row.get("score_metric") or ""),
        "alignment_score": best.get("score"),
        "alignment_mean_z": best.get("final_map_meanz") or best.get("mean_z"),
        "alignment_map_cc_score": best.get("map_cc_score"),
        "map_path": str(map_path),
        "gt_model_path": str(gt_model_path),
        "fitted_path": str(best.get("fitted_path") or ""),
        "pipeline_status": str(row.get("status") or ""),
    }


def _score_one(
    *,
    job: dict[str, Any],
    chimerax_path: Path,
    work_root: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    """Run the ChimeraX Q-score for one (GT, identified, map) job and return an output row."""
    out = dict(job)
    started = time.perf_counter()
    try:
        import qscore

        if str(job.get("pipeline_status") or "").lower() != "ok":
            raise RuntimeError(f"pipeline_status={job.get('pipeline_status')}")
        gt_model_path = Path(str(job["gt_model_path"]))
        fitted_path = Path(str(job["fitted_path"]))
        map_path = Path(str(job["map_path"]))
        for label, path in (("gt_model_path", gt_model_path), ("fitted_path", fitted_path), ("map_path", map_path)):
            if not path.exists():
                raise FileNotFoundError(f"{label} missing: {path}")
        gt_q, identified_q, error, elapsed = qscore.compute_qscore_pair(
            gt_model_path=gt_model_path,
            identified_model_path=fitted_path,
            map_path=map_path,
            chimerax_path=chimerax_path,
            work_root=work_root / str(job.get("case_key") or "unknown"),
            timeout_sec=timeout_sec,
        )
        out["elapsed_s"] = elapsed
        if error:
            out["status"] = "error"
            out["error"] = error
            return out
        out["qscore_ground_truth"] = format_float(gt_q)
        out["qscore_identified"] = format_float(identified_q)
        if gt_q is not None and identified_q is not None:
            out["qscore_diff_identified_minus_gt"] = format_float(float(identified_q) - float(gt_q))
        out["status"] = "ok"
        out["error"] = ""
    except Exception as exc:
        out["status"] = "error"
        out["error"] = repr(exc)
        out["elapsed_s"] = float(time.perf_counter() - started)
    return out


def compute_qscores(
    *,
    summary_json_path: Path,
    csv_path: Path,
    output_csv: Path,
    output_json: Path,
    chimerax_path: Path,
    work_root: Path,
    timeout_sec: int,
    workers: int,
) -> list[dict[str, Any]]:
    """Run Q-score for every target in ``summary_json_path`` in parallel, writing CSV/JSON incrementally.

    Returns the result rows in the same order as the pipeline summary, so
    downstream plotting and analysis preserve the input ordering.
    """
    summary_rows = _summary_rows(summary_json_path)
    csv_rows = _csv_rows(csv_path)
    if len(summary_rows) > len(csv_rows):
        raise ValueError(f"Summary has {len(summary_rows)} rows but CSV has only {len(csv_rows)} rows")
    jobs = [_qscore_job(row, csv_rows[idx]) for idx, row in enumerate(summary_rows)]
    results: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}
    try:
        with ThreadPoolExecutor(max_workers=max(1, int(workers)), thread_name_prefix="qscore") as executor:
            future_to_key = {
                executor.submit(
                    _score_one,
                    job=job,
                    chimerax_path=chimerax_path,
                    work_root=work_root,
                    timeout_sec=timeout_sec,
                ): str(job.get("case_key") or idx)
                for idx, job in enumerate(jobs)
            }
            for completed, future in enumerate(as_completed(future_to_key), start=1):
                row = future.result()
                by_key[str(row.get("case_key") or future_to_key[future])] = row
                results = [by_key.get(str(job.get("case_key") or idx), {}) for idx, job in enumerate(jobs)]
                results = [row for row in results if row]
                write_qscore_csv(output_csv, results)
                write_json(output_json, results)
                print(
                    f"[qscore] {completed}/{len(jobs)} {row.get('case_key')} "
                    f"status={row.get('status')} gt={row.get('qscore_ground_truth')} identified={row.get('qscore_identified')}",
                    flush=True,
                )
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    return [by_key[str(job.get("case_key") or idx)] for idx, job in enumerate(jobs)]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse benchmark CLI args, merging the JSON config with CLI overrides."""
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Benchmark JSON config path. Defaults to configs/benchmark_all.json.",
    )
    config_args, _ = config_parser.parse_known_args(argv)
    config_path = resolve_method_path(config_args.config, must_exist=True)

    parser = argparse.ArgumentParser(
        parents=[config_parser],
        description="Run the full struct-fetch benchmark, compute ChimeraX Q-scores, and generate the Q-score plot."
    )
    parser.set_defaults(
        csv=None,
        entries=None,
        run_dir=None,
        max_experimental_tries=None,
        include_alphafold=None,
        run_pipeline=None,
        compute_qscore=None,
        generate_plot=None,
        chimerax=None,
        qscore_workers=None,
        qscore_timeout_sec=None,
    )
    parser.add_argument("--csv", help="Benchmark CSV. Overrides config csv.")
    parser.add_argument("--entries", type=int, help="Rows to process. Use 0 for all rows.")
    parser.add_argument("--run-dir", help="Run directory. Empty config value creates runs/benchmark_all_<timestamp>.")
    parser.add_argument("--max-experimental-tries", type=int)
    parser.add_argument("--include-alphafold", dest="include_alphafold", action="store_true", default=None)
    parser.add_argument("--skip-alphafold", dest="include_alphafold", action="store_false", default=None)
    parser.add_argument("--run-pipeline", dest="run_pipeline", action="store_true", default=None)
    parser.add_argument(
        "--skip-pipeline",
        dest="run_pipeline",
        action="store_false",
        default=None,
        help="Reuse an existing run-dir/data/logs/summary.json.",
    )
    parser.add_argument("--compute-qscore", dest="compute_qscore", action="store_true", default=None)
    parser.add_argument("--skip-qscore", dest="compute_qscore", action="store_false", default=None)
    parser.add_argument("--generate-plot", dest="generate_plot", action="store_true", default=None)
    parser.add_argument("--skip-plot", dest="generate_plot", action="store_false", default=None)
    parser.add_argument("--chimerax", help="ChimeraX executable path, or set CHIMERAX_PATH.")
    parser.add_argument("--qscore-workers", type=int)
    parser.add_argument("--qscore-timeout-sec", type=int)
    parser.add_argument("--randomize-candidate-start", dest="randomize_candidate_start", action="store_true", default=None)
    parser.add_argument("--no-randomize-candidate-start", dest="randomize_candidate_start", action="store_false", default=None)
    parser.add_argument("--random-start-margin-A", dest="random_start_margin_A", type=float)
    parser.add_argument("--keep-only-best-models", dest="keep_only_best_models", action="store_true", default=None)
    parser.add_argument("--keep-all-models", dest="keep_only_best_models", action="store_false", default=None)
    cli_args = parser.parse_args(argv)
    settings = _merged_settings(config_path, cli_args)
    settings["config"] = str(config_path)
    return argparse.Namespace(**settings)


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for the full benchmark workflow (pipeline + Q-score + plot)."""
    args = parse_args(argv)
    import pipeline
    import qscore

    run_dir = resolve_method_path(args.run_dir, must_exist=False) if str(args.run_dir).strip() else _timestamped_run_dir()
    data_dir = run_dir / "data"
    logs_dir = data_dir / "logs"
    qscore_dir = data_dir / "qscore"
    pipeline_config = pipeline.RunConfig(
        run_dir=run_dir,
        data_dir=data_dir,
        source_dir=data_dir / "sources",
        fitted_dir=data_dir / "fitted_models",
        logs_dir=logs_dir,
        entries=int(args.entries),
        max_experimental_tries=args.max_experimental_tries,
        include_alphafold=bool(args.include_alphafold),
        recommended_contour_level=None,
        randomize_candidate_start=bool(args.randomize_candidate_start),
        random_start_margin_A=float(args.random_start_margin_A),
        keep_only_best_models=bool(args.keep_only_best_models),
    )
    csv_path = resolve_method_path(args.csv, must_exist=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    with tee_output(logs_dir / "terminal.log"):
        print(f"[benchmark] terminal log -> {(logs_dir / 'terminal.log').resolve().as_posix()}", flush=True)
        write_json(
            data_dir / "benchmark_config.json",
            {
                "config_file": args.config,
                "resolved_settings": vars(args),
                "csv": csv_path,
                "run_dir": run_dir,
                "pipeline_config": pipeline_config.__dict__,
                "score_metric": pipeline.FINAL_SCORE_METRIC,
                "run_pipeline": bool(args.run_pipeline),
                "compute_qscore": bool(args.compute_qscore),
                "generate_plot": bool(args.generate_plot),
            },
        )

        if args.run_pipeline:
            print(f"[benchmark] running pipeline -> {data_dir}", flush=True)
            pipeline.run_csv(
                csv_path,
                run_config=pipeline_config,
            )

        summary_json_path = logs_dir / "summary.json"
        if not summary_json_path.exists():
            raise FileNotFoundError(f"Pipeline summary not found: {summary_json_path}")

        qscore_csv = qscore_dir / "benchmark_qscores.csv"
        qscore_json = qscore_dir / "benchmark_qscores.json"
        plot_png = qscore_dir / "benchmark_qscores.png"

        if args.compute_qscore:
            print(f"[benchmark] computing qscore -> {qscore_csv}", flush=True)
            chimerax_path = qscore.find_chimerax(args.chimerax or None)
            compute_qscores(
                summary_json_path=summary_json_path,
                csv_path=csv_path,
                output_csv=qscore_csv,
                output_json=qscore_json,
                chimerax_path=chimerax_path,
                work_root=qscore_dir / "_tmp",
                timeout_sec=int(args.qscore_timeout_sec),
                workers=max(1, int(args.qscore_workers)),
            )

        if args.generate_plot:
            if not qscore_csv.exists():
                raise FileNotFoundError(f"Q-score CSV not found: {qscore_csv}")
            from plot_qscore import plot_qscores

            plot_qscores(qscore_csv, plot_png, title="Fetching + Alignment Q-scores")
            print(f"[benchmark] plot -> {plot_png.resolve().as_posix()}", flush=True)

        print(f"[benchmark] run data -> {data_dir.resolve().as_posix()}", flush=True)


if __name__ == "__main__":
    main()
