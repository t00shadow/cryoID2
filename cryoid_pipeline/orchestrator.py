from __future__ import annotations

import csv
import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping

from C_Unet_model_predicting import stage as stage_c_prediction
from D_BLAST_searching import stage as stage_d_identification
from H_Backbone_tracing import stage as stage_h_tracing
from I_Structure_fitting import stage as stage_i_structure

from .models import (
    PipelineResult,
    PredictionBundle,
    QueryBundle,
    RequestedSequenceMode,
    SearchHit,
    SequenceMode,
    StageResult,
)
from .runtime_config import RuntimeConfig
from .compute import cpu_thread_limit
from .weights import resolve_weights, pipeline_percent, DOWNLOAD_START, DOWNLOAD_END
from .structure_selection import (
    StructureSelection,
    resolve_effective_structure_selection,
    select_ranked_rows,
)
from .user_config import validate_protein_min_trace_length, validate_structure_selection


ProgressCallback = Callable[[str], None]
PercentCallback = Callable[[int], None]
CancelCheck = Callable[[], bool]
CancellationCheckpoint = Callable[[], None]
StructFetchRunner = Callable[[Path, Path, ProgressCallback, PercentCallback], list[Path]]
UnetRunner = Callable[[Path, Path, ProgressCallback], PredictionBundle]
ProteinQueryRunner = Callable[[PredictionBundle, Path, ProgressCallback], QueryBundle]
ProteinBlastRunner = Callable[
    [Path, Path, Path, ProgressCallback], "stage_d_identification.BlastSearchResult"
]
RnaQueryRunner = Callable[[PredictionBundle, Path, ProgressCallback], QueryBundle]
RnaRankingRunner = Callable[[Path, Path, Path, ProgressCallback], Any]
RnaStructureRunner = Callable[..., Any]

ALIGNMENT_SUMMARY_FIELDS = [
    "source_type",
    "rank",
    "sequence_rank",
    "sequence_id",
    "sequence",
    "model_id",
    "status",
    "error",
    "score",
    "score_metric",
    "source_model",
    "aligned_model",
    "source_run",
    "retained",
    "is_best_for_type",
]


class PipelineCancelled(RuntimeError):
    pass


class CryoIDPipeline:
    """Mode-aware sequence-identification and structure-fetch backend used by the GUI."""

    def __init__(
        self,
        *,
        project_dir: Path | None = None,
        struct_fetch_runner: StructFetchRunner | None = None,
        unet_runner: UnetRunner | None = None,
        protein_query_runner: ProteinQueryRunner | None = None,
        protein_blast_runner: ProteinBlastRunner | None = None,
        rna_query_runner: RnaQueryRunner | None = None,
        rna_ranking_runner: RnaRankingRunner | None = None,
        rna_structure_runner: RnaStructureRunner | None = None,
        unet_weights_dir: Path | str | None = None,
        unet_device: str | None = None,
        cpu_threads: int | str | None = None,
        unet_rcl: float = 0.0,
        protein_min_trace_length: int = 10,
        chimerax_path: Path | str | None = None,
        sequence_mode: str = "auto",
        structure_selection: str | int | float | None = None,
        keep_tmp: bool = False,
        user_config: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = RuntimeConfig.from_env(
            project_dir=project_dir,
            unet_weights_dir=unet_weights_dir,
            unet_device=unet_device,
            cpu_threads=cpu_threads,
            user_config=user_config,
        )
        self.project_dir = self.config.project_dir
        self.output_root = self.project_dir / "runs"
        self.struct_fetch_runner = struct_fetch_runner
        self.unet_runner = unet_runner
        self.protein_query_runner = protein_query_runner
        self.protein_blast_runner = protein_blast_runner
        self.rna_query_runner = rna_query_runner
        self.rna_ranking_runner = rna_ranking_runner
        self.rna_structure_runner = rna_structure_runner
        self.unet_weights_dir = self.config.unet_weights_dir
        self.unet_device = self.config.unet_device
        self.cpu_threads = self.config.cpu_threads
        self.unet_rcl = stage_c_prediction.validate_rcl(unet_rcl)
        self.protein_min_trace_length = validate_protein_min_trace_length(
            protein_min_trace_length
        )
        self.chimerax_path = str(chimerax_path) if chimerax_path else ""
        self.sequence_mode: RequestedSequenceMode = str(sequence_mode).strip().lower()  # type: ignore[assignment]
        if self.sequence_mode not in {"auto", "protein", "rna"}:
            raise ValueError("Sequence mode must be auto, protein, or rna.")
        self.structure_selection = (
            validate_structure_selection(structure_selection)
            if structure_selection is not None and str(structure_selection).strip()
            else None
        )
        self.keep_tmp = bool(keep_tmp)
        self.blast_bin_dir = stage_d_identification.runtime_blast_bin_dir()

    def run(
        self,
        density_map: str | Path,
        candidate_pool: str | Path,
        *,
        progress: ProgressCallback | None = None,
        progress_percent: PercentCallback | None = None,
        status: Callable[[dict[str, Any]], None] | None = None,
        should_cancel: CancelCheck | None = None,
        command_line: str | None = None,
        stop_after_identification: bool = False,
    ) -> PipelineResult:
        with cpu_thread_limit(self.cpu_threads):
            return self._run(
                density_map, candidate_pool, progress=progress, progress_percent=progress_percent,
                status=status, should_cancel=should_cancel, command_line=command_line,
                stop_after_identification=stop_after_identification,
            )

    def _run(
        self,
        density_map: str | Path,
        candidate_pool: str | Path,
        *,
        progress: ProgressCallback | None = None,
        progress_percent: PercentCallback | None = None,
        status: Callable[[dict[str, Any]], None] | None = None,
        should_cancel: CancelCheck | None = None,
        command_line: str | None = None,
        stop_after_identification: bool = False,
    ) -> PipelineResult:
        progress = progress or (lambda _message: None)
        progress_percent = progress_percent or (lambda _value: None)
        should_cancel = should_cancel or (lambda: False)
        self._ensure_project_dirs()
        run_dir = self._new_run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "tmp").mkdir(parents=True, exist_ok=True)
        log_path = run_dir / "pipeline.log"
        log_path.write_text("", encoding="utf-8")
        run_config_path = run_dir / "run_config.json"
        stages: list[StageResult] = []
        output_files: list[Path] = []
        requested_mode: RequestedSequenceMode = self.sequence_mode
        effective_mode: SequenceMode | None = None
        ranking_path: Path | None = None
        density_path = Path(str(density_map)).expanduser().resolve()
        pool_path = Path(str(candidate_pool)).expanduser().resolve()
        self._write_run_config(
            run_config_path,
            run_dir=run_dir,
            density_map=density_path,
            candidate_pool=pool_path,
            stages=stages,
            output_files=output_files,
            status="running",
            selected_hit=None,
            log_path=log_path,
            requested_mode=requested_mode,
            effective_mode=effective_mode,
            ranking_path=ranking_path,
        )

        def record_log(message: str, *, emit_to_user: bool = True) -> None:
            text = str(message)
            timestamp = datetime.now().isoformat(timespec="seconds")
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"[{timestamp}] {text}\n")
            if emit_to_user:
                progress(text)

        def emit(message: str) -> None:
            record_log(message, emit_to_user=True)

        def set_stage(name: str, device: str = "cpu") -> None:
            record_log(f"Stage: {name}; device: {device}", emit_to_user=True)
            if status is not None:
                status({"name": name, "device": device})

        current_percent = -1
        with_download = False

        def publish_percent(value: float) -> None:
            nonlocal current_percent
            next_percent = max(0, min(100, int(value)))
            if next_percent > current_percent:
                current_percent = next_percent
                record_log(f"Progress: {next_percent}%", emit_to_user=False)
                progress_percent(next_percent)

        def set_percent(value: int) -> None:
            publish_percent(pipeline_percent(value, with_download))

        def weights_status(event: dict) -> None:
            nonlocal with_download
            if event["name"] == "Downloading weights":
                with_download = True
                fraction = event["completed_bytes"] / max(1, event["total_bytes"])
                publish_percent(DOWNLOAD_START + (DOWNLOAD_END - DOWNLOAD_START) * fraction)
            if status is not None:
                status(event)

        def check_cancel() -> None:
            if should_cancel():
                raise PipelineCancelled("Pipeline run was cancelled.")

        try:
            set_stage("Input validation")
            emit("Starting cryoID pipeline.")
            emit(f"CPU threads: {self.cpu_threads if self.cpu_threads is not None else 'Auto (library defaults)'}")
            if command_line:
                emit(f"Command: {command_line}")
            set_percent(0)
            emit(f"Project directory: {self.project_dir}")
            emit(f"Run directory: {run_dir}")
            emit(f"Pipeline log: {log_path}")
            emit(f"BLAST+ bin folder: {self.blast_bin_dir or 'not detected'}")
            validation = self._validate_inputs(density_path, pool_path)
            stages.append(validation)
            emit(f"{validation.name}: {validation.message}")
            if validation.status != "ok":
                set_percent(100)
                emit("Run stopped during input validation.")
                return self._finish(
                    run_dir,
                    stages,
                    output_files,
                    selected_hit=None,
                    log_path=log_path,
                    run_config_path=run_config_path,
                    density_map=density_path,
                    candidate_pool=pool_path,
                    run_status="blocked",
                )
            set_percent(2)
            check_cancel()

            try:
                requested_mode, effective_mode, _records = stage_d_identification.resolve_sequence_mode(
                    pool_path, requested_mode
                )
            except stage_d_identification.SequenceModeError as exc:
                mode_stage = StageResult("Sequence mode", "blocked_missing_input", str(exc))
                stages.append(mode_stage)
                emit(f"{mode_stage.name}: {mode_stage.message}")
                return self._finish(
                    run_dir,
                    stages,
                    output_files,
                    selected_hit=None,
                    log_path=log_path,
                    run_config_path=run_config_path,
                    density_map=density_path,
                    candidate_pool=pool_path,
                    run_status="blocked",
                    requested_mode=requested_mode,
                    effective_mode=None,
                    ranking_path=None,
                )
            mode_message = (
                f"Auto-detected sequence mode: {effective_mode}."
                if requested_mode == "auto"
                else f"Using explicitly selected sequence mode: {effective_mode}."
            )
            stages.append(
                StageResult(
                    "Sequence mode",
                    "ok",
                    mode_message,
                    details={"requested_mode": requested_mode, "effective_mode": effective_mode},
                )
            )
            emit(mode_message)
            effective_rcl = self._effective_unet_rcl(effective_mode)
            emit(f"UNet RCL: {effective_rcl:g}.")
            structure_selection = self._effective_structure_selection(effective_mode)
            if structure_selection.kind == "count":
                emit(f"Structure candidates: top {int(structure_selection.numeric)} ranked sequence(s).")
            else:
                emit(
                    "Structure candidates: top "
                    f"{float(structure_selection.numeric) * 100:g}% of ranked sequences."
                )

            # Resolve before prediction/provenance checks, so both entrypoints
            # use the same files and cleared caches recover automatically.
            if self.unet_runner is None:
                self.unet_weights_dir = resolve_weights(
                    self.unet_weights_dir, log=emit, status=weights_status, check_cancel=check_cancel,
                )
                check_cancel()
            set_percent(3)
            set_stage("U-Net prediction", self.unet_device)
            prediction_stage, prediction_bundle = self._run_unet_predictions(
                density_path,
                run_dir,
                emit,
                set_percent,
                check_cancel,
                effective_mode,
            )
            stages.append(prediction_stage)
            if prediction_bundle is not None:
                output_files.extend(prediction_bundle.output_files)
            emit(f"{prediction_stage.name}: {prediction_stage.message}")
            set_percent(55)
            check_cancel()

            set_stage("Backbone tracing / query generation")
            if effective_mode == "protein":
                query_stage, query_bundle = self._run_protein_query_generation(
                    prediction_bundle=prediction_bundle,
                    run_dir=run_dir,
                    progress=emit,
                )
            else:
                query_stage, query_bundle = self._run_rna_query_generation(
                    prediction_bundle=prediction_bundle,
                    run_dir=run_dir,
                    progress=emit,
                )
            stages.append(query_stage)
            if query_bundle is not None:
                output_files.extend(query_bundle.output_files)
            emit(f"{query_stage.name}: {query_stage.message}")
            set_percent(57)
            check_cancel()

            set_stage("BLAST+ sequence search" if effective_mode == "protein" else "RNA sequence ranking")
            if effective_mode == "protein":
                search_stage, selected_hit, search_outputs, ranking_path = self._run_protein_blast(
                    query_bundle=query_bundle,
                    candidate_pool=pool_path,
                    run_dir=run_dir,
                    progress=emit,
                )
            else:
                search_stage, selected_hit, search_outputs, ranking_path = self._run_rna_ranking(
                    query_bundle=query_bundle,
                    candidate_pool=pool_path,
                    run_dir=run_dir,
                    progress=emit,
                    check_cancel=check_cancel,
                )
            stages.append(search_stage)
            output_files.extend(search_outputs)
            emit(f"{search_stage.name}: {search_stage.message}")
            set_percent(59)
            check_cancel()

            if stop_after_identification:
                stages.append(
                    StageResult(
                        "Pipeline scope",
                        "ok",
                        f"Stopped after {effective_mode} identification as requested; structure fetching was not started.",
                        details={"stop_after_identification": True},
                    )
                )
                result = self._finish(
                    run_dir,
                    stages,
                    output_files,
                    selected_hit=selected_hit,
                    log_path=log_path,
                    run_config_path=run_config_path,
                    density_map=density_path,
                    candidate_pool=pool_path,
                    run_status=self._run_status(stages),
                    requested_mode=requested_mode,
                    effective_mode=effective_mode,
                    ranking_path=ranking_path,
                )
                emit(f"Wrote run config: {run_config_path}")
                set_percent(100)
                return result

            set_stage("Structure fetching / fitting")
            if effective_mode == "protein":
                struct_stage, struct_outputs = self._run_struct_fetch(
                    run_dir=run_dir,
                    density_map=density_path,
                    prediction_bundle=prediction_bundle,
                    ranking_path=ranking_path,
                    structure_selection=structure_selection,
                    progress=emit,
                    progress_percent=set_percent,
                    check_cancel=check_cancel,
                )
            else:
                struct_stage, struct_outputs = self._run_rna_structure_fetch(
                    run_dir=run_dir,
                    density_map=density_path,
                    candidate_pool=pool_path,
                    prediction_bundle=prediction_bundle,
                    ranking_path=ranking_path,
                    structure_selection=structure_selection,
                    progress=emit,
                    progress_percent=set_percent,
                    check_cancel=check_cancel,
                )
            stages.append(struct_stage)
            output_files.extend(struct_outputs)
            emit(f"{struct_stage.name}: {struct_stage.message}")
            set_percent(98)

            set_stage("Finalizing outputs")
            result = self._finish(
                run_dir,
                stages,
                output_files,
                selected_hit=selected_hit,
                log_path=log_path,
                run_config_path=run_config_path,
                density_map=density_path,
                candidate_pool=pool_path,
                run_status=self._run_status(stages),
                requested_mode=requested_mode,
                effective_mode=effective_mode,
                ranking_path=ranking_path,
            )
            emit(f"Wrote run config: {run_config_path}")
            set_percent(100)
            return result
        except PipelineCancelled:
            set_stage("Cancelling / cleanup")
            stages.append(StageResult("Run", "skipped", "Pipeline run cancelled by user."))
            emit("Run cancelled. Cleaning tmp/intermediate outputs.")
            try:
                removed = self._cleanup_tmp_and_intermediates(run_dir)
                output_files = [path for path in output_files if path.exists()]
                stages.append(
                    StageResult(
                        "Cleanup",
                        "ok",
                        f"Removed {len(removed)} tmp/intermediate item(s).",
                        details={"removed": [str(path) for path in removed]},
                    )
                )
            except Exception as cleanup_exc:
                stages.append(StageResult("Cleanup", "error", f"Could not clean unfinished outputs: {cleanup_exc!r}"))
                emit(f"Cleanup failed: {cleanup_exc!r}")
            result = self._finish(
                run_dir,
                stages,
                output_files,
                log_path=log_path,
                run_config_path=run_config_path,
                density_map=density_path,
                candidate_pool=pool_path,
                run_status="cancelled",
                requested_mode=requested_mode,
                effective_mode=effective_mode,
                ranking_path=ranking_path,
            )
            set_percent(100)
            return result
        except Exception as exc:
            stages.append(StageResult("Run", "error", repr(exc)))
            emit(f"Run failed: {exc!r}")
            result = self._finish(
                run_dir,
                stages,
                output_files,
                log_path=log_path,
                run_config_path=run_config_path,
                density_map=density_path,
                candidate_pool=pool_path,
                run_status="error",
                requested_mode=requested_mode,
                effective_mode=effective_mode,
                ranking_path=ranking_path,
            )
            set_percent(100)
            return result

    def _new_run_dir(self) -> Path:
        self.output_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = self.output_root / f"run_{stamp}"
        counter = 1
        while candidate.exists():
            counter += 1
            candidate = self.output_root / f"run_{stamp}_{counter}"
        return candidate

    def _ensure_project_dirs(self) -> None:
        for directory in (
            self.project_dir / "unetpredictions",
            self.project_dir / "runs",
            self.project_dir / "best_overall",
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _effective_unet_rcl(self, mode: SequenceMode) -> float:
        """Return the configured contour literally for every sequence mode."""

        # Retain this resolver as the single provenance/config call site, but
        # do not silently substitute a mode-specific value.
        _ = mode
        return self.unet_rcl

    def _effective_structure_selection(self, mode: SequenceMode) -> StructureSelection:
        return resolve_effective_structure_selection(self.structure_selection, mode)

    @staticmethod
    def _ranking_row_count(ranking_path: Path | None) -> int:
        if ranking_path is None or not ranking_path.is_file():
            return 0
        try:
            with ranking_path.open("r", encoding="utf-8", newline="") as handle:
                return sum(1 for _row in csv.DictReader(handle, delimiter="\t"))
        except OSError:
            return 0

    def _cleanup_tmp_and_intermediates(self, run_dir: Path) -> list[Path]:
        if self.keep_tmp:
            return []
        removed: list[Path] = []
        for path in (
            run_dir / "tmp",
            run_dir / "intermediates",
            run_dir / "results" / "data",
            run_dir / "results" / "tmp",
        ):
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(path)
        for path in (
            run_dir / "struct_fetch_input.csv",
        ):
            if not path.exists():
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
            removed.append(path)
        return removed

    def _cleanup_unfinished_products(self, run_dir: Path, *, preserve: list[Path]) -> list[Path]:
        run_root = run_dir.resolve()
        preserved: set[Path] = set()
        for path in preserve:
            try:
                resolved = Path(path).resolve()
                resolved.relative_to(run_root)
            except (OSError, ValueError):
                continue
            preserved.add(resolved)

        removed: list[Path] = []

        def has_preserved_descendant(path: Path) -> bool:
            resolved = path.resolve()
            for preserved_path in preserved:
                try:
                    preserved_path.relative_to(resolved)
                except ValueError:
                    continue
                return True
            return False

        def remove_children(directory: Path) -> None:
            for child in list(directory.iterdir()):
                resolved_child = child.resolve()
                if resolved_child in preserved:
                    continue
                if child.is_dir() and has_preserved_descendant(child):
                    remove_children(child)
                    try:
                        next(child.iterdir())
                    except StopIteration:
                        child.rmdir()
                        removed.append(child)
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
                removed.append(child)

        remove_children(run_root)
        return removed

    def _validate_inputs(self, density_map: Path, candidate_pool: Path) -> StageResult:
        missing = []
        if not density_map.is_file():
            missing.append(f"density map not found: {density_map}")
        if not candidate_pool.is_file():
            missing.append(f"candidate pool not found: {candidate_pool}")
        if missing:
            return StageResult("Input validation", "blocked_missing_input", "; ".join(missing))
        allowed_density = {".map", ".mrc", ".ccp4"}
        allowed_pool = {".fasta", ".fa", ".faa", ".fna"}
        if density_map.suffix.lower() not in allowed_density:
            return StageResult(
                "Input validation",
                "blocked_missing_input",
                f"Density map must be one of {sorted(allowed_density)}.",
            )
        if candidate_pool.suffix.lower() not in allowed_pool:
            return StageResult(
                "Input validation",
                "blocked_missing_input",
                f"Candidate pool must be FASTA. Got {candidate_pool.suffix}.",
            )
        try:
            records = stage_d_identification.read_sequence_pool(candidate_pool)
        except (OSError, UnicodeError, ValueError) as exc:
            return StageResult("Input validation", "blocked_missing_input", f"Invalid FASTA candidate pool: {exc}")
        if not records:
            return StageResult("Input validation", "blocked_missing_input", "Candidate FASTA contains no sequences.")
        record_ids = [record.record_id for record in records]
        if len(record_ids) != len(set(record_ids)):
            return StageResult(
                "Input validation",
                "blocked_missing_input",
                "Candidate FASTA identifiers must be unique for BLAST database creation.",
            )
        return StageResult(
            "Input validation",
            "ok",
            "Density map and candidate pool were found.",
            outputs={"density_map": str(density_map), "candidate_pool": str(candidate_pool)},
        )

    def _run_unet_predictions(
        self,
        density_map: Path,
        run_dir: Path,
        progress: ProgressCallback,
        progress_percent: PercentCallback,
        check_cancel: CancellationCheckpoint,
        mode: SequenceMode,
    ) -> tuple[StageResult, PredictionBundle | None]:
        effective_rcl = self._effective_unet_rcl(mode)
        output_root = self.project_dir / "unetpredictions"
        tmp_root = run_dir / "tmp" / "unet"
        required_paths = self._required_prediction_paths(density_map, output_root, mode)
        prefix = self._prediction_prefix(density_map)
        provenance_path = self._prediction_manifest_for_mode(output_root, prefix, mode)
        missing_before = {
            path.resolve()
            for path in required_paths
            if not path.is_file() or path.stat().st_size <= 0
        }
        if all(path.is_file() and path.stat().st_size > 0 for path in required_paths):
            try:
                provenance_ok, provenance_reason, _manifest = stage_c_prediction.verify_prediction_manifest(
                    manifest_path=provenance_path,
                    density_map=density_map,
                    rcl=effective_rcl,
                    prediction_paths=required_paths,
                    weights_dir=self.unet_weights_dir,
                )
            except (OSError, KeyError, TypeError, ValueError) as exc:
                provenance_ok = False
                provenance_reason = f"provenance verification failed: {exc}"
            if provenance_ok:
                bundle = self._prediction_bundle(
                    density_map, output_root, mode, [*required_paths, provenance_path]
                )
                progress(
                    f"UNet inference: reusing verified project prediction maps from "
                    f"{output_root}"
                )
                progress_percent(55)
                return (
                    StageResult(
                        "Prediction maps",
                        "ok",
                        "Reused provenance-verified project-level UNet prediction maps.",
                        outputs=self._prediction_stage_outputs(bundle, provenance_path),
                        details={
                            "layout": bundle.layout,
                            "device": "reused",
                            "weights_dir": str(self.unet_weights_dir or ""),
                            "configured_rcl": self.unet_rcl,
                            "rcl": effective_rcl,
                            "provenance": provenance_reason,
                            "tmp_dir": str(tmp_root),
                            "generated_file_count": 0,
                            "effective_mode": mode,
                        },
                    ),
                    bundle,
                )
            progress(f"UNet inference: cached predictions will be regenerated because {provenance_reason}.")

        progress(f"UNet inference: starting checkpoint prediction for {density_map}")
        progress(f"UNet inference: writing prediction maps to {output_root}")
        progress(f"UNet inference: writing intermediate maps to {tmp_root}")
        progress_percent(5)
        prediction_backups = self._backup_existing_prediction_maps(density_map, output_root, tmp_root)
        try:
            check_cancel()
            if self.unet_runner is not None:
                tmp_root.mkdir(parents=True, exist_ok=True)
                bundle = self.unet_runner(density_map, output_root, progress)
                bundle.mode = mode
                check_cancel()
                device = "custom"
                weights_dir = ""
                generated_files = bundle.output_files
                progress_percent(55)
            else:
                result = stage_c_prediction.run_unet_inference(
                    density_map,
                    output_root,
                    weights_dir=self.unet_weights_dir,
                    device=self.unet_device,
                    recl=effective_rcl,
                    tmp_root=tmp_root,
                    progress=progress,
                    progress_percent=progress_percent,
                    should_cancel=check_cancel,
                    mode=mode,
                )
                bundle = result.bundle
                device = result.device
                weights_dir = str(result.weights_dir)
                generated_files = result.generated_files
            source_bundle = bundle
            bundle = self._prediction_bundle(density_map, output_root, mode, list(required_paths))
            bundle.layout = source_bundle.layout
            bundle.density_dir = source_bundle.density_dir
            generated_files = [
                path
                for path in generated_files
                if path.resolve() in missing_before or path.resolve() not in prediction_backups
            ]
            manifest = stage_c_prediction.build_prediction_manifest(
                density_map=density_map,
                rcl=effective_rcl,
                prediction_paths=required_paths,
                weights_dir=self.unet_weights_dir,
                source="custom_runner" if self.unet_runner is not None else "checkpoint_inference",
            )
            stage_c_prediction.write_prediction_manifest(provenance_path, manifest)
            if provenance_path not in bundle.output_files:
                bundle.output_files.append(provenance_path)
            check_cancel()
            progress_percent(55)
            progress(f"UNet inference: completed on {device}; generated {len(generated_files)} file(s)")
            return (
                StageResult(
                    "Prediction maps",
                    "ok",
                    "Generated missing project-level UNet prediction maps from checkpoint weights.",
                    outputs=self._prediction_stage_outputs(bundle, provenance_path),
                    details={
                        "layout": bundle.layout,
                        "device": device,
                        "weights_dir": weights_dir,
                        "configured_rcl": self.unet_rcl,
                        "rcl": effective_rcl,
                        "tmp_dir": str(tmp_root),
                        "generated_file_count": len(generated_files),
                        "effective_mode": mode,
                    },
                ),
                bundle,
            )
        except PipelineCancelled:
            self._restore_prediction_backups(prediction_backups)
            progress("UNet inference: abort requested.")
            raise
        except (FileNotFoundError, KeyError, ImportError, ModuleNotFoundError) as exc:
            self._restore_prediction_backups(prediction_backups)
            return (
                StageResult(
                    "Prediction maps",
                    "blocked_missing_input",
                    f"UNet checkpoint inference could not start: {exc}",
                    outputs={"raw_prediction_dir": str(output_root)},
                ),
                None,
            )
        except Exception as exc:
            self._restore_prediction_backups(prediction_backups)
            return (
                StageResult(
                    "Prediction maps",
                    "error",
                    f"UNet checkpoint inference failed: {exc!r}",
                    outputs={"raw_prediction_dir": str(output_root)},
                ),
                None,
            )

    @staticmethod
    def _prediction_prefix(map_path: Path) -> str:
        name = map_path.name
        return name[:-4] if name.lower().endswith(".mrc") else name

    @staticmethod
    def _prediction_manifest_for_mode(
        output_root: Path,
        prefix: str,
        mode: SequenceMode,
    ) -> Path:
        # All 15 probability maps are generated together and validated as one
        # mode-independent bundle.
        _ = mode
        return stage_c_prediction.prediction_manifest_path(output_root, prefix)

    def _prediction_bundle(
        self,
        density_map: Path,
        output_root: Path,
        mode: SequenceMode,
        output_files: list[Path],
    ) -> PredictionBundle:
        prefix = self._prediction_prefix(density_map)
        possibility = output_root
        key_paths = [
            possibility / f"{prefix}_key_atoms_{idx}_possibility.mrc" for idx in range(1, 4)
        ]
        secondary_paths = [
            possibility / f"{prefix}_secondary_strctures_{idx}_possibility.mrc"
            for idx in range(1, 5)
        ]
        residue_paths = [
            possibility / f"{prefix}_residue_types_6letter_{idx}_possibility.mrc"
            for idx in range(1, 9)
        ]
        if mode == "protein":
            key_channels = {1: key_paths[0]}
            secondary_channels = {index: secondary_paths[index - 1] for index in range(1, 4)}
            residue_channels = {index: residue_paths[index - 1] for index in range(1, 7)}
        else:
            key_channels = {2: key_paths[1], 3: key_paths[2]}
            secondary_channels = {4: secondary_paths[3]}
            residue_channels = {7: residue_paths[6], 8: residue_paths[7]}
        return PredictionBundle(
            layout="checkpoint_inference",
            density_map=density_map,
            raw_prediction_dir=output_root,
            raw_key_atoms_1_path=key_paths[0] if mode == "protein" else None,
            secondary_structure_paths=list(secondary_channels.values()),
            residue_type_paths=list(residue_channels.values()),
            density_dir=None,
            output_files=output_files,
            mode=mode,
            key_atom_paths=list(key_channels.values()),
            key_atom_channels=key_channels,
            secondary_structure_channels=secondary_channels,
            residue_type_channels=residue_channels,
        )

    @staticmethod
    def _prediction_stage_outputs(bundle: PredictionBundle, provenance_path: Path) -> dict[str, str]:
        outputs = {
            "raw_prediction_dir": str(bundle.raw_prediction_dir),
            "provenance_manifest": str(provenance_path),
        }
        for channel, path in bundle.key_atom_channels.items():
            outputs[f"key_atoms_{channel}"] = str(path)
        for channel, path in bundle.secondary_structure_channels.items():
            outputs[f"secondary_strctures_{channel}"] = str(path)
        for channel, path in bundle.residue_type_channels.items():
            outputs[f"residue_types_6letter_{channel}"] = str(path)
        return outputs

    def _required_prediction_paths(
        self,
        density_map: Path,
        output_root: Path,
        mode: SequenceMode = "protein",
    ) -> list[Path]:
        prefix = self._prediction_prefix(density_map)
        possibility = output_root
        _ = mode
        return [
            *[
                possibility / f"{prefix}_key_atoms_{idx}_possibility.mrc"
                for idx in range(1, 4)
            ],
            *[
                possibility / f"{prefix}_secondary_strctures_{idx}_possibility.mrc"
                for idx in range(1, 5)
            ],
            *[
                possibility / f"{prefix}_residue_types_6letter_{idx}_possibility.mrc"
                for idx in range(1, 9)
            ],
        ]

    def _backup_existing_prediction_maps(self, density_map: Path, output_root: Path, tmp_root: Path) -> dict[Path, Path]:
        prefix = self._prediction_prefix(density_map)
        possibility = output_root
        backups: dict[Path, Path] = {}
        if not possibility.is_dir():
            return backups
        backup_dir = tmp_root / "existing_project_predictions"
        candidates = [
            *sorted(possibility.glob(f"{prefix}_*_possibility.mrc")),
            stage_c_prediction.prediction_manifest_path(output_root, prefix),
        ]
        for path in candidates:
            if not path.is_file():
                continue
            backup_dir.mkdir(parents=True, exist_ok=True)
            backup = backup_dir / path.name
            path.replace(backup)
            backups[path.resolve()] = backup.resolve()
        return backups

    def _restore_prediction_backups(self, backups: dict[Path, Path]) -> None:
        for original, backup in backups.items():
            if backup.is_file():
                original.parent.mkdir(parents=True, exist_ok=True)
                if original.exists():
                    original.unlink()
                backup.replace(original)

    @staticmethod
    def _run_status(stages: list[StageResult]) -> str:
        if any(stage.status == "error" for stage in stages):
            return "error"
        if any(str(stage.status).startswith("blocked") for stage in stages):
            return "blocked"
        return "complete"

    def _run_protein_query_generation(
        self,
        *,
        prediction_bundle: PredictionBundle | None,
        run_dir: Path,
        progress: ProgressCallback,
    ) -> tuple[StageResult, QueryBundle | None]:
        if prediction_bundle is None:
            return (
                StageResult(
                    "Protein query generation",
                    "blocked_missing_input",
                    "Protein query generation requires the key-atom, secondary-structure, and residue prediction maps.",
                ),
                None,
            )
        try:
            if self.protein_query_runner is not None:
                bundle = self.protein_query_runner(prediction_bundle, run_dir, progress)
            else:
                bundle = stage_h_tracing.run_protein_query_generation(
                    prediction_bundle,
                    run_dir,
                    progress,
                    min_trace_length=self.protein_min_trace_length,
                )
            if not bundle.query_fasta.is_file() or bundle.query_fasta.stat().st_size <= 0:
                raise FileNotFoundError(f"Query generator did not create {bundle.query_fasta}.")
            return (
                StageResult(
                    "Protein query generation",
                    "ok",
                    f"Generated protein queries from clustered and traced prediction maps: {bundle.query_fasta.name}.",
                    outputs={"query_fasta": str(bundle.query_fasta)},
                    details={
                        "source": bundle.source,
                        "output_file_count": len(bundle.output_files),
                        "minimum_trace_length": self.protein_min_trace_length,
                    },
                ),
                bundle,
            )
        except (FileNotFoundError, ImportError, ModuleNotFoundError) as exc:
            return (
                StageResult(
                    "Protein query generation",
                    "blocked_missing_input",
                    str(exc),
                ),
                None,
            )
        except Exception as exc:
            return (
                StageResult(
                    "Protein query generation",
                    "error",
                    f"Protein query generation failed: {exc!r}",
                ),
                None,
            )

    def _run_protein_blast(
        self,
        *,
        query_bundle: QueryBundle | None,
        candidate_pool: Path,
        run_dir: Path,
        progress: ProgressCallback,
    ) -> tuple[StageResult, SearchHit | None, list[Path], Path | None]:
        if query_bundle is None:
            return (
                StageResult(
                    "Protein candidate BLAST",
                    "blocked_missing_input",
                    "No generated protein query FASTA is available for candidate-pool BLAST.",
                ),
                None,
                [],
                None,
            )
        try:
            if self.protein_blast_runner is not None:
                result = self.protein_blast_runner(
                    query_bundle.query_fasta,
                    candidate_pool,
                    run_dir,
                    progress,
                )
            else:
                result = stage_d_identification.run_protein_blast_search(
                    query_fasta=query_bundle.query_fasta,
                    candidate_pool=candidate_pool,
                    run_dir=run_dir,
                    progress=progress,
                    blast_bin_dir=self.blast_bin_dir,
                    cpu_threads=self.cpu_threads,
                )
            top_row = result.ranking[0] if result.ranking else {}
            ranking_path = stage_d_identification.write_sequence_ranking(
                run_dir,
                mode="protein",
                rows=stage_d_identification.protein_ranking_rows(
                    result.ranking,
                    stage_d_identification.read_sequence_pool(candidate_pool),
                ),
            )
            outputs = [*result.outputs, ranking_path]
            return (
                StageResult(
                    "Protein candidate BLAST",
                    "ok",
                    f"Selected top candidate-pool sequence '{result.selected_hit.record_id}'.",
                    outputs={
                        "summary": str(result.outputs[0]) if result.outputs else "",
                        "sequence_ranking": str(ranking_path),
                    },
                    details={
                        "mode": "reduced_alphabet_blastp",
                        "reverse_queries": True,
                        "gap_open": result.gap_open,
                        "gap_extend": result.gap_extend,
                        "top_result": top_row,
                    },
                ),
                result.selected_hit,
                outputs,
                ranking_path,
            )
        except FileNotFoundError as exc:
            return (
                StageResult("Protein candidate BLAST", "blocked_missing_input", str(exc)),
                None,
                [],
                None,
            )
        except Exception as exc:
            return (
                StageResult(
                    "Protein candidate BLAST",
                    "error",
                    f"Protein candidate BLAST failed: {exc!r}",
                ),
                None,
                [],
                None,
            )

    def _run_rna_query_generation(
        self,
        *,
        prediction_bundle: PredictionBundle | None,
        run_dir: Path,
        progress: ProgressCallback,
    ) -> tuple[StageResult, QueryBundle | None]:
        if prediction_bundle is None:
            return (
                StageResult(
                    "RNA query generation",
                    "blocked_missing_input",
                    "RNA query generation requires phosphate, sugar, scaffold, and A/U prediction maps.",
                ),
                None,
            )
        try:
            if self.rna_query_runner is not None:
                bundle = self.rna_query_runner(prediction_bundle, run_dir, progress)
            else:
                bundle = stage_h_tracing.run_rna_query_generation(
                    prediction_bundle, run_dir, progress
                )
            if not bundle.query_fasta.is_file() or bundle.query_fasta.stat().st_size <= 0:
                raise FileNotFoundError(f"RNA query generator did not create {bundle.query_fasta}.")
            return (
                StageResult(
                    "RNA query generation",
                    "ok",
                    f"Generated RNA queries from phosphate/sugar traces: {bundle.query_fasta.name}.",
                    outputs={"query_fasta": str(bundle.query_fasta)},
                    details={"source": bundle.source, "output_file_count": len(bundle.output_files)},
                ),
                bundle,
            )
        except (FileNotFoundError, ImportError, ModuleNotFoundError) as exc:
            return StageResult("RNA query generation", "blocked_missing_input", str(exc)), None
        except Exception as exc:
            return (
                StageResult("RNA query generation", "error", f"RNA query generation failed: {exc!r}"),
                None,
            )

    def _run_rna_ranking(
        self,
        *,
        query_bundle: QueryBundle | None,
        candidate_pool: Path,
        run_dir: Path,
        progress: ProgressCallback,
        check_cancel: CancellationCheckpoint,
    ) -> tuple[StageResult, SearchHit | None, list[Path], Path | None]:
        if query_bundle is None:
            return (
                StageResult(
                    "RNA candidate ranking",
                    "blocked_missing_input",
                    "No generated RNA query FASTA is available for hard-maximin ranking.",
                ),
                None,
                [],
                None,
            )
        try:
            if self.rna_ranking_runner is not None:
                result = self.rna_ranking_runner(
                    query_bundle.query_fasta, candidate_pool, run_dir, progress
                )
            else:
                result = stage_d_identification.rank_rna_candidates(
                    query_fasta=query_bundle.query_fasta,
                    candidate_pool=candidate_pool,
                    run_dir=run_dir,
                    progress=progress,
                    check_cancel=check_cancel,
                )
            ranking_path = stage_d_identification.write_sequence_ranking(
                run_dir, mode="rna", rows=result.ranking
            )
            outputs = [*result.outputs, ranking_path]
            return (
                StageResult(
                    "RNA candidate ranking",
                    "ok",
                    f"Selected top hard-maximin RNA sequence '{result.selected_hit.record_id}'.",
                    outputs={
                        "sequence_ranking": str(ranking_path),
                    },
                    details={
                        "mode": "balanced_hard_maximin",
                        "ranked_sequence_count": len(result.ranking),
                        "top_result": result.ranking[0] if result.ranking else {},
                    },
                ),
                result.selected_hit,
                outputs,
                ranking_path,
            )
        except PipelineCancelled:
            raise
        except FileNotFoundError as exc:
            return StageResult("RNA candidate ranking", "blocked_missing_input", str(exc)), None, [], None
        except Exception as exc:
            return (
                StageResult("RNA candidate ranking", "error", f"RNA ranking failed: {exc!r}"),
                None,
                [],
                None,
            )

    def _run_rna_structure_fetch(
        self,
        *,
        run_dir: Path,
        density_map: Path,
        candidate_pool: Path,
        prediction_bundle: PredictionBundle | None,
        ranking_path: Path | None,
        structure_selection: StructureSelection,
        progress: ProgressCallback,
        progress_percent: PercentCallback,
        check_cancel: CancellationCheckpoint,
    ) -> tuple[StageResult, list[Path]]:
        if prediction_bundle is None or ranking_path is None or not ranking_path.is_file():
            return (
                StageResult(
                    "RNA structure lookup and fitting",
                    "blocked_missing_input",
                    "RNA structure processing requires completed predictions and sequence ranking.",
                ),
                [],
            )
        output_dir = run_dir / "tmp" / "structure_fetch"
        try:
            check_cancel()
            progress_percent(60)
            if self.rna_structure_runner is not None:
                result = self.rna_structure_runner(
                    project_dir=self.project_dir,
                    run_dir=run_dir,
                    density_map=density_map,
                    candidate_pool=candidate_pool,
                    prediction_bundle=prediction_bundle,
                    ranking_path=ranking_path,
                    structure_selection=structure_selection,
                    progress=progress,
                    progress_percent=progress_percent,
                    check_cancel=check_cancel,
                    blast_bin_dir=self.blast_bin_dir,
                )
            else:
                result = stage_i_structure.run_rna_structure_pipeline(
                    project_dir=self.project_dir,
                    run_dir=run_dir,
                    density_map=density_map,
                    prediction_bundle=prediction_bundle,
                    ranking_path=ranking_path,
                    structure_selection=structure_selection,
                    progress=progress,
                    progress_percent=progress_percent,
                    check_cancel=check_cancel,
                    blast_bin_dir=self.blast_bin_dir,
                    cpu_threads=self.cpu_threads,
                    keep_tmp=self.keep_tmp,
                )
            normalized_outputs = self._normalize_rna_structure_results(
                run_dir=run_dir,
                work_dir=output_dir,
                ranking_path=ranking_path,
            )
            if result.selected_structure_count:
                message = (
                    f"Selected the top {result.selected_locus_count} of {result.ranked_locus_count} "
                    f"ranked RNA sequences; {result.structure_bearing_count} had qualifying public "
                    f"structures and {result.fit_success_count} structure(s) were fitted."
                )
            else:
                message = (
                    f"Selected the top {result.selected_locus_count} of {result.ranked_locus_count} "
                    "ranked RNA sequences, but none had qualifying public structures to fit."
                )
            return (
                StageResult(
                    "RNA structure lookup and fitting",
                    "ok",
                    message,
                    outputs={"models_dir": str(run_dir / "models"), "sequence_ranking": str(ranking_path)},
                    details={
                        "structure_selection": structure_selection.details(
                            result.ranked_locus_count
                        ),
                        "ranked_locus_count": result.ranked_locus_count,
                        "selected_locus_count": result.selected_locus_count,
                        "structure_bearing_count": result.structure_bearing_count,
                        "selected_structure_count": result.selected_structure_count,
                        "fit_success_count": result.fit_success_count,
                        "sequence_ranking_preserved": True,
                    },
                ),
                [
                    *[Path(path).resolve() for path in result.outputs if Path(path).is_file()],
                    *normalized_outputs,
                ],
            )
        except PipelineCancelled:
            raise
        except Exception as exc:
            partial = (
                [path.resolve() for path in output_dir.rglob("*") if path.is_file()]
                if output_dir.is_dir()
                else []
            )
            return (
                StageResult(
                    "RNA structure lookup and fitting",
                    "error",
                    f"RNA structure processing failed after sequence ranking was saved: {exc!r}",
                    outputs={"results_dir": str(output_dir), "sequence_ranking": str(ranking_path)},
                    details={"sequence_ranking_preserved": True},
                ),
                partial,
            )

    def _run_struct_fetch(
        self,
        *,
        run_dir: Path,
        density_map: Path,
        prediction_bundle: PredictionBundle | None,
        ranking_path: Path | None,
        structure_selection: StructureSelection,
        progress: ProgressCallback,
        progress_percent: PercentCallback,
        check_cancel: CancellationCheckpoint,
    ) -> tuple[StageResult, list[Path]]:
        check_cancel()
        if ranking_path is None or not ranking_path.is_file():
            return (
                StageResult(
                    "Struct-fetch",
                    "blocked_missing_input",
                    "No sequence ranking is available for protein struct-fetch.",
                ),
                [],
            )
        if prediction_bundle is None:
            return (
                StageResult(
                    "Struct-fetch",
                    "blocked_missing_input",
                    "No prediction bundle is available for struct-fetch.",
                ),
                [],
            )

        outputs: list[Path] = []
        (
            csv_path,
            selected_sequences_path,
            selected_sequence_files,
            selected_count,
            ranking_count,
        ) = self._write_struct_fetch_csv(
            run_dir=run_dir,
            density_map=density_map,
            prediction_bundle=prediction_bundle,
            ranking_path=ranking_path,
            structure_selection=structure_selection,
        )
        outputs.append(selected_sequences_path)
        outputs.extend(selected_sequence_files.values())

        struct_run_dir = run_dir / "tmp" / "structure_fetch" / "work"
        tmp_dir = run_dir / "tmp" / "structure_fetch" / "runtime"
        try:
            check_cancel()
            progress("Struct-fetch: starting candidate fetch and rawfield alignment.")
            progress_percent(60)
            if self.struct_fetch_runner is not None:
                struct_outputs = self.struct_fetch_runner(csv_path, struct_run_dir, progress, progress_percent)
                check_cancel()
            else:
                struct_outputs = stage_i_structure.run_protein_structure_pipeline(
                    csv_path=csv_path,
                    run_dir=struct_run_dir,
                    tmp_dir=tmp_dir,
                    progress=progress,
                    check_cancel=check_cancel,
                    keep_tmp=self.keep_tmp,
                )
                progress_percent(98)
            outputs.extend(path for path in struct_outputs if path.exists())
            processed_count, fitted_target_count = self._protein_structure_summary_counts(
                struct_run_dir
            )
            normalized_outputs = self._normalize_struct_fetch_results(
                run_dir, struct_run_dir, selected_sequence_files
            )
            outputs.extend(path for path in normalized_outputs if path.exists())
            return (
                StageResult(
                    "Struct-fetch",
                    "ok",
                    f"Processed the top {selected_count} of {ranking_count} ranked protein "
                    f"sequences; {fitted_target_count} sequence(s) produced fitted structures.",
                    outputs={
                        "models_dir": str(run_dir / "models"),
                        "alignment_summary": str(run_dir / "alignment_summary.csv"),
                        "selected_sequences": str(run_dir / "selected_structure_sequences.fasta"),
                    },
                    details={
                        "tmp_dir": str(tmp_dir),
                        "structure_selection": structure_selection.details(ranking_count),
                        "processed_sequence_count": processed_count,
                        "fitted_sequence_count": fitted_target_count,
                    },
                ),
                outputs,
            )
        except PipelineCancelled:
            raise
        except Exception as exc:
            outputs.extend(
                path
                for path in stage_i_structure.collect_protein_structure_outputs(struct_run_dir)
                if path not in outputs
            )
            return (
                StageResult(
                    "Struct-fetch",
                    "error",
                    f"Struct-fetch pipeline failed: {exc!r}",
                    outputs={
                        "models_dir": str(run_dir / "models"),
                        "alignment_summary": str(run_dir / "alignment_summary.csv"),
                    },
                    details={"tmp_dir": str(tmp_dir)},
                ),
                outputs,
            )

    @staticmethod
    def _protein_structure_summary_counts(struct_run_dir: Path) -> tuple[int, int]:
        summary_path = struct_run_dir / "data" / "logs" / "summary.csv"
        if not summary_path.is_file():
            return 0, 0
        try:
            with summary_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
        except OSError:
            return 0, 0
        fitted = sum(
            str(row.get("status") or "").strip().lower() == "ok"
            and int(float(row.get("ok_candidate_count") or 0)) > 0
            for row in rows
        )
        return len(rows), int(fitted)

    def _write_struct_fetch_csv(
        self,
        *,
        run_dir: Path,
        density_map: Path,
        prediction_bundle: PredictionBundle,
        ranking_path: Path,
        structure_selection: StructureSelection,
    ) -> tuple[Path, Path, dict[str, Path], int, int]:
        with ranking_path.open("r", encoding="utf-8", newline="") as handle:
            ranking = list(csv.DictReader(handle, delimiter="\t"))
        selected_rows = select_ranked_rows(ranking, structure_selection)
        if not selected_rows:
            raise ValueError("Protein sequence ranking contains no rows to send to struct-fetch.")
        for row in selected_rows:
            if not str(row.get("candidate_id") or "").strip() or not str(row.get("sequence") or "").strip():
                raise ValueError("Protein sequence ranking contains a selected row without an ID or sequence.")

        csv_path = run_dir / "tmp" / "structure_fetch" / "struct_fetch_input.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        selected_sequences_path = run_dir / "tmp" / "structure_fetch" / "selected_for_fetch.fasta"
        per_sequence_dir = run_dir / "tmp" / "structure_fetch" / "candidate_sequences"
        per_sequence_dir.mkdir(parents=True, exist_ok=True)
        selected_sequence_files: dict[str, Path] = {}
        with selected_sequences_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in selected_rows:
                candidate_id = str(row["candidate_id"]).strip()
                rank = str(row.get("rank") or "")
                handle.write(f">{candidate_id} rank={rank}\n")
                sequence = str(row["sequence"]).strip()
                for offset in range(0, len(sequence), 80):
                    handle.write(sequence[offset : offset + 80] + "\n")
                per_sequence_path = (
                    per_sequence_dir
                    / f"rank_{int(rank):04d}_{self._sanitize_filename(candidate_id)}.fasta"
                )
                with per_sequence_path.open("w", encoding="utf-8", newline="\n") as per_handle:
                    per_handle.write(f">{candidate_id} rank={rank}\n")
                    for offset in range(0, len(sequence), 80):
                        per_handle.write(sequence[offset : offset + 80] + "\n")
                selected_sequence_files[candidate_id] = per_sequence_path.resolve()
        fieldnames = [
            "gene_id",
            "sequence_rank",
            "sequence",
            "density_dir",
            "map_path",
            "raw_prediction_dir",
            "raw_key_atoms_1_path",
        ]
        density_dir = prediction_bundle.density_dir or prediction_bundle.raw_prediction_dir
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in selected_rows:
                writer.writerow(
                    {
                        "gene_id": str(row["candidate_id"]).strip(),
                        "sequence_rank": str(row.get("rank") or ""),
                        "sequence": str(row["sequence"]).strip(),
                        "density_dir": str(density_dir),
                        "map_path": str(density_map),
                        "raw_prediction_dir": str(prediction_bundle.raw_prediction_dir),
                        "raw_key_atoms_1_path": str(prediction_bundle.raw_key_atoms_1_path),
                    }
                )
        return (
            csv_path,
            selected_sequences_path,
            selected_sequence_files,
            len(selected_rows),
            len(ranking),
        )

    @staticmethod
    def _source_type_from_row(row: dict[str, Any]) -> str | None:
        kind = str(row.get("kind") or row.get("source_type") or "").strip().lower()
        label = str(row.get("label") or row.get("model_id") or "").lower()
        if kind in {"af", "alphafold"} or "__af_" in label:
            return "alphafold"
        if kind in {"pdb", "experimental"} or "__pdb_" in label:
            return "experimental"
        return None

    @staticmethod
    def _score_from_row(row: dict[str, Any], *keys: str) -> float | None:
        for key in keys or ("score", "raw_field_score", "optimization_score", "alignment_score"):
            value = row.get(key)
            if value in (None, ""):
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _path_from_row(value: object, *, base: Path) -> Path | None:
        if value in (None, ""):
            return None
        text = str(value).strip()
        if not text:
            return None
        path = Path(text).expanduser()
        candidates = [path] if path.is_absolute() else [base / path, base.parent / path, path]
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()
        return path.resolve() if path.is_absolute() else None

    @staticmethod
    def _sanitize_filename(value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
        return cleaned.strip("._") or "model"

    def _copy_unique(self, source: Path | None, dest_dir: Path, *, preferred_name: str | None = None) -> Path | None:
        if source is None or not source.is_file():
            return None
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_name = preferred_name or source.name
        candidate = dest_dir / dest_name
        if candidate.exists():
            try:
                if source.resolve() == candidate.resolve():
                    return candidate
            except OSError:
                pass
            stem = candidate.stem
            suffix = candidate.suffix
            counter = 2
            while candidate.exists():
                candidate = dest_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        shutil.copy2(source, candidate)
        return candidate.resolve()

    def _copy_overwrite(self, source: Path | None, destination: Path) -> Path | None:
        if source is None or not source.is_file():
            return None
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        return destination.resolve()

    def _alignment_rows_from_candidate_logs(self, struct_run_dir: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        logs_dir = struct_run_dir / "data" / "logs"
        if not logs_dir.is_dir():
            return rows
        for csv_path in sorted(logs_dir.glob("*__candidate_scores.csv")):
            try:
                with csv_path.open("r", encoding="utf-8", newline="") as handle:
                    raw_rows = list(csv.DictReader(handle))
            except OSError:
                continue
            for raw in raw_rows:
                source_type = self._source_type_from_row(raw)
                score = self._score_from_row(raw)
                aligned = self._path_from_row(raw.get("fitted_path"), base=struct_run_dir)
                if source_type is None:
                    continue
                status = str(raw.get("status") or "").strip().lower()
                model_id = str(
                    raw.get("candidate_id")
                    or raw.get("id")
                    or raw.get("label")
                    or (aligned.stem if aligned is not None else "model")
                )
                rows.append(
                    {
                        "source_type": source_type,
                        "sequence_rank": str(raw.get("sequence_rank") or ""),
                        "sequence_id": str(raw.get("sequence_id") or ""),
                        "model_id": model_id,
                        "status": status or ("ok" if score is not None else "error"),
                        "error": str(raw.get("error") or ""),
                        "score": score,
                        "score_metric": str(raw.get("score_metric") or ""),
                        "source_model": self._path_from_row(raw.get("source_model"), base=struct_run_dir),
                        "aligned_model": aligned,
                    }
                )
        return rows

    def _alignment_rows_from_summary_csv(self, struct_run_dir: Path) -> list[dict[str, Any]]:
        summary_csv = struct_run_dir / "data" / "logs" / "summary.csv"
        if not summary_csv.is_file():
            return []
        try:
            with summary_csv.open("r", encoding="utf-8", newline="") as handle:
                raw_rows = list(csv.DictReader(handle))
        except OSError:
            return []

        rows: list[dict[str, Any]] = []
        for raw in raw_rows:
            score_metric = str(raw.get("score_metric") or "")
            for prefix, source_type in (("best_af", "alphafold"), ("best_pdb", "experimental")):
                score = self._score_from_row(raw, f"{prefix}_score")
                aligned = self._path_from_row(raw.get(f"{prefix}_fitted_path"), base=struct_run_dir)
                if score is None or aligned is None:
                    continue
                rows.append(
                    {
                        "source_type": source_type,
                        "sequence_rank": str(raw.get("sequence_rank") or ""),
                        "sequence_id": str(raw.get("target") or ""),
                        "model_id": str(raw.get(f"{prefix}_id") or aligned.stem),
                        "status": "ok",
                        "error": "",
                        "score": score,
                        "score_metric": score_metric,
                        "source_model": self._path_from_row(raw.get(f"{prefix}_source_model"), base=struct_run_dir),
                        "aligned_model": aligned,
                    }
                )
        return rows

    def _collect_alignment_rows(self, struct_run_dir: Path) -> list[dict[str, Any]]:
        rows = self._alignment_rows_from_candidate_logs(struct_run_dir)
        if not rows:
            rows = self._alignment_rows_from_summary_csv(struct_run_dir)
        seen: set[tuple[str, str, str, str]] = set()
        unique: list[dict[str, Any]] = []
        for row in rows:
            aligned = row.get("aligned_model")
            aligned_key = str(aligned).lower() if aligned else ""
            key = (
                str(row.get("source_type") or ""),
                str(row.get("sequence_id") or ""),
                str(row.get("model_id") or ""),
                aligned_key,
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(row)
        return unique

    def _normalize_struct_fetch_results(
        self,
        run_dir: Path,
        struct_run_dir: Path,
        selected_sequence_files: dict[str, Path] | None = None,
    ) -> list[Path]:
        models_dir = run_dir / "models"
        for source_type in ("alphafold", "experimental"):
            (models_dir / source_type / "fetched_models").mkdir(parents=True, exist_ok=True)
            (models_dir / source_type / "aligned_models").mkdir(parents=True, exist_ok=True)

        rows = self._collect_alignment_rows(struct_run_dir)
        sorted_rows = sorted(
            rows,
            key=lambda item: (
                str(item.get("source_type") or ""),
                -float(item["score"]) if item.get("score") is not None else float("inf"),
            ),
        )
        best_by_type: dict[str, dict[str, Any]] = {}
        copied_rows: list[dict[str, str]] = []
        ranks: dict[str, int] = {"alphafold": 0, "experimental": 0}
        outputs: list[Path] = []

        for row in sorted_rows:
            source_type = str(row.get("source_type") or "")
            aligned_path = row.get("aligned_model") if isinstance(row.get("aligned_model"), Path) else None
            if (
                source_type in ranks
                and str(row.get("status") or "").lower() == "ok"
                and row.get("score") is not None
                and aligned_path is not None
                and aligned_path.is_file()
            ):
                current = best_by_type.get(source_type)
                if current is None or float(row["score"]) > float(current["score"]):
                    best_by_type[source_type] = row

        for row in sorted_rows:
            source_type = str(row.get("source_type") or "")
            if source_type not in ranks:
                continue
            ranks[source_type] += 1
            source_path = row.get("source_model") if isinstance(row.get("source_model"), Path) else None
            aligned_path = row.get("aligned_model") if isinstance(row.get("aligned_model"), Path) else None
            retained = best_by_type.get(source_type) is row
            source_copy: Path | None = None
            aligned_copy: Path | None = None
            if retained:
                model_id = self._sanitize_filename(str(row.get("model_id") or "model"))
                source_copy = self._copy_unique(
                    source_path,
                    models_dir / source_type / "fetched_models",
                    preferred_name=f"{model_id}{source_path.suffix}" if source_path is not None else None,
                )
                aligned_copy = self._copy_unique(
                    aligned_path,
                    models_dir / source_type / "aligned_models",
                    preferred_name=f"{model_id}{aligned_path.suffix}" if aligned_path is not None else None,
                )
                outputs.extend(path for path in (source_copy, aligned_copy) if path is not None)
            sequence_id = str(row.get("sequence_id") or "")
            if not sequence_id and selected_sequence_files:
                sequence_id = next(iter(selected_sequence_files))
            sequence_path = (selected_sequence_files or {}).get(sequence_id)
            sequence = self._read_first_fasta_sequence(sequence_path)
            copied = {
                "source_type": source_type,
                "rank": str(ranks[source_type]),
                "sequence_rank": str(row.get("sequence_rank") or ""),
                "sequence_id": sequence_id,
                "sequence": sequence,
                "model_id": str(row.get("model_id") or ""),
                "status": str(row.get("status") or ""),
                "error": str(row.get("error") or ""),
                "score": f"{float(row['score']):.6f}" if row.get("score") is not None else "",
                "score_metric": str(row.get("score_metric") or ""),
                "source_model": str(source_copy or source_path or ""),
                "aligned_model": str(aligned_copy or aligned_path or ""),
                "source_run": str(run_dir),
                "retained": "true" if retained else "false",
                "is_best_for_type": "true" if retained else "false",
            }
            copied_rows.append(copied)

        summary_path = run_dir / "alignment_summary.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ALIGNMENT_SUMMARY_FIELDS)
            writer.writeheader()
            for row in copied_rows:
                writer.writerow(row)
        outputs.append(summary_path)
        selected_path = run_dir / "selected_structure_sequences.fasta"
        with selected_path.open("w", encoding="utf-8", newline="\n") as handle:
            for source_type in ("alphafold", "experimental"):
                best = next(
                    (
                        item for item in copied_rows
                        if item["source_type"] == source_type and item["is_best_for_type"] == "true"
                    ),
                    None,
                )
                if best is None or not best.get("sequence"):
                    continue
                handle.write(
                    f">{best['sequence_id']} source={source_type} sequence_rank={best['sequence_rank']}\n"
                )
                sequence = best["sequence"]
                for offset in range(0, len(sequence), 80):
                    handle.write(sequence[offset : offset + 80] + "\n")
        outputs.append(selected_path)
        outputs.extend(self._update_best_overall(run_dir, copied_rows))
        self._cleanup_tmp_and_intermediates(run_dir)
        return outputs

    def _normalize_rna_structure_results(
        self,
        *,
        run_dir: Path,
        work_dir: Path,
        ranking_path: Path,
    ) -> list[Path]:
        models_dir = run_dir / "models" / "experimental"
        fetched_dir = models_dir / "fetched_models"
        aligned_dir = models_dir / "aligned_models"
        fetched_dir.mkdir(parents=True, exist_ok=True)
        aligned_dir.mkdir(parents=True, exist_ok=True)

        with ranking_path.open("r", encoding="utf-8", newline="") as handle:
            ranking_rows = list(csv.DictReader(handle, delimiter="\t"))
        ranking_by_id = {
            str(row.get("candidate_id") or "").upper(): row for row in ranking_rows
        }
        fit_summary = work_dir / "fit_summary.json"
        try:
            raw_rows = json.loads(fit_summary.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            raw_rows = []
        if not isinstance(raw_rows, list):
            raw_rows = []
        rows = sorted(
            [row for row in raw_rows if isinstance(row, dict)],
            key=lambda row: -float(row["score"]) if row.get("score") is not None else float("inf"),
        )
        best = next(
            (
                row for row in rows
                if str(row.get("status") or "").lower() == "ok"
                and row.get("score") is not None
                and Path(str(row.get("fitted_path") or "")).is_file()
            ),
            None,
        )
        outputs: list[Path] = []
        summary_rows: list[dict[str, str]] = []
        for index, row in enumerate(rows, start=1):
            retained = row is best
            query_id = str(row.get("query") or "")
            ranking_row = ranking_by_id.get(query_id.upper(), {})
            sequence_id = str(ranking_row.get("candidate_id") or query_id)
            source_path = Path(str(row.get("source_model") or ""))
            aligned_path = Path(str(row.get("fitted_path") or ""))
            source_copy: Path | None = None
            aligned_copy: Path | None = None
            if retained:
                model_id = self._sanitize_filename(str(row.get("pdb_chain") or "model"))
                source_copy = self._copy_unique(
                    source_path if source_path.is_file() else None,
                    fetched_dir,
                    preferred_name=f"{model_id}{source_path.suffix}" if source_path.suffix else None,
                )
                aligned_copy = self._copy_unique(
                    aligned_path if aligned_path.is_file() else None,
                    aligned_dir,
                    preferred_name=f"{model_id}{aligned_path.suffix}" if aligned_path.suffix else None,
                )
                outputs.extend(path for path in (source_copy, aligned_copy) if path is not None)
            summary_rows.append(
                {
                    "source_type": "experimental",
                    "rank": str(index),
                    "sequence_rank": str(row.get("original_rank") or ranking_row.get("rank") or ""),
                    "sequence_id": sequence_id,
                    "sequence": str(ranking_row.get("sequence") or ""),
                    "model_id": str(row.get("pdb_chain") or ""),
                    "status": str(row.get("status") or ""),
                    "error": str(row.get("error") or ""),
                    "score": f"{float(row['score']):.6f}" if row.get("score") is not None else "",
                    "score_metric": str(row.get("score_metric") or ""),
                    "source_model": str(source_copy or source_path if source_path.is_file() else ""),
                    "aligned_model": str(aligned_copy or aligned_path if aligned_path.is_file() else ""),
                    "source_run": str(run_dir),
                    "retained": "true" if retained else "false",
                    "is_best_for_type": "true" if retained else "false",
                }
            )

        summary_path = run_dir / "alignment_summary.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ALIGNMENT_SUMMARY_FIELDS)
            writer.writeheader()
            writer.writerows(summary_rows)
        outputs.append(summary_path)

        selected_path = run_dir / "selected_structure_sequences.fasta"
        with selected_path.open("w", encoding="utf-8", newline="\n") as handle:
            if best is not None:
                query_id = str(best.get("query") or "")
                ranking_row = ranking_by_id.get(query_id.upper(), {})
                sequence_id = str(ranking_row.get("candidate_id") or query_id)
                sequence = str(ranking_row.get("sequence") or "")
                if sequence:
                    handle.write(
                        f">{sequence_id} source=experimental sequence_rank="
                        f"{best.get('original_rank') or ranking_row.get('rank') or ''}\n"
                    )
                    for offset in range(0, len(sequence), 80):
                        handle.write(sequence[offset : offset + 80] + "\n")
        outputs.append(selected_path)
        outputs.extend(self._update_best_overall(run_dir, summary_rows))
        self._cleanup_tmp_and_intermediates(run_dir)
        return outputs

    @staticmethod
    def _read_first_fasta_sequence(path: Path | None) -> str:
        if path is None or not path.is_file():
            return ""
        return "".join(
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith(">")
        )

    def _update_best_overall(self, run_dir: Path, alignment_rows: list[dict[str, str]]) -> list[Path]:
        del run_dir, alignment_rows
        best_dir = self.project_dir / "best_overall"
        best_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        candidates: dict[str, list[dict[str, str]]] = {"alphafold": [], "experimental": []}
        for summary_path in sorted((self.project_dir / "runs").glob("run_*/alignment_summary.csv")):
            try:
                with summary_path.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
            except OSError:
                continue
            for row in rows:
                source_type = str(row.get("source_type") or "")
                model_path = Path(row.get("aligned_model") or "")
                if (
                    source_type in candidates
                    and row.get("is_best_for_type") == "true"
                    and row.get("sequence")
                    and self._score_from_row(row, "score") is not None
                    and model_path.is_file()
                ):
                    candidates[source_type].append(row)

        winners: list[dict[str, str]] = []
        for source_type in ("alphafold", "experimental"):
            if not candidates[source_type]:
                continue
            winner = max(candidates[source_type], key=lambda row: float(row["score"]))
            source_model = Path(winner["aligned_model"])
            for old_model in best_dir.glob(f"best_{source_type}_model.*"):
                old_model.unlink()
            destination = best_dir / f"best_{source_type}_model{source_model.suffix}"
            copied = self._copy_overwrite(source_model, destination)
            if copied is not None:
                outputs.append(copied)
                winners.append(winner)

        if winners:
            highest = max(winners, key=lambda row: float(row["score"]))
            selected_path = best_dir / "best_structure_selected_sequence.fasta"
            sequence = highest["sequence"]
            with selected_path.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(
                    f">{highest['sequence_id']} source={highest['source_type']} "
                    f"sequence_rank={highest['sequence_rank']}\n"
                )
                for offset in range(0, len(sequence), 80):
                    handle.write(sequence[offset : offset + 80] + "\n")
            outputs.append(selected_path)
        return outputs

    def _finish(
        self,
        run_dir: Path,
        stages: list[StageResult],
        output_files: list[Path],
        *,
        selected_hit: SearchHit | None = None,
        log_path: Path | None = None,
        run_config_path: Path | None = None,
        density_map: Path | None = None,
        candidate_pool: Path | None = None,
        run_status: str = "complete",
        requested_mode: RequestedSequenceMode | None = None,
        effective_mode: SequenceMode | None = None,
        ranking_path: Path | None = None,
    ) -> PipelineResult:
        self._cleanup_tmp_and_intermediates(run_dir)
        final_outputs = list(output_files)
        if ranking_path is not None and ranking_path.is_file() and ranking_path not in final_outputs:
            final_outputs.append(ranking_path)
        if log_path is not None and log_path.is_file() and log_path not in final_outputs:
            final_outputs.append(log_path)
        if run_config_path is None:
            run_config_path = run_dir / "run_config.json"
        if run_config_path not in final_outputs:
            final_outputs.append(run_config_path)
        final_outputs = [path for path in final_outputs if Path(path).exists() or Path(path) == run_config_path]
        self._write_run_config(
            run_config_path,
            run_dir=run_dir,
            density_map=density_map,
            candidate_pool=candidate_pool,
            stages=stages,
            output_files=final_outputs,
            status=run_status,
            selected_hit=selected_hit,
            log_path=log_path,
            requested_mode=requested_mode or self.sequence_mode,
            effective_mode=effective_mode,
            ranking_path=ranking_path,
        )
        final_outputs = [path for path in final_outputs if Path(path).exists()]
        return PipelineResult(
            run_dir=run_dir,
            stages=stages,
            selected_hit=selected_hit,
            output_files=final_outputs,
            mode=effective_mode,
            ranking_path=ranking_path.resolve() if ranking_path is not None and ranking_path.is_file() else None,
        )

    def _write_run_config(
        self,
        run_config_path: Path,
        *,
        run_dir: Path,
        density_map: Path | None,
        candidate_pool: Path | None,
        stages: list[StageResult],
        output_files: list[Path],
        status: str,
        selected_hit: SearchHit | None,
        log_path: Path | None,
        requested_mode: RequestedSequenceMode | None = None,
        effective_mode: SequenceMode | None = None,
        ranking_path: Path | None = None,
    ) -> None:
        payload = {
            "status": status,
            "project_dir": str(self.project_dir),
            "run_dir": str(run_dir),
            "density_map": str(density_map) if density_map is not None else "",
            "candidate_pool": str(candidate_pool) if candidate_pool is not None else "",
            "weights_path": str(self.unet_weights_dir or ""),
            "unet_device": self.unet_device,
            "configured_rcl": self.unet_rcl,
            "rcl": self._effective_unet_rcl(effective_mode) if effective_mode else self.unet_rcl,
            "protein_min_trace_length": self.protein_min_trace_length,
            "chimerax_path": self.chimerax_path,
            "requested_mode": requested_mode or self.sequence_mode,
            "effective_mode": effective_mode or "",
            "configured_structure_selection": self.structure_selection or "",
            "keep_tmp": self.keep_tmp,
            "cpu_threads": self.cpu_threads if self.cpu_threads is not None else "auto",
            "structure_selection": (
                self._effective_structure_selection(effective_mode).details(
                    self._ranking_row_count(ranking_path)
                )
                if effective_mode
                else {}
            ),
            "blast_bin_dir": str(self.blast_bin_dir or ""),
            "blast_source": "bundled",
            "sequence_ranking": str(ranking_path) if ranking_path is not None else "",
            "log_path": str(log_path) if log_path is not None else "",
            "selected_hit": None
            if selected_hit is None
            else {
                "query_id": selected_hit.query_id,
                "record_id": selected_hit.record_id,
                "score": selected_hit.score,
                "description": selected_hit.description,
            },
            "stages": [
                {
                    "name": stage.name,
                    "status": stage.status,
                    "message": stage.message,
                    "outputs": stage.outputs,
                    "details": stage.details,
                }
                for stage in stages
            ],
            "output_files": [str(path) for path in output_files],
        }
        run_config_path.parent.mkdir(parents=True, exist_ok=True)
        run_config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
