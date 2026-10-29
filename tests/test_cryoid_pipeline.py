from __future__ import annotations

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from cryoid_pipeline import CryoIDPipeline, PipelineCancelled
from cryoid_pipeline.models import PredictionBundle, QueryBundle, SearchHit
from D_BLAST_searching.protein_search import BlastSearchResult
from C_Unet_model_predicting.prediction_provenance import (
    build_prediction_manifest,
    prediction_manifest_path,
    write_prediction_manifest,
)
from D_BLAST_searching.sequence_search import read_sequence_pool
from cryoid_pipeline.runtime_config import RuntimeConfig, default_repo_root
from G_GUI.local_config import gui_config_view, shared_config_view
from cryoid_pipeline.user_config import load_config, save_config, ensure_config
from G_GUI.session_files import (
    collect_chimerax_project_files,
    collect_project_display_files,
    ensure_project_dirs,
    flatten_chimerax_session_files,
    selected_sequence_ranking,
    sequence_ranking_files,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class CryoIDProjectPipelineTests(unittest.TestCase):
    def _write(self, path: Path, text: str = "x") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def _make_density_case(self, root: Path, name: str = "emd_0001.mrc") -> Path:
        return self._write(root / "inputs" / name)

    def _make_pool(self, root: Path, record_id: str = "top_hit", sequence: str = "MAGSTIVQQ") -> Path:
        return self._write(root / "inputs" / "pool.fasta", f">{record_id}\n{sequence}\n")

    def _fake_unet_runner(self, density_map: Path, output_root: Path, progress) -> PredictionBundle:
        progress("fake UNet inference started")
        prefix = density_map.name[:-4] if density_map.name.lower().endswith(".mrc") else density_map.name
        possibility = output_root
        possibility.mkdir(parents=True, exist_ok=True)
        key_paths = [
            self._write(possibility / f"{prefix}_key_atoms_{idx}_possibility.mrc", f"key{idx}")
            for idx in range(1, 4)
        ]
        raw_key = key_paths[0]
        ss_paths = [
            self._write(possibility / f"{prefix}_secondary_strctures_{idx}_possibility.mrc", f"ss{idx}")
            for idx in range(1, 5)
        ]
        residue_paths = [
            self._write(possibility / f"{prefix}_residue_types_6letter_{idx}_possibility.mrc", f"residue{idx}")
            for idx in range(1, 9)
        ]
        return PredictionBundle(
            layout="checkpoint_inference",
            density_map=density_map,
            raw_prediction_dir=output_root,
            raw_key_atoms_1_path=raw_key,
            secondary_structure_paths=ss_paths,
            residue_type_paths=residue_paths,
            output_files=[*key_paths, *ss_paths, *residue_paths],
        )

    def _fake_query_runner(self, _bundle: PredictionBundle, run_dir: Path, progress) -> QueryBundle:
        progress("fake protein query generation started")
        query = self._write(
            run_dir / "tmp" / "generated_queries.fasta",
            ">trace_1\nGPLKYW\n",
        )
        return QueryBundle(query_fasta=query, source="fake_tracing", output_files=[query])

    def _write_prediction_provenance(
        self,
        density_map: Path,
        output_root: Path,
        *,
        rcl: float = 0.0,
        source: str = "test_fixture",
    ) -> Path:
        required = CryoIDPipeline(project_dir=output_root.parent)._required_prediction_paths(
            density_map, output_root
        )
        prefix = CryoIDPipeline._prediction_prefix(density_map)
        path = prediction_manifest_path(output_root, prefix)
        manifest = build_prediction_manifest(
            density_map=density_map,
            rcl=rcl,
            prediction_paths=required,
            weights_dir=None,
            source=source,
        )
        return write_prediction_manifest(path, manifest)

    def _fake_blast_runner(self, _query: Path, pool: Path, run_dir: Path, progress) -> BlastSearchResult:
        progress("fake protein BLAST started")
        selected = read_sequence_pool(pool)[0]
        hit = SearchHit(
            query_id="fake_query",
            record_id=selected.record_id,
            score=10.0,
            sequence=selected.sequence,
            description=selected.description,
        )
        blast_dir = run_dir / "tmp" / "sequence_search"
        summary = self._write(
            blast_dir / "protein_candidates_summary.txt",
            "Ranking, candidate, length, ave_identity %, composite_E\n"
            f"1\t{selected.record_id}\t{len(selected.sequence)}\t100%\t1e-10\n",
        )
        return BlastSearchResult(
            selected_hit=hit,
            outputs=[summary],
            ranking=[
                {
                    "rank": "1",
                    "candidate": selected.record_id,
                    "length": str(len(selected.sequence)),
                    "average_identity": "100%",
                    "composite_evalue": "1e-10",
                }
            ],
            gap_open=15,
            gap_extend=3,
        )

    def _cancelling_unet_runner(self, density_map: Path, output_root: Path, progress) -> PredictionBundle:
        progress("fake UNet inference started")
        self._write(output_root / "partial_prediction.mrc")
        raise PipelineCancelled("fake unet cancellation")

    def _struct_fetch_runner_factory(self, *, af_score: float = 5.0, pdb_score: float = 4.0):
        def runner(csv_path: Path, run_dir: Path, progress, progress_percent) -> list[Path]:
            progress("fake struct-fetch started")
            progress_percent(90)
            source_af = self._write(run_dir / "data" / "sources" / "case" / "af" / "af_source.pdb", f"af-source-{af_score}")
            source_pdb = self._write(run_dir / "data" / "sources" / "case" / "pdb" / "pdb_source.cif", f"pdb-source-{pdb_score}")
            aligned_af = self._write(run_dir / "data" / "fitted_models" / "af_fit.pdb", f"af-fit-{af_score}")
            aligned_pdb = self._write(run_dir / "data" / "fitted_models" / "pdb_fit.cif", f"pdb-fit-{pdb_score}")
            logs_dir = run_dir / "data" / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)
            summary_csv = logs_dir / "summary.csv"
            with summary_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=[
                        "status",
                        "ok_candidate_count",
                        "score_metric",
                        "best_af_id",
                        "best_af_score",
                        "best_af_source_model",
                        "best_af_fitted_path",
                        "best_pdb_id",
                        "best_pdb_score",
                        "best_pdb_source_model",
                        "best_pdb_fitted_path",
                    ],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "status": "ok",
                        "ok_candidate_count": "2",
                        "score_metric": "rawfield",
                        "best_af_id": "AF_A",
                        "best_af_score": str(af_score),
                        "best_af_source_model": str(source_af),
                        "best_af_fitted_path": str(aligned_af),
                        "best_pdb_id": "PDB_A",
                        "best_pdb_score": str(pdb_score),
                        "best_pdb_source_model": str(source_pdb),
                        "best_pdb_fitted_path": str(aligned_pdb),
                    }
                )
            candidate_scores = logs_dir / "case__candidate_scores.csv"
            with candidate_scores.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["status", "kind", "candidate_id", "score", "score_metric", "source_model", "fitted_path"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "status": "ok",
                        "kind": "AF",
                        "candidate_id": "AF_A",
                        "score": str(af_score),
                        "score_metric": "rawfield",
                        "source_model": str(source_af),
                        "fitted_path": str(aligned_af),
                    }
                )
                writer.writerow(
                    {
                        "status": "ok",
                        "kind": "PDB",
                        "candidate_id": "PDB_A",
                        "score": str(pdb_score),
                        "score_metric": "rawfield",
                        "source_model": str(source_pdb),
                        "fitted_path": str(aligned_pdb),
                    }
                )
            return [summary_csv, candidate_scores, source_af, source_pdb, aligned_af, aligned_pdb]

        return runner

    def test_backend_uses_current_project_default_and_ignores_retired_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.dict(os.environ, {
                "CRYOID_CONFIG": str(root / "config.toml"),
                "CRYOID_OUTPUT_ROOT": str(root / "obsolete-output"),
                "CRYOID_REPO_ROOT": str(root / "obsolete-repo"),
                "CRYOID_TEST_QUERY_FASTA": str(root / "obsolete-query.fasta"),
            }, clear=True):
                config = RuntimeConfig.from_env(unet_device="cpu")
                pipeline = CryoIDPipeline(unet_device="cpu")
            self.assertEqual(config.repo_root, default_repo_root().resolve())
            self.assertEqual(config.project_dir, config.repo_root / "cryoid_results")
            self.assertEqual(pipeline.output_root, config.project_dir / "runs")
            self.assertFalse(hasattr(config, "output_root"))
            self.assertFalse(hasattr(config, "test_query_fasta"))

    def test_project_run_layout_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "MyCryoIDProject"
            density_map = self._make_density_case(root)
            pool = self._make_pool(root)
            result = CryoIDPipeline(
                project_dir=project,
                struct_fetch_runner=self._struct_fetch_runner_factory(),
                unet_runner=self._fake_unet_runner,
                protein_query_runner=self._fake_query_runner,
                protein_blast_runner=self._fake_blast_runner,
            ).run(density_map, pool)

            self.assertTrue(result.run_dir.parent.samefile(project / "runs"))
            self.assertTrue(result.run_dir.name.startswith("run_"))
            self.assertTrue((result.run_dir / "run_config.json").is_file())
            run_config = json.loads(
                (result.run_dir / "run_config.json").read_text(encoding="utf-8")
            )
            self.assertEqual(run_config["structure_selection"]["input"], "1")
            self.assertEqual(run_config["structure_selection"]["selected_count"], 1)
            self.assertTrue((result.run_dir / "pipeline.log").is_file())
            self.assertFalse((result.run_dir / "selected_sequence.fasta").exists())
            self.assertEqual(result.ranking_path, result.run_dir / "sequence_ranking.tsv")
            self.assertTrue(result.ranking_path.is_file())
            self.assertTrue((result.run_dir / "alignment_summary.csv").is_file())
            self.assertTrue(any((result.run_dir / "models" / "alphafold" / "aligned_models").iterdir()))
            self.assertTrue(any((result.run_dir / "models" / "experimental" / "aligned_models").iterdir()))
            self.assertTrue((project / "unetpredictions" / "emd_0001_key_atoms_1_possibility.mrc").is_file())
            self.assertTrue((project / "best_overall" / "best_alphafold_model.pdb").is_file())
            self.assertTrue((project / "best_overall" / "best_experimental_model.cif").is_file())
            self.assertTrue((project / "best_overall" / "best_structure_selected_sequence.fasta").is_file())
            self.assertFalse((project / "best_overall" / "best_overall_summary.csv").exists())
            self.assertFalse((result.run_dir / "tmp").exists())
            self.assertFalse((result.run_dir / "unet_predictions").exists())
            self.assertFalse((result.run_dir / "struct_fetch").exists())
            self.assertFalse((project / "cache").exists())
            self.assertFalse((project / "results").exists())

            jsons = sorted(path.relative_to(result.run_dir).as_posix() for path in result.run_dir.rglob("*.json"))
            logs = sorted(path.relative_to(result.run_dir).as_posix() for path in result.run_dir.rglob("*.log"))
            self.assertEqual(jsons, ["run_config.json"])
            self.assertEqual(logs, ["pipeline.log"])

    def test_real_pipeline_events_keep_whole_run_estimate_available(self) -> None:
        from G_GUI.run_status import RunStatus

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            density = self._make_density_case(root)
            pool = self._make_pool(root)
            now = [0.0]
            timing = RunStatus(clock=lambda: now[0], history_path=root / "timing.json")
            timing.start("cpu")
            seen_stages, seen_progress = [], []

            def status(event):
                now[0] += 0.1
                timing.set_stage(event)
                seen_stages.append(timing.stage)
                self.assertTrue(timing.eta.startswith("~"))

            def progress(value):
                now[0] += 0.1
                timing.update_progress(value)
                seen_progress.append(timing.percent)
                self.assertLess(timing.percent, 100)
                self.assertGreater(timing.remaining_seconds, 0)

            def log(message):
                timing.consume_log(message)

            result = CryoIDPipeline(
                project_dir=root / "project", unet_device="cpu",
                struct_fetch_runner=self._struct_fetch_runner_factory(),
                unet_runner=self._fake_unet_runner,
                protein_query_runner=self._fake_query_runner,
                protein_blast_runner=self._fake_blast_runner,
            ).run(density, pool, status=status, progress=log, progress_percent=progress)
            self.assertTrue(all(stage.status == "ok" for stage in result.stages))
            self.assertEqual(seen_stages, ["Input validation", "U-Net prediction",
                "Backbone tracing / query generation", "BLAST+ sequence search",
                "Structure fetching / fitting", "Finalizing outputs"])
            self.assertEqual(seen_progress, sorted(seen_progress))
            timing.finish("Complete")
            self.assertEqual(timing.percent, 100)
            self.assertEqual(timing.eta, "00:00:00")
            self.assertTrue((root / "timing.json").is_file())

    def test_default_protein_blast_runner_uses_supported_stage_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            density_map = self._make_density_case(root)
            pool = self._make_pool(root)
            calls: list[Path] = []

            def production_blast(
                *,
                query_fasta: Path,
                candidate_pool: Path,
                run_dir: Path,
                progress,
                gap_open: int = 15,
                gap_extend: int = 3,
                blast_bin_dir: Path | None = None,
                cpu_threads: int | None = None,
            ) -> BlastSearchResult:
                del gap_open, gap_extend, blast_bin_dir, cpu_threads
                calls.append(query_fasta)
                return self._fake_blast_runner(
                    query_fasta,
                    candidate_pool,
                    run_dir,
                    progress,
                )

            with patch(
                "cryoid_pipeline.orchestrator.stage_d_identification.run_protein_blast_search",
                production_blast,
            ):
                result = CryoIDPipeline(
                    project_dir=root / "project",
                    struct_fetch_runner=self._struct_fetch_runner_factory(),
                    unet_runner=self._fake_unet_runner,
                    protein_query_runner=self._fake_query_runner,
                ).run(density_map, pool)

            self.assertTrue(result.ok)
            self.assertEqual(len(calls), 1)
            blast_stage = next(
                stage for stage in result.stages if stage.name == "Protein candidate BLAST"
            )
            self.assertEqual(blast_stage.status, "ok")

    def test_keep_tmp_retains_intermediates_and_tested_models(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "MyCryoIDProject"
            density_map = self._make_density_case(root)
            pool = self._make_pool(root)
            base_runner = self._struct_fetch_runner_factory()

            def struct_runner(csv_path: Path, run_dir: Path, progress, progress_percent):
                self._write(
                    run_dir.parent.parent / "sequences_with_structures.tsv",
                    "pdb_id\tchain_id\tsequence\n1ABC\tA\tMAGSTIVQQ\n",
                )
                return base_runner(csv_path, run_dir, progress, progress_percent)

            result = CryoIDPipeline(
                project_dir=project,
                keep_tmp=True,
                struct_fetch_runner=struct_runner,
                unet_runner=self._fake_unet_runner,
                protein_query_runner=self._fake_query_runner,
                protein_blast_runner=self._fake_blast_runner,
            ).run(density_map, pool)

            self.assertTrue(result.ok)
            self.assertTrue((result.run_dir / "tmp").is_dir())
            self.assertTrue((result.run_dir / "tmp" / "sequences_with_structures.tsv").is_file())
            tested = result.run_dir / "tmp" / "structure_fetch" / "work" / "data"
            self.assertTrue((tested / "sources" / "case" / "af" / "af_source.pdb").is_file())
            self.assertTrue((tested / "fitted_models" / "pdb_fit.cif").is_file())
            config = json.loads((result.run_dir / "run_config.json").read_text(encoding="utf-8"))
            self.assertTrue(config["keep_tmp"])

    def test_existing_project_predictions_are_reused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            density_map = self._make_density_case(root)
            pool = self._make_pool(root)
            ensure_project_dirs(project)
            self._fake_unet_runner(density_map, project / "unetpredictions", lambda _message: None)
            self._write_prediction_provenance(density_map, project / "unetpredictions")

            def fail_if_called(_density_map: Path, _output_root: Path, _progress) -> PredictionBundle:
                raise AssertionError("UNet runner should not be called when required maps exist")

            result = CryoIDPipeline(
                project_dir=project,
                struct_fetch_runner=self._struct_fetch_runner_factory(),
                unet_runner=fail_if_called,
                protein_query_runner=self._fake_query_runner,
                protein_blast_runner=self._fake_blast_runner,
            ).run(density_map, pool)

            prediction_stage = next(stage for stage in result.stages if stage.name == "Prediction maps")
            self.assertEqual(prediction_stage.status, "ok")
            self.assertIn("Reused provenance-verified", prediction_stage.message)

    def test_partial_unverified_project_predictions_are_regenerated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            density_map = self._make_density_case(root)
            pool = self._make_pool(root)
            ensure_project_dirs(project)
            existing_key = self._write(
                project / "unetpredictions" / "emd_0001_key_atoms_1_possibility.mrc",
                "original-key",
            )

            result = CryoIDPipeline(
                project_dir=project,
                struct_fetch_runner=self._struct_fetch_runner_factory(),
                unet_runner=self._fake_unet_runner,
                protein_query_runner=self._fake_query_runner,
                protein_blast_runner=self._fake_blast_runner,
            ).run(density_map, pool)

            self.assertTrue(result.ok)
            self.assertEqual(existing_key.read_text(encoding="utf-8"), "key1")
            self.assertTrue((project / "unetpredictions" / "emd_0001_secondary_strctures_1_possibility.mrc").is_file())

    def test_changed_rcl_invalidates_prediction_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            density_map = self._make_density_case(root)
            pool = self._make_pool(root)
            ensure_project_dirs(project)
            self._fake_unet_runner(density_map, project / "unetpredictions", lambda _message: None)
            self._write_prediction_provenance(density_map, project / "unetpredictions", rcl=0.0)
            calls = 0

            def regenerating_runner(map_path: Path, output_root: Path, progress) -> PredictionBundle:
                nonlocal calls
                calls += 1
                return self._fake_unet_runner(map_path, output_root, progress)

            result = CryoIDPipeline(
                project_dir=project,
                unet_rcl=0.25,
                struct_fetch_runner=self._struct_fetch_runner_factory(),
                unet_runner=regenerating_runner,
                protein_query_runner=self._fake_query_runner,
                protein_blast_runner=self._fake_blast_runner,
            ).run(density_map, pool)

            self.assertTrue(result.ok)
            self.assertEqual(calls, 1)
            run_config = json.loads((result.run_dir / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(run_config["rcl"], 0.25)

    def test_archived_predictions_can_be_verified_without_historical_weights_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            density_map = self._make_density_case(root)
            pool = self._make_pool(root)
            weights = root / "configured_weights"
            weights.mkdir()
            ensure_project_dirs(project)
            self._fake_unet_runner(density_map, project / "unetpredictions", lambda _message: None)
            self._write_prediction_provenance(
                density_map,
                project / "unetpredictions",
                rcl=0.25,
                source="archived_paper_predictions",
            )

            def fail_if_called(*_args, **_kwargs):
                raise AssertionError("Checksummed archived predictions should be reused")

            result = CryoIDPipeline(
                project_dir=project,
                unet_weights_dir=weights,
                unet_rcl=0.25,
                struct_fetch_runner=self._struct_fetch_runner_factory(),
                unet_runner=fail_if_called,
                protein_query_runner=self._fake_query_runner,
                protein_blast_runner=self._fake_blast_runner,
            ).run(density_map, pool)

            prediction_stage = next(stage for stage in result.stages if stage.name == "Prediction maps")
            self.assertEqual(prediction_stage.status, "ok")
            self.assertIn("archived weights fingerprint is unavailable", prediction_stage.details["provenance"])

    def test_csv_candidate_pool_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            density_map = self._make_density_case(root)
            pool = self._write(root / "inputs" / "pool.csv", "gene_id,sequence\nhit,ACDE\n")
            result = CryoIDPipeline(project_dir=root / "project").run(density_map, pool)

            self.assertEqual(result.stages[0].status, "blocked_missing_input")
            self.assertIn("must be FASTA", result.stages[0].message)

    def test_best_overall_updates_only_when_new_score_is_higher(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            density_map = self._make_density_case(root)
            first_pool = self._make_pool(root, "first_hit", "AAAA")
            first = CryoIDPipeline(
                project_dir=project,
                sequence_mode="protein",
                struct_fetch_runner=self._struct_fetch_runner_factory(af_score=5.0, pdb_score=4.0),
                unet_runner=self._fake_unet_runner,
                protein_query_runner=self._fake_query_runner,
                protein_blast_runner=self._fake_blast_runner,
            ).run(density_map, first_pool)
            first_ranking_contents = first.ranking_path.read_bytes()
            af_best = project / "best_overall" / "best_alphafold_model.pdb"
            self.assertEqual(af_best.read_text(encoding="utf-8"), "af-fit-5.0")

            second_pool = self._write(root / "inputs" / "pool2.fasta", ">second_hit\nBBBB\n")
            second = CryoIDPipeline(
                project_dir=project,
                sequence_mode="protein",
                struct_fetch_runner=self._struct_fetch_runner_factory(af_score=3.0, pdb_score=8.0),
                unet_runner=self._fake_unet_runner,
                protein_query_runner=self._fake_query_runner,
                protein_blast_runner=self._fake_blast_runner,
            ).run(density_map, second_pool)

            self.assertEqual(af_best.read_text(encoding="utf-8"), "af-fit-5.0")
            self.assertEqual((project / "best_overall" / "best_experimental_model.cif").read_text(encoding="utf-8"), "pdb-fit-8.0")
            selected_text = (project / "best_overall" / "best_structure_selected_sequence.fasta").read_text(encoding="utf-8")
            self.assertIn(">second_hit source=experimental", selected_text)
            self.assertTrue(selected_text.rstrip().endswith("BBBB"))
            self.assertNotEqual(first.run_dir, second.run_dir)
            self.assertNotEqual(first.ranking_path, second.ranking_path)
            self.assertTrue(first.ranking_path.is_file())
            self.assertTrue(second.ranking_path.is_file())
            self.assertEqual(first.ranking_path.read_bytes(), first_ranking_contents)

    def test_stop_after_identification_does_not_start_struct_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            density_map = self._make_density_case(root)
            pool = self._make_pool(root, "identified_hit", "GPLKYW")

            def fail_struct_fetch(*_args, **_kwargs):
                raise AssertionError("Struct-fetch must not run in identification-only mode")

            result = CryoIDPipeline(
                project_dir=project,
                struct_fetch_runner=fail_struct_fetch,
                unet_runner=self._fake_unet_runner,
                protein_query_runner=self._fake_query_runner,
                protein_blast_runner=self._fake_blast_runner,
            ).run(density_map, pool, stop_after_identification=True)

            statuses = {stage.name: stage.status for stage in result.stages}
            self.assertEqual(statuses["Protein query generation"], "ok")
            self.assertEqual(statuses["Protein candidate BLAST"], "ok")
            self.assertEqual(statuses["Pipeline scope"], "ok")
            self.assertNotIn("Struct-fetch", statuses)
            self.assertEqual(result.selected_hit.record_id, "identified_hit")
            self.assertTrue(result.ranking_path.is_file())
            self.assertIn(result.ranking_path, result.output_files)

    def test_protein_fraction_sends_multiple_top_ranked_sequences_to_struct_fetch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            density_map = self._make_density_case(root)
            pool = self._write(
                root / "inputs" / "pool.fasta",
                ">hit1\nAAAA\n>hit2\nBBBB\n>hit3\nCCCC\n>hit4\nDDDD\n",
            )

            def blast_runner(_query: Path, candidate_pool: Path, run_dir: Path, _progress) -> BlastSearchResult:
                records = read_sequence_pool(candidate_pool)
                ranking = [
                    {
                        "rank": str(index),
                        "candidate": record.record_id,
                        "length": str(len(record.sequence)),
                        "average_identity": f"{100 - index}%",
                        "composite_evalue": f"1e-{11 - index}",
                    }
                    for index, record in enumerate(records, 1)
                ]
                selected = records[0]
                summary = self._write(
                    run_dir / "tmp" / "sequence_search" / "protein_candidates_summary.txt",
                    "summary",
                )
                return BlastSearchResult(
                    SearchHit("q", selected.record_id, 10.0, selected.sequence),
                    [summary],
                    ranking,
                    15,
                    3,
                )

            captured_ids: list[str] = []
            base_runner = self._struct_fetch_runner_factory()

            def struct_runner(csv_path: Path, run_dir: Path, progress, progress_percent):
                with csv_path.open("r", encoding="utf-8", newline="") as handle:
                    captured_ids.extend(row["gene_id"] for row in csv.DictReader(handle))
                return base_runner(csv_path, run_dir, progress, progress_percent)

            result = CryoIDPipeline(
                project_dir=root / "project",
                sequence_mode="protein",
                structure_selection="0.5",
                struct_fetch_runner=struct_runner,
                unet_runner=self._fake_unet_runner,
                protein_query_runner=self._fake_query_runner,
                protein_blast_runner=blast_runner,
            ).run(density_map, pool)

            self.assertTrue(result.ok)
            self.assertEqual(captured_ids, ["hit1", "hit2"])
            selected_sequences = result.run_dir / "selected_structure_sequences.fasta"
            self.assertEqual(selected_sequences.read_text(encoding="utf-8").count(">"), 2)
            config = json.loads((result.run_dir / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["structure_selection"]["kind"], "fraction")
            self.assertEqual(config["structure_selection"]["selected_count"], 2)

    def test_cancellation_after_ranking_preserves_run_specific_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            density_map = self._make_density_case(root)
            pool = self._make_pool(root, "ranked_hit", "GPLKYW")

            def cancel_during_structure_fetch(*_args, **_kwargs):
                raise PipelineCancelled("cancel after ranking")

            result = CryoIDPipeline(
                project_dir=root / "project",
                struct_fetch_runner=cancel_during_structure_fetch,
                unet_runner=self._fake_unet_runner,
                protein_query_runner=self._fake_query_runner,
                protein_blast_runner=self._fake_blast_runner,
            ).run(density_map, pool)

            self.assertEqual(next(stage for stage in result.stages if stage.name == "Run").status, "skipped")
            self.assertTrue(result.ranking_path.is_file())
            self.assertIn(result.ranking_path, result.output_files)
            config = json.loads((result.run_dir / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["status"], "cancelled")
            self.assertEqual(config["sequence_ranking"], str(result.ranking_path))

    def test_project_display_and_chimerax_files_are_project_level(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            ensure_project_dirs(project)
            input_map = self._write(root / "emd_0001.map")
            af_model = self._write(project / "best_overall" / "best_alphafold_model.pdb", "af")
            pdb_model = self._write(project / "best_overall" / "best_experimental_model.cif", "pdb")
            map_file = self._write(project / "unetpredictions" / "map.mrc")
            hidden_run_file = self._write(project / "runs" / "run_1" / "models" / "alphafold" / "aligned_models" / "model.pdb")
            newest_ranking = self._write(
                project / "runs" / "run_2" / "sequence_ranking.tsv",
                "rank\tmode\tcandidate_id\n1\tprotein\ttop_hit\n",
            )
            older_ranking = self._write(
                project / "runs" / "run_1" / "sequence_ranking.tsv",
                "rank\tmode\tcandidate_id\n1\tprotein\tolder_hit\n",
            )
            run_pdb = self._write(project / "runs" / "run_2" / "models" / "experimental" / "aligned_models" / "pdb.cif")
            with (project / "runs" / "run_2" / "alignment_summary.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["source_type", "score", "aligned_model", "is_best_for_type"])
                writer.writeheader()
                writer.writerow({"source_type": "experimental", "score": "7", "aligned_model": str(run_pdb), "is_best_for_type": "true"})
            self._write(project / "runs" / "run_2" / "run_config.json", "{}")
            self._write(project / "runs" / "run_1" / "run_config.json", "{}")
            hidden_log = self._write(project / "runs" / "run_1" / "pipeline.log")

            display_files = collect_project_display_files(project)
            self.assertIn(af_model.resolve(), display_files)
            self.assertIn(pdb_model.resolve(), display_files)
            self.assertIn(map_file.resolve(), display_files)
            self.assertNotIn(newest_ranking.resolve(), display_files)
            self.assertNotIn(hidden_run_file.resolve(), display_files)
            self.assertNotIn(hidden_log.resolve(), display_files)
            self.assertEqual(
                sequence_ranking_files(project),
                [newest_ranking.resolve(), older_ranking.resolve()],
            )
            self.assertEqual(
                selected_sequence_ranking(project, None),
                newest_ranking.resolve(),
            )
            self.assertEqual(
                selected_sequence_ranking(project, older_ranking.parent),
                older_ranking.resolve(),
            )
            self.assertEqual(
                selected_sequence_ranking(project, project / "runs" / "missing"),
                newest_ranking.resolve(),
            )

            groups = collect_chimerax_project_files(project, input_map)
            self.assertEqual(flatten_chimerax_session_files(groups), [input_map.resolve(), pdb_model.resolve()])

    def test_abort_cleans_tmp_and_preserves_only_run_log_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            density_map = self._make_density_case(root)
            pool = self._make_pool(root)
            result = CryoIDPipeline(
                project_dir=project,
                struct_fetch_runner=self._struct_fetch_runner_factory(),
                unet_runner=self._cancelling_unet_runner,
            ).run(density_map, pool)

            statuses = {stage.name: stage.status for stage in result.stages}
            self.assertEqual(statuses["Run"], "skipped")
            self.assertTrue((result.run_dir / "run_config.json").is_file())
            self.assertTrue((result.run_dir / "pipeline.log").is_file())
            self.assertFalse((result.run_dir / "tmp").exists())
            self.assertEqual([path.name for path in result.run_dir.rglob("*.json")], ["run_config.json"])
            self.assertEqual([path.name for path in result.run_dir.rglob("*.log")], ["pipeline.log"])

    def test_gui_config_adapter_round_trips_shared_toml(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "config.toml"
            self.assertEqual(gui_config_view(load_config(config_path))["project_dir"], "")
            ensured = gui_config_view(ensure_config(config_path))
            self.assertTrue(config_path.is_file())
            self.assertEqual(ensured["weights_path"], "")
            saved = save_config(shared_config_view(
                {
                    "project_dir": "p",
                    "weights_path": "w",
                    "sequence_ranking_run_dir": "r",
                },
                ), config_path,
            )
            loaded = gui_config_view(load_config(config_path))
            self.assertEqual(saved["project_dir"], str(Path("p").resolve()))
            self.assertEqual(loaded["weights_path"], str(Path("w").resolve()))
            self.assertEqual(loaded["sequence_ranking_run_dir"], str(Path("r").resolve()))
            self.assertEqual(
                set(loaded),
                {
                    "project_dir",
                    "last_run_dir",
                    "sequence_ranking_run_dir",
                    "weights_path",
                    "device",
                    "cpu_threads",
                    "rcl",
                    "protein_min_trace_length",
                    "chimerax_path",
                    "last_density_map",
                    "last_candidate_pool",
                    "sequence_mode",
                    "structure_selection",
                    "keep_tmp",
                },
            )

    def test_e1_d3_paper_fixture_expected_sequence_is_pool_top_hit(self) -> None:
        test_data = REPO_ROOT / "tests" / "fixtures" / "protein_e1d3"
        expected = read_sequence_pool(test_data / "text.fasta")
        pool = {
            record.record_id: record
            for record in read_sequence_pool(test_data / "malaria_pool.fasta")
        }
        self.assertEqual(len(expected), 1)
        expected_record = expected[0]
        self.assertEqual(expected_record.record_id, "PF3D7_1446200.1_len605")
        self.assertIn(expected_record.record_id, pool)
        self.assertEqual(pool[expected_record.record_id].sequence, expected_record.sequence)

        summary_lines = (
            test_data / "expected" / "E1_D3_queries_summary.txt"
        ).read_text(encoding="utf-8").splitlines()
        first_result = summary_lines[1].split("\t")
        self.assertEqual(first_result[0], "1")
        self.assertEqual(first_result[1], expected_record.record_id)


if __name__ == "__main__":
    unittest.main()
