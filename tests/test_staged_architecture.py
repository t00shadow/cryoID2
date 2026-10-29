from __future__ import annotations

import os
import csv
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import pytest

from C_Unet_model_predicting import stage as stage_c
from D_BLAST_searching import stage as stage_d
from D_BLAST_searching import protein_search
from D_BLAST_searching.reduced_alphabet import search_pool
from H_Backbone_tracing import stage as stage_h
from I_Structure_fitting import stage as stage_i
from I_Structure_fitting.rna.rawfield import link_large_artifact
from cryoid_pipeline.artifacts import link_input_file
from cryoid_pipeline import CryoIDPipeline, __version__


def test_lettered_stages_expose_production_contracts():
    assert callable(stage_c.run_unet_inference)
    assert callable(stage_h.run_protein_query_generation)
    assert callable(stage_h.run_rna_query_generation)
    assert callable(stage_d.run_protein_blast_search)
    assert callable(stage_d.rank_rna_candidates)
    assert callable(stage_i.run_rna_structure_pipeline)
    assert callable(stage_i.run_protein_structure_pipeline)


def test_release_version_matches_project_metadata():
    metadata = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    assert __version__ == metadata["project"]["version"] == "1.0.0"


def test_stage_d_imports_in_a_fresh_interpreter_without_orchestrator_cycle():
    completed = subprocess.run(
        [sys.executable, "-c", "from D_BLAST_searching.protein_search import BlastSearchResult"],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_protein_search_requests_a_full_pool_ranking(tmp_path, monkeypatch):
    query = tmp_path / "query.fasta"
    pool = tmp_path / "pool.fasta"
    query.write_text(">q\nAAAA\n", encoding="utf-8")
    pool.write_text(">A\nAAAA\n>B\nBBBB\n>C\nCCCC\n", encoding="utf-8")
    captured = {}

    def fake_search(params):
        captured["num_candidate"] = params.num_candidate
        output = Path(params.outdir) / "protein_candidates_summary.txt"
        output.write_text(
            "Ranking, candidate, length, ave_identity %, composite_E\n"
            "1\tA\t4\t100%\t1e-10\n"
            "2\tB\t4\t90%\t1e-9\n"
            "3\tC\t4\t80%\t1e-8\n",
            encoding="utf-8",
        )
        return "A"

    monkeypatch.setattr(search_pool, "Search", fake_search)
    monkeypatch.setattr(protein_search, "_blast_path", lambda _path: nullcontext())
    result = protein_search.run_protein_blast_search(
        query_fasta=query,
        candidate_pool=pool,
        run_dir=tmp_path / "run",
        progress=lambda _message: None,
    )

    assert captured["num_candidate"] == 3
    assert len(result.ranking) == 3


def test_protein_search_appends_pool_records_without_reportable_blast_hits(tmp_path, monkeypatch):
    query = tmp_path / "query.fasta"
    pool = tmp_path / "pool.fasta"
    query.write_text(">q\nAAAA\n", encoding="utf-8")
    pool.write_text(">A\nAAAA\n>B\nBBBB\n>C\nCCCC\n", encoding="utf-8")

    def fake_search(params):
        output = Path(params.outdir) / "protein_candidates_summary.txt"
        output.write_text(
            "Ranking, candidate, length, ave_identity %, composite_E\n"
            "1\tA\t4\t100%\t1e-10\n",
            encoding="utf-8",
        )
        return "A"

    monkeypatch.setattr(search_pool, "Search", fake_search)
    monkeypatch.setattr(protein_search, "_blast_path", lambda _path: nullcontext())
    result = protein_search.run_protein_blast_search(
        query_fasta=query,
        candidate_pool=pool,
        run_dir=tmp_path / "run",
        progress=lambda _message: None,
    )

    assert [row["candidate"] for row in result.ranking] == ["A", "B", "C"]
    assert [row["rank"] for row in result.ranking] == ["1", "2", "3"]
    assert result.ranking[-1]["composite_evalue"] == "inf"


def test_large_input_helper_links_without_copying(tmp_path):
    source = tmp_path / "external" / "pool.fasta"
    source.parent.mkdir()
    source.write_text(">candidate\nACGU\n", encoding="utf-8")
    destination = tmp_path / "run" / "candidate_pool.fasta"

    link_input_file(source, destination)

    assert destination.read_bytes() == source.read_bytes()
    assert os.path.samefile(source, destination)


def test_rna_fitter_rejects_same_sized_unlinked_prediction(tmp_path):
    source = tmp_path / "source.mrc"
    destination = tmp_path / "alias.mrc"
    source.write_bytes(b"prediction")
    destination.write_bytes(b"unrelated!")

    with pytest.raises(FileExistsError, match="not linked"):
        link_large_artifact(source, destination)


def test_prediction_backup_moves_and_restores_files(tmp_path):
    project = tmp_path / "project"
    pipeline = CryoIDPipeline(project_dir=project)
    density_map = tmp_path / "map.mrc"
    density_map.write_bytes(b"map")
    output_root = project / "unetpredictions"
    output_root.mkdir(parents=True)
    prediction = output_root / "map_key_atoms_1_possibility.mrc"
    prediction.write_bytes(b"prediction")
    tmp_root = project / "runs" / "run" / "tmp" / "unet"

    backups = pipeline._backup_existing_prediction_maps(density_map, output_root, tmp_root)

    assert not prediction.exists()
    assert list(backups.values())[0].read_bytes() == b"prediction"
    pipeline._restore_prediction_backups(backups)
    assert prediction.read_bytes() == b"prediction"
    assert not list(backups.values())[0].exists()


def test_structure_stage_owns_direct_protein_api():
    assert stage_i.run_protein_structure_pipeline.__module__ == "I_Structure_fitting.stage"


def test_protein_structure_stage_processes_every_selected_csv_row(tmp_path, monkeypatch):
    input_csv = tmp_path / "selected.csv"
    input_csv.write_text(
        "gene_id,sequence,density_dir,map_path,raw_prediction_dir,raw_key_atoms_1_path\n"
        "A,AAAA,d,d/map.mrc,p,p/key.mrc\n"
        "B,BBBB,d,d/map.mrc,p,p/key.mrc\n",
        encoding="utf-8",
    )
    captured = {}

    def fake_run_csv(_csv_path, *, run_config):
        captured["entries"] = run_config.entries
        run_config.logs_dir.mkdir(parents=True, exist_ok=True)
        with (run_config.logs_dir / "summary.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["target", "status", "ok_candidate_count"]
            )
            writer.writeheader()
            writer.writerow({"target": "A", "status": "ok", "ok_candidate_count": 1})
            writer.writerow({"target": "B", "status": "error", "ok_candidate_count": 0})
        return []

    monkeypatch.setattr(stage_i, "run_csv", fake_run_csv)
    stage_i.run_protein_structure_pipeline(
        csv_path=input_csv,
        run_dir=tmp_path / "run",
        tmp_dir=tmp_path / "tmp",
        progress=lambda _message: None,
        check_cancel=lambda: None,
    )

    assert captured["entries"] == 0


def test_gui_assets_are_included_in_distribution():
    repo_root = Path(__file__).parents[1]
    with (repo_root / "pyproject.toml").open("rb") as handle:
        package_data = tomllib.load(handle)["tool"]["setuptools"]["package-data"]

    assert "svg_assets/*" in package_data["G_GUI"]
    assert (repo_root / "G_GUI" / "svg_assets" / "upload-sign-svgrepo-com.svg").is_file()
