"""Completed checkpoints survive aborts and invalidate only dependent stages."""
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cryoid_pipeline import CryoIDPipeline, PipelineCancelled
from cryoid_pipeline.checkpoints import Checkpoints, atomic_json, project_lock


@pytest.fixture
def case(tmp_path):
    from test_cryoid_pipeline import CryoIDProjectPipelineTests
    helper = CryoIDProjectPipelineTests()
    density = helper._make_density_case(tmp_path)
    pool = helper._make_pool(tmp_path)
    counts = Counter()
    state = {"abort": None}
    structure = helper._struct_fetch_runner_factory()

    def interrupt(stage, run):
        if state["abort"] == stage:
            helper._write(run / "tmp" / "partial_model.pdb", "unfinished")
            raise PipelineCancelled(f"Abort during {stage}")

    def prediction(density, output, progress):
        counts["prediction"] += 1
        return helper._fake_unet_runner(density, output, progress)

    def tracing(bundle, run, progress):
        counts["tracing"] += 1
        interrupt("tracing", run)
        return helper._fake_query_runner(bundle, run, progress)

    def search(query, pool, run, progress):
        counts["search"] += 1
        interrupt("search", run)
        return helper._fake_blast_runner(query, pool, run, progress)

    def fitting(csv, run, progress, percent):
        counts["structure"] += 1
        interrupt("structure", run)
        helper._write(run / "data" / "fitted_models" / "not_retained.pdb", "losing candidate")
        return structure(csv, run, progress, percent)

    runners = dict(unet_runner=prediction, protein_query_runner=tracing,
                   protein_blast_runner=search, struct_fetch_runner=fitting)
    for name, runner in runners.items():
        runner.checkpoint_id = f"fixture-{name}-v1"

    def run(**settings):
        return CryoIDPipeline(project_dir=tmp_path / "project", sequence_mode="protein",
                              **runners, **settings).run(density, pool)

    return SimpleNamespace(run=run, counts=counts, state=state, density=density, pool=pool,
                           project=tmp_path / "project", runners=runners)


def assert_ok(result):
    assert all(stage.status == "ok" for stage in result.stages), [(s.name, s.status, s.message) for s in result.stages]


@pytest.mark.parametrize("abort,expected", [
    ("tracing", {"prediction": 1, "tracing": 2, "search": 1, "structure": 1}),
    ("search", {"prediction": 1, "tracing": 1, "search": 2, "structure": 1}),
    ("structure", {"prediction": 1, "tracing": 1, "search": 1, "structure": 2}),
])
@pytest.mark.parametrize("keep_tmp", [False, True])
def test_abort_resumes_only_after_completed_stages(case, abort, expected, keep_tmp):
    case.state["abort"] = abort
    stopped = case.run(keep_tmp=keep_tmp)
    assert any(s.status == "skipped" for s in stopped.stages)
    assert (stopped.run_dir / "tmp").exists() == keep_tmp
    assert not (stopped.run_dir / "checkpoints" / "complete.json").exists()
    for stage in ("tracing", "search"):
        exists = (stopped.run_dir / "checkpoints" / stage / "manifest.json").is_file()
        assert exists == (abort == "structure" or abort == "search" and stage == "tracing")
    case.state["abort"] = None
    resumed = case.run(keep_tmp=keep_tmp)
    assert_ok(resumed)
    assert case.counts == expected
    assert resumed.ranking_path.is_file()
    assert (resumed.run_dir / "checkpoints/complete.json").is_file()
    assert (resumed.run_dir / "tmp").exists() == keep_tmp
    assert bool(list(resumed.run_dir.rglob("not_retained.pdb"))) == keep_tmp
    assert list((resumed.run_dir / "models").rglob("*.pdb"))


def test_completed_run_is_noop_and_runtime_only_settings_do_not_invalidate(case):
    first = case.run()
    assert_ok(first)
    files = {p: p.read_bytes() for p in first.run_dir.rglob("*") if p.is_file()}
    for settings in ({}, {"keep_tmp": True}, {"cpu_threads": 1}, {"unet_device": "cpu"},
                     {"chimerax_path": "different-viewer"}):
        result = case.run(**settings)
        assert result.run_dir == first.run_dir
        assert case.counts == dict.fromkeys(("prediction", "tracing", "search", "structure"), 1)
        assert not (result.run_dir / "tmp").exists()  # Deleted temporary files are not recreated.
        assert {p: p.read_bytes() for p in first.run_dir.rglob("*") if p.is_file()} == files
    assert len(list((case.project / "runs").iterdir())) == 1


@pytest.mark.parametrize("setting,expected", [
    ({"unet_rcl": .2}, {"prediction": 2, "tracing": 2, "search": 2, "structure": 2}),
    ({"protein_min_trace_length": 20}, {"prediction": 1, "tracing": 2, "search": 2, "structure": 2}),
    ({"structure_selection": "0.5"}, {"prediction": 1, "tracing": 1, "search": 1, "structure": 2}),
])
def test_settings_invalidate_from_earliest_affected_stage(case, setting, expected):
    assert_ok(case.run())
    assert_ok(case.run(**setting))
    assert case.counts == expected


def test_candidate_content_change_only_restarts_search_and_fitting(case):
    assert_ok(case.run())
    case.pool.write_text(">different\nGPLKYW\n")
    result = case.run()
    assert_ok(result)
    assert case.counts == {"prediction": 1, "tracing": 1, "search": 2, "structure": 2}
    assert result.selected_hit.record_id == "different"


def test_map_content_change_at_same_path_restarts_all_stages(case):
    assert_ok(case.run())
    case.density.write_text("different map bytes")
    assert_ok(case.run())
    assert case.counts == dict.fromkeys(("prediction", "tracing", "search", "structure"), 2)


def test_versioned_inference_change_restarts_predictions_and_downstream(case):
    assert_ok(case.run())
    case.runners["unet_runner"].checkpoint_id = "updated-model-implementation"
    assert_ok(case.run())
    assert case.counts == dict.fromkeys(("prediction", "tracing", "search", "structure"), 2)


@pytest.mark.parametrize("damage,expected", [
    ("query", {"prediction": 1, "tracing": 2, "search": 1, "structure": 1}),
    ("ranking", {"prediction": 1, "tracing": 1, "search": 2, "structure": 1}),
    ("model", {"prediction": 1, "tracing": 1, "search": 1, "structure": 2}),
    ("tracing_manifest", {"prediction": 1, "tracing": 2, "search": 1, "structure": 1}),
])
def test_damaged_checkpoints_are_not_reused(case, damage, expected):
    first = case.run()
    assert_ok(first)
    path = {"query": next((first.run_dir / "checkpoints/tracing").rglob("*.fasta")),
            "ranking": first.ranking_path,
            "model": next((first.run_dir / "models").rglob("*.pdb")),
            "tracing_manifest": first.run_dir / "checkpoints/tracing/manifest.json"}[damage]
    path.write_text("damaged data")
    result = case.run()
    assert_ok(result)
    # Recomputed byte-identical upstream results can still authorize downstream
    # checkpoints; changed/corrupt outputs themselves are never reused.
    assert case.counts == expected
    assert all("damaged" not in p.read_text() for p in (case.project / "best_overall").glob("*model.*"))
    if damage != "model":
        with (result.run_dir / "alignment_summary.csv").open() as handle:
            assert str(result.run_dir) in handle.read()


def test_missing_completion_marker_reuses_completed_fitting(case):
    import csv
    first = case.run()
    assert_ok(first)
    (first.run_dir / "checkpoints/complete.json").unlink()
    second = case.run()
    assert_ok(second)
    assert second.run_dir != first.run_dir
    assert case.counts == dict.fromkeys(("prediction", "tracing", "search", "structure"), 1)
    with (second.run_dir / "alignment_summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert all(Path(row["source_run"]) == second.run_dir for row in rows)
    assert all(Path(row["aligned_model"]).is_relative_to(second.run_dir) for row in rows if row["retained"] == "true")


def test_legacy_outputs_without_manifests_are_recomputed_safely(case):
    first = case.run()
    for manifest in (first.run_dir / "checkpoints").rglob("*.json"):
        manifest.unlink()
    assert_ok(case.run())
    assert case.counts == {"prediction": 1, "tracing": 2, "search": 2, "structure": 2}


def test_partial_search_commit_is_never_a_checkpoint(case, monkeypatch):
    from cryoid_pipeline import checkpoints
    original = checkpoints.atomic_json
    def interrupted(path, value):
        if value.get("stage") == "search":
            raise OSError("Simulated interruption before commit")
        original(path, value)
    with monkeypatch.context() as patch:
        patch.setattr(checkpoints, "atomic_json", interrupted)
        first = case.run()
    assert any(s.status == "error" for s in first.stages)
    assert (first.run_dir / "checkpoints/tracing/manifest.json").is_file()
    assert not (first.run_dir / "checkpoints/search/manifest.json").exists()
    assert_ok(case.run())
    assert case.counts == {"prediction": 1, "tracing": 1, "search": 2, "structure": 1}


@pytest.mark.parametrize("failure_status", ["error", "download_error"])
def test_transient_structure_errors_are_retryable(case, failure_status):
    import csv
    original = case.runners["struct_fetch_runner"]
    corrupt = [True]
    def fitting(*args):
        outputs = original(*args)
        if corrupt[0]:
            scores = next(p for p in outputs if p.name.endswith("candidate_scores.csv"))
            with scores.open(newline="") as handle:
                reader = csv.DictReader(handle)
                fields, rows = reader.fieldnames, list(reader)
            rows[0].update(status=failure_status, score="nan", source_model="", fitted_path="")
            with scores.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
        return outputs
    fitting.checkpoint_id = original.checkpoint_id
    case.runners["struct_fetch_runner"] = fitting
    first = case.run()
    assert not (first.run_dir / "checkpoints/structure.json").exists()
    assert not (first.run_dir / "checkpoints/complete.json").exists()
    corrupt[0] = False
    assert_ok(case.run())
    assert case.counts == {"prediction": 1, "tracing": 1, "search": 1, "structure": 2}


@pytest.mark.parametrize("malformed", [[], None, {"schema": "cryoid-prediction-provenance", "schema_version": []}])
def test_malformed_prediction_provenance_regenerates_identical_results(case, malformed):
    first = case.run()
    assert_ok(first)
    original_ranking = first.ranking_path.read_bytes()
    manifest = next((case.project / "unetpredictions").glob("*_prediction_provenance.json"))
    manifest.write_text(json.dumps(malformed))
    restored = case.run()
    assert_ok(restored)
    assert restored.selected_hit == first.selected_hit
    assert restored.ranking_path.read_bytes() == original_ranking
    assert case.counts == {"prediction": 2, "tracing": 1, "search": 1, "structure": 1}


@pytest.mark.parametrize("filename", ["scan[1].mrc", "scan.mrc", "scan[1].map"])
@pytest.mark.parametrize("restore", [True, False])
def test_prediction_backups_only_touch_the_exact_map(tmp_path, filename, restore):
    pipeline = CryoIDPipeline.__new__(CryoIDPipeline)
    output = tmp_path / "predictions"
    output.mkdir()
    density = Path(filename)
    expected = pipeline._required_prediction_paths(density, output)
    for path in expected:
        path.write_bytes(b"original predictions")
    prefix = pipeline._prediction_prefix(density)
    unrelated = [output / f"scan1_key_atoms_1_possibility.mrc",
                 output / f"{prefix}_other_key_atoms_1_possibility.mrc"]
    for path in unrelated:
        path.write_bytes(b"other map")
    before = {p: p.read_bytes() for p in unrelated}
    backups = pipeline._backup_existing_prediction_maps(density, output, tmp_path / "tmp")
    assert set(backups) == set(expected)
    assert all(not p.exists() for p in expected)
    assert {p: p.read_bytes() for p in unrelated} == before
    if restore:
        pipeline._restore_prediction_backups(backups)
        assert all(p.read_bytes() == b"original predictions" for p in expected)
    else:
        # Successful inference replaces only this map's outputs; old backups
        # remain in tmp for normal finalization to discard.
        for path in expected:
            path.write_bytes(b"new predictions")
        assert all(p.read_bytes() == b"original predictions" for p in backups.values())
    assert {p: p.read_bytes() for p in unrelated} == before


def test_atomic_commit_keeps_old_marker_on_interruption(monkeypatch, tmp_path):
    marker = tmp_path / "manifest.json"
    atomic_json(marker, {"completed": "old"})
    monkeypatch.setattr("cryoid_pipeline.checkpoints.os.replace", lambda *args: (_ for _ in ()).throw(OSError("interrupted")))
    with pytest.raises(OSError):
        atomic_json(marker, {"completed": "unfinished new"})
    assert json.loads(marker.read_text()) == {"completed": "old"}
    assert list(tmp_path.iterdir()) == [marker]


def test_project_lock_is_released_after_exception(tmp_path):
    with project_lock(tmp_path):
        with pytest.raises(OSError, match="Another CryoID run"):
            with project_lock(tmp_path):
                pass
    with pytest.raises(RuntimeError):
        with project_lock(tmp_path):
            raise RuntimeError("killed work")
    with project_lock(tmp_path):
        pass


def test_real_blast_outputs_survive_abort_and_are_reused(case, monkeypatch):
    from cryoid_pipeline.models import QueryBundle
    from D_BLAST_searching import stage
    case.pool.write_text(">matching\nGPLKYWGPLKYWGPLKYW\n>other\nAAAAAAAAAAAAAAAAAA\n")
    def tracing(bundle, run, progress):
        case.counts["tracing"] += 1
        path = run / "tmp/tracing/generated_queries.fasta"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(">q\nGPLKYWGPLKYWGPLKYW\n")
        return QueryBundle(path, "fixture", [path])
    tracing.checkpoint_id = "real-blast-traces-v1"
    case.runners["protein_query_runner"] = tracing
    case.runners["protein_blast_runner"] = None
    original = stage.run_protein_blast_search
    def search(**kwargs):
        case.counts["search"] += 1
        return original(**kwargs)
    monkeypatch.setattr(stage, "run_protein_blast_search", search)
    case.state["abort"] = "structure"
    first = case.run()
    assert any(s.status == "skipped" for s in first.stages), [(s.name, s.message) for s in first.stages]
    ranking = first.ranking_path.read_bytes()
    assert list((first.run_dir / "checkpoints/search").glob("*alignment.txt"))
    case.state["abort"] = None
    resumed = case.run()
    assert_ok(resumed)
    assert resumed.ranking_path.read_bytes() == ranking
    assert case.counts["search"] == 1
    assert case.run().run_dir == resumed.run_dir


def test_production_inference_and_rna_ranking_checkpoint_chain(tmp_path, monkeypatch):
    """Use production inference/provenance with a tiny stand-in network, real
    RNA ranking, and deterministic fitting; no network or multi-GB weights."""
    from test_unet_inference import UnetInferenceTests, FakeUNET
    from C_Unet_model_predicting import stage as prediction
    from D_BLAST_searching import stage as search
    from cryoid_pipeline.models import QueryBundle
    from I_Structure_fitting.rna.structures import RnaStructureResult
    helper = UnetInferenceTests()
    density, weights, pool = tmp_path / "map.mrc", tmp_path / "weights", tmp_path / "pool.fasta"
    helper._write_map(density)
    helper._write_weights(weights, full_rna_channels=True)
    pool.write_text(">rna_a\nAUGCAUGCAUGC\n>rna_b\nAAAAAAAAAAAA\n")
    counts, abort = Counter(), [True]
    inference, ranking = prediction.run_unet_inference, search.rank_rna_candidates
    def predict(*args, **kwargs):
        counts["prediction"] += 1
        return inference(*args, **kwargs)
    def rank(**kwargs):
        counts["search"] += 1
        return ranking(**kwargs)
    monkeypatch.setattr("C_Unet_model_predicting.unet_inference.UNET", FakeUNET)
    monkeypatch.setattr(prediction, "run_unet_inference", predict)
    monkeypatch.setattr(search, "rank_rna_candidates", rank)

    def tracing(bundle, run, progress):
        counts["tracing"] += 1
        path = run / "tmp/tracing/generated_queries.fasta"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(">q\nAUGCAUGCAUGC\n")
        return QueryBundle(path, "fixture", [path])

    def fitting(**kwargs):
        counts["structure"] += 1
        if abort[0]:
            raise PipelineCancelled("Abort RNA fitting")
        work = kwargs["run_dir"] / "tmp/structure_fetch"
        work.mkdir(parents=True, exist_ok=True)
        source, model = work / "source.cif", work / "fitted.cif"
        source.write_text("data_source\n")
        model.write_text("data_fitted\n")
        import csv
        with kwargs["ranking_path"].open(newline="") as handle:
            top = next(csv.DictReader(handle, delimiter="\t"))
        summary = work / "fit_summary.json"
        summary.write_text(json.dumps([dict(status="ok", score=1.0, query=top["candidate_id"],
                                           source_model=str(source), fitted_path=str(model), pdb_chain="test_A")]))
        return RnaStructureResult([summary, source, model], 2, 1, 1, 1, 1, {})

    tracing.checkpoint_id, fitting.checkpoint_id = "rna-traces-v1", "rna-fits-v1"
    def run(**settings):
        return CryoIDPipeline(project_dir=tmp_path / "project", unet_weights_dir=weights, unet_device="cpu",
                              rna_query_runner=tracing, rna_structure_runner=fitting,
                              **settings).run(density, pool)
    first = run()
    assert any(s.status == "skipped" for s in first.stages)
    assert (first.run_dir / "checkpoints/search/ranking_summary.json").is_file()
    abort[0] = False
    resumed = run()
    assert_ok(resumed)
    assert counts == {"prediction": 1, "tracing": 1, "search": 1, "structure": 2}
    assert run(protein_min_trace_length=99).run_dir == resumed.run_dir  # Protein-only setting is irrelevant to RNA.
    metadata = weights / "class_weight_for_training.txt"
    data = json.loads(metadata.read_text())
    data["key_atoms"][0] = .25
    metadata.write_text(json.dumps(data))
    assert_ok(run())
    assert counts == {"prediction": 2, "tracing": 2, "search": 2, "structure": 3}
