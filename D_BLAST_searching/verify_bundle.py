"""Compare scientific outputs from reference and bundled BLAST, using local maps.

Run --prepare on the machine with the existing prediction maps, then run
--reference-bin PATH on each platform. All generated files stay under this stage.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
sys.path.insert(0, str(REPO))
os.environ.setdefault("MPLBACKEND", "Agg")
WORK = ROOT / "validation" / "work"

from D_BLAST_searching.blast_tools import bundled_platform_key, bundled_blast_bin_dir


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def prepare():
    from H_Backbone_tracing.protein.pipeline import run_protein_query_generation
    from cryoid_pipeline.models import PredictionBundle
    profiles = [
        ("protein_e1d3", "test_data/protein_e1d3/E1_D3_map.mrc", "test_data/protein_e1d3/malaria_pool.fasta",
         "experiments/10_25_protein_final/cryoid_results/unetpredictions"),
        ("protein_4zx8", "experiments/alpha_experiment/4ZX8_chain_A_molmap_2p7A.mrc", "experiments/alpha_experiment/malaria_pool.fasta",
         "experiments/alpha_experiment/10_22_generated_from_chain_A/cryoid_results/unetpredictions"),
        ("protein_alphafold", "experiments/alpha_experiment/A0A024V0B1_v6_molmap_2p7A.mrc", "experiments/alpha_experiment/malaria_pool.fasta",
         "experiments/alpha_experiment/10_22_generated_from_AF/cryoid_results/unetpredictions"),
    ]
    manifest = []
    for name, map_name, pool, predictions in profiles:
        density = REPO / map_name
        prefix = density.name[:-4] if density.suffix.lower() == ".mrc" else density.name
        directory = REPO / predictions
        key = directory / f"{prefix}_key_atoms_1_possibility.mrc"
        secondary = [directory / f"{prefix}_secondary_strctures_{i}_possibility.mrc" for i in range(1, 4)]
        residues = [directory / f"{prefix}_residue_types_6letter_{i}_possibility.mrc" for i in range(1, 7)]
        bundle = PredictionBundle("existing_predictions", density, directory, key, secondary, residues)
        print(f"Generating queries from existing maps: {name}", flush=True)
        generated = run_protein_query_generation(bundle, WORK / "queries" / name, lambda message: print(message, flush=True))
        query = WORK / "inputs" / f"{name}.fasta"
        query.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generated.query_fasta, query)
        manifest.append({"name": name, "mode": "protein", "query": query.relative_to(REPO).as_posix(), "pool": pool,
                         "query_sha256": sha(query), "source": "fresh production backbone tracing from existing U-Net maps",
                         "prediction_maps": [{"path": p.relative_to(REPO).as_posix(), "sha256": sha(p)} for p in [key, *secondary, *residues]]})
    # This archived RNA case predates the current five-channel tracing inputs.
    # Keep its frozen, map-derived production queries as a ranking control.
    query = REPO / "experiments/rna_blastn_20261007/inputs/current_generated_queries.fasta"
    manifest.append({"name": "rna_emd47655", "mode": "rna", "query": query.relative_to(REPO).as_posix(),
                     "pool": "experiments/rna_blastn_20261007/inputs/reduced_noncds_pool.fasta",
                     "query_sha256": sha(query), "source": "frozen production queries derived from archived predicted maps"})
    for case in manifest:
        case["candidate_pool_sha256"] = sha(REPO / case["pool"])
        case["query_count"] = sum(line.startswith(">") for line in (REPO / case["query"]).read_text().splitlines())
    (ROOT / "validation" / "inputs.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def ranking(case, directory, blast, *, check_existing=False, cpu_threads=None):
    if check_existing:
        path = directory / "sequence_ranking.tsv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        return {"ranking_sha256": sha(path), "ranked_fasta_sha256": sha(directory / "ranked_candidates.fasta"),
                "rows": len(rows), "top_candidate": rows[0]["candidate_id"],
                "seconds": None, "checked_existing_outputs": True}
    from D_BLAST_searching.protein_search import run_protein_blast_search
    from D_BLAST_searching.rna_ranking import rank_rna_candidates
    from D_BLAST_searching.sequence_ranking import protein_ranking_rows, write_sequence_ranking
    from D_BLAST_searching.sequence_search import read_sequence_pool
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with (directory / "console.log").open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
        if case["mode"] == "protein":
            result = run_protein_blast_search(query_fasta=REPO / case["query"], candidate_pool=REPO / case["pool"],
                                             run_dir=directory, progress=lambda message: None, blast_bin_dir=blast,
                                             cpu_threads=cpu_threads)
            rows = protein_ranking_rows(result.ranking, read_sequence_pool(REPO / case["pool"]))
        else:
            result = rank_rna_candidates(query_fasta=REPO / case["query"], candidate_pool=REPO / case["pool"],
                                         run_dir=directory, progress=lambda message: None)
            rows = result.ranking
        path = write_sequence_ranking(directory, mode=case["mode"], rows=rows)
    fasta = directory / "ranked_candidates.fasta"
    fasta.write_text("".join(f">{row['candidate_id']}\n{row['sequence']}\n" for row in rows), encoding="utf-8", newline="\n")
    return {"ranking_sha256": sha(path), "ranked_fasta_sha256": sha(fasta), "rows": len(rows),
            "top_candidate": str(rows[0]["candidate_id"]), "seconds": time.perf_counter() - started}


def rna_blast(directory, blast, *, check_existing=False, cpu_threads=None):
    from I_Structure_fitting.rna.structures import BLAST_FIELDS
    suffix = ".exe" if sys.platform == "win32" else ""
    database = REPO / "experiments/rna_blastn_20261007/database/pdb_rna_chains.fasta"
    query = REPO / "experiments/rna_blastn_20261007/run_current_queries/tmp/structure_fetch/ranked_loci.fasta"
    if not query.exists():
        query = REPO / "experiments/rna_blastn_20261007/inputs/reduced_noncds_pool.fasta"
    directory.mkdir(parents=True, exist_ok=True)
    prefix = directory / "rna"
    started = time.perf_counter()
    output = directory / "hits.tsv"
    if not check_existing:
        subprocess.run([str(blast / ("makeblastdb" + suffix)), "-in", str(database), "-dbtype", "nucl", "-parse_seqids", "-out", str(prefix)],
                       check=True, capture_output=True, text=True)
        command = [str(blast / ("blastn" + suffix)), "-task", "blastn", "-query", str(query), "-db", str(prefix),
                   "-evalue", "1e-3", "-max_target_seqs", "100", "-outfmt", "6 " + " ".join(BLAST_FIELDS), "-out", str(output)]
        if cpu_threads is not None:
            command.extend(["-num_threads", str(cpu_threads)])
        subprocess.run(command, check=True, capture_output=True, text=True)
    return {"hits_sha256": sha(output), "hit_rows": sum(1 for _ in output.open()), "query_sha256": sha(query),
            "database_fasta_sha256": sha(database), "seconds": None if check_existing else time.perf_counter() - started}


def compare(reference, cross_report, *, check_existing=False):
    key = bundled_platform_key()
    bundle = bundled_blast_bin_dir()
    cases = json.loads((ROOT / "validation" / "inputs.json").read_text())
    report = {"platform": platform.platform(), "reference_bin": str(reference), "bundled_bin": bundle.relative_to(REPO).as_posix(),
              "comparison_scope": "reference versus bundled on this operating system",
              "checked_existing_outputs": check_existing,
              "version": subprocess.check_output([str(bundle / ("blastp.exe" if sys.platform == "win32" else "blastp")), "-version"], text=True), "cases": {}}
    cross = json.loads(Path(cross_report).read_text()) if cross_report else None
    for case in cases:
        assert sha(REPO / case["query"]) == case["query_sha256"]
        if "candidate_pool_sha256" in case:
            assert sha(REPO / case["pool"]) == case["candidate_pool_sha256"]
        results = {}
        # Exercise production's automatic selection for the bundled run.
        for label, executable in (("reference", reference), ("bundled", None)):
            results[label] = ranking(case, WORK / key / case["name"] / label, executable, check_existing=check_existing)
        for filename in ("sequence_ranking.tsv", "ranked_candidates.fasta"):
            case_dir = WORK / key / case["name"]
            assert (case_dir / "reference" / filename).read_bytes() == (case_dir / "bundled" / filename).read_bytes()
        for field in ("ranking_sha256", "ranked_fasta_sha256"):
            assert results["reference"][field] == results["bundled"][field], (case["name"], field, results)
            if cross and field == "ranked_fasta_sha256":
                assert results["bundled"][field] == cross["cases"][case["name"]]["bundled"][field], (case["name"], "cross-platform", field)
        if cross:
            cross_key = Path(cross["bundled_bin"]).parts[-2]
            previous = WORK / cross_key / case["name"] / "bundled" / "sequence_ranking.tsv"
            current = WORK / key / case["name"] / "bundled" / "sequence_ranking.tsv"
            differences = []
            if previous.exists():
                with previous.open(newline="") as left, current.open(newline="") as right:
                    for old, new in zip(csv.DictReader(left, delimiter="\t"), csv.DictReader(right, delimiter="\t")):
                        for field in old:
                            if old[field] != new[field]:
                                # libm/Python can format the last decimal differently
                                # across OS builds. Never relax the same-OS comparison.
                                assert field in {"primary_score", "balanced_score", "mean_percentile", "mean_identity"}
                                assert math.isclose(float(old[field]), float(new[field]), rel_tol=1e-14, abs_tol=1e-14)
                                differences.append({"rank": new["rank"], "field": field,
                                                    "reference_platform": old[field], "this_platform": new[field]})
            results["cross_platform"] = {
                "ranked_fasta_exact": True,
                "ranking_tsv_exact": results["bundled"]["ranking_sha256"] == cross["cases"][case["name"]]["bundled"]["ranking_sha256"],
                "numeric_format_differences": differences,
            }
        report["cases"][case["name"]] = results
        print(f"EXACT MATCH {key} {case['name']}: {results['bundled']['rows']} ranked sequences", flush=True)
    rna = {label: rna_blast(WORK / key / "rna_structure_lookup" / label, executable, check_existing=check_existing)
           for label, executable in (("reference", reference), ("bundled", bundle))}
    assert rna["reference"]["hits_sha256"] == rna["bundled"]["hits_sha256"], "RNA BLASTn output differs"
    if cross:
        previous = WORK / Path(cross["bundled_bin"]).parts[-2] / "rna_structure_lookup" / "bundled" / "hits.tsv"
        current = WORK / key / "rna_structure_lookup" / "bundled" / "hits.tsv"
        assert previous.read_text().splitlines() == current.read_text().splitlines(), "Cross-platform BLASTn hits differ"
        report["cross_platform_rna_structure_lookup"] = {
            "identical_hit_rows": True,
            "byte_identical": rna["bundled"]["hits_sha256"] == cross["rna_structure_lookup"]["bundled"]["hits_sha256"],
        }
    report["rna_structure_lookup"] = rna
    report["exact_match"] = True
    (ROOT / "validation" / f"{key}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"EXACT MATCH {key} RNA structure lookup: {rna['bundled']['hit_rows']} BLASTn hits", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--reference-bin", type=Path)
    parser.add_argument("--reference-archive", type=Path, help="Extract an independent reference from the verified NCBI archive.")
    parser.add_argument("--cross-report", type=Path)
    parser.add_argument("--check-existing", action="store_true", help="Recheck completed saved outputs without repeating searches; no new timings are recorded.")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    if args.reference_archive:
        key = bundled_platform_key()
        metadata = json.loads((ROOT / "bundled" / "manifest.json").read_text())[key]
        assert sha(args.reference_archive) == metadata["archive_sha256"]
        directory = WORK / key / "archive_reference" / "bin"
        directory.mkdir(parents=True, exist_ok=True)
        names = {"blastp", "blastn", "makeblastdb"}
        with tarfile.open(args.reference_archive) as archive:
            for member in archive:
                if Path(member.name).name in names and member.isfile():
                    destination = directory / Path(member.name).name
                    stream = archive.extractfile(member)
                    with stream, destination.open("wb") as handle:
                        shutil.copyfileobj(stream, handle)
                    destination.chmod(0o755)
        args.reference_bin = directory
    if args.reference_bin:
        compare(args.reference_bin.resolve(), args.cross_report, check_existing=args.check_existing)


if __name__ == "__main__":
    main()
