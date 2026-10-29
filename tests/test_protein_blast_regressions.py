from __future__ import annotations

import csv
import math
import random
from argparse import Namespace
from decimal import Decimal
from pathlib import Path

import pytest
from matplotlib import pyplot as plt

from D_BLAST_searching.protein_search import run_protein_blast_search
from D_BLAST_searching.reduced_alphabet.evalues import (
    alignment_log10_evalue,
    format_log10_evalue,
    negative_log10_composite,
)
from D_BLAST_searching.reduced_alphabet.search_pool import BlastSeq, Calculate, Filter
from D_BLAST_searching.sequence_ranking import protein_ranking_rows, write_sequence_ranking
from D_BLAST_searching.sequence_search import read_sequence_pool


def _sequence(length: int) -> str:
    return "".join(random.Random(231).choices("ACDEFGHIKLMNPQRSTVWY", k=length))


def _run(tmp_path: Path, sequence: str, candidates: list[tuple[str, str]]):
    query = tmp_path / "query.fasta"
    pool = tmp_path / "pool.fasta"
    query.write_text(f">query\n{sequence}\n", encoding="utf-8")
    original = "".join(f">{header}\n{residues}\n" for header, residues in candidates)
    pool.write_text(original, encoding="utf-8")
    try:
        result = run_protein_blast_search(
            query_fasta=query, candidate_pool=pool, run_dir=tmp_path / "run",
            progress=lambda _message: None, cpu_threads=1,
        )
    finally:
        plt.close("all")
    assert pool.read_text(encoding="utf-8") == original
    return result, pool


@pytest.mark.parametrize("identifier", [
    "plain_id", "sp|P12345|TEST_HUMAN", "tr|Q12345|TEST_HUMAN", "lcl|local_protein",
])
def test_bundled_blast_preserves_original_fasta_identifiers(tmp_path, identifier):
    sequence = _sequence(100)
    result, pool = _run(tmp_path, sequence, [
        (f"{identifier} Original protein description", sequence),
        ("unmatched Description without a match", "W" * 100),
    ])
    assert result.selected_hit.record_id == identifier
    assert result.selected_hit.sequence == sequence
    assert result.selected_hit.description == f"{identifier} Original protein description"
    assert {row["candidate"] for row in result.ranking} == {identifier, "unmatched"}
    rows = protein_ranking_rows(result.ranking, read_sequence_pool(pool))
    assert rows[0]["candidate_id"] == identifier
    assert rows[0]["description"] == result.selected_hit.description
    for output in result.outputs:
        if output.suffix == ".txt":
            assert "cryoid_seq_" not in output.read_text(encoding="utf-8")


def test_accession_collisions_and_ties_keep_original_identifiers(tmp_path):
    sequence = _sequence(100)
    identifiers = ["tr|P12345|TWO", "sp|P12345|ONE", "P12345", "lcl|P12345"]
    result, _pool = _run(tmp_path, sequence, [(identifier, sequence) for identifier in identifiers])
    # Internal BLAST tokens must not merge records sharing an accession or
    # change the established original-ID ordering for equal significance.
    assert [row["candidate"] for row in result.ranking] == sorted(identifiers)


def test_ids_differing_only_in_case_retain_their_own_lengths(tmp_path):
    sequence = _sequence(100)
    result, _pool = _run(tmp_path, sequence, [("same", sequence), ("Same", sequence * 2)])
    assert result.selected_hit.record_id == "same"
    assert {row["candidate"]: row["length"] for row in result.ranking} == {
        "same": "100", "Same": "200",
    }


def test_old_database_with_raw_identifiers_is_not_reused(tmp_path):
    legacy = tmp_path / "run/tmp/sequence_search/work/candidate_pool_db/candidate_pool_db.pin"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"An old database that contains raw FASTA identifiers")
    sequence = _sequence(100)
    result, _pool = _run(tmp_path, sequence, [("sp|P12345|TEST_HUMAN", sequence)])
    assert result.selected_hit.record_id == "sp|P12345|TEST_HUMAN"
    assert legacy.exists()


@pytest.mark.parametrize("length", [200, 400, 800])
def test_real_zero_evalue_hits_finish_and_keep_distinct_significance(tmp_path, length):
    sequence = _sequence(length)
    weaker = "".join(
        ("W" if residue != "W" else "A") if index % 13 == 0 else residue
        for index, residue in enumerate(sequence)
    )
    result, pool = _run(tmp_path, sequence, [
        ("a_weaker", weaker), ("sp|P12345|Z_STRONGER", sequence),
    ])
    assert result.selected_hit.record_id == "sp|P12345|Z_STRONGER"
    assert math.isfinite(result.selected_hit.score)
    raw_files = list((tmp_path / "run").rglob("*_blast.txt"))
    alignments = [
        line.split("\t") for path in raw_files
        for line in path.read_text().splitlines() if line and not line.startswith("#")
    ]
    assert any(float(fields[14]) == 0 for fields in alignments)
    if length >= 400:
        assert len({fields[1] for fields in alignments if float(fields[14]) == 0}) == 2
    assert float(result.ranking[0]["composite_log10_evalue"]) < float(
        result.ranking[1]["composite_log10_evalue"]
    )
    assert Decimal(result.ranking[0]["composite_evalue"]) > 0
    rows = protein_ranking_rows(result.ranking, read_sequence_pool(pool))
    assert rows[0]["primary_score"] == result.selected_hit.score
    assert rows[0]["primary_score"] > rows[1]["primary_score"]
    if length == 800:
        assert result.selected_hit.score > 308
        assert float(result.ranking[0]["composite_evalue"]) == 0
    output = write_sequence_ranking(tmp_path / "run", mode="protein", rows=rows)
    with output.open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle, delimiter="\t"))
    assert float(saved[0]["primary_score"]) == result.selected_hit.score
    assert saved[0]["candidate_id"] == result.selected_hit.record_id


def _alignment(candidate, bit_score):
    return BlastSeq("\t".join(map(str, [
        "query", candidate, 200, 200, 200, 0, 0, 0, 1, 200, 1, 200,
        "A" * 200, "A" * 200, 0.0, bit_score,
    ])), dbsize=1000)


def test_zero_evalues_choose_strongest_alignment_instead_of_first():
    weak = _alignment("candidate", 700)
    strong = _alignment("candidate", 1000)
    for field in (weak, strong):
        field.set_prot_len(200)
    params = Namespace(
        evalue=1.0, query_info=[["query", 200, "unused", 0]], reverse=False,
    )
    result = Calculate([weak, strong], params)
    expected = alignment_log10_evalue(0, bit_score=1000, search_space=200000) + math.log10(0.2)
    assert result[0].composite_log10_evalue == pytest.approx(expected)
    filtered = Filter([strong, weak], Namespace(verbose=False, length=[1, 100000]))
    assert filtered == [strong]


def test_legacy_positive_evalue_alignment_remains_supported():
    line = "\t".join(map(str, [
        "query", "candidate", 30, 30, 30, 0, 0, 0, 1, 30, 1, 30,
        "A" * 30, "A" * 30, "1e-20",
    ]))
    assert BlastSeq(line).log10_evalue == -20


@pytest.mark.parametrize("log_value, formatted", [
    (-1000.25, "5.6e-1001"), (1000.25, "1.8e+1000"),
    (0.0, "1.0e+00"), (-0.001, "1.0e+00"),
])
def test_composite_output_handles_values_outside_float_range(log_value, formatted):
    assert format_log10_evalue(log_value) == formatted
    assert negative_log10_composite({"composite_evalue": formatted}) == pytest.approx(-log_value, abs=0.01)


def test_zero_evalue_requires_statistical_context():
    with pytest.raises(ValueError, match="bit score and positive search space"):
        alignment_log10_evalue(0)
    assert alignment_log10_evalue(1e-200) == -200


def test_compound_evalue_overflow_does_not_crash_ranking():
    field = _alignment("long_candidate", 1000)
    field.set_prot_len(100000)
    params = Namespace(
        evalue=1.0, query_info=[[f"unmatched_{i}", 200, "unused", 0] for i in range(200)],
        reverse=False,
    )
    result = Calculate([field], params)
    assert result[0].composite_log10_evalue == pytest.approx(400)
    assert result[0].composite_evalue == math.inf
    assert format_log10_evalue(result[0].composite_log10_evalue) == "1.0e+400"
