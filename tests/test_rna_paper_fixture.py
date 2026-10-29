from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from D_BLAST_searching.rna_ranking import rank_rna_candidates
from D_BLAST_searching.sequence_search import read_sequence_pool


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "rna_emd47655"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RnaPaperFixtureTests(unittest.TestCase):
    def test_frozen_inputs_have_expected_hashes_and_query_lengths(self) -> None:
        queries = FIXTURE / "expected_queries.fasta"
        current_queries = FIXTURE / "current_generated_queries.fasta"
        pool = FIXTURE / "reduced_noncds_pool.fasta"
        ground_truth = FIXTURE / "ground_truth.fasta"

        self.assertEqual(
            _sha256(queries),
            "5088f62b01281662e253c5176eb6ff24e0fd99298d0d0dad030d0dbdfb8b1cad",
        )
        self.assertEqual(
            _sha256(current_queries),
            "fde2f88b9716ba6c7b056ca7bf2680dfb4d53039fa3917fd344eb73003018531",
        )
        self.assertEqual(
            _sha256(pool),
            "1ee4427aac471865e3d276c18384f7c029dd98545139c237a08d94fe41a7cfa3",
        )
        self.assertEqual(
            _sha256(ground_truth),
            "8e00ecf804c0792067667d66a5fa8c2b56710357270ed362b3a22a4a3a269442",
        )
        self.assertEqual(
            [len(record.sequence) for record in read_sequence_pool(queries)],
            [105, 96, 55, 44, 35, 23, 14],
        )
        self.assertEqual(
            [len(record.sequence) for record in read_sequence_pool(current_queries)],
            [105, 96, 55, 44, 35, 23, 14],
        )
        self.assertEqual(len(read_sequence_pool(pool)), 2538)
        self.assertEqual(len(read_sequence_pool(ground_truth)), 1)

    def test_hard_maximin_reproduces_ground_truth_rank_four(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "results" / "identification" / "rna").mkdir(parents=True)
            result = rank_rna_candidates(
                query_fasta=FIXTURE / "expected_queries.fasta",
                candidate_pool=FIXTURE / "reduced_noncds_pool.fasta",
                run_dir=run_dir,
                progress=lambda _message: None,
            )

        self.assertEqual(len(result.ranking), 2538)
        self.assertEqual(
            [row["candidate_id"] for row in result.ranking[:4]],
            ["seq02778", "seq02774", "seq02300", "seq03296"],
        )
        target = result.ranking[3]
        self.assertEqual(target["rank"], 4)
        self.assertEqual(target["members"], "seq03296")
        self.assertAlmostEqual(float(target["balanced_score"]), 0.7549251379038613)

    def test_fresh_production_queries_rank_ground_truth_third(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "results" / "identification" / "rna").mkdir(parents=True)
            result = rank_rna_candidates(
                query_fasta=FIXTURE / "current_generated_queries.fasta",
                candidate_pool=FIXTURE / "reduced_noncds_pool.fasta",
                run_dir=run_dir,
                progress=lambda _message: None,
            )

        self.assertEqual(len(result.ranking), 2538)
        target = next(
            row for row in result.ranking if row["candidate_id"] == "seq03296"
        )
        self.assertEqual(target["rank"], 3)
        self.assertAlmostEqual(float(target["balanced_score"]), 0.7675334909377463)


if __name__ == "__main__":
    unittest.main()
