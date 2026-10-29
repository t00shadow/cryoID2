from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cryoid_pipeline.models import PredictionBundle, QueryBundle, SearchHit
from cryoid_pipeline.orchestrator import CryoIDPipeline
from cryoid_pipeline.structure_selection import parse_structure_selection
from D_BLAST_searching.rna_ranking import (
    RnaRankingResult,
    average_percentile,
    maximum_local_identity,
)
from D_BLAST_searching.sequence_mode import SequenceModeError, resolve_sequence_mode
from I_Structure_fitting.rna.structures import RnaStructureResult, select_structure_hits


class SequenceModeTests(unittest.TestCase):
    def _pool(self, root: Path, text: str) -> Path:
        path = root / "pool.fasta"
        path.write_text(text, encoding="utf-8")
        return path

    def test_auto_detects_rna_with_u_or_t_and_protein(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(resolve_sequence_mode(self._pool(root, ">r\nACGUUN\n"))[1], "rna")
            self.assertEqual(resolve_sequence_mode(self._pool(root, ">r\nACGTTN\n"))[1], "rna")
            self.assertEqual(resolve_sequence_mode(self._pool(root, ">p\nMELQIFK\n"))[1], "protein")

    def test_auto_blocks_mixed_pool_and_explicit_override_is_authoritative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = self._pool(root, ">rna_like\nACGU\n>protein\nMELQ\n")
            with self.assertRaisesRegex(SequenceModeError, "mixed or alphabetically ambiguous"):
                resolve_sequence_mode(pool, "auto")
            self.assertEqual(resolve_sequence_mode(pool, "protein")[1], "protein")

    def test_all_probability_channels_share_one_validated_prediction_bundle(self) -> None:
        pipeline = CryoIDPipeline(project_dir=Path("project"))
        density = Path("emd_1.map")
        output = Path("predictions")
        protein = [path.name for path in pipeline._required_prediction_paths(density, output, "protein")]
        rna = [path.name for path in pipeline._required_prediction_paths(density, output, "rna")]
        self.assertEqual(len(protein), 15)
        self.assertEqual(protein, rna)
        self.assertTrue(any("key_atoms_1" in name for name in protein))
        self.assertTrue(any("key_atoms_2" in name for name in rna))
        self.assertTrue(any("key_atoms_3" in name for name in rna))
        self.assertTrue(any("secondary_strctures_4" in name for name in rna))
        self.assertTrue(any("residue_types_6letter_8" in name for name in rna))
        self.assertEqual(
            pipeline._prediction_manifest_for_mode(output, "emd_1.map", "protein"),
            pipeline._prediction_manifest_for_mode(output, "emd_1.map", "rna"),
        )
        protein_bundle = pipeline._prediction_bundle(density, output, "protein", [])
        rna_bundle = pipeline._prediction_bundle(density, output, "rna", [])
        self.assertEqual(set(protein_bundle.key_atom_channels), {1})
        self.assertEqual(set(protein_bundle.secondary_structure_channels), {1, 2, 3})
        self.assertEqual(set(protein_bundle.residue_type_channels), set(range(1, 7)))
        self.assertEqual(set(rna_bundle.key_atom_channels), {2, 3})
        self.assertEqual(set(rna_bundle.secondary_structure_channels), {4})
        self.assertEqual(set(rna_bundle.residue_type_channels), {7, 8})
        self.assertIsNone(rna_bundle.raw_key_atoms_1_path)

    def test_zero_rcl_is_literal_in_both_modes(self) -> None:
        pipeline = CryoIDPipeline(project_dir=Path("project"), unet_rcl=0.0)
        self.assertEqual(pipeline._effective_unet_rcl("protein"), 0.0)
        self.assertEqual(pipeline._effective_unet_rcl("rna"), 0.0)

        explicit = CryoIDPipeline(project_dir=Path("project"), unet_rcl=0.08)
        self.assertEqual(explicit._effective_unet_rcl("protein"), 0.08)
        self.assertEqual(explicit._effective_unet_rcl("rna"), 0.08)


class RnaSelectionTests(unittest.TestCase):
    def test_count_and_fraction_spelling_are_distinct(self) -> None:
        self.assertEqual(parse_structure_selection("1").kind, "count")
        self.assertEqual(parse_structure_selection("1").selected_count(12), 1)
        self.assertEqual(parse_structure_selection("1.0").kind, "fraction")
        self.assertEqual(parse_structure_selection("1.0").selected_count(12), 12)
        self.assertEqual(parse_structure_selection("0.25").selected_count(10), 3)
        self.assertEqual(parse_structure_selection("20").selected_count(12), 12)
        with self.assertRaises(ValueError):
            parse_structure_selection("2.5")

    def test_ranking_percentiles_are_tie_aware(self) -> None:
        np.testing.assert_allclose(
            average_percentile(np.array([1.0, 1.0, 2.0])),
            [0.5, 0.5, 1.0],
        )

    def test_local_identity_accepts_reversed_trace_direction(self) -> None:
        candidate = np.array([0, 0, 1, 0], dtype=np.int8)
        query = np.array([1, 0, 0], dtype=np.int8)
        self.assertEqual(maximum_local_identity(candidate, query), 1.0)

    def test_fraction_selects_ranked_rows_before_structure_lookup(self) -> None:
        ranking = [
            {"rank": str(index), "candidate_id": f"SEQ{index}"}
            for index in range(1, 21)
        ]
        hits = {
            f"SEQ{index}": [
                {
                    "query": f"SEQ{index}",
                    "pdb_chain": f"P{index}_A",
                    "structure_sequence": f"ACGU{index}",
                }
            ]
            for index in (2, 4, 6, 8, 10, 12, 14, 16, 18, 20)
        }
        selected_loci, structured, selected = select_structure_hits(
            ranking, hits, parse_structure_selection("0.25")
        )
        self.assertEqual(
            [row["candidate_id"] for row in selected_loci],
            [f"SEQ{i}" for i in range(1, 6)],
        )
        self.assertEqual([row["candidate_id"] for row in structured], ["SEQ2", "SEQ4"])
        self.assertEqual(selected[0]["query"], "SEQ2")

    def test_structure_sequences_are_deduplicated_globally_and_limited_per_locus(self) -> None:
        ranking = [{"rank": "1", "candidate_id": "A"}, {"rank": "2", "candidate_id": "B"}]
        hits = {
            "A": [
                {"query": "A", "pdb_chain": f"1AA_{i}", "structure_sequence": f"SEQ{i}"}
                for i in range(5)
            ],
            "B": [
                {"query": "B", "pdb_chain": "2BB_X", "structure_sequence": "SEQ0"},
                {"query": "B", "pdb_chain": "2BB_Y", "structure_sequence": "UNIQUE"},
            ],
        }
        _selected_loci, _structured, selected = select_structure_hits(
            ranking, hits, parse_structure_selection("1.0")
        )
        self.assertEqual([row["pdb_chain"] for row in selected], ["1AA_0", "1AA_1", "1AA_2", "2BB_Y"])


class RnaPipelineIntegrationTests(unittest.TestCase):
    @staticmethod
    def _write(path: Path, text: str = "x") -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def _unet(self, density: Path, output: Path, _progress) -> PredictionBundle:
        prefix = density.name
        possibility = output
        keys = [self._write(possibility / f"{prefix}_key_atoms_{i}_possibility.mrc") for i in range(1, 4)]
        secondary = [
            self._write(possibility / f"{prefix}_secondary_strctures_{i}_possibility.mrc")
            for i in range(1, 5)
        ]
        residues = [
            self._write(possibility / f"{prefix}_residue_types_6letter_{i}_possibility.mrc")
            for i in range(1, 9)
        ]
        return PredictionBundle(
            "checkpoint_inference",
            density,
            output,
            keys[0],
            secondary,
            residues,
            output_files=[*keys, *secondary, *residues],
            mode="rna",
            key_atom_paths=keys,
        )

    def test_rna_branch_saves_ranking_before_structure_runner_and_never_calls_protein(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            density = self._write(root / "emd_1.map")
            pool = self._write(root / "pool.fasta", ">rna_a\nACGUACGU\n>rna_b\nAAAACCCC\n")

            def query_runner(_bundle, run_dir, _progress):
                query = self._write(run_dir / "tmp" / "generated_queries.fasta", ">q\nAAUU\n")
                return QueryBundle(query, "fake_rna", [query])

            def ranking_runner(_queries, _pool, run_dir, _progress):
                selected = SearchHit("q", "RNA_A", 0.9, "ACGUACGU")
                rows = [
                    {
                        "rank": 1,
                        "candidate_id": "RNA_A",
                        "members": "RNA_A",
                        "best_strand": "RNA_A",
                        "sequence": "ACGUACGU",
                        "length": 8,
                        "primary_score_name": "balanced_hard_maximin",
                        "primary_score": 0.9,
                        "score_direction": "higher_is_better",
                        "balanced_score": 0.9,
                        "mean_percentile": 0.95,
                        "mean_identity": 0.8,
                        "description": "",
                    }
                ]
                return RnaRankingResult(selected, rows, [])

            def structure_runner(**kwargs):
                ranking_path = kwargs["ranking_path"]
                self.assertEqual(kwargs["structure_selection"].raw, "0.5")
                self.assertTrue(ranking_path.is_file())
                with ranking_path.open(encoding="utf-8", newline="") as handle:
                    self.assertEqual(next(csv.DictReader(handle, delimiter="\t"))["candidate_id"], "RNA_A")
                summary = self._write(kwargs["run_dir"] / "tmp" / "structure_fetch" / "selection_summary.json", "{}")
                return RnaStructureResult([summary], 1, 1, 1, 1, 0, {})

            def forbidden(*_args, **_kwargs):
                raise AssertionError("Protein pipeline must not run in RNA mode")

            result = CryoIDPipeline(
                project_dir=root / "project",
                sequence_mode="auto",
                unet_runner=self._unet,
                rna_query_runner=query_runner,
                rna_ranking_runner=ranking_runner,
                rna_structure_runner=structure_runner,
                protein_query_runner=forbidden,
                protein_blast_runner=forbidden,
                struct_fetch_runner=forbidden,
            ).run(density, pool)

            self.assertEqual(result.mode, "rna")
            self.assertEqual(result.ranking_path, result.run_dir / "sequence_ranking.tsv")
            self.assertTrue(result.ranking_path.is_file())
            config = json.loads((result.run_dir / "run_config.json").read_text(encoding="utf-8"))
            self.assertEqual(config["requested_mode"], "auto")
            self.assertEqual(config["effective_mode"], "rna")
            self.assertEqual(config["configured_rcl"], 0.0)
            self.assertEqual(config["rcl"], 0.0)
            self.assertEqual(config["sequence_ranking"], str(result.ranking_path))
            self.assertEqual(config["structure_selection"]["input"], "0.5")
            self.assertEqual(config["structure_selection"]["selected_count"], 1)


if __name__ == "__main__":
    unittest.main()
