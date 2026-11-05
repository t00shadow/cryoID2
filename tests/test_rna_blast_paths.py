import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from I_Structure_fitting.rna.structures import (
    PublicRnaDatabase,
    run_rna_structure_pipeline,
)


def test_rna_search_handles_project_path_with_spaces():
    sequence = "ACGT" * 30

    with tempfile.TemporaryDirectory() as temp:
        project = Path(temp) / "RNA for Star"
        run_dir = project / "runs" / "sample run"
        run_dir.mkdir(parents=True)

        ranking_path = project / "ranking.tsv"
        ranking_path.write_text(
            f"rank\tcandidate_id\tsequence\n"
            f"1\tcandidate1\t{sequence}\n"
        )

        database_dir = project / "cache" / "rna"
        database_dir.mkdir(parents=True)

        fasta = database_dir / "pdb_rna_chains.fasta"
        fasta.write_text(f">pdb|1ABC|A\n{sequence}\n")

        prefix = database_dir / "pdb_rna_chains"
        prefix.with_suffix(".nin").write_bytes(b"test database")
        prefix.with_suffix(".nsq").write_bytes(b"test database")

        database = PublicRnaDatabase(
            fasta, prefix, {"active_snapshot": "test"}, []
        )

        ranked_row = {
            "rank": "1",
            "candidate_id": "candidate1",
            "sequence": sequence,
        }

        class FakeSelection:
            def details(self, total):
                return {"total": total}

        def fake_blast(command, **kwargs):
            for flag in ("-query", "-db", "-out"):
                value = command[command.index(flag) + 1]
                assert not any(c.isspace() for c in value), (flag, value)

            query = Path(command[command.index("-query") + 1])
            db = Path(command[command.index("-db") + 1])
            output = Path(command[command.index("-out") + 1])

            assert query.is_file()
            assert db.with_suffix(".nin").is_file()

            output.write_text(
                "candidate1\tpdb|1ABC|A\t217\t1e-20"
                "\t100\t120\t120\t120\t100\n"
            )
            return SimpleNamespace(
                returncode=0, stdout="", stderr=""
            )

        with (
            patch(
                "I_Structure_fitting.rna.structures.ensure_public_rna_database",
                return_value=database,
            ),
            patch(
                "I_Structure_fitting.rna.structures._executable",
                return_value="blastn",
            ),
            patch(
                "I_Structure_fitting.rna.structures.select_ranked_rows",
                return_value=[ranked_row],
            ),
            patch(
                "I_Structure_fitting.rna.structures.select_structure_hits",
                return_value=([ranked_row], [], []),
            ) as select_hits,
            patch(
                "I_Structure_fitting.rna.structures.subprocess.run",
                side_effect=fake_blast,
            ),
        ):
            run_rna_structure_pipeline(
                project_dir=project,
                run_dir=run_dir,
                density_map=project / "map.mrc",
                prediction_bundle=None,
                ranking_path=ranking_path,
                structure_selection=FakeSelection(),
                progress=lambda message: None,
                progress_percent=lambda percent: None,
                check_cancel=lambda: None,
            )

        result_path = (
            run_dir / "tmp" / "structure_fetch"
            / "vs_public_pdb_rna.tsv"
        )

        assert result_path.is_file()
        assert "candidate1" in result_path.read_text()

        parsed_hits = select_hits.call_args.args[1]
        assert "CANDIDATE1" in parsed_hits
        assert parsed_hits["CANDIDATE1"]
