import tempfile
import csv
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from I_Structure_fitting.rna.structures import (
    PublicRnaDatabase,
    run_rna_structure_pipeline,
)


@pytest.mark.parametrize("subject,chain", [("pdb|1ABC|A", "A"), ("1abc_a", "a"), ("pdb|1abc|a", "a")])
def test_rna_search_handles_project_path_with_spaces(subject, chain):
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
        # The lower-case chain must not overwrite the upper-case chain.
        fasta.write_text(f">pdb|1ABC|A\n{'A' * 120}\n>pdb|1abc|a\n{'C' * 120}\n")

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
                f"candidate1\t{subject}\t217\t1e-20\t100\t120\t120\t120\t100\n"
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
        hit = parsed_hits["candidate1"][0]
        assert hit["chain_id"] == chain
        assert hit["structure_sequence"] == ("A" if chain == "A" else "C") * 120


def test_rna_structure_selection_keeps_query_ids_differing_only_in_case():
    from I_Structure_fitting.rna.structures import select_structure_hits
    from cryoid_pipeline.structure_selection import parse_structure_selection

    ranking = [{"rank": "1", "candidate_id": "rna"}, {"rank": "2", "candidate_id": "RNA"}]
    hits = {
        "rna": [{"query": "rna", "pdb_chain": "1ABC_A", "structure_sequence": "AAAA"}],
        "RNA": [{"query": "RNA", "pdb_chain": "1ABC_a", "structure_sequence": "CCCC"}],
    }
    _, structured, selected = select_structure_hits(ranking, hits, parse_structure_selection("1.0"))
    assert structured == ranking
    assert [(row["query"], row["pdb_chain"]) for row in selected] == [("rna", "1ABC_A"), ("RNA", "1ABC_a")]


@pytest.mark.parametrize("keep_tmp", [True, False])
def test_rna_fitting_keeps_case_sensitive_chains_and_files_distinct(tmp_path, monkeypatch, keep_tmp):
    import gemmi
    from I_Structure_fitting.rna import rawfield
    from I_Structure_fitting.protein.core import aligner

    structure = gemmi.Structure()
    model = gemmi.Model("1")
    for name, x in [("A", 1.0), ("a", 2.0)]:
        chain = gemmi.Chain(name)
        residue = gemmi.Residue()
        residue.name, residue.seqid = "A", gemmi.SeqId(1, " ")
        atom = gemmi.Atom()
        atom.name, atom.pos = "P", gemmi.Position(x, 0, 0)
        residue.add_atom(atom)
        chain.add_residue(residue)
        model.add_chain(chain)
    structure.add_model(model)
    source = tmp_path / "source.cif"
    structure.make_mmcif_document().write_file(str(source))
    hits = tmp_path / "hits.tsv"
    with hits.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["query", "pdb_chain", "pdb_id", "chain_id"], delimiter="\t")
        writer.writeheader()
        for chain in ("A", "a"):
            writer.writerow(dict(query="candidate", pdb_chain=f"1ABC_{chain}", pdb_id="1ABC", chain_id=chain))
    def download(pdb, path):
        path.parent.mkdir(parents=True, exist_ok=True)
        return shutil.copy2(source, path)

    monkeypatch.setattr(rawfield, "download_structure", download)
    monkeypatch.setattr(rawfield, "prepare_prediction_aliases", lambda **kwargs: (tmp_path, source))
    observed = []

    def fit(density, isolated, fitted, **kwargs):
        chain = gemmi.read_structure(str(isolated))[0][0]
        observed.append((chain.name, chain[0][0].pos.x, str(isolated).casefold(), str(fitted).casefold()))
        shutil.copy2(isolated, fitted)
        return dict(score=3.0 if chain.name == "A" else 2.0,
                    source_model=str(isolated), fitted_path=str(fitted))

    monkeypatch.setattr(aligner, "align_model_to_map", fit)
    rows = rawfield.fit_rna_structures(hits_path=hits, structures_dir=tmp_path / "downloads",
                                     density_map=source, phosphate_map=source, scaffold_map=source,
                                     output_dir=tmp_path / "output", emdb_number="1", keep_tmp=keep_tmp)
    assert [(name, x) for name, x, *_ in observed] == [("A", 1.0), ("a", 2.0)]
    assert len({item[2] for item in observed}) == len({item[3] for item in observed}) == 2
    assert all(row["status"] == "ok" for row in rows)
    winner = Path(rows[0]["fitted_path"])
    assert gemmi.read_structure(str(winner))[0][0].name == "A"
    assert winner.exists()
