from __future__ import annotations

import logging
import os
import shutil
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from cryoid_pipeline.artifacts import link_input_file
from cryoid_pipeline.models import SearchHit

from .sequence_search import read_fasta, read_sequence_pool
from .blast_tools import runtime_blast_bin_dir, validate_blast_bin_dir
from .reduced_alphabet.evalues import negative_log10_composite


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class BlastSearchResult:
    selected_hit: SearchHit
    outputs: list[Path]
    ranking: list[dict[str, str]]
    gap_open: int
    gap_extend: int
    ranking_rows: list[dict[str, object]] | None = None
    ranking_strategy: str | None = None


def _require_file(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


@contextmanager
def _blast_path(blast_bin_dir: Path | None) -> Iterator[None]:
    directory = validate_blast_bin_dir(blast_bin_dir) if blast_bin_dir is not None else runtime_blast_bin_dir()
    original = os.environ.get("PATH", "")
    os.environ["PATH"] = str(directory) + os.pathsep + original
    try:
        missing = [name for name in ("blastp", "makeblastdb") if shutil.which(name) is None]
        if missing:
            raise FileNotFoundError(
                "NCBI BLAST+ executable(s) not found: "
                + ", ".join(missing)
                + ". Reinstall CryoID to restore its bundled BLAST+ tools."
            )
        yield
    finally:
        os.environ["PATH"] = original


def _parse_summary(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        next(handle, None)
        for line in handle:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) < 5:
                continue
            rows.append(
                {
                    "rank": fields[0],
                    "candidate": fields[1],
                    "length": fields[2],
                    "average_identity": fields[3],
                    "composite_evalue": fields[4],
                }
            )
            if len(fields) > 5:
                rows[-1]["composite_log10_evalue"] = fields[5]
    return rows


def run_protein_blast_search(
    *,
    query_fasta: Path,
    candidate_pool: Path,
    run_dir: Path,
    progress: ProgressCallback,
    gap_open: int = 15,
    gap_extend: int = 3,
    blast_bin_dir: Path | None = None,
    cpu_threads: int | None = None,
) -> BlastSearchResult:
    output_dir = run_dir / "tmp" / "sequence_search"
    work_dir = output_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    query_copy = link_input_file(query_fasta, work_dir / "generated_queries.fasta")
    pool_copy = link_input_file(candidate_pool, work_dir / "candidate_pool.fasta")
    candidate_records = read_sequence_pool(candidate_pool)
    if not candidate_records:
        raise ValueError("Protein candidate pool contains no sequences.")

    log_path = output_dir / "blast_search.log"
    queries = read_fasta(query_fasta)
    if not queries or all(len(query.sequence) == 1 for query in queries):
        from .sequence_ranking import protein_ranking_rows

        reason = "no query fragments are available" if not queries else "only single-residue query fragments are available"
        message = (
            f"Protein identification: {reason}; "
            "preserving FASTA input order and selecting the first candidate. "
            "No BLAST alignment evidence or sequence scores were calculated."
        )
        ranking = [
            {"rank": str(index), "candidate": record.record_id, "length": str(len(record.sequence)),
             "average_identity": "", "composite_evalue": "inf"}
            for index, record in enumerate(candidate_records, 1)
        ]
        summary_path = output_dir / "protein_candidates_summary.txt"
        summary_path.write_text(
            "Ranking\tcandidate\tlength\tave_identity %\tcomposite_E\n"
            + "".join(f"{r['rank']}\t{r['candidate']}\t{r['length']}\t\tinf\n" for r in ranking),
            encoding="utf-8",
        )
        log_path.write_text(message + "\n", encoding="utf-8")
        rows = protein_ranking_rows(ranking, candidate_records)
        for row in rows:
            row.update(primary_score_name="input_order", primary_score="")
        first = candidate_records[0]
        progress(message)
        return BlastSearchResult(
            SearchHit("generated_protein_queries", first.record_id, 0., first.sequence, first.description),
            [summary_path, log_path], ranking, gap_open, gap_extend,
            ranking_rows=rows, ranking_strategy=("input_order_no_queries" if not queries else "input_order_single_residue_queries"),
        )

    if blast_bin_dir is None:
        blast_bin_dir = runtime_blast_bin_dir()
    from D_BLAST_searching.reduced_alphabet.search_pool import Search

    logger = logging.getLogger(f"cryoid.protein_blast.{id(work_dir)}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers = [handler]
    params = Namespace(
        queryfile=str(query_copy),
        candidate_pool=pool_copy,
        outfile="protein_candidates",
        outdir=str(output_dir),
        reverse=True,
        length=[1, 100000],
        evalue=1.0,
        gapscore=[gap_open, gap_extend],
        # Preserve the complete ranking produced for the supplied pool. The
        # structure selector is applied later to this ordered result, rather
        # than silently imposing the historical top-20 cap here.
        num_candidate=len(candidate_records),
        verbose=True,
        logger=logger,
        cpu_threads=cpu_threads,
    )

    progress(
        f"Protein identification: running reverse, gapped BLASTp "
        f"(gap open/extend {gap_open}/{gap_extend})."
    )
    try:
        with _blast_path(blast_bin_dir):
            selected_id = Search(params)
    finally:
        handler.close()
        logger.handlers.clear()

    summary_path = _require_file(output_dir / "protein_candidates_summary.txt", "BLAST summary")
    ranking = _parse_summary(summary_path)
    if not selected_id or not ranking:
        raise RuntimeError("Protein BLAST completed without a ranked candidate.")
    if selected_id != ranking[0]["candidate"]:
        raise RuntimeError(
            f"BLAST return value {selected_id!r} does not match summary top hit {ranking[0]['candidate']!r}."
        )
    pool_records = {record.record_id: record for record in candidate_records}
    # BLAST can omit a pool record when none of the generated queries produces
    # a reportable local alignment.  The durable ranking is still required to
    # be a complete reordering of the input FASTA, so retain those records at
    # the bottom in their original FASTA order.  They deliberately have an
    # infinite composite E-value and zero identity.
    ranked_ids = {row["candidate"] for row in ranking}
    for record in candidate_records:
        if record.record_id in ranked_ids:
            continue
        ranking.append(
            {
                "rank": str(len(ranking) + 1),
                "candidate": record.record_id,
                "length": str(len(record.sequence)),
                "average_identity": "0%",
                "composite_evalue": "inf",
            }
        )
    selected_record = pool_records.get(selected_id)
    if selected_record is None:
        raise RuntimeError(f"BLAST selected {selected_id!r}, which is absent from the candidate pool.")
    score = negative_log10_composite(ranking[0])
    selected_hit = SearchHit(
        query_id="generated_protein_queries",
        record_id=selected_record.record_id,
        score=score if isinstance(score, float) else 0.0,
        sequence=selected_record.sequence,
        description=selected_record.description,
    )
    outputs = [
        summary_path,
        output_dir / "protein_candidates_alignment.txt",
        output_dir / "protein_candidates_results.png",
        log_path,
    ]
    outputs = [path.resolve() for path in outputs if path.is_file()]
    progress(f"Protein identification: selected top candidate {selected_hit.record_id}.")
    return BlastSearchResult(
        selected_hit=selected_hit,
        outputs=outputs,
        ranking=ranking,
        gap_open=gap_open,
        gap_extend=gap_extend,
    )
