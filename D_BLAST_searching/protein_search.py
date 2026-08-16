from __future__ import annotations

import logging
import math
import os
import shutil
from argparse import Namespace
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from cryoid_pipeline.artifacts import link_input_file
from cryoid_pipeline.models import SearchHit

from .sequence_search import read_sequence_pool


ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class BlastSearchResult:
    selected_hit: SearchHit
    outputs: list[Path]
    ranking: list[dict[str, str]]
    gap_open: int
    gap_extend: int


def _require_file(path: Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"Missing {label}: {resolved}")
    return resolved


@contextmanager
def _blast_path(blast_bin_dir: Path | None) -> Iterator[None]:
    original = os.environ.get("PATH", "")
    if blast_bin_dir is not None:
        os.environ["PATH"] = str(blast_bin_dir) + os.pathsep + original
    try:
        missing = [name for name in ("blastp", "makeblastdb") if shutil.which(name) is None]
        if missing:
            raise FileNotFoundError(
                "NCBI BLAST+ executable(s) not found: "
                + ", ".join(missing)
                + ". Install BLAST+ or set CRYOID_BLAST_BIN_DIR."
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
    return rows


def run_protein_blast_search(
    *,
    query_fasta: Path,
    candidate_pool: Path,
    run_dir: Path,
    repo_root: Path,
    progress: ProgressCallback,
    gap_open: int = 15,
    gap_extend: int = 3,
    blast_bin_dir: Path | None = None,
) -> BlastSearchResult:
    del repo_root  # Kept in the stable stage API; stage D no longer resolves repository scripts.
    if blast_bin_dir is None:
        blast_bin_value = os.environ.get("CRYOID_BLAST_BIN_DIR", "").strip()
        blast_bin_dir = Path(blast_bin_value).expanduser().resolve() if blast_bin_value else None
    from D_BLAST_searching.reduced_alphabet.search_pool import Search

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
    try:
        composite_evalue = float(ranking[0]["composite_evalue"])
        score = -math.log10(max(composite_evalue, 1e-300))
    except (TypeError, ValueError):
        score = 0.0
    selected_hit = SearchHit(
        query_id="generated_protein_queries",
        record_id=selected_record.record_id,
        score=score,
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
