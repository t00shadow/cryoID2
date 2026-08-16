from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from cryoid_pipeline.models import SearchHit


_DEGENERATE_ALPHABET = set("GPZMYWX")
_AA_TO_DEGENERATE = {
    "G": "G",
    "A": "G",
    "S": "G",
    "C": "G",
    "V": "G",
    "T": "G",
    "I": "G",
    "U": "G",
    "P": "P",
    "L": "Z",
    "D": "Z",
    "N": "Z",
    "E": "Z",
    "Q": "Z",
    "M": "Z",
    "B": "Z",
    "K": "M",
    "R": "M",
    "H": "Y",
    "F": "Y",
    "Y": "Y",
    "W": "W",
    "X": "X",
    "J": "X",
    "O": "X",
    "Z": "Z",
}


@dataclass(frozen=True)
class SequenceRecord:
    record_id: str
    sequence: str
    description: str = ""


def normalize_sequence(value: str) -> str:
    return re.sub(r"[^A-Za-z*]", "", str(value or "")).upper().replace("*", "")


def to_degenerate_sequence(value: str) -> str:
    return "".join(_AA_TO_DEGENERATE.get(ch, "X") for ch in normalize_sequence(value))


def sequence_looks_degenerate(value: str) -> bool:
    seq = normalize_sequence(value)
    return bool(seq) and set(seq) <= _DEGENERATE_ALPHABET


def read_fasta(path: Path) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    current_id = ""
    current_desc = ""
    chunks: list[str] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id:
                    records.append(
                        SequenceRecord(current_id, normalize_sequence("".join(chunks)), current_desc)
                    )
                header = line[1:].strip()
                current_desc = header
                current_id = header.split()[0] if header else f"record_{len(records) + 1}"
                chunks = []
            else:
                chunks.append(line)
    if current_id:
        records.append(SequenceRecord(current_id, normalize_sequence("".join(chunks)), current_desc))
    return [record for record in records if record.sequence]


def read_sequence_pool(path: Path) -> list[SequenceRecord]:
    suffix = path.suffix.lower()
    if suffix in {".fasta", ".fa", ".faa", ".fna"}:
        return read_fasta(path)
    raise ValueError(f"Unsupported candidate pool type: {path.suffix}. Use FASTA.")


def _kmers(seq: str, k: int) -> set[str]:
    if len(seq) < k:
        return {seq} if seq else set()
    return {seq[idx : idx + k] for idx in range(0, len(seq) - k + 1)}


def kmer_overlap_score(query: str, target: str) -> float:
    query = normalize_sequence(query)
    target = normalize_sequence(target)
    if not query or not target:
        return 0.0
    if query in target:
        return 1.0
    if target in query:
        return max(0.5, len(target) / max(len(query), 1))
    k = max(2, min(5, len(query) // 8 or 2))
    query_kmers = _kmers(query, k)
    target_kmers = _kmers(target, k)
    if not query_kmers or not target_kmers:
        return 0.0
    overlap = len(query_kmers & target_kmers) / len(query_kmers)
    length_penalty = min(len(query), len(target)) / max(len(query), len(target))
    return float((0.85 * overlap) + (0.15 * length_penalty))


def search_pool(
    queries: Iterable[SequenceRecord],
    pool: Iterable[SequenceRecord],
    *,
    top_n: int = 25,
) -> list[SearchHit]:
    query_records = list(queries)
    pool_records = list(pool)
    hits: list[SearchHit] = []
    for query in query_records:
        query_seq = normalize_sequence(query.sequence)
        use_degenerate = sequence_looks_degenerate(query_seq)
        scored_query = query_seq
        for candidate in pool_records:
            candidate_seq = to_degenerate_sequence(candidate.sequence) if use_degenerate else candidate.sequence
            score = kmer_overlap_score(scored_query, candidate_seq)
            hits.append(
                SearchHit(
                    query_id=query.record_id,
                    record_id=candidate.record_id,
                    score=score,
                    sequence=candidate.sequence,
                    description=candidate.description,
                )
            )
    hits.sort(key=lambda hit: (hit.score, -abs(len(hit.sequence) - len(query_records[0].sequence)) if query_records else 0), reverse=True)
    return hits[:top_n]
