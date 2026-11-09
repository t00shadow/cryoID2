from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from cryoid_pipeline.models import SearchHit

from .sequence_search import read_sequence_pool


ProgressCallback = Callable[[str], None]
CancellationCheckpoint = Callable[[], None]
RNA_TRANSLATION = str.maketrans({"A": "A", "G": "A", "C": "U", "T": "U", "U": "U", "N": "A"})


@dataclass
class RnaRankingResult:
    selected_hit: SearchHit
    ranking: list[dict[str, object]]
    outputs: list[Path]


def binary_reduced(sequence: str) -> np.ndarray:
    normalized = sequence.upper().replace("U", "T")
    reduced = normalized.translate(RNA_TRANSLATION)
    return np.fromiter((letter == "U" for letter in reduced), dtype=np.int8)


def maximum_local_identity(candidate: np.ndarray, query: np.ndarray) -> float:
    scores: list[float] = []
    for oriented in (query, query[::-1]):
        if len(candidate) >= len(oriented):
            windows = np.lib.stride_tricks.sliding_window_view(candidate, len(oriented))
            scores.append(float(np.max(np.mean(windows == oriented, axis=1))))
        else:
            windows = np.lib.stride_tricks.sliding_window_view(oriented, len(candidate))
            scores.append(float(np.max(np.mean(windows == candidate, axis=1))))
    return max(scores)


def average_percentile(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    result = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        result[order[start:end]] = (((start + 1) + end) / 2.0) / len(values)
        start = end
    return result


def rank_rna_candidates(
    *,
    query_fasta: Path,
    candidate_pool: Path,
    run_dir: Path,
    progress: ProgressCallback,
    check_cancel: CancellationCheckpoint | None = None,
) -> RnaRankingResult:
    check = check_cancel or (lambda: None)
    queries = read_sequence_pool(query_fasta)
    from .candidate_groups import canonical_records
    candidates = canonical_records(candidate_pool)
    if not queries or not candidates:
        raise ValueError("RNA ranking requires at least one query and one candidate.")
    query_arrays = [binary_reduced(record.sequence) for record in queries]
    candidate_arrays = [binary_reduced(record.sequence) for record in candidates]
    identities = np.empty((len(candidates), len(queries)), dtype=float)
    progress(f"RNA ranking: scoring {len(candidates)} candidates against {len(queries)} traces.")
    for index, candidate in enumerate(candidate_arrays):
        if index % 25 == 0:
            check()
        identities[index] = [maximum_local_identity(candidate, query) for query in query_arrays]
    percentiles = np.column_stack(
        [average_percentile(identities[:, column]) for column in range(len(queries))]
    )

    scores: list[dict[str, object]] = []
    for index, _record in enumerate(candidates):
        scores.append({
            "balanced_score": float(np.min(percentiles[index])),
            "mean_percentile": float(np.mean(percentiles[index])),
            "mean_identity": float(np.mean(identities[index])),
        })

    ranking: list[dict[str, object]] = []
    for input_index, record in enumerate(candidates):
        ranking.append(
            {
                "candidate_id": record.record_id,
                "members": record.record_id,
                "best_strand": record.record_id,
                "sequence": record.sequence,
                "length": len(record.sequence),
                "description": record.description,
                "primary_score_name": "balanced_hard_maximin",
                "primary_score": scores[input_index]["balanced_score"],
                "score_direction": "higher_is_better",
                **scores[input_index],
                "input_index": input_index,
            }
        )
    # Python's sort is stable, so exact score ties retain input FASTA order.
    ranking.sort(
        key=lambda row: (
            -float(row["balanced_score"]),
            -float(row["mean_percentile"]),
            -float(row["mean_identity"]),
        )
    )
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank
    top = ranking[0]
    selected = SearchHit(
        query_id="generated_rna_queries",
        record_id=str(top["candidate_id"]),
        score=float(top["balanced_score"]),
        sequence=str(top["sequence"]),
        description=str(top["description"]),
    )
    diagnostics = Path(run_dir) / "tmp" / "sequence_search" / "ranking_summary.json"
    diagnostics.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.write_text(
        json.dumps(
            {
                "design": "ground-truth-free hard maximin over every supplied FASTA record",
                "query_count": len(queries),
                "candidate_count": len(candidates),
                "ranked_record_count": len(ranking),
                "top_candidate": selected.record_id,
                "top_score": selected.score,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    progress(f"RNA ranking: selected top locus {selected.record_id}.")
    return RnaRankingResult(selected_hit=selected, ranking=ranking, outputs=[diagnostics])
