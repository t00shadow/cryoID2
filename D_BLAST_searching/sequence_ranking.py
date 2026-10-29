from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping

from cryoid_pipeline.models import SequenceMode
from .sequence_search import SequenceRecord
from .reduced_alphabet.evalues import negative_log10_composite


RANKING_FIELDS = [
    "rank",
    "mode",
    "candidate_id",
    "members",
    "sequence",
    "length",
    "primary_score_name",
    "primary_score",
    "score_direction",
    "description",
    "average_identity",
    "composite_evalue",
    "best_strand",
    "balanced_score",
    "mean_percentile",
    "mean_identity",
]


def write_sequence_ranking(
    run_dir: Path,
    *,
    mode: SequenceMode,
    rows: Iterable[Mapping[str, object]],
) -> Path:
    destination = Path(run_dir) / "sequence_ranking.tsv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RANKING_FIELDS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            payload = {field: row.get(field, "") for field in RANKING_FIELDS}
            payload["mode"] = mode
            writer.writerow(payload)
    temporary.replace(destination)
    return destination.resolve()


def protein_ranking_rows(
    ranking: list[dict[str, str]],
    records: list[SequenceRecord],
) -> list[dict[str, object]]:
    by_id = {record.record_id: record for record in records}
    rows: list[dict[str, object]] = []
    for source in ranking:
        candidate_id = str(source.get("candidate") or "")
        record = by_id.get(candidate_id)
        composite = str(source.get("composite_evalue") or "")
        primary = negative_log10_composite(source)
        rows.append(
            {
                "rank": source.get("rank", len(rows) + 1),
                "candidate_id": candidate_id,
                "members": candidate_id,
                "sequence": record.sequence if record else "",
                "length": len(record.sequence) if record else source.get("length", ""),
                "primary_score_name": "negative_log10_composite_evalue",
                "primary_score": primary,
                "score_direction": "higher_is_better",
                "description": record.description if record else "",
                "average_identity": source.get("average_identity", ""),
                "composite_evalue": composite,
            }
        )
    return rows
