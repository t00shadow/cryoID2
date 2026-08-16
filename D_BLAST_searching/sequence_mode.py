from __future__ import annotations

from pathlib import Path

from cryoid_pipeline.models import RequestedSequenceMode, SequenceMode
from .sequence_search import SequenceRecord, read_sequence_pool


RNA_ALPHABET = frozenset("ACGTUN")
PROTEIN_ALPHABET = frozenset("ABCDEFGHIKLMNPQRSTVWXYZJUO")
VALID_MODES = frozenset({"auto", "protein", "rna"})


class SequenceModeError(ValueError):
    """Raised when a sequence pool cannot be assigned to one pipeline safely."""


def validate_requested_mode(value: str | None) -> RequestedSequenceMode:
    mode = str(value or "auto").strip().lower()
    if mode not in VALID_MODES:
        raise SequenceModeError("Sequence mode must be auto, protein, or rna.")
    return mode  # type: ignore[return-value]


def _is_rna_record(record: SequenceRecord) -> bool:
    return bool(record.sequence) and set(record.sequence) <= RNA_ALPHABET


def validate_records_for_mode(records: list[SequenceRecord], mode: SequenceMode) -> None:
    if mode == "rna":
        invalid = [record.record_id for record in records if not _is_rna_record(record)]
        if invalid:
            preview = ", ".join(invalid[:5])
            raise SequenceModeError(
                "RNA mode accepts only A/C/G/T/U/N sequences; invalid record(s): " + preview
            )
        return

    invalid = [
        record.record_id
        for record in records
        if not record.sequence or not set(record.sequence) <= PROTEIN_ALPHABET
    ]
    if invalid:
        preview = ", ".join(invalid[:5])
        raise SequenceModeError("Protein mode contains invalid amino-acid record(s): " + preview)


def resolve_sequence_mode(
    candidate_pool: Path,
    requested_mode: str | None = "auto",
) -> tuple[RequestedSequenceMode, SequenceMode, list[SequenceRecord]]:
    requested = validate_requested_mode(requested_mode)
    records = read_sequence_pool(Path(candidate_pool))
    if not records:
        raise SequenceModeError("Candidate FASTA contains no sequences.")

    if requested != "auto":
        effective: SequenceMode = requested
        validate_records_for_mode(records, effective)
        return requested, effective, records

    rna_flags = [_is_rna_record(record) for record in records]
    if all(rna_flags):
        return requested, "rna", records
    if not any(rna_flags):
        return requested, "protein", records
    raise SequenceModeError(
        "Auto mode found a mixed or alphabetically ambiguous candidate pool. "
        "Select Protein or RNA explicitly in Advanced Settings."
    )
