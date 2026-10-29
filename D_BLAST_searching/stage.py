"""Stable production API for stage D (sequence identification)."""

from .blast_tools import (
    blast_executable,
    bundled_blast_bin_dir,
    validate_blast_bin_dir,
)
from .protein_search import (
    BlastSearchResult,
    run_protein_blast_search,
)
from .rna_ranking import RnaRankingResult, rank_rna_candidates
from .sequence_mode import (
    SequenceModeError,
    resolve_sequence_mode,
    validate_requested_mode,
)
from .sequence_ranking import (
    protein_ranking_rows,
    write_sequence_ranking,
)
from .sequence_search import SequenceRecord, read_sequence_pool

__all__ = [
    "BlastSearchResult",
    "RnaRankingResult",
    "SequenceModeError",
    "SequenceRecord",
    "blast_executable",
    "bundled_blast_bin_dir",
    "protein_ranking_rows",
    "rank_rna_candidates",
    "read_sequence_pool",
    "resolve_sequence_mode",
    "run_protein_blast_search",
    "validate_blast_bin_dir",
    "validate_requested_mode",
    "write_sequence_ranking",
]
