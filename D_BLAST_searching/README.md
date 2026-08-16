# D - Sequence identification

Production entry point: `D_BLAST_searching.stage`.

- `reduced_alphabet/`: protein reduced-code BLASTP search and aggregation.
- `protein_search.py`: protein search result assembly.
- `rna_ranking.py`: reverse-complement-collapsed hard-maximin RNA ranking.
- `sequence_mode.py`: Auto/Protein/RNA FASTA validation and resolution.
- `sequence_ranking.py`: run-specific TSV serialization.
- `blast_tools.py`: BLAST+ validation, configuration, and autodetection.

This stage owns sequence ranking but not trace generation or structure lookup.
