#!/usr/bin/env python3
"""
Submit a hardcoded protein sequence to NCBI BLAST (blastp vs Swiss-Prot) using Biopython
and print the top UniProt accession.

Requires: biopython  (pip install biopython)
Network: calls NCBI BLAST (qblast), so it needs outbound internet.
"""

from __future__ import annotations

import sys
from typing import Optional

from Bio.Blast import NCBIWWW, NCBIXML

# Example: human ACTB (beta-actin) partial sequence
QUERY_SEQ = (
    "MDDDIAALVVDNGSGMCKAGFAGDDAPRAVFPSIVGRPRHQGVMVGMGQKDSYVGDEAQ"
    "SKRGILTLKYPIEHGIVTNWDDMEKIWHHTFYNELRVAPEEHPVLLTEAPLNPKANREL"
)


def run_blast(seq: str) -> Optional[str]:
    """
    Run BLASTP against Swiss-Prot and return the top UniProt accession (if any).
    """
    try:
        # db="swissprot" hits UniProt/Swiss-Prot at NCBI
        handle = NCBIWWW.qblast("blastp", "swissprot", seq)
    except Exception as e:
        print(f"BLAST submission failed: {e}", file=sys.stderr)
        return None

    try:
        blast_record = NCBIXML.read(handle)
    except Exception as e:
        print(f"Failed to parse BLAST XML: {e}", file=sys.stderr)
        return None

    if not blast_record.alignments:
        return None

    top = blast_record.alignments[0]
    # Typical hit.id: 'sp|P60709.2|ACTB_HUMAN'
    parts = top.hit_id.split("|")
    acc = None
    if len(parts) >= 2 and parts[1]:
        acc = parts[1].split(".")[0]  # strip version if present
    else:
        acc = top.hit_id
    return acc


def main():
    print("Submitting BLASTP vs Swiss-Prot via Biopython...")
    acc = run_blast(QUERY_SEQ)
    if acc:
        print(f"Top UniProt accession: {acc}")
    else:
        print("No accession returned.")
        sys.exit(1)


if __name__ == "__main__":
    main()
