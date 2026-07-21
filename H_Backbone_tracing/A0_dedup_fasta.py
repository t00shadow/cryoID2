from pathlib import Path

def dedup_fasta(input_path, output_path=None):
    """
    Remove duplicate sequences from a FASTA file and renumber headers.
    Keeps the first occurrence of each unique sequence.
    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path.with_name(input_path.stem + "_unique.fasta")

    sequences = {}
    current_header = None
    current_seq = []

    # --- read fasta ---
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # save previous sequence
                if current_header and current_seq:
                    seq = "".join(current_seq)
                    sequences.setdefault(seq, current_header)
                current_header = line
                current_seq = []
            else:
                current_seq.append(line)
        # last sequence
        if current_header and current_seq:
            seq = "".join(current_seq)
            sequences.setdefault(seq, current_header)

    # --- write unique fasta ---
    with open(output_path, "w", encoding="utf-8") as f:
        for i, seq in enumerate(sequences.keys(), start=1):
            f.write(f">prediction {i}\n{seq}\n\n")

    print(f"✅ Wrote {len(sequences)} unique sequences to {output_path}")

if __name__ == "__main__":
    import sys
    input_file = "malaria_predictions_FIXED.fasta"
    output_file = "malaria_predictions_FIXED_unique.fasta"
    dedup_fasta(input_file, output_file)
