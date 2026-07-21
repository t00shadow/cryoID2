# Define amino acid groups
DEGEN_GROUPS_PROTEIN = {
    "G": ["G", "A", "S", "C", "V", "T", "I"],  # Group 1
    "P": ["P"],                                # Group 2
    "L": ["L", "D", "N", "E", "Q", "M"],      # Group 3
    "K": ["K", "R"],                           # Group 4
    "Y": ["H", "F", "Y"],                      # Group 5
    "W": ["W"],                                # Group 6
}

# Build lookup table
RESIDUE_TO_GROUP = {aa: group for group, aas in DEGEN_GROUPS_PROTEIN.items() for aa in aas}


def convert_to_group_codes(sequence):
    """Convert a sequence string into group codes. Non-AA chars remain as-is."""
    return "".join(RESIDUE_TO_GROUP.get(aa, aa) for aa in sequence)


def process_fasta(input_fasta, output_fasta):
    """Read a FASTA, convert sequences to group codes, and write output FASTA."""
    with open(input_fasta, "r") as infile, open(output_fasta, "w") as outfile:
        for line in infile:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            if line.startswith(">"):
                outfile.write(line + "\n")  # write header as-is
            elif line.isalpha():
                grouped_seq = convert_to_group_codes(line.upper())
                outfile.write(grouped_seq + "\n")
            else:
                # skip lines with non-alphabetic chars
                pass


if __name__ == "__main__":
    print(convert_to_group_codes("MKVIKTLSIINFFIFVTFNIKNESKYSNTFINNAYNMSIRRSMAESKPSTGAGGSAGGSAGGSAGGSAGGSAGGSAGSGDGNGADAEGSSSTPATTTTTKTTTTTTTTNDAEASTSTSSENPNHKNAETNPKGKGEVQEPNQANKETQNNSNVQQDSQTKSNVPPTQDADTKSPTAQPEQAENSAPTAEQTESPELQSAPENKGTGQHGHMHGSRNNHPQNTSDSQKECTDGNKENCGAATSLLNNSSNIASINKFVVLISATLVLSFAIFI"))
    assert(False)


    input_fasta = r"G:\.shortcut-targets-by-id\1VHPX-gidYYLEewDizIvgtImMbleuqE0r\CryoID2\WorkFolder\BackboneTracing\UnetPrediction\real_malaria\3-E1_D3\search_pool_200_700.fasta"
    output_fasta = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\results_3_26_26\pool_converted_to_degen_groups\search_pool_200_700_DEGEN_GROUPS.fasta"
    process_fasta(input_fasta, output_fasta)
    print(f"Converted sequences written to {output_fasta}")