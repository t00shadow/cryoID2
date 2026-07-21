from pathlib import Path

# Dictionary to convert 3-letter amino acid codes to 1-letter codes
d3to1 = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

# TODO: consider adding this to A0_utils.py
#^ This function is useful beyond just pdbtofasta, so consider adding these all to the same utils file, and just separate their categories with comments or prefixes in fxn names
def find_pdb_files(directory, pattern="*.pdb"):
    """
    Find all PDB files in a directory matching the given pattern using pathlib.
    
    Parameters:
        directory (str or Path): The directory to search in.
        pattern (str): The glob pattern to match files.
    
    Returns:
        list of Path: Paths to matching PDB files.
    """
    directory = Path(directory)
    
    return list(directory.glob(pattern))


# Helper function
# TODO: modify this so it accepts a pattern too, might need two patterns? Left and right, leaving middle leftover.
# TODO: consider renaming to smth like "extract_index_from_file_name" or "extract_file_index_from_name" so easier to understand if move this to A0_utils.py
def extract_number_from_file_name(path):
    """
    Extract the number between the second underscore and first period.

    Parameters:
        path (pathlib.Path or str): The path to the file whose number should be extracted.
    
    Returns:
        int: The extracted number from the filename.

    Example: step3_output_10.pdb_#43.pdb -> 10
    """
    path = Path(path)    # lmao this looks so goofy
    parts = path.stem.split('_')  # path.stem removes the suffix
    # parts = ['step3', 'output', '10.pdb', '#43']  <- example
    # the number is usually in the 3rd element (index 2)
    num_part = parts[2]  
    # remove non-digit characters if any (like '.pdb')
    num_str = ''.join(filter(str.isdigit, num_part))
    
    return int(num_str)


def sort_pdb_files_by_natural_number(pdb_files):
    """
    Sort a list of PDB file paths numerically based on the number in the filename (natural number sorting).

    The function assumes that each filename contains an integer between underscores,
    e.g., 'step3_output_10.pdb_#43.pdb'. Sorting is done based on this number (10 in this example).

    Parameters:
        pdb_files (list of pathlib.Path or str): List of PDB file paths to sort.

    Returns:
        list of pathlib.Path: Sorted list of PDB file paths in ascending numerical order.
    """
    return sorted(pdb_files, key=extract_number_from_file_name)


# Helper function
def pdb_to_fasta(pdb_file):
    """
    Convert a single PDB file to a FASTA string of 1-letter amino acid codes.
    
    Parameters:
        pdb_file (str or Path): Path to the PDB file.
    
    Returns:
        str: Sequence as 1-letter amino acid codes.
    """
    sequence = []
    pdb_file = Path(pdb_file)
    with pdb_file.open('r') as f:
        for line in f:
            if line.startswith("ATOM") and line[13:15].strip() == "CA":  # only alpha carbons
                resname = line[17:20].strip()
                sequence.append(d3to1.get(resname, "X"))  # use 'X' for unknown residues
    
    return "".join(sequence)


def write_fasta_file(pdb_files, output_file="output.fasta", verbose=False, include_reverse=False):
    """
    Convert multiple PDB files to FASTA and write them to a file.
    
    Parameters:
        pdb_files (list of Path): List of PDB file paths.
        output_file (str or Path): Path to write the FASTA sequences.
        verbase (boolean): Flag to turn on successful print message.
        include_reverse (boolean): Flag to include reverse of each FASTA sequence (since direction may be unknown)
    """
    output_file = Path(output_file)
    with output_file.open('w') as out_f:
        for i, pdb_file in enumerate(pdb_files, start=1):
            fasta_seq = pdb_to_fasta(pdb_file)
            out_f.write(f">prediction {i}\n")
            out_f.write(fasta_seq + "\n")
            if include_reverse:     # Could also put all the reversed ones at the end. In that case move this out of the for loop. Make a second for loop under the if condition with the exact same loop conditions.
                out_f.write(f">prediction {i} REVERSED\n")    # If need a new line btwn original and revesed, add it at the start of this string. 
                out_f.write(fasta_seq[::-1] + "\n")
            out_f.write("\n")
    
    if verbose:
        print(f"Wrote {len(pdb_files)} sequences to {output_file}")


# Example usage
if __name__ == "__main__":
    folder = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\10_7_25\overwritten_here_but_recovered_from_chimerax_session"
    pattern = "step3_output_*.pdb"
    output_file = "malaria_predictions.fasta"
    # output_file_plus_reversed = "malaria_predictions_plus_reversed.fasta"    #! unneeded
    
    pdb_files = find_pdb_files(folder, pattern)
    pdb_files = sort_pdb_files_by_natural_number(pdb_files)
    for file in pdb_files:
        print(file)    # file name
    print("=" * 50)

    print(f"Found {len(pdb_files)} files.")
    write_fasta_file(pdb_files, output_file=output_file, verbose=True)
    # write_fasta_file(pdb_files, output_file=output_file_plus_reversed, verbose=True, include_reverse=True)    #! unneeded
    print(f"Converted {len(pdb_files)} PDB files to FASTA format.")
