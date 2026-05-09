import os

def replace_hetatm_c_with_ca(input_pdb, output_pdb):
    """
    Reads a PDB file and writes a modified version where:
    - 'HETATM' records are replaced with 'ATOM'
    - Atom name 'C' is replaced with 'CA'
    """
    with open(input_pdb, "r") as infile, open(output_pdb, "w") as outfile:
        for line in infile:
            # Only process lines that are ATOM or HETATM
            if line.startswith(("ATOM", "HETATM")):
                # Replace HETATM -> ATOM          #! uncomment to keep HETATM so can use CONECT rows
                if line.startswith("HETATM"):
                    line = "ATOM  " + line[6:]  # keep formatting

                # Extract atom name (columns 13–16, i.e. line[12:16])
                atom_name = line[12:16].strip()
                if atom_name == "C":      #! change this to CA and below to C if want to use C instead. extend this fxn so it's easier to swap atom names, and atom/hetatm type. input should be just CA and ATOM for example and it switches all cases to that, actually make a SECOND FXN that does that, and keep this (ehh lowk dont even need this tho). Maybe split it into 2 functions?
                    # Replace with 'CA' and keep alignment
                    line = line[:12] + " CA " + line[16:]

            outfile.write(line)


if __name__ == "__main__":
    # input_pdb = "input.pdb"
    # input_pdb = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\clustering_v2\emd_34843_agglomerative_cd_4.2_combined_fullGraph_20250927_235412.pdb"
    # input_pdb = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\outputs\one_off_files_for_michael\emd_34843_agglomerative_cd_4.2_combined_fullGraph_20250927_235412_WITH_RESIDUES.pdb"
    input_pdb = r"C:\Users\noelu\programming\python_projects\cryoID_v2\traced_backbone3_updated_residues.pdb"
    # output_pdb = "output_with_CA.pdb"
    base, ext = os.path.splitext(os.path.basename(input_pdb))
    output_pdb = f"{base}_standardized_format{ext}"
    # output_pdb = os.path.join("H_Backbone_tracing/outputs/one_off_files_for_michael", output_pdb)
    replace_hetatm_c_with_ca(input_pdb, output_pdb)
    print(f"Modified PDB written to {output_pdb}")
