# Various tests to verify PDB and CIF parsing functions


###* Test 1: custom cif parser vs biopython
# Comparing my own cif parsing function vs biopython
# CONCLUSION: both have same results so mine is fine, but just use the biopython one from now on. 
# Had my own custom one earlier bc was part of a larger script that did other stuff and wanted a smaller executable (gonna package everything into a desktop app), 
# buttt i think all of it can be done in biopython so just switch over

import numpy as np

cif_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\8hk3.cif"

from parse_cif_file_v3_4__FINAL import load_cif_lines, parse_ca_atoms
cif = load_cif_lines(cif_file)
ca_coords_my_custom_fxn, info = parse_ca_atoms(cif)
# print(ca_coords_my_custom_fxn)

from R_A0_utils import parse_calpha_CIF
ca_coords_biopython = parse_calpha_CIF(cif_file)
# print(ca_coords_biopython)

print("TEST 1")
print("Is my function the same as biopython?", np.array_equal(ca_coords_my_custom_fxn, ca_coords_biopython))




###* Test 2: pdb vs cif atom order (both using biopython)

pdb_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\8HK3.pdb"

# from Bio.PDB import PDBParser
# parser = PDBParser()
# structure = parser.get_structure("structure_id", pdb_file)
# for atom in structure.get_atoms():
#     print(atom.get_name(), atom.get_coord())
#     # print(atom.get_parent().get_resname(), atom.get_parent().get_id())     #NOTE can get residues from Atom module
# # for residue in structure.get_residues():
# #     print(residue.get_resname(), residue.get_id())

# from Bio.PDB import MMCIFParser
# parser = MMCIFParser()
# structure = parser.get_structure("structure_id", cif_file)
# for model in structure:
#     for chain in model:
#         for residue in chain:
#             for atom in residue:
#                 print(atom.get_name(), atom.get_coord())
#                 # print(atom.get_parent().get_resname(), atom.get_parent().get_id())     #NOTE can get residues from Atom module

from R_A0_utils import parse_calpha_PDB
ca_coords_PDB = parse_calpha_PDB(pdb_file)
# print(ca_coords_PDB)

from R_A0_utils import parse_calpha_CIF
ca_coords_CIF = parse_calpha_CIF(cif_file)
# print(ca_coords_CIF)

print("TEST 2")
print("Is atom order the same for PDB and CIF?", np.array_equal(ca_coords_PDB, ca_coords_CIF))



# from R_A0_utils import parse_calpha_CIF_ss_filter
# ca_coords_CIF_helix = parse_calpha_CIF_ss_filter(cif_file, ss_filter="H")
