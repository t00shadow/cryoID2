import numpy as np
import networkx as nx
from parse_PDB_file__FINAL import extract_ca_and_ss  # import your previous script

def build_helix_graph(coords, ss, info, max_dist=4.0):
    """
    Build a graph of CA atoms labeled 'H' where edges connect atoms that could
    belong to the same helix (spatially close and roughly sequential along chain).
    """
    helix_indices = np.where(ss == 'H')[0]
    G = nx.Graph()

    for idx in helix_indices:
        G.add_node(idx, chain=info[idx][0], resseq=info[idx][1], coord=coords[idx])

    for i in helix_indices:
        for j in helix_indices:
            if i >= j:
                continue
            chain_i, res_i = info[i]
            chain_j, res_j = info[j]

            if chain_i != chain_j:
                continue
            # Only connect atoms that are consecutive or close in residue number
            if abs(res_i - res_j) <= 2:
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist <= max_dist:
                    G.add_edge(i, j, weight=dist)

    return G

def trace_helices(coords, ss, info):
    """
    Trace helices by finding connected components in the helix graph.
    Returns a list of helices, each as a list of indices into coords/info.
    """
    G = build_helix_graph(coords, ss, info)
    components = list(nx.connected_components(G))

    helices = []
    for comp in components:
        sorted_comp = sorted(comp, key=lambda idx: info[idx][1])
        helices.append(sorted_comp)
    return helices

if __name__ == "__main__":
    pdb_file = r"C:\Users\noelu\programming\python_projects\cryoID_v2\H_Backbone_tracing\small_prototypes\8HK3.pdb"
    
    # Extract CA positions and secondary structure
    coords, ss = extract_ca_and_ss(pdb_file)
    
    # Prepare info list (chain, resseq)
    from Bio.PDB import PDBParser
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("S", pdb_file)
    info = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if 'CA' in residue:
                    info.append((chain.id, residue.id[1]))

    # Trace helices
    helices_traced = trace_helices(coords, ss, info)

    # Print results
    print(f"Found {len(helices_traced)} helices:")
    for i, helix in enumerate(helices_traced, 1):
        chain = info[helix[0]][0]
        start_res = info[helix[0]][1]
        end_res = info[helix[-1]][1]
        print(f"Helix {i}: chain {chain}, residues {start_res}-{end_res}, length {len(helix)}")
