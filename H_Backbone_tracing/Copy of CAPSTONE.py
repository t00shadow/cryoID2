import math
import numpy as np
# import pdbreader    # unused

import openmm as mm
from openmm import app, unit, Platform

# OR-Tools
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

# Matplotlib for 3D plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from joblib import load

###############################################################################
# 1) Utility: Distance in Å (OpenMM Vec3) & Adding Error for testing
###############################################################################

def distance_in_angstroms(pos1, pos2):
    """
    pos1, pos2 : OpenMM Vec3 (with nm units).
    Returns float: distance in Å.
    """
    diff = pos2 - pos1
    dx_nm = diff[0].value_in_unit(unit.nanometer)
    dy_nm = diff[1].value_in_unit(unit.nanometer)
    dz_nm = diff[2].value_in_unit(unit.nanometer)
    dist_nm = math.sqrt(dx_nm*dx_nm + dy_nm*dy_nm + dz_nm*dz_nm)
    return dist_nm * 10.0

def gerror(data, std_dev, mean=0):
    """
    mean is the mean of the noise, keeping it at 0 keeps it centered
    std_dev is 2Å for testing, you can change
    """
    noise = np.random.normal(mean,std_dev,data.shape)
    noisy_data = data + noise
    return noisy_data

###############################################################################
# 2) Reading PDB Coordinates
###############################################################################


def load_pdb_coordinates(pdb_file, gerrorTesting=0, std_dev=0.0):
    """
    Reads a PDB file and returns an Nx3 numpy array of coordinates (in Å).
    
    Optionally, Gaussian noise can be added to the coordinates if gerrorTesting is 1.
    
    Parameters:
      pdb_file      : Path to the PDB file.
      gerrorTesting : (Optional) If set to 1, Gaussian noise is added. Default is 0.
      std_dev       : (Optional) Standard deviation for the Gaussian noise (in Å). Default is 0.0.
    
    Returns:
      coords : Nx3 numpy array of coordinates (with noise added if requested).
    """
    x_list, y_list, z_list = [], [], []

    with open(pdb_file, 'r') as f:
        for line in f:
            record = line[0:6].strip()
            if record in ("ATOM", "HETATM"):
                x_val = float(line[30:38])
                y_val = float(line[38:46])
                z_val = float(line[46:54])
                x_list.append(x_val)
                y_list.append(y_val)
                z_list.append(z_val)

    coords = np.column_stack((x_list, y_list, z_list))  # shape (N, 3)
    
    if gerrorTesting == 1:
        coords = gerror(coords, std_dev)
    
    return coords


###############################################################################
# 3) Build Distance Matrix (No segments)
###############################################################################

def build_distance_matrix(coords, min_dist=4.0, max_dist=8.0):
    N = len(coords)
    LARGE_COST = 1e6
    dist_matrix = np.zeros((N, N), dtype=np.float64)

    for i in range(N):
        for j in range(N):
            if i == j:
                dist_matrix[i, j] = LARGE_COST
            else:
                dx = coords[i,0] - coords[j,0]
                dy = coords[i,1] - coords[j,1]
                dz = coords[i,2] - coords[j,2]
                dist = math.sqrt(dx*dx + dy*dy + dz*dz)
                if dist < min_dist or dist > max_dist:
                    dist_matrix[i,j] = LARGE_COST
                else:
                    dist_matrix[i,j] = dist
    return dist_matrix

###############################################################################
# 4) OR-Tools TSP (GUIDED_LOCAL_SEARCH)
###############################################################################

def solve_tsp_ortools(dist_matrix, time_limit=30):
    N = len(dist_matrix)
    manager = pywrapcp.RoutingIndexManager(N, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node   = manager.IndexToNode(to_index)
        cost = dist_matrix[from_node][to_node]
        return int(cost * 1000)

    cb_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(cb_idx)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    search_params.time_limit.seconds = time_limit

    solution = routing.SolveWithParameters(search_params)
    if not solution:
        print("No TSP solution found.")
        return None

    # Extract cycle
    route_cycle = []
    index = routing.Start(0)
    while not routing.IsEnd(index):
        route_cycle.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    return route_cycle

###############################################################################
# 5) Multi-run with Node Permutations
###############################################################################

def permute_dist_matrix(dist_matrix, seed):
    np.random.seed(seed)
    N = len(dist_matrix)
    perm   = np.random.permutation(N)
    invp   = np.argsort(perm)

    dist_perm = np.zeros_like(dist_matrix)
    for i in range(N):
        for j in range(N):
            dist_perm[i,j] = dist_matrix[perm[i], perm[j]]

    return dist_perm, perm, invp

def solve_tsp_multi_seed(dist_matrix, n_runs=3, time_limit=5):
    best_route = None
    best_cost  = float('inf')
    N = len(dist_matrix)

    for seed in range(1, n_runs+1):
        dist_perm, perm, invp = permute_dist_matrix(dist_matrix, seed)
        route_perm = solve_tsp_ortools(dist_perm, time_limit=time_limit)
        if route_perm is None:
            continue

        # Convert back to original indices
        route_original = [perm[r] for r in route_perm]

        # Compute cost in the original matrix
        cost = 0.0
        for i in range(len(route_original)-1):
            cost += dist_matrix[route_original[i], route_original[i+1]]
        cost += dist_matrix[route_original[-1], route_original[0]]

        print(f"Seed={seed}, route_original={route_original}, cost={cost:.2f}")
        if cost < best_cost:
            best_cost = cost
            best_route = route_original

    return best_route

###############################################################################
# 6) Filter Bonds and Connect Ends
###############################################################################

def filter_route_by_bond_length(route, coords, max_bond_length=7.0):
    allowed_pairs = []
    for i in range(len(route) - 1):
        A = route[i]
        B = route[i+1]
        dx = coords[A,0] - coords[B,0]
        dy = coords[A,1] - coords[B,1]
        dz = coords[A,2] - coords[B,2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        if dist <= max_bond_length:
            allowed_pairs.append((A, B))

    # Connect the first and last points if within the bond length
    first, last = route[0], route[-1]
    dx = coords[first,0] - coords[last,0]
    dy = coords[first,1] - coords[last,1]
    dz = coords[first,2] - coords[last,2]
    dist = math.sqrt(dx*dx + dy*dy + dz*dz)
    if dist <= max_bond_length:
        allowed_pairs.append((last, first))

    return allowed_pairs

###############################################################################
# 7) Build Minimal OpenMM System (Only for Allowed Pairs)
###############################################################################

def build_minimal_system(coords, route, allowed_pairs):
    system = mm.System()
    topology = app.Topology()
    chain = topology.addChain("A")

    node_to_atom_idx = {}
    positions = []

    for atom_idx, node_idx in enumerate(route):
        residue = topology.addResidue("PHO", chain)
        atom = topology.addAtom(f"P_{node_idx}", app.Element.getByAtomicNumber(15), residue)
        system.addParticle(30.97 * unit.dalton)

        node_to_atom_idx[node_idx] = atom_idx

        x_nm = coords[node_idx,0]/10.0
        y_nm = coords[node_idx,1]/10.0
        z_nm = coords[node_idx,2]/10.0
        positions.append(mm.Vec3(x_nm,y_nm,z_nm)*unit.nanometer)

    bond_force = mm.HarmonicBondForce()
    system.addForce(bond_force)

    k_bond = 1.0 * unit.kilocalories_per_mole / (unit.angstrom**2)

    for (A, B) in allowed_pairs:
        idxA = node_to_atom_idx[A]
        idxB = node_to_atom_idx[B]
        dx   = coords[A,0] - coords[B,0]
        dy   = coords[A,1] - coords[B,1]
        dz   = coords[A,2] - coords[B,2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)
        bond_force.addBond(idxA, idxB, dist*unit.angstrom, k_bond)

    rest_force = mm.CustomExternalForce(
        "0.5 * k_rest * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)"
    )
    rest_force.addPerParticleParameter("k_rest")
    rest_force.addPerParticleParameter("x0")
    rest_force.addPerParticleParameter("y0")
    rest_force.addPerParticleParameter("z0")
    system.addForce(rest_force)

    k_rest = 5.0 * unit.kilocalories_per_mole / (unit.angstrom**2)
    for atom_idx, node_idx in enumerate(route):
        x0_nm = coords[node_idx,0]/10.0
        y0_nm = coords[node_idx,1]/10.0
        z0_nm = coords[node_idx,2]/10.0
        rest_force.addParticle(atom_idx, [
            k_rest,
            x0_nm*unit.nanometer,
            y0_nm*unit.nanometer,
            z0_nm*unit.nanometer
        ])

    return system, topology, positions

###############################################################################
# 8) Minimize & MD
###############################################################################

def minimize_and_score(system, topology, positions, nsteps=500):
    integrator = mm.VerletIntegrator(0.001*unit.picoseconds)
    platform = Platform.getPlatformByName("Reference")
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)

    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
    simulation.step(nsteps)

    state = simulation.context.getState(getEnergy=True, getPositions=True)
    final_energy = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    return final_energy, state.getPositions(asNumpy=True)

###############################################################################
# 9) 3D Plot, Only Allowed Pairs
###############################################################################

def plot_3d_path(coords, route, allowed_pairs):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    path_coords = coords[route]
    ax.scatter(path_coords[:,0], path_coords[:,1], path_coords[:,2],
               color='blue', s=40, marker='o')

    for (A, B) in allowed_pairs:
        x_vals = [coords[A,0], coords[B,0]]
        y_vals = [coords[A,1], coords[B,1]]
        z_vals = [coords[A,2], coords[B,2]]
        ax.plot(x_vals, y_vals, z_vals, color='red')

    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")
    ax.set_zlabel("Z (Å)")
    plt.title("Phosphate TSP Path (Filtered Bonds)")
    plt.show()

def save_path_as_pdb(coords, path, pairs, output_file="Large_Error_PDBOUT.pdb"):
    with open(output_file, "w") as pdb_file:
        pdb_file.write("REMARK  TSP Path Saved as PDB\n")
        pdb_file.write("REMARK  Generated by RNA TSP Pipeline\n")

        atom_mapping = {idx: i + 1 for i, idx in enumerate(path)}

        for atom_number, idx in enumerate(path, start=1):
            x, y, z = coords[idx]
            pdb_file.write(
                f"ATOM  {atom_number:5d}  P   TSP A{atom_number:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           P\n"
            )

        pdb_file.write("REMARK  Connectivity\n")
        for A, B in pairs:
            if A in atom_mapping and B in atom_mapping:
                pdb_file.write(f"CONECT{atom_mapping[A]:5d}{atom_mapping[B]:5d}\n")

        pdb_file.write("TER\nEND\n")
    print(f"TSP path saved as {output_file}")

###############################################################################
# 10) ANGLE-BASED FEATURE EXTRACTION (6 features)
###############################################################################

def angle_degrees(coords, a, b, c):
    """
    Computes the angle at 'b' formed by (a->b->c) in degrees.
    """
    v1 = coords[a] - coords[b]
    v2 = coords[c] - coords[b]
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-12 or norm2 < 1e-12:
        return 0.0
    dot = np.dot(v1, v2)
    cos_theta = dot / (norm1 * norm2)
    cos_theta = max(min(cos_theta, 1.0), -1.0)
    return math.degrees(math.acos(cos_theta))

def local_angle_sum(route, coords, idx):
    """
    Sum angles at idx-1, idx, idx+1 if valid
    Each angle is formed by (k-1, k, k+1).
    """
    n = len(route)
    total_ang = 0.0

    # angle at (idx-1)
    if idx-1 > 0:
        a = route[idx-2]
        b = route[idx-1]
        c = route[idx]
        total_ang += angle_degrees(coords, a, b, c)

    # angle at (idx)
    if 0 < idx < n-1:
        a = route[idx-1]
        b = route[idx]
        c = route[idx+1]
        total_ang += angle_degrees(coords, a, b, c)

    # angle at (idx+1)
    if idx+1 < n-1:
        a = route[idx]
        b = route[idx+1]
        c = route[idx+2]
        total_ang += angle_degrees(coords, a, b, c)

    return total_ang


def extract_move_features(route, i, j, dist_matrix, coords):
    """
    6-feature version:
    [i, j, old_sum_dist, new_sum_dist, old_sum_ang, new_sum_ang]
    Must match how you trained your ML model.
    """
    n = len(route)

    # OLD local distance around i
    old_i_minus = dist_matrix[route[i-1], route[i]] if i>0 else 0.0
    old_i_plus  = dist_matrix[route[i], route[i+1]] if i<n-1 else 0.0
    # OLD local distance around j
    old_j_minus = dist_matrix[route[j-1], route[j]] if j>0 else 0.0
    old_j_plus  = dist_matrix[route[j], route[j+1]] if j<n-1 else 0.0
    old_sum_dist = old_i_minus + old_i_plus + old_j_minus + old_j_plus

    # Swap
    new_route = route[:]
    new_route[i], new_route[j] = new_route[j], new_route[i]

    new_i_minus = dist_matrix[new_route[i-1], new_route[i]] if i>0 else 0.0
    new_i_plus  = dist_matrix[new_route[i], new_route[i+1]] if i<n-1 else 0.0
    new_j_minus = dist_matrix[new_route[j-1], new_route[j]] if j>0 else 0.0
    new_j_plus  = dist_matrix[new_route[j], new_route[j+1]] if j<n-1 else 0.0
    new_sum_dist = new_i_minus + new_i_plus + new_j_minus + new_j_plus

    # OLD angles near i,j
    old_sum_ang = local_angle_sum(route, coords, i) + local_angle_sum(route, coords, j)
    # NEW angles near i,j
    new_sum_ang = local_angle_sum(new_route, coords, i) + local_angle_sum(new_route, coords, j)

    return [i, j, old_sum_dist, new_sum_dist, old_sum_ang, new_sum_ang]

###############################################################################
# 11) MC with the 6-feature extraction
###############################################################################

def compute_route_cost(route, dist_matrix):
    """
    Summation of dist_matrix along route as a cycle (simple).
    """
    cost = 0.0
    for k in range(len(route)-1):
        cost += dist_matrix[route[k], route[k+1]]
    cost += dist_matrix[route[-1], route[0]]
    return cost

def ml_guided_monte_carlo(initial_route, dist_matrix, coords, ml_model,
                          n_iters=1000, temperature=300.0, skip_threshold=10.0):
    """
    6-feature MC refinement:
    1) random swap
    2) extract 6-feature (distance+angle)
    3) predict delta cost
    4) skip / do real cost / Metropolis
    """
    import random

    kB = 0.001987
    beta = 1.0/(kB * temperature)

    route = initial_route[:]
    best_route = route[:]
    current_cost = compute_route_cost(route, dist_matrix)
    best_cost = current_cost
    n = len(route)

    for step in range(n_iters):
        i, j = random.sample(range(n), 2)
        feats = extract_move_features(route, i, j, dist_matrix, coords)
        predicted_delta = ml_model.predict([feats])[0]

        if predicted_delta > skip_threshold:
            continue  # skip if predicted too large

        # actual cost check
        new_route = route[:]
        new_route[i], new_route[j] = new_route[j], new_route[i]
        new_cost = compute_route_cost(new_route, dist_matrix)
        delta_cost = new_cost - current_cost

        if delta_cost < 0:
            route = new_route
            current_cost = new_cost
            if current_cost < best_cost:
                best_route = route[:]
                best_cost = current_cost
        else:
            import math
            accept_prob = math.exp(-beta*delta_cost)
            if random.random() < accept_prob:
                route = new_route
                current_cost = new_cost
                if current_cost < best_cost:
                    best_route = route[:]
                    best_cost = current_cost

    return best_route, best_cost

#### Accuracy

def extract_edges_from_route(route):
    """
    Given a route (list of original indices) representing a cyclic order,
    extract a set of edges. Each edge is represented as a tuple of sorted node indices.
    """
    edges = set()
    n = len(route)
    for i in range(n):
        j = (i + 1) % n  # ensures cyclic connectivity (last connects to first)
        edge = tuple(sorted((route[i], route[j])))
        edges.add(edge)
    return edges

def compare_graphs(reference_connections, generated_route):
    """
    Compares the reference graph connectivity to that generated by the TSP route.

    Parameters:
      reference_connections: Iterable of edges (tuples) representing the correct connectivity.
      generated_route      : TSP route in the original index space.

    Returns:
      accuracy: the fraction of reference edges that are present in the generated route.
    """
    generated_edges = extract_edges_from_route(generated_route)
    reference_edges = {tuple(sorted(edge)) for edge in reference_connections}
    matching_edges = generated_edges.intersection(reference_edges)
    accuracy = len(matching_edges) / len(reference_edges) if reference_edges else 0

    print(f"Total Reference Edges: {len(reference_edges)}")
    print(f"Total Generated Edges: {len(generated_edges)}")
    print(f"Matching Edges: {len(matching_edges)}")
    return accuracy



###############################################################################
# Main
###############################################################################

def trace_backbone(molecule_type: str, max_bond_length: float | None = None, *, input_file: str, output_file: str, joblib_file: str | None = None):
    """
    Traces the backbone of either a nucleic acid or protein and returns a path to the result.

    Parameters:
        molecule_type (str):               The type of molecule to trace a backbone for. Options are "nucleic_acid" or "protein"
        max_bond_length (float | None):    (Optional) Max bond length for the corresponding molecule type. Defaults are 12 angstroms for nucleic acid and 6 angstroms for protein.
        input_file (str):                  Path of the input file (PDB format).
        output_file (str):                 Path to write the output file (PDB format).
        joblib_file (str | None):          (Optional) ML model file for MC refinement.

    Returns:
        output_file (str):                 Path of the output file (PDB format). Yes, it's the same as the parameter.
    """

    coords = load_pdb_coordinates(input_file, 0)
    print(f"Total number of coordinates: {len(coords)}")
    
    dist_matrix = build_distance_matrix(coords, min_dist=3.0, max_dist=13.0)
    best_cycle = solve_tsp_multi_seed(dist_matrix, n_runs=10, time_limit=10)

    if best_cycle is None:
        raise RuntimeError("No TSP solution found.")
    
    # Ensure the route is cyclic (remove duplicate endpoint if present)
    route_path = best_cycle[:-1] if best_cycle[-1] == best_cycle[0] else best_cycle

    # Attempt to load ML model for MC refinement
    if joblib_file is not None:
        try:
            ml_model = load(joblib_file)
            print("Loaded ML model with 6-feature extraction")

            # Perform MC using the 6-feature code
            refined_route, refined_cost = ml_guided_monte_carlo(
                route_path, dist_matrix, coords, ml_model,
                n_iters=10000, temperature=300.0, skip_threshold=15.0
            )
            print(f"Refined route cost: {refined_cost:.2f}")
            route_path = refined_route

        except FileNotFoundError:
            print(f"ML model {joblib_file} not found. Skipping MC refinement.")

    # Define reference connectivity based on the sequential order (0, 1, 2, ..., n-1)
    n = len(coords)
    reference_connections = [(i, i+1) for i in range(n-1)]
    # Add the cyclic edge connecting the last and first atoms
    reference_connections.append((n-1, 0))

    # Compute and print bond order accuracy
    try:
        bond_order_accuracy = compare_graphs(reference_connections, refined_route)
        print(f"Bond order accuracy: {bond_order_accuracy*100:.1f}%")
    except:
        print("Didnt load ML model earlier (the .joblib file), no bond order accuracy available.")

    # Continue with filtering bonds and simulation building
    if max_bond_length is None:
        if molecule_type == "nucleic_acid":
            max_bond_length = 12.0    # RNA
        elif molecule_type == "protein":
            max_bond_length = 6.0    # protein
    print(f"Sanity check: max_bond_length = {max_bond_length}")
    allowed_pairs = filter_route_by_bond_length(route_path, coords, max_bond_length)

    system, topology, positions = build_minimal_system(coords, route_path, allowed_pairs)
    final_energy, final_positions = minimize_and_score(system, topology, positions, nsteps=500)

    # plot_3d_path(coords, route_path, allowed_pairs)     # Can disable if using ChimeraX
    save_path_as_pdb(coords, route_path, allowed_pairs, output_file=output_file)
    print(f"Final energy: {final_energy:.3f} kcal/mol")



if __name__ == "__main__":
    # "Ground truth" PDB files
    # input_file = "C:/Users/david/Desktop/Capstone/Large.pdb"    
    # input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\test_files\6hlq_pho_prediction.pdb"

    # PDB files from clustering Unet predictions
    input_file = r"C:\Users\noelu\Python Projects\cryoID_v2\Z_Developmental_Clustering\output\meanshift_clusters_threshold_0_25.pdb"
    
    # joblib_file = "my_trained_model.joblib"      #* used to refine the backbone after TSP

    output_file = r"C:\Users\noelu\Python Projects\cryoID_v2\H_Backbone_tracing\tracing_outputs\protein_backbone.pdb"

    # Note: handle file I/O stuff like the naming of input/output files OUTSIDE of the trace_backbone() function
    trace_backbone("protein", input_file=input_file, output_file=output_file)