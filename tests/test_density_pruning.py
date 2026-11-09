from collections import Counter
import numpy as np

from H_Backbone_tracing.common.density_pruning import edge_density_score, limit_edges_global, repair_cycles
from H_Backbone_tracing.rna.tracing import edge_density_score as rna_score


def test_finite_cylinder_matches_existing_scaffold_geometry():
    points = np.array([[0., 0., 0.], [0., 0., 4.]])
    voxels = np.array([[0., 0., 1.], [0., .5, 2.], [0., 2., 2.], [0., 0., 5.]])
    assert edge_density_score(points, voxels, 0, 1, 1.) == .5
    assert edge_density_score(points, voxels, 0, 1, 1.) == rna_score(points, voxels, 0, 1, 1., 'normalized_by_length')


def test_indexed_scores_match_all_voxel_scoring():
    rng = np.random.default_rng(917)
    points = rng.uniform(0, 12, (15, 3)).astype(np.float32)
    voxels = rng.uniform(-1, 13, (1500, 3))
    edges = [(i, j) for i in range(len(points)) for j in range(i + 1, len(points))]
    _, _, scores = limit_edges_global(points, edges, voxels)
    assert scores == {e: edge_density_score(points, voxels, *e, 1.) for e in edges}


def test_branch_pruning_checks_both_endpoints_and_support():
    points = np.array([[0., 0., 0.], [2., 0., 0.], [0., 2., 0.], [0., 0., 2.]])
    voxels = np.array([[1., 0., 0.], [1.5, 0., 0.], [0., 1., 0.]])
    edges, _, _ = limit_edges_global(points, [(0, 1), (0, 2), (0, 3)], voxels, radius=.2)
    assert set(edges) == {(0, 1), (0, 2)}
    assert max(Counter(n for edge in edges for n in edge).values()) <= 2


def test_cycle_repair_opens_cycle_and_reconnects_supported_paths():
    cycle = {(0, 1), (1, 2), (0, 2)}
    scores = {edge: 5. for edge in cycle}
    scores.update({(0, 3): 3., (1, 4): 3.})
    result = repair_cycles(5, cycle, set(scores), scores)
    assert set(result) == {(0, 2), (1, 2), (0, 3), (1, 4)}
    assert sum(scores[e] for e in result) >= sum(scores[e] for e in cycle)
    assert max(Counter(n for edge in result for n in edge).values()) <= 2


def test_cycle_without_non_decreasing_repair_is_preserved_and_terminates():
    cycle = {(0, 1), (1, 2), (0, 2)}
    assert repair_cycles(3, cycle, cycle, {e: 1. for e in cycle}) == sorted(cycle)


def test_input_edge_order_does_not_change_density_pruning():
    points = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    edges = [(i, j) for i in range(4) for j in range(i + 1, 4)]
    empty = np.empty((0, 3))
    assert limit_edges_global(points, edges, empty) == limit_edges_global(points, edges[::-1], empty)
