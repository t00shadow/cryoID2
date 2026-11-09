"""Scaffold-supported graph pruning from Leon's protein tracer.

Ported from origin/leon, commit 8e5d4dc (betterpruning_fix2).  Keep the
finite-cylinder support and global endpoint degree checks, then try local
cycle repairs without reducing total scaffold support.  Spatial indexing
only narrows the voxel candidates; the cylinder test is unchanged.
"""
from __future__ import annotations

from collections import defaultdict, deque
from math import fsum

import numpy as np
from scipy.spatial import cKDTree


def edge_density_score(coords, voxels, i, j, radius):
    """Count scaffold voxels in the finite edge cylinder, divided by length."""
    a, b = coords[i], coords[j]
    segment = b - a
    length = float(np.linalg.norm(segment))
    if length == 0 or len(voxels) == 0:
        return 0.0
    unit = segment / length
    offsets = np.asarray(voxels) - a
    projection = offsets @ unit
    bounded = (projection >= 0) & (projection <= length)
    perpendicular = offsets[bounded] - projection[bounded, None] * unit
    return float(np.count_nonzero(np.linalg.norm(perpendicular, axis=1) <= radius)) / length


def limit_edges_global(coords, edges, scaffold_voxels, *, radius=1.0, max_degree=2):
    """Greedily retain supported edges while checking BOTH endpoint degrees.

    Return retained pairs, the original pairs and their cached support scores.
    Density ties use Leon's descending (score, i, j) ordering.
    """
    if radius <= 0 or not np.isfinite(radius):
        raise ValueError("Scaffold cylinder radius must be finite and positive.")
    if isinstance(max_degree, bool) or int(max_degree) != max_degree or max_degree < 1:
        raise ValueError("Maximum degree must be a positive integer.")
    pairs = {tuple(sorted((int(e[0]), int(e[1])))) for e in edges if e[0] != e[1]}
    voxels = np.asarray(scaffold_voxels)
    tree = cKDTree(voxels) if len(voxels) else None
    scored = []
    scores = {}
    for i, j in sorted(pairs):
        if not (0 <= i < len(coords) and 0 <= j < len(coords)):
            raise ValueError("Graph edge references a missing coordinate.")
        a, b = np.asarray(coords[i], dtype=float), np.asarray(coords[j], dtype=float)
        # This enclosing ball is deliberately larger than the cylinder.
        # Every candidate still undergoes the exact finite-cylinder test.
        candidates = (tree.query_ball_point((a + b) / 2, np.linalg.norm(b - a) / 2 + radius)
                      if tree is not None else [])
        score = edge_density_score(coords, voxels[candidates], i, j, radius)
        scores[(i, j)] = score
        scored.append((score, i, j))
    degree = defaultdict(int)
    kept = []
    for _, i, j in sorted(scored, reverse=True):
        if degree[i] < max_degree and degree[j] < max_degree:
            kept.append((i, j))
            degree[i] += 1
            degree[j] += 1
    return kept, pairs, scores


def repair_cycles(n_nodes, edges, superset_edges, edge_scores, *, max_degree=2):
    """Apply Leon's non-decreasing-support, local cycle-repair rule.

    Removing a cycle edge frees two endpoints.  Refill them using original
    superset edges, rejecting new cycles and degree violations.  Accept the
    first repair whose total support does not decrease.  If none qualifies,
    retain that cycle and stop, as in the upstream fix2 algorithm.  This is
    not an unconditional cycle-breaking operation.

    Explicit edge ordering resolves upstream set-order ambiguity reproducibly.
    Every accepted change removes a cycle and cannot introduce another one.
    """
    if max_degree != 2:
        raise ValueError("Cycle repair requires a degree-two graph.")
    current = {tuple(sorted((int(e[0]), int(e[1])))) for e in edges}
    superset = {tuple(sorted((int(e[0]), int(e[1])))) for e in superset_edges}

    def adjacency(edge_set):
        adj = defaultdict(set)
        for u, v in edge_set:
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def graph_score(edge_set):
        return fsum(edge_scores.get(e, 0.0) for e in sorted(edge_set))

    def path_exists(adj, start, goal):
        queue = deque([start])
        visited = {start}
        while queue:
            node = queue.popleft()
            if node == goal:
                return True
            for neighbor in adj[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return False

    def find_cycle(adj):
        visited = set()
        for start in range(n_nodes):
            if start in visited:
                continue
            stack, component = [start], []
            while stack:
                node = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                stack.extend(sorted(adj[node], reverse=True))
            if len(component) > 2 and all(len(adj[n]) == 2 for n in component):
                return sorted({tuple(sorted((n, other))) for n in component for other in adj[n]})
        return None

    while True:
        base_adj = adjacency(current)
        cycle = find_cycle(base_adj)
        if cycle is None:
            break
        base_score = graph_score(current)
        repaired = None
        for removed in cycle:
            trial = current - {removed}
            adj = defaultdict(set, {node: set(neighbors) for node, neighbors in base_adj.items()})
            u, v = removed
            adj[u].remove(v)
            adj[v].remove(u)
            candidates = [e for e in superset - trial if u in e or v in e]
            candidates.sort(key=lambda e: (-edge_scores.get(e, 0.0), e))
            for x, y in candidates:
                if len(adj[x]) >= max_degree or len(adj[y]) >= max_degree or path_exists(adj, x, y):
                    continue
                trial.add((x, y))
                adj[x].add(y)
                adj[y].add(x)
            if trial != current and graph_score(trial) >= base_score:
                repaired = trial
                break
        if repaired is None:
            break
        current = repaired
    return sorted(current)
