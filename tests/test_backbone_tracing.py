from __future__ import annotations

import unittest

import numpy as np

from H_Backbone_tracing.rna.tracing import (
    edge_density_score,
)


def scalar_edge_density_score(
    coords: np.ndarray,
    voxels: np.ndarray,
    i: int,
    j: int,
    radius: float,
) -> tuple[int, float]:
    a, b = coords[i], coords[j]
    ab = b - a
    length = np.linalg.norm(ab)
    unit = ab / length
    supported = 0
    for point in voxels:
        projection = np.dot(point - a, unit)
        if 0 <= projection <= length:
            closest = a + projection * unit
            supported += int(np.linalg.norm(point - closest) <= radius)
    return supported, supported / length


class BackboneTracingTests(unittest.TestCase):
    def test_vectorized_edge_score_matches_scalar_calculation(self) -> None:
        coords = np.asarray(
            [
                [1.0, 2.0, 3.0],
                [4.5, 5.0, 7.0],
            ],
            dtype=np.float32,
        )
        rng = np.random.default_rng(42)
        voxels = rng.uniform(-2.0, 10.0, size=(500, 3))
        radius = 1.25
        expected_count, expected_normalized = scalar_edge_density_score(
            coords,
            voxels,
            0,
            1,
            radius,
        )

        self.assertEqual(
            edge_density_score(coords, voxels, 0, 1, radius, metric="raw_count"),
            expected_count,
        )
        self.assertAlmostEqual(
            edge_density_score(
                coords,
                voxels,
                0,
                1,
                radius,
                metric="normalized_by_length",
            ),
            expected_normalized,
        )


if __name__ == "__main__":
    unittest.main()
