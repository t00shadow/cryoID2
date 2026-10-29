from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from H_Backbone_tracing.rna.tracing import (
    edge_density_score,
)
from H_Backbone_tracing.rna.pipeline import read_component


class RnaQueryGenerationTests(unittest.TestCase):
    def test_vectorized_edge_density_matches_scalar_geometry(self):
        coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]])
        voxels = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.5, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 5.0]]
        )
        self.assertEqual(edge_density_score(coords, voxels, 0, 1, 1.0), 2)
        self.assertAlmostEqual(
            edge_density_score(coords, voxels, 0, 1, 1.0, "normalized_by_length"),
            0.5,
        )

    def test_component_atoms_preserve_historical_pdb_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "component.pdb"
            path.write_text(
                "HETATM    1  C   UNK     1       1.000   0.000   0.000  1.00  0.00           C\n"
                "HETATM    2  C   UNK     1       0.000   0.000   0.000  1.00  0.00           C\n"
                "HETATM    3  C   UNK     1       2.000   0.000   0.000  1.00  0.00           C\n"
                "CONECT    1    2\n"
                "CONECT    1    3\n"
            )
            ordered = read_component(path)
        self.assertEqual([coordinate[2] for coordinate in ordered], [1.0, 0.0, 2.0])


if __name__ == "__main__":
    unittest.main()
