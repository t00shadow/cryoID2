from __future__ import annotations

import tempfile
import shutil
import unittest
from pathlib import Path
from unittest import mock

import gemmi
import mrcfile
import numpy as np


from I_Structure_fitting.protein.core import map_utils


class MapInfoPlacementTests(unittest.TestCase):
    def _write_map(
        self,
        path: Path,
        *,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        starts: tuple[int, int, int] = (0, 0, 0),
        voxel: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        with mrcfile.new(path, overwrite=True) as mrc:
            mrc.set_data(np.zeros((4, 5, 6), dtype=np.float32))
            mrc.voxel_size = voxel
            mrc.header.origin.x = origin[0]
            mrc.header.origin.y = origin[1]
            mrc.header.origin.z = origin[2]
            mrc.header.nxstart = starts[0]
            mrc.header.nystart = starts[1]
            mrc.header.nzstart = starts[2]

    def test_structured_origin_is_read_and_takes_precedence_over_starts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "explicit_origin.mrc"
            self._write_map(
                path,
                origin=(203.0, 173.0, 191.0),
                starts=(7, 8, 9),
                voxel=(1.5, 2.0, 2.5),
            )

            with mrcfile.open(path, permissive=True) as mrc:
                self.assertEqual(map_utils._safe_get_origin(mrc), (203.0, 173.0, 191.0))
                info = map_utils.MapInfo(mrc)

            self.assertEqual(info.placement_source, "origin")
            np.testing.assert_allclose([info.eff_x0, info.eff_y0, info.eff_z0], [203.0, 173.0, 191.0])
            x, y, z = info.grid_to_world_many(np.array([0]), np.array([0]), np.array([0]))
            np.testing.assert_allclose([x[0], y[0], z[0]], [203.0, 173.0, 191.0])

    def test_start_indices_place_map_when_explicit_origin_is_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "start_placement.mrc"
            self._write_map(path, starts=(7, 8, 9), voxel=(1.5, 2.0, 2.5))

            with mrcfile.open(path, permissive=True) as mrc:
                info = map_utils.MapInfo(mrc)

            self.assertEqual(info.placement_source, "start")
            np.testing.assert_allclose([info.eff_x0, info.eff_y0, info.eff_z0], [10.5, 16.0, 22.5])

    def test_world_array_coordinate_round_trip_uses_explicit_origin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "round_trip.mrc"
            self._write_map(path, origin=(-12.0, 4.0, 30.0), voxel=(1.5, 2.0, 2.5))

            with mrcfile.open(path, permissive=True) as mrc:
                info = map_utils.MapInfo(mrc)

            ix = np.array([0.0, 2.0, 5.0])
            iy = np.array([0.0, 3.0, 4.0])
            iz = np.array([0.0, 1.0, 3.0])
            x, y, z = info.grid_to_world_many(ix, iy, iz)
            coords = info.world_to_array_frac_many(np.column_stack([x, y, z]))
            np.testing.assert_allclose(coords[info.axX], ix)
            np.testing.assert_allclose(coords[info.axY], iy)
            np.testing.assert_allclose(coords[info.axZ], iz)


class TransformedStructureWriteTests(unittest.TestCase):
    def test_long_paths_preserve_format_and_transformed_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            source = root / "source.pdb"
            source.write_text(
                "ATOM      1  CA  ALA A   1       1.000   2.000   3.000  1.00 20.00           C  \n"
                "ATOM      2  CA  GLY A   2       4.000   5.000   6.000  1.00 20.00           C  \n"
                "END\n",
                encoding="ascii",
            )
            # Exceed 260 characters even with POSIX's shorter /tmp prefix.
            deep = root / ("a" * 80) / ("b" * 80) / ("c" * 80)
            rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
            translation = np.array([10., 20., 30.])
            expected = np.array([[8., 21., 33.], [5., 24., 36.]])
            for suffix in (".pdb", ".cif"):
                for write_tmp in (None, deep / "structure_writes"):
                    with self.subTest(suffix=suffix, configured_tmp=write_tmp is not None):
                        output = deep / "models" / ("aligned" + suffix)
                        self.assertGreater(len(str(output)), 260)
                        with mock.patch.object(map_utils, "_STRUCTURE_WRITE_TMP_DIR", write_tmp):
                            map_utils.write_transformed_structure(source, output, rotation, translation)
                        structure = gemmi.read_structure(str(map_utils._extended_path(output)))
                        np.testing.assert_allclose(map_utils.collect_ca_coords(structure), expected)
                        self.assertFalse(list(map_utils._extended_path(deep).rglob("*.tmp")))
            original = gemmi.read_structure(str(source))
            np.testing.assert_allclose(
                map_utils.collect_ca_coords(original), [[1., 2., 3.], [4., 5., 6.]],
            )
            self.assertTrue(deep.is_relative_to(root))
            shutil.rmtree(map_utils._extended_path(deep))


if __name__ == "__main__":
    unittest.main()
