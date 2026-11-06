"""Predictions must describe the same physical volume for every MRC axis order."""
from itertools import permutations
import json

import mrcfile
import numpy as np
import pytest

from C_Unet_model_predicting.unet_inference import normalize_map, write_prediction_map
from C_Unet_model_predicting.prediction_provenance import (
    build_prediction_manifest, verify_prediction_manifest, write_prediction_manifest,
)
from I_Structure_fitting.protein.core.map_utils import MapInfo


def write_map(path, data, axes=(1, 2, 3), voxel=(1, 1, 1), origin=(0, 0, 0)):
    storage = tuple((3, 2, 1).index(axis) for axis in axes[::-1])
    with mrcfile.new(path) as mrc:
        mrc.set_data(np.ascontiguousarray(data.transpose(storage)))
        mrc.header.mapc, mrc.header.mapr, mrc.header.maps = axes
        mrc.header.mx, mrc.header.my, mrc.header.mz = data.shape[::-1]
        mrc.voxel_size = voxel
        mrc.header.origin = origin
        starts = {1: 2, 2: 4, 3: 6}
        mrc.header.nxstart, mrc.header.nystart, mrc.header.nzstart = (starts[axis] for axis in axes)


def geometry(path):
    with mrcfile.open(path) as mrc:
        data = np.squeeze(mrc.data).copy()
        index = np.unravel_index(np.argmax(data), data.shape)
        peak = MapInfo(mrc).grid_to_world_many(*index[::-1])
        return data, tuple(float(value) for value in peak), mrc.header.copy()


@pytest.mark.parametrize("axes", list(permutations((1, 2, 3))))
@pytest.mark.parametrize("voxel", [(1, 1, 1), (1.5, 2, 2.5)])
@pytest.mark.parametrize("origin", [(0, 0, 0), (12, -4, 30)])
def test_equivalent_mrc_storage_orders_have_identical_normalization_and_output_geometry(tmp_path, axes, voxel, origin):
    raw = np.arange(1, 61, dtype=np.float32).reshape(3, 4, 5)
    canonical, permuted = tmp_path / "canonical.mrc", tmp_path / "permuted.mrc"
    write_map(canonical, raw, voxel=voxel, origin=origin)
    write_map(permuted, raw, axes=axes, voxel=voxel, origin=origin)
    expected, factors = normalize_map(canonical)
    actual, actual_factors = normalize_map(permuted)
    np.testing.assert_array_equal(actual, expected)
    assert actual_factors == factors
    for is_int in (False, True):
        output_a, output_b = tmp_path / f"a-{is_int}.mrc", tmp_path / f"b-{is_int}.mrc"
        write_prediction_map(canonical, expected, output_a, factors, is_int=is_int)
        write_prediction_map(permuted, actual, output_b, actual_factors, is_int=is_int)
        data_a, peak_a, header_a = geometry(output_a)
        data_b, peak_b, header_b = geometry(output_b)
        np.testing.assert_array_equal(data_a, data_b)
        for field in ("nx", "ny", "nz", "mx", "my", "mz", "mapc", "mapr", "maps",
                      "nxstart", "nystart", "nzstart", "cella", "origin", "ispg"):
            np.testing.assert_array_equal(header_a[field], header_b[field])
        np.testing.assert_allclose(peak_a, peak_b, atol=1e-6)
        assert (int(header_b.mapc), int(header_b.mapr), int(header_b.maps)) == (1, 2, 3)
        if voxel == (1, 1, 1):
            np.testing.assert_allclose(geometry(permuted)[1], peak_b, atol=1e-6)


def test_invalid_mrc_axis_mapping_is_reported_before_inference(tmp_path):
    source = tmp_path / "invalid.mrc"
    write_map(source, np.ones((3, 4, 5), dtype=np.float32))
    with mrcfile.open(source, mode="r+") as mrc:
        mrc.header.mapr = 1
    with pytest.raises(ValueError, match="permutation"):
        normalize_map(source)


@pytest.mark.parametrize("axes", [(1, 2, 3), (2, 1, 3)])
def test_axis_fix_invalidates_only_affected_old_prediction_caches(tmp_path, axes):
    source, output = tmp_path / "input.mrc", tmp_path / "prediction.mrc"
    write_map(source, np.arange(1, 61, dtype=np.float32).reshape(3, 4, 5), axes=axes)
    normalized, factors = normalize_map(source)
    write_prediction_map(source, normalized, output, factors)
    manifest_path = tmp_path / "provenance.json"
    manifest = build_prediction_manifest(density_map=source, rcl=0, prediction_paths=[output],
                                         weights_dir=None, source="checkpoint_inference")
    arguments = dict(manifest_path=manifest_path, density_map=source, rcl=0,
                     prediction_paths=[output], weights_dir=None)
    write_prediction_manifest(manifest_path, manifest)
    assert verify_prediction_manifest(**arguments)[0]
    manifest.pop("normalization")
    write_prediction_manifest(manifest_path, manifest)
    valid, reason, _ = verify_prediction_manifest(**arguments)
    assert valid == (axes == (1, 2, 3)), reason


@pytest.mark.parametrize("field,value", [
    (None, []), (None, None), (None, "invalid"),
    ("schema_version", []), ("schema_version", True),
    ("density_map", []), ("density_map", "invalid"),
    ("rcl", {}), ("rcl", "nan"),
    ("outputs", None), ("outputs", {}), ("outputs", [None]),
    ("outputs", [{"name": []}]),
    ("weights", ["invalid"]), ("source", ["invalid"]),
])
def test_malformed_provenance_is_a_cache_miss(tmp_path, field, value):
    source, output = tmp_path / "input.mrc", tmp_path / "prediction.mrc"
    source.write_bytes(b"input map")
    output.write_bytes(b"prediction")
    manifest = build_prediction_manifest(density_map=source, rcl=0, prediction_paths=[output],
                                         weights_dir=None, source="checkpoint_inference")
    if field is None:
        manifest = value
    else:
        manifest[field] = value
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    valid, reason, _ = verify_prediction_manifest(manifest_path=path, density_map=source, rcl=0,
                                                prediction_paths=[output], weights_dir=None)
    assert not valid and reason
