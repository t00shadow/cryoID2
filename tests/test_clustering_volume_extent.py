"""Full-volume defaults must retain boundary densities without changing smaller cases."""
from dataclasses import replace

import mrcfile
import numpy as np
import pytest

from H_Backbone_tracing.common.clustering import ClusteringConfig, run_clustering


def write_volume(path, data):
    with mrcfile.new(path) as mrc:
        mrc.set_data(data)
        mrc.voxel_size = 1


@pytest.mark.parametrize("method", ["meanshift", "agglomerative"])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_clustering_includes_densities_past_999_on_every_axis(tmp_path, method, axis):
    shape = [2, 2, 2]
    shape[axis] = 1002
    data = np.zeros(shape, dtype=np.float32)
    region = [slice(None)] * 3
    region[axis] = slice(999, 1002)
    data[tuple(region)] = 1
    source = tmp_path / "input.mrc"
    write_volume(source, data)
    config = ClusteringConfig(str(source), clustering_method=method, manual_file_name="probe", output_dir=str(tmp_path / "default"))
    full = run_clustering(config)
    explicit = run_clustering(replace(config, output_dir=str(tmp_path / "full"),
                                       z_slice=(0, shape[0]), y_slice=(0, shape[1]), x_slice=(0, shape[2])))
    cropped = run_clustering(replace(config, output_dir=str(tmp_path / "cropped"),
                                      **{("z_slice", "y_slice", "x_slice")[axis]: (0, 999)}))
    assert len(full["centroids"]) > 0
    assert len(cropped["centroids"]) == 0
    np.testing.assert_array_equal(full["centroids"], explicit["centroids"])
    assert full["output_path"].read_bytes() == explicit["output_path"].read_bytes()
    assert np.all(full["centroids"][:, axis] >= 999)


@pytest.mark.parametrize("method", ["meanshift", "agglomerative"])
def test_existing_small_volume_clusters_and_pdb_are_unchanged(tmp_path, method):
    data = np.zeros((12, 14, 16), dtype=np.float32)
    data[1:3, 2:4, 3:5] = 1
    data[8:10, 9:11, 10:12] = 1
    source = tmp_path / "input.mrc"
    write_volume(source, data)
    config = ClusteringConfig(str(source), clustering_method=method, manual_file_name="probe", output_dir=str(tmp_path / "default"))
    full = run_clustering(config)
    previous = run_clustering(replace(config, output_dir=str(tmp_path / "previous"),
                                      z_slice=(0, 999), y_slice=(0, 999), x_slice=(0, 999)))
    assert len(full["centroids"]) == 2
    np.testing.assert_array_equal(full["centroids"], previous["centroids"])
    assert full["output_path"].read_bytes() == previous["output_path"].read_bytes()


def test_explicit_region_still_limits_clustering_and_preserves_global_indices(tmp_path):
    data = np.zeros((1008, 2, 2), dtype=np.float32)
    data[1:3] = 1
    data[1002:1005] = 1
    source = tmp_path / "input.mrc"
    write_volume(source, data)
    result = run_clustering(ClusteringConfig(str(source), manual_file_name="probe",
                            output_dir=str(tmp_path), z_slice=(1002, 1005)))
    np.testing.assert_allclose(result["centroids"], [[1003, .5, .5]])
