from unittest.mock import patch

import pytest

from cryoid_pipeline.runtime_config import default_unet_device
from cryoid_pipeline.user_config import resolve_device, validate_device


@pytest.mark.parametrize(
    "cuda_available,mps_available,expected",
    [
        (True, True, "cuda:0"),
        (True, False, "cuda:0"),
        (False, True, "mps"),
        (False, False, "cpu"),
    ],
)
def test_automatic_device_selection(
    cuda_available, mps_available, expected
):
    with (
        patch("torch.cuda.is_available", return_value=cuda_available),
        patch(
            "torch.backends.mps.is_available",
            return_value=mps_available,
        ),
    ):
        assert default_unet_device() == expected


def test_explicit_mps_selection():
    assert validate_device("mps") == "mps"
    assert resolve_device("mps", {}) == "mps"


def test_mps_device_normalization():
    assert validate_device(" MPS ") == "mps"


def test_invalid_device_still_rejected():
    with pytest.raises(ValueError):
        validate_device("gpu")
