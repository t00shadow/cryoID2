from unittest.mock import patch

import pytest

from cryoid_pipeline.runtime_config import default_unet_device, mps_unet_problem
from cryoid_pipeline.user_config import resolve_device, validate_device


@pytest.mark.parametrize(
    "cuda_available,mps_available,execution_error,expected",
    [
        (True, True, None, "cuda:0"),
        (True, False, None, "cuda:0"),
        (False, True, None, "mps"),
        (False, False, None, "cpu"),
        (False, True, RuntimeError("ConvTranspose3D is not supported on MPS"), "cpu"),
    ],
)
def test_automatic_device_selection(
    cuda_available, mps_available, execution_error, expected
):
    with (
        patch("torch.cuda.is_available", return_value=cuda_available),
        patch(
            "torch.backends.mps.is_available",
            return_value=mps_available,
        ),
        patch("cryoid_pipeline.preflight.exercise_torch", side_effect=execution_error) as exercise,
    ):
        assert default_unet_device() == expected
        assert exercise.call_count == int(not cuda_available and mps_available)


@pytest.mark.parametrize("value", ["mps", " MPS "])
def test_explicit_mps_selection(value):
    from C_Unet_model_predicting.unet_inference import select_device
    assert validate_device(value) == "mps"
    assert resolve_device(value, {}) == "mps"
    with patch("C_Unet_model_predicting.unet_inference.mps_unet_problem", return_value=None):
        assert select_device(resolve_device(value, {})) == "mps"


@pytest.mark.parametrize("available", [True, False])
def test_explicit_mps_checks_compatibility_before_loading_weights(available):
    from C_Unet_model_predicting.unet_inference import select_device
    with (
        patch("torch.backends.mps.is_available", return_value=available),
        patch("cryoid_pipeline.preflight.exercise_torch",
              side_effect=RuntimeError("ConvTranspose3D is not supported on MPS")),
    ):
        problem = mps_unet_problem()
        assert problem and ("ConvTranspose3D" in problem if available else "unavailable" in problem)
        with pytest.raises(ValueError, match="Select CPU or Auto"):
            select_device("mps")


def test_mps_backend_missing_in_older_torch(monkeypatch):
    import torch
    monkeypatch.delattr(torch.backends, "mps")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert default_unet_device() == "cpu"


def test_real_mps_unet_execution_when_available():
    import torch
    if not torch.backends.mps.is_available():
        pytest.skip("Requires a Mac with an available MPS backend")
    from C_Unet_model_predicting.unet_inference import select_device
    problem = mps_unet_problem()
    if problem:
        with pytest.raises(ValueError, match="Select CPU or Auto"):
            select_device("mps")
        assert default_unet_device() == "cpu"
    else:
        assert select_device("mps") == "mps"
