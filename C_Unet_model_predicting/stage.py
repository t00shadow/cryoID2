"""Production API for stage C (U-Net prediction)."""

from .prediction_provenance import (
    build_prediction_manifest,
    prediction_manifest_path,
    validate_rcl,
    verify_prediction_manifest,
    write_prediction_manifest,
)


def run_unet_inference(*args, **kwargs):
    """Run checkpoint inference without importing Torch during CLI startup."""

    from .unet_inference import run_unet_inference as implementation

    return implementation(*args, **kwargs)


def validate_weights(*args, **kwargs):
    """Validate production checkpoints using the canonical implementation."""

    from .unet_inference import validate_weights as implementation

    return implementation(*args, **kwargs)

__all__ = [
    "build_prediction_manifest",
    "prediction_manifest_path",
    "run_unet_inference",
    "validate_rcl",
    "validate_weights",
    "verify_prediction_manifest",
    "write_prediction_manifest",
]
