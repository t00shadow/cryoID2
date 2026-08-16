# C - U-Net prediction

Production entry point: `C_Unet_model_predicting.stage`.

- `model.py`: checkpoint-compatible U-Net definition.
- `unet_inference.py`: ordered three-checkpoint inference and mode-specific
  prediction bundle creation.
- `prediction_provenance.py`: weights/input fingerprints and safe prediction
  cache reuse.
- `stage.py`: the API imported by the orchestrator.

Production weights are external and configured with `cryoid config set
weights-dir PATH`.
