from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import mrcfile
import numpy as np
import torch
from scipy.ndimage import zoom

from C_Unet_model_predicting.model import UNET
from cryoid_pipeline.models import PredictionBundle, SequenceMode
from cryoid_pipeline.runtime_config import default_unet_device, resolve_unet_weights_dir


MODEL_TYPES = ("secondary_strctures", "key_atoms", "residue_types_6letter")
GRID_SIZE = 64
CUBE_SIZE = 50
PAD_SIZE = 7

ProgressCallback = Callable[[str], None]
PercentCallback = Callable[[int], None]
CancellationCheckpoint = Callable[[], None]


@dataclass(frozen=True)
class UnetInferenceResult:
    bundle: PredictionBundle
    device: str
    weights_dir: Path
    generated_files: list[Path] = field(default_factory=list)


def default_weights_dir() -> Path | None:
    return resolve_unet_weights_dir()


def select_device(device: str | None = None) -> str:
    if device:
        return device
    return default_unet_device()


def load_weights_metadata(weights_dir: Path) -> tuple[dict[str, list[float]], dict[str, int]]:
    class_weight_path = weights_dir / "class_weight_for_training.txt"
    epoch_path = weights_dir / "epoch_num_path.txt"
    with class_weight_path.open("r", encoding="utf-8") as handle:
        class_weights = json.load(handle)
    with epoch_path.open("r", encoding="utf-8") as handle:
        epochs = json.load(handle)
    return class_weights, epochs


def _checkpoint_path(weights_dir: Path, model_type: str, epoch: int) -> Path:
    return weights_dir / "checkpoint" / model_type / f"{model_type}.epoch{epoch}.pth.tar"


def _load_checkpoint_data(checkpoint: Path, device: str):
    try:
        return torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint, map_location=device)


def validate_weights(weights_dir: Path) -> None:
    class_weights, epochs = load_weights_metadata(weights_dir)
    missing: list[Path] = []
    for model_type in MODEL_TYPES:
        if model_type not in class_weights:
            raise KeyError(f"Missing class weights for {model_type}")
        if model_type not in epochs:
            raise KeyError(f"Missing checkpoint epoch for {model_type}")
        checkpoint = _checkpoint_path(weights_dir, model_type, int(epochs[model_type]))
        if not checkpoint.is_file():
            missing.append(checkpoint)
    if missing:
        raise FileNotFoundError("Missing UNet checkpoint(s): " + ", ".join(str(path) for path in missing))


def load_checkpoints_for_device(
    weights_dir: Path,
    device: str,
    *,
    progress: ProgressCallback | None = None,
) -> None:
    validate_weights(weights_dir)
    class_weights, epochs = load_weights_metadata(weights_dir)
    in_channels = 1
    for model_type in MODEL_TYPES:
        out_channels = len(class_weights[model_type])
        checkpoint = _checkpoint_path(weights_dir, model_type, int(epochs[model_type]))
        if progress:
            progress(f"Loading {model_type}: in={in_channels} out={out_channels} epoch={epochs[model_type]}")
        model = UNET(in_channels, out_channels).to(device)
        checkpoint_data = _load_checkpoint_data(checkpoint, device)
        state_dict = checkpoint_data["state_dict"] if isinstance(checkpoint_data, dict) and "state_dict" in checkpoint_data else checkpoint_data
        model.load_state_dict(state_dict)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        in_channels += out_channels


def _prediction_prefix(map_path: Path) -> str:
    name = map_path.name
    return name[:-4] if name.lower().endswith(".mrc") else name


def normalize_map(map_path: Path, recl: float = 0.0) -> tuple[np.ndarray, list[float]]:
    with mrcfile.mmap(map_path, mode="r", permissive=True) as mrc:
        map_data = np.array(mrc.data, dtype=np.float32)
        zoom_factors = [float(mrc.voxel_size.z), float(mrc.voxel_size.y), float(mrc.voxel_size.x)]
        map_data = zoom(map_data, zoom_factors)

        count_good = int(np.sum(map_data > max(0.0, float(recl))))
        count_total = int(map_data.size)
        if count_total == 0:
            raise ValueError(f"Empty map data: {map_path}")

        if recl > 0.0:
            bottom_percentile = (1 - (count_good / 0.85) / count_total) * 100
        else:
            bottom_percentile = (1 - (count_good * 0.7) / count_total) * 100
        bottom_percentile = max(0.0, min(100.0, float(bottom_percentile)))
        value_bottom = np.percentile(map_data, bottom_percentile)
        map_data -= max(float(value_bottom), 0.0)

        positive_data = map_data[map_data >= 0]
        if positive_data.size == 0:
            raise ValueError(f"Map has no non-negative density values after normalization: {map_path}")
        positive_data_99 = float(np.percentile(positive_data, 99))
        if positive_data_99 == 0.0:
            raise ValueError(f"Empty map: 99th percentile of positive density is zero: {map_path}")
        map_data = np.clip(map_data / positive_data_99, 0.0, 1.0)

        orientation = np.array(
            [int(mrc.header.mapc) - 1, int(mrc.header.mapr) - 1, int(mrc.header.maps) - 1],
            dtype=np.int32,
        )
        map_data = np.transpose(map_data, orientation)
    return map_data.astype(np.float32, copy=False), zoom_factors


def predict_location(
    map_data: np.ndarray,
    model: torch.nn.Module,
    out_channels: int,
    device: str,
    *,
    progress: ProgressCallback | None = None,
    cube_progress: Callable[[int, int], None] | None = None,
    should_cancel: CancellationCheckpoint | None = None,
    label: str = "",
    cube_size: int = CUBE_SIZE,
    pad_size: int = PAD_SIZE,
) -> np.ndarray:
    check_cancel = should_cancel or (lambda: None)
    check_cancel()
    map_shape = map_data.shape
    cube_nums = [map_shape[i] // cube_size + 1 for i in range(1, len(map_shape))]
    num_cubes = cube_nums[0] * cube_nums[1] * cube_nums[2]
    larger_shape = [cube_size * cube_nums[i] + pad_size * 2 for i in range(len(cube_nums))]
    larger_shape.insert(0, map_shape[0])

    reshaped_map_data = np.zeros(tuple(larger_shape), np.float32)
    reshaped_map_data[
        :,
        pad_size : pad_size + map_shape[1],
        pad_size : pad_size + map_shape[2],
        pad_size : pad_size + map_shape[3],
    ] = map_data

    reshaped_tensor = torch.from_numpy(reshaped_map_data)
    model_shape = tuple([out_channels, *larger_shape[-3:]])
    reshaped_model = torch.zeros(model_shape, dtype=torch.float32)

    model.eval()
    completed = 0
    with torch.inference_mode():
        for zz in range(cube_nums[0]):
            z_start = cube_size * zz
            for yy in range(cube_nums[1]):
                y_start = cube_size * yy
                for xx in range(cube_nums[2]):
                    check_cancel()
                    x_start = cube_size * xx
                    cube = reshaped_tensor[
                        None,
                        :,
                        z_start : GRID_SIZE + z_start,
                        y_start : GRID_SIZE + y_start,
                        x_start : GRID_SIZE + x_start,
                    ].to(device)
                    pred_data = torch.softmax(model(cube), 1).cpu()
                    reshaped_model[
                        :,
                        pad_size + z_start : GRID_SIZE - pad_size + z_start,
                        pad_size + y_start : GRID_SIZE - pad_size + y_start,
                        pad_size + x_start : GRID_SIZE - pad_size + x_start,
                    ] = pred_data[
                        0,
                        :,
                        pad_size : GRID_SIZE - pad_size,
                        pad_size : GRID_SIZE - pad_size,
                        pad_size : GRID_SIZE - pad_size,
                    ]
                    completed += 1
                    if progress:
                        progress(f"UNet {label}: predicted cube {completed}/{num_cubes}")
                    if cube_progress:
                        cube_progress(completed, num_cubes)
                    check_cancel()

    return (
        reshaped_model[
            :,
            pad_size : pad_size + map_shape[1],
            pad_size : pad_size + map_shape[2],
            pad_size : pad_size + map_shape[3],
        ]
        .numpy()
        .astype(np.float32, copy=False)
    )


def write_prediction_map(
    input_map: Path,
    model_data: np.ndarray,
    output_map: Path,
    zoom_factors: list[float],
    *,
    is_int: bool = False,
) -> Path:
    output_map.parent.mkdir(parents=True, exist_ok=True)
    if output_map.exists():
        output_map.unlink()
    shutil.copyfile(input_map, output_map)
    with mrcfile.open(output_map, mode="r+", permissive=True) as mrc:
        data = model_data.astype(np.int8 if is_int else np.float32, copy=False)
        mrc.set_data(data)
        mrc.header.nzstart = int(mrc.header.nzstart * zoom_factors[0])
        mrc.header.nystart = int(mrc.header.nystart * zoom_factors[1])
        mrc.header.nxstart = int(mrc.header.nxstart * zoom_factors[2])
        mrc.header.ispg = 401
        mrc.update_header_from_data()
        mrc.update_header_stats()
    return output_map


def _write_model_outputs(
    *,
    input_map: Path,
    output_root: Path,
    label_root: Path,
    model_type: str,
    out_channels: int,
    probabilities: np.ndarray,
    labels: np.ndarray,
    zoom_factors: list[float],
) -> list[Path]:
    prefix = _prediction_prefix(input_map)
    outputs: list[Path] = []
    for channel_num in range(1, out_channels):
        probability = probabilities[channel_num, :, :, :]
        probability_path = output_root / f"{prefix}_{model_type}_{channel_num}_possibility.mrc"
        outputs.append(write_prediction_map(input_map, probability, probability_path, zoom_factors))

        label_data = np.where(labels == channel_num, 1, 0)
        label_path = label_root / f"{prefix}_{model_type}_{channel_num}.mrc"
        outputs.append(write_prediction_map(input_map, label_data, label_path, zoom_factors, is_int=True))
    return outputs


def run_unet_inference(
    density_map: Path,
    output_root: Path,
    *,
    weights_dir: Path | None = None,
    device: str | None = None,
    tmp_root: Path | None = None,
    recl: float = 0.0,
    progress: ProgressCallback | None = None,
    progress_percent: PercentCallback | None = None,
    should_cancel: CancellationCheckpoint | None = None,
    mode: SequenceMode = "protein",
) -> UnetInferenceResult:
    check_cancel = should_cancel or (lambda: None)
    check_cancel()
    weights = (weights_dir or default_weights_dir())
    if weights is None:
        raise FileNotFoundError(
            "UNet weights directory was not found. Set the weights path in Advanced Settings "
            "or set CRYOID_UNET_WEIGHTS_DIR."
        )
    weights = weights.expanduser().resolve()
    validate_weights(weights)
    check_cancel()

    selected_device = select_device(device)
    class_weights, epochs = load_weights_metadata(weights)
    check_cancel()
    label_root = (tmp_root.expanduser().resolve() / "unrefined") if tmp_root else output_root / "unrefined"
    output_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)

    check_cancel()
    if progress:
        progress(f"UNet inference: loading density map on {selected_device}")
    if progress_percent:
        progress_percent(5)
    map_data, zoom_factors = normalize_map(density_map, recl=recl)
    check_cancel()
    if progress_percent:
        progress_percent(8)
    model_input = np.expand_dims(map_data, axis=0)
    generated_files: list[Path] = []

    unet_progress_start = 8
    unet_progress_end = 55
    for model_index, model_type in enumerate(MODEL_TYPES):
        check_cancel()
        model_start = unet_progress_start + round((unet_progress_end - unet_progress_start) * model_index / len(MODEL_TYPES))
        model_end = unet_progress_start + round((unet_progress_end - unet_progress_start) * (model_index + 1) / len(MODEL_TYPES))
        out_channels = len(class_weights[model_type])
        in_channels = model_input.shape[0]
        checkpoint = _checkpoint_path(weights, model_type, int(epochs[model_type]))
        if progress:
            progress(f"UNet inference: loading {model_type} checkpoint {checkpoint.name}")
        if progress_percent:
            progress_percent(model_start)
        model = UNET(in_channels, out_channels).to(selected_device)
        try:
            check_cancel()
            checkpoint_data = _load_checkpoint_data(checkpoint, selected_device)
            state_dict = checkpoint_data["state_dict"] if isinstance(checkpoint_data, dict) and "state_dict" in checkpoint_data else checkpoint_data
            model.load_state_dict(state_dict)
            check_cancel()

            if progress:
                progress(f"UNet inference: running {model_type}")

            def emit_cube_percent(completed: int, total: int, *, start: int = model_start, end: int = model_end) -> None:
                if progress_percent:
                    progress_percent(start + round((end - start) * completed / max(total, 1)))

            probabilities = predict_location(
                model_input,
                model,
                out_channels,
                selected_device,
                progress=progress,
                cube_progress=emit_cube_percent,
                should_cancel=check_cancel,
                label=model_type,
            )
            check_cancel()
            labels = np.argmax(probabilities, axis=0)
            generated_files.extend(
                _write_model_outputs(
                    input_map=density_map,
                    output_root=output_root,
                    label_root=label_root,
                    model_type=model_type,
                    out_channels=out_channels,
                    probabilities=probabilities,
                    labels=labels,
                    zoom_factors=zoom_factors,
                )
            )
            check_cancel()
            if progress_percent:
                progress_percent(model_end)

            model_input = np.concatenate((model_input, probabilities), axis=0)
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    prefix = _prediction_prefix(density_map)
    check_cancel()
    if mode not in {"protein", "rna"}:
        raise ValueError("UNet inference mode must be protein or rna.")
    key_atom_paths = [
        output_root / f"{prefix}_key_atoms_{idx}_possibility.mrc"
        for idx in range(1, len(class_weights["key_atoms"]))
    ]
    raw_key_atoms_1_path = key_atom_paths[0]
    secondary_structure_paths = [
        output_root / f"{prefix}_secondary_strctures_{idx}_possibility.mrc"
        for idx in range(1, len(class_weights["secondary_strctures"]))
    ]
    residue_type_paths = [
        output_root / f"{prefix}_residue_types_6letter_{idx}_possibility.mrc"
        for idx in range(1, len(class_weights["residue_types_6letter"]))
    ]
    required_paths = (
        [key_atom_paths[0], *secondary_structure_paths[:3], *residue_type_paths[:6]]
        if mode == "protein"
        else [key_atom_paths[1], key_atom_paths[2], secondary_structure_paths[3], residue_type_paths[6], residue_type_paths[7]]
    )
    missing = [
        path
        for path in required_paths
        if not path.is_file() or path.stat().st_size <= 0
    ]
    if missing:
        raise FileNotFoundError("UNet inference did not write required prediction map(s): " + ", ".join(str(path) for path in missing))
    if progress_percent:
        progress_percent(unet_progress_end)

    if mode == "protein":
        exposed_key_channels = {1: key_atom_paths[0]}
        exposed_secondary_channels = {
            index: secondary_structure_paths[index - 1] for index in range(1, 4)
        }
        exposed_residue_channels = {
            index: residue_type_paths[index - 1] for index in range(1, 7)
        }
    else:
        exposed_key_channels = {2: key_atom_paths[1], 3: key_atom_paths[2]}
        exposed_secondary_channels = {4: secondary_structure_paths[3]}
        exposed_residue_channels = {7: residue_type_paths[6], 8: residue_type_paths[7]}

    bundle = PredictionBundle(
        layout="checkpoint_inference",
        density_map=density_map,
        raw_prediction_dir=output_root,
        raw_key_atoms_1_path=raw_key_atoms_1_path if mode == "protein" else None,
        secondary_structure_paths=list(exposed_secondary_channels.values()),
        residue_type_paths=list(exposed_residue_channels.values()),
        density_dir=None,
        output_files=[path.resolve() for path in required_paths],
        mode=mode,
        key_atom_paths=list(exposed_key_channels.values()),
        key_atom_channels=exposed_key_channels,
        secondary_structure_channels=exposed_secondary_channels,
        residue_type_channels=exposed_residue_channels,
    )
    return UnetInferenceResult(
        bundle=bundle,
        device=selected_device,
        weights_dir=weights,
        generated_files=generated_files,
    )
