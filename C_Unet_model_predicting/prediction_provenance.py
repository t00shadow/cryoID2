from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import mrcfile


SCHEMA_VERSION = 1
NORMALIZATION = "canonical-zyx-v1"


def validate_rcl(value: float | int | str | None) -> float:
    try:
        rcl = 0.0 if value in (None, "") else float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("RCL must be a finite number greater than or equal to zero.") from exc
    if not math.isfinite(rcl) or rcl < 0.0:
        raise ValueError("RCL must be a finite number greater than or equal to zero.")
    return rcl


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_manifest_path(output_root: Path, prefix: str) -> Path:
    return Path(output_root) / f"{prefix}_prediction_provenance.json"


def _checkpoint_paths(weights_dir: Path) -> list[Path]:
    epoch_path = weights_dir / "epoch_num_path.txt"
    epochs = json.loads(epoch_path.read_text(encoding="utf-8"))
    paths = [
        weights_dir / "class_weight_for_training.txt",
        epoch_path,
    ]
    for model_type in ("secondary_strctures", "key_atoms", "residue_types_6letter"):
        epoch = int(epochs[model_type])
        paths.append(
            weights_dir
            / "checkpoint"
            / model_type
            / f"{model_type}.epoch{epoch}.pth.tar"
        )
    return paths


def weights_fingerprint(weights_dir: Path | None) -> dict[str, Any] | None:
    if weights_dir is None:
        return None
    root = Path(weights_dir).expanduser().resolve()
    files: list[dict[str, Any]] = []
    combined = hashlib.sha256()
    for path in _checkpoint_paths(root):
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Missing UNet provenance input: {resolved}")
        # Hub snapshots link files to blobs outside the snapshot directory.
        # Identify each logical weights file by its snapshot-relative name,
        # while hashing the resolved contents below.
        relative = path.relative_to(root).as_posix()
        file_hash = sha256_file(resolved)
        size = resolved.stat().st_size
        combined.update(relative.encode("utf-8"))
        combined.update(b"\0")
        combined.update(file_hash.encode("ascii"))
        combined.update(b"\0")
        files.append({"path": relative, "sha256": file_hash, "size": size})
    return {
        "root": str(root),
        "algorithm": "sha256-selected-checkpoints-v1",
        "sha256": combined.hexdigest(),
        "files": files,
    }


def build_prediction_manifest(
    *,
    density_map: Path,
    rcl: float,
    prediction_paths: Iterable[Path],
    weights_dir: Path | None,
    source: str,
) -> dict[str, Any]:
    map_path = Path(density_map).expanduser().resolve()
    outputs = []
    for path in prediction_paths:
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file() or resolved.stat().st_size <= 0:
            raise FileNotFoundError(f"Missing prediction output for provenance: {resolved}")
        outputs.append(
            {
                "name": resolved.name,
                "sha256": sha256_file(resolved),
                "size": resolved.stat().st_size,
            }
        )
    return {
        "schema": "cryoid-prediction-provenance",
        "schema_version": SCHEMA_VERSION,
        "normalization": NORMALIZATION,
        "source": str(source),
        "density_map": {
            "path": str(map_path),
            "sha256": sha256_file(map_path),
            "size": map_path.stat().st_size,
        },
        "rcl": validate_rcl(rcl),
        "weights": weights_fingerprint(weights_dir),
        "outputs": outputs,
    }


def write_prediction_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def verify_prediction_manifest(
    *,
    manifest_path: Path,
    density_map: Path,
    rcl: float,
    prediction_paths: Iterable[Path],
    weights_dir: Path | None,
) -> tuple[bool, str, dict[str, Any] | None]:
    path = Path(manifest_path)
    if not path.is_file():
        return False, "prediction provenance manifest is missing", None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return False, f"prediction provenance manifest is unreadable: {exc}", None
    if not isinstance(manifest, dict):
        return False, "prediction provenance must contain a JSON object", None
    if manifest.get("schema") != "cryoid-prediction-provenance":
        return False, "prediction provenance schema is not recognized", manifest
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != SCHEMA_VERSION:
        return False, "prediction provenance schema version does not match", manifest

    map_meta = manifest.get("density_map")
    if not isinstance(map_meta, dict):
        return False, "prediction density-map metadata is invalid", manifest
    try:
        if "rcl" not in manifest:
            return False, "prediction provenance RCL is missing", manifest
        stored_rcl = validate_rcl(manifest["rcl"])
    except ValueError:
        return False, "prediction provenance RCL is invalid", manifest
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or not outputs or not all(
        isinstance(item, dict) and isinstance(item.get("name"), str) and item["name"]
        and isinstance(item.get("sha256"), str) for item in outputs
    ):
        return False, "prediction output metadata is invalid", manifest
    if len({item["name"] for item in outputs}) != len(outputs):
        return False, "prediction output metadata contains duplicate filenames", manifest
    recorded_weights = manifest.get("weights")
    if recorded_weights is not None and not isinstance(recorded_weights, dict):
        return False, "prediction weights metadata is invalid", manifest
    if "source" in manifest and not isinstance(manifest["source"], str):
        return False, "prediction source metadata is invalid", manifest
    map_path = Path(density_map).expanduser().resolve()
    if map_meta.get("sha256") != sha256_file(map_path):
        return False, "input density-map checksum changed", manifest
    normalization = manifest.get("normalization")
    if normalization is None:
        # Existing standard-axis predictions are unchanged by the axis fix.
        # Older nonstandard-axis predictions must be regenerated.
        with mrcfile.open(map_path, header_only=True, permissive=True) as mrc:
            axes = tuple(int(value) for value in (mrc.header.mapc, mrc.header.mapr, mrc.header.maps))
        if axes != (1, 2, 3):
            return False, "prediction normalization predates the MRC axis correction", manifest
    elif normalization != NORMALIZATION:
        return False, "prediction normalization version does not match", manifest
    if stored_rcl != validate_rcl(rcl):
        return False, "RCL changed", manifest

    expected_outputs = {
        str(item.get("name")): item
        for item in outputs
    }
    requested_paths = [Path(item).expanduser().resolve() for item in prediction_paths]
    if set(expected_outputs) != {item.name for item in requested_paths}:
        return False, "prediction output set changed", manifest
    for output in requested_paths:
        if not output.is_file() or output.stat().st_size <= 0:
            return False, f"prediction output is missing: {output.name}", manifest
        output_meta = expected_outputs[output.name]
        if output_meta.get("sha256") != sha256_file(output):
            return False, f"prediction checksum changed: {output.name}", manifest

    if weights_dir is not None:
        if not recorded_weights:
            if manifest.get("source") in {"archived_paper_predictions", "imported_predictions"}:
                return (
                    True,
                    "map, RCL, and prediction checksums match; archived weights fingerprint is unavailable",
                    manifest,
                )
            return False, "configured weights are absent from prediction provenance", manifest
        current_weights = weights_fingerprint(weights_dir)
        if recorded_weights.get("sha256") != current_weights.get("sha256"):
            return False, "UNet weights changed", manifest
    return True, "map, RCL, prediction, and available weights provenance match", manifest
