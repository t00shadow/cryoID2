"""Resolve local weights or recover the published bundle in the Hub cache.

This module deliberately has no GUI or numerical-library imports. Downloads
run in the CLI process, which is also the GUI's background pipeline process.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable


PUBLISHED_REPO = "testaccount153/testupload"
PUBLISHED_REVISION = "3ae72b857d69e16dbfecc3af875733de55ef39a7"
MODELS = ("secondary_strctures", "key_atoms", "residue_types_6letter")
# Sizes belong to the pinned revision, not a moving branch. Apart from allowing
# aggregate byte progress, these detect truncated cache files without loading
# several GB of Torch checkpoints on every startup.
PUBLISHED_FILES = {
    "class_weight_for_training.txt": 299,
    "epoch_num_path.txt": 96,
    "checkpoint/secondary_strctures/secondary_strctures.epoch76.pth.tar": 1083734902,
    "checkpoint/key_atoms/key_atoms.epoch11.pth.tar": 1083834610,
    "checkpoint/residue_types_6letter/residue_types_6letter.epoch64.pth.tar": 1083925226,
}
DOWNLOAD_START, DOWNLOAD_END = 2, 12


def _weights_directory(directory: Path) -> Path:
    """Keep nested checkpoint reads usable with long Windows cache locations."""
    if os.name != "nt":
        return directory
    absolute = str(directory.absolute())
    if absolute.startswith("\\\\?\\") or len(absolute) + max(map(len, PUBLISHED_FILES)) + 1 <= 250:
        return directory
    return Path("\\\\?\\UNC\\" + absolute[2:] if absolute.startswith("\\\\") else "\\\\?\\" + absolute)


def pipeline_percent(value: float, with_download: bool) -> float:
    if with_download and value >= 3:
        return DOWNLOAD_END + (100 - DOWNLOAD_END) * (value - 3) / 97
    return value


def is_published_path(directory: Path) -> bool:
    return (directory.name == "train_val_data" and directory.parent.parent.name == "snapshots"
            and directory.parent.parent.parent.name == "models--" + PUBLISHED_REPO.replace("/", "--"))


def weights_problems(directory: Path) -> list[str]:
    """Check metadata and required checkpoint files without importing Torch."""
    directory = _weights_directory(directory)
    problems, metadata = [], {}
    for filename in ("class_weight_for_training.txt", "epoch_num_path.txt"):
        try:
            data = json.loads((directory / filename).read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Metadata must contain a dictionary")
            metadata[filename] = data
        except FileNotFoundError:
            problems.append(f"Missing metadata file: {filename}.")
        except (OSError, ValueError, UnicodeError) as exc:
            problems.append(f"Cannot read {filename}: {exc}")
    for model in MODELS:
        label = {"secondary_strctures": "secondary structures", "key_atoms": "key atoms",
                 "residue_types_6letter": "residue types"}[model]
        classes = metadata.get("class_weight_for_training.txt")
        if classes is not None and (not isinstance(classes.get(model), list) or not classes[model]):
            problems.append(f"Class weights for {label} are missing or invalid.")
        epochs = metadata.get("epoch_num_path.txt")
        if epochs is None:
            continue
        try:
            epoch = int(epochs[model])
        except (KeyError, ValueError, TypeError, OverflowError):
            problems.append(f"The checkpoint number for {label} is missing or invalid.")
            continue
        checkpoint = directory / "checkpoint" / model / f"{model}.epoch{epoch}.pth.tar"
        try:
            with checkpoint.open("rb") as handle:
                if not handle.read(1):
                    problems.append(f"The {label} checkpoint is empty. Download that weights file again.")
        except FileNotFoundError:
            problems.append(f"Missing {label} checkpoint: {checkpoint.relative_to(directory)}.")
        except OSError as exc:
            problems.append(f"Cannot read {checkpoint.relative_to(directory)}: {exc}")
    if is_published_path(directory) and directory.parent.name == PUBLISHED_REVISION:
        for filename, size in PUBLISHED_FILES.items():
            try:
                if (directory / filename).stat().st_size != size:
                    problems.append(f"Published file has an incorrect size: {filename}")
            except OSError:
                pass  # The readable-file checks above describe missing files.
    return problems


def auto_download_needed(value: str | Path | None) -> bool:
    if not value or not str(value).strip():
        return True
    directory = _weights_directory(Path(value).expanduser())
    return not directory.exists() or (is_published_path(directory) and bool(weights_problems(directory)))


def resolve_weights(
    configured: str | Path | None,
    *,
    log: Callable[[str], None],
    status: Callable[[dict], None],
    check_cancel: Callable[[], None],
) -> Path:
    """Use a valid custom folder; otherwise download/recover published weights.

    A missing saved path falls back to autodownload. An existing custom folder
    with invalid contents produces an error instead of silently changing models.
    Hub downloads publish complete files atomically. Completed files are reused
    on retry. Only successfully validated paths are sent back to configuration.
    """
    check_cancel()
    configured_path = _weights_directory(Path(configured).expanduser().resolve()) if configured and str(configured).strip() else None
    if configured_path is not None and not auto_download_needed(configured_path):
        problems = weights_problems(configured_path)
        if problems:
            raise ValueError(f"Invalid custom weights folder {configured_path}: " + "; ".join(problems))
        log(f"Weights: using configured folder {configured_path}")
        return configured_path
    if configured_path is not None:
        log(f"Weights: saved folder is missing or incomplete: {configured_path}. Recovering published weights automatically.")

    # HTTP gives cancellable byte callbacks during downloads and network retries.
    # Xet's native reconstruction callbacks cannot reliably propagate Python
    # cancellation exceptions. Set this before the lazy Hub import.
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub import constants
        from tqdm.auto import tqdm
    except ImportError as exc:
        raise RuntimeError("Automatic weights download requires huggingface-hub. Reinstall CryoID's dependencies or configure a local weights folder.") from exc

    constants.HF_HUB_DISABLE_XET = True  # Also applies if the Hub was imported earlier.
    directory = _weights_directory(Path(constants.HF_HUB_CACHE) / ("models--" + PUBLISHED_REPO.replace("/", "--"))
                                   / "snapshots" / PUBLISHED_REVISION / "train_val_data")
    if not weights_problems(directory):
        log(f"Weights: using cached published weights {directory}")
        status({"name": "Weights ready", "device": "cpu", "weights_path": str(directory), "automatic": True})
        return directory

    total = sum(PUBLISHED_FILES.values())
    completed, transferred = 0, 0
    started = time.monotonic()
    last_update, last_log = -float("inf"), -float("inf")

    def report(*, force=False):
        nonlocal last_update, last_log
        check_cancel()
        now = time.monotonic()
        if not force and now - last_update < 0.2:
            return
        last_update = now
        # Tiny metadata transfers do not predict multi-GB checkpoint speed.
        rate = transferred / max(0.001, now - started) if transferred >= 1024**2 else 0.0
        remaining = 0.0 if completed >= total else (max(0, total - completed) / rate if rate > 0 else None)
        status({"name": "Downloading weights", "device": "Network", "completed_bytes": completed,
                "total_bytes": total, "remaining_seconds": remaining})
        if force or now - last_log >= 5:
            last_log = now
            detail = f"; {rate / 1024**2:.1f} MiB/s; download remaining ~{remaining:.0f}s" if remaining is not None else "; estimating download time"
            log(f"Weights download: {100 * completed / total:.1f}% ({completed / 1024**2:.1f}/{total / 1024**2:.1f} MiB){detail}")

    log(f"Weights: downloading published bundle from Hugging Face ({total / 1024**3:.2f} GiB).")
    log(f"Weights source: {PUBLISHED_REPO} @ {PUBLISHED_REVISION}")
    log(f"Weights cache: {directory}")
    report(force=True)
    try:
        for filename, size in PUBLISHED_FILES.items():
            check_cancel()
            target = directory / filename
            try:
                valid_size = target.stat().st_size == size
            except OSError:
                valid_size = False
            if valid_size and filename.startswith("checkpoint/"):
                completed += size
                log(f"Weights: reusing cached {filename}")
                report(force=True)
                continue
            log(f"Weights: downloading {filename}")
            base = completed

            class DownloadProgress(tqdm):
                def __init__(self, *args, **kwargs):
                    nonlocal completed
                    kwargs.pop("name", None)
                    kwargs["disable"] = True  # Structured events replace terminal bars.
                    super().__init__(*args, **kwargs)
                    self._received = int(kwargs.get("initial", 0))
                    completed = base + min(size, self._received)
                    report(force=True)

                def update(self, n=1):
                    nonlocal completed, transferred
                    delta = int(n or 0)
                    self._received += delta
                    transferred += max(0, delta)
                    completed = base + max(0, min(size, self._received))
                    report()
                    return super().update(n)

            path = Path(hf_hub_download(
                repo_id=PUBLISHED_REPO, revision=PUBLISHED_REVISION,
                filename="train_val_data/" + filename, tqdm_class=DownloadProgress,
                force_download=target.exists(),
            ))
            check_cancel()
            if path.stat().st_size != size:
                raise ValueError(f"Incomplete published weights file: {filename}")
            completed = base + size
            report(force=True)
        problems = weights_problems(directory)
        if problems:
            raise ValueError("; ".join(problems))
    except Exception as exc:
        check_cancel()  # Cancellation must remain cancellation, not a download error.
        log(f"Weights download failed: {exc}")
        raise RuntimeError(f"Could not download published weights: {exc}. Check network access and free disk space, then retry; completed files are retained.") from exc
    log(f"Weights: download complete; files verified at {directory}")
    status({"name": "Weights ready", "device": "cpu", "weights_path": str(directory), "automatic": True})
    return directory
