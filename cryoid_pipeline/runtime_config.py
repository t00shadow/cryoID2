from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path


ENV_REPO_ROOT = "CRYOID_REPO_ROOT"
ENV_OUTPUT_ROOT = "CRYOID_OUTPUT_ROOT"
ENV_UNET_WEIGHTS_DIR = "CRYOID_UNET_WEIGHTS_DIR"
ENV_UNET_DEVICE = "CRYOID_UNET_DEVICE"
ENV_TEST_QUERY_FASTA = "CRYOID_TEST_QUERY_FASTA"


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _path_from_env(name: str) -> Path | None:
    value = os.environ.get(name)
    if not value:
        return None
    return Path(value).expanduser().resolve()


def resolve_unet_weights_dir() -> Path | None:
    env_path = _path_from_env(ENV_UNET_WEIGHTS_DIR)
    if env_path is not None and env_path.is_dir():
        return env_path
    return None


def default_unet_device() -> str:
    env_device = os.environ.get(ENV_UNET_DEVICE)
    if env_device:
        return env_device
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def cuda_status_message() -> str:
    try:
        import torch
    except ImportError:
        return "CUDA: torch is not installed; using CPU."
    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            return f"CUDA: detected ({device_name})."
        return "CUDA: not detected; using CPU."
    except Exception as exc:
        return f"CUDA: detection failed ({exc!r}); using CPU."


def display_problem() -> str | None:
    if platform.system().lower() in {"windows", "darwin"}:
        return None
    qpa_platform = os.environ.get("QT_QPA_PLATFORM", "").lower()
    if qpa_platform in {"offscreen", "minimal"}:
        return None
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return None
    return "No GUI display was detected. Use SSH X11 forwarding, VNC, or set DISPLAY/WAYLAND_DISPLAY."


@dataclass(frozen=True)
class RuntimeConfig:
    repo_root: Path
    output_root: Path
    unet_weights_dir: Path | None
    unet_device: str
    test_query_fasta: Path | None

    @classmethod
    def from_env(
        cls,
        *,
        repo_root: Path | None = None,
        output_root: Path | None = None,
        unet_weights_dir: Path | None = None,
        unet_device: str | None = None,
    ) -> "RuntimeConfig":
        resolved_repo_root = (
            Path(repo_root).expanduser().resolve()
            if repo_root is not None
            else _path_from_env(ENV_REPO_ROOT) or default_repo_root().resolve()
        )
        resolved_output_root = (
            Path(output_root).expanduser().resolve()
            if output_root is not None
            else _path_from_env(ENV_OUTPUT_ROOT) or resolved_repo_root / "runs"
        )
        resolved_weights_dir = (
            Path(unet_weights_dir).expanduser().resolve()
            if unet_weights_dir is not None
            else resolve_unet_weights_dir()
        )
        resolved_device = unet_device or default_unet_device()
        return cls(
            repo_root=resolved_repo_root,
            output_root=resolved_output_root,
            unet_weights_dir=resolved_weights_dir,
            unet_device=resolved_device,
            test_query_fasta=_path_from_env(ENV_TEST_QUERY_FASTA),
        )
