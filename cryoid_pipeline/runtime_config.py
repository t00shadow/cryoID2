from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .user_config import load_config, resolve_device, resolve_project_dir, resolve_weights_dir, resolve_cpu_threads


def default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_unet_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    if mps_unet_problem() is None:
        return "mps"
    return "cpu"


def mps_unet_problem() -> str | None:
    """Check actual U-Net operations, since MPS availability alone is insufficient."""
    try:
        import torch
        backend = getattr(getattr(torch, "backends", None), "mps", None)
        if backend is None or not backend.is_available():
            return "Apple Metal (MPS) is unavailable in this PyTorch environment."
        from .preflight import exercise_torch
        exercise_torch("mps")
    except Exception as exc:
        return f"Apple Metal (MPS) cannot run U-Net in this environment: {type(exc).__name__}: {exc}"
    return None


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
    project_dir: Path
    unet_weights_dir: Path | None
    unet_device: str
    cpu_threads: int | None

    @classmethod
    def from_env(
        cls,
        *,
        project_dir: Path | None = None,
        unet_weights_dir: Path | str | None = None,
        unet_device: str | None = None,
        cpu_threads: int | str | None = None,
        user_config: Mapping[str, Any] | None = None,
    ) -> "RuntimeConfig":
        config = load_config() if user_config is None else user_config
        root = default_repo_root().resolve()
        project = (Path(project_dir).expanduser().resolve() if project_dir is not None
                   else resolve_project_dir(None, config, default=root))
        return cls(
            repo_root=root,
            project_dir=project,
            unet_weights_dir=resolve_weights_dir(unet_weights_dir, config),
            unet_device=resolve_device(unet_device, config) or default_unet_device(),
            cpu_threads=resolve_cpu_threads(cpu_threads, config),
        )
