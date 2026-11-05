"""Choose pinned installation builds without needing an installed PyTorch.

CUDA is queried in a child process: a broken vendor driver must not crash or
hang the installer. The driver API enumerates devices visible to this process,
including CUDA_VISIBLE_DEVICES restrictions used by cluster schedulers.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


@dataclass(frozen=True)
class TorchProfile:
    name: str
    requirement: str | None
    index: str | None
    reason: str

    @property
    def cuda(self) -> bool:
        return self.name.startswith("cu")


def cpu_profile(target: str, reason: str) -> TorchProfile:
    if target.startswith("osx"):
        return TorchProfile("conda-cpu", None, None, reason)
    return TorchProfile("cpu", "torch==2.5.1+cpu", "https://download.pytorch.org/whl/cpu", reason)


def clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def probe_cuda() -> dict:
    if os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}:
        return {"devices": [], "reason": "CUDA devices are hidden in this session."}
    try:
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--probe"],
            capture_output=True, text=True, timeout=20, env=clean_environment(),
        )
        if result.returncode:
            return {"devices": [], "reason": f"CUDA driver probe failed (exit {result.returncode})."}
        payload = json.loads(result.stdout)
        # Validate the subprocess wire format before making installation choices.
        if not isinstance(payload, dict) or not isinstance(payload.get("devices"), list):
            raise ValueError("invalid driver probe response")
        for device in payload["devices"]:
            major, minor = device["capability"]
            if type(major) is not int or type(minor) is not int or major < 1 or minor < 0:
                raise ValueError("invalid GPU capability")
        if payload["devices"] and type(payload.get("driver_api")) is not int:
            raise ValueError("invalid CUDA driver version")
        return payload
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        return {"devices": [], "reason": f"CUDA driver could not be inspected: {exc}"}


def select_torch_profile(target: str, hardware: dict, libc: tuple[str, str] | None = None) -> TorchProfile:
    if target.startswith("osx"):
        return cpu_profile(target, "Mac installations use Conda PyTorch on CPU.")
    devices = hardware.get("devices", [])
    if not devices:
        return cpu_profile(target, hardware.get("reason", "No usable NVIDIA GPU is visible in this session."))
    capabilities = [tuple(device["capability"]) for device in devices]
    driver = hardware["driver_api"]
    legacy = {(5, 0), (5, 2), (5, 3), (6, 0), (6, 1), (6, 2), (7, 0), (7, 2),
              (7, 5), (8, 0), (8, 6), (8, 7), (8, 9), (9, 0)}
    if all(capability in legacy for capability in capabilities) and driver >= 11000:
        # CUDA 11.x minor compatibility also depends on driver patch level and
        # kernel features. Real execution is mandatory before accepting it.
        return TorchProfile("cu118", "torch==2.5.1+cu118", "https://download.pytorch.org/whl/cu118",
                            "Visible NVIDIA GPUs support the legacy CUDA 11.8 build.")
    modern = {(7, 5), (8, 0), (8, 6), (8, 7), (8, 9), (9, 0), (10, 0), (12, 0)}
    if all(capability in modern for capability in capabilities) and driver >= 12080:
        if target == "linux-64":
            name, release = libc or platform.libc_ver()
            try:
                compatible = name == "glibc" and tuple(map(int, release.split(".")[:2])) >= (2, 28)
            except ValueError:
                compatible = False
            if not compatible:
                return cpu_profile(target, "These GPUs need the CUDA 12.8 wheel, which requires Linux glibc >=2.28.")
        return TorchProfile("cu128", "torch==2.7.1+cu128", "https://download.pytorch.org/whl/cu128",
                            "Visible NVIDIA GPUs and driver support the CUDA 12.8 build.")
    return cpu_profile(target, "No pinned CUDA build matches every visible GPU and the installed driver; using CPU.")


def _driver_probe() -> dict:
    import ctypes

    try:
        library = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")

        def call(name, arguments, *values):
            function = getattr(library, name)
            function.argtypes = arguments
            function.restype = ctypes.c_int
            code = function(*values)
            if code:
                raise RuntimeError(f"{name} returned CUDA error {code}")

        integer = ctypes.c_int
        pointer = ctypes.POINTER(integer)
        call("cuInit", [ctypes.c_uint], 0)
        version, count = integer(), integer()
        call("cuDriverGetVersion", [pointer], ctypes.byref(version))
        call("cuDeviceGetCount", [pointer], ctypes.byref(count))
        devices = []
        for ordinal in range(count.value):
            device, major, minor = integer(), integer(), integer()
            name = ctypes.create_string_buffer(256)
            call("cuDeviceGet", [pointer, integer], ctypes.byref(device), ordinal)
            call("cuDeviceGetName", [ctypes.POINTER(ctypes.c_char), integer, integer], name, len(name), device)
            call("cuDeviceComputeCapability", [pointer, pointer, integer],
                 ctypes.byref(major), ctypes.byref(minor), device)
            devices.append({"name": name.value.decode(errors="replace"),
                            "capability": [major.value, minor.value]})
        return {"driver_api": version.value, "devices": devices}
    except (OSError, AttributeError, RuntimeError) as exc:
        return {"devices": [], "reason": f"No usable CUDA driver: {exc}"}


if __name__ == "__main__":
    print(json.dumps(_driver_probe()))
