"""Background setup checks for the Run button and setup details dialog."""
from __future__ import annotations

import importlib
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from cryoid_pipeline import __version__
from D_BLAST_searching.blast_tools import (
    BLAST_PROGRAMS, bundled_blast_bin_dir, blast_executable,
)


@dataclass
class Check:
    summary: str
    detail: str
    ok: bool
    technical_detail: str = ""


@dataclass
class Diagnostics:
    blast: Check
    compute: Check
    weights: Check
    device: str
    lines: list[str] = field(default_factory=list)
    dependency_issues: list[str] = field(default_factory=list)
    devices: list[tuple[str, str]] = field(default_factory=list)

    @property
    def issues(self) -> list[str]:
        issues = [f"{name}: {check.detail}" for name, check in (
            ("BLAST+", self.blast), ("U-Net device", self.compute), ("U-Net weights", self.weights),
        ) if not check.ok]
        issues.extend(self.dependency_issues)
        return issues


def check_weights(value: str) -> Check:
    from cryoid_pipeline.weights import auto_download_needed, weights_problems
    try:
        if auto_download_needed(value):
            return Check("Automatic download", "Published weights will be downloaded or recovered automatically when the run starts.", True)
        directory = Path(value).expanduser().resolve()
        problems = weights_problems(directory)
    except (OSError, ValueError) as exc:
        return Check("Folder unavailable", f"Choose a valid weights folder or clear the field to autodownload. {exc}", False)
    if problems:
        return Check("Needs attention", "\n".join(problems) +
                     f"\nChoose the complete weights folder or clear Custom weights path to autodownload.\nFolder: {directory}", False)
    return Check("Files verified", f"{directory}\nMetadata and all three checkpoint files are present and readable. "
                 "Model compatibility is checked when the checkpoints are loaded.", True)


def check_blast_tools() -> Check:
    try:
        return Check("Bundled / CPU", str(bundled_blast_bin_dir()), True)
    except RuntimeError as exc:
        return Check("Unsupported platform", str(exc), False)
    except (OSError, ValueError):
        return Check("Needs repair", "CryoID's bundled BLAST+ tools are missing or cannot be opened. "
                     "Reinstall CryoID to restore the search tools.", False)


def folder_write_issue(value: str) -> str | None:
    if not value.strip():
        return "Choose a folder where CryoID can save results."
    try:
        target = Path(value).expanduser().resolve()
        ancestor = target
        while not ancestor.exists() and ancestor != ancestor.parent:
            ancestor = ancestor.parent
        if not ancestor.is_dir():
            return f"This path points to a file. Choose a folder instead.\nLocation: {target}"
        if not os.access(ancestor, os.W_OK):
            return f"CryoID cannot write to this location. Choose a writable folder.\nLocation: {target}"
    except (OSError, ValueError):
        return "This location cannot be opened. Choose a valid, writable folder."
    return None


def collect_diagnostics(weights_path: str, device: str | None) -> Diagnostics:
    devices = []
    dependency_issues = []
    lines = [
        f"CryoID2 version {__version__}",
        f"Operating system: {platform.platform()}",
        f"Python: {platform.python_version()} ({sys.executable})",
        f"CPU: {platform.processor() or platform.machine()}",
        "Dependency checks:",
    ]
    # Import as well as checking versions, so broken native libraries are reported.
    for module, distribution in (
        ("Bio", "biopython"), ("gemmi", "gemmi"), ("matplotlib", "matplotlib"),
        ("mrcfile", "mrcfile"), ("numpy", "numpy"), ("pandas", "pandas"),
        ("parasail", "parasail"), ("requests", "requests"),
        ("sklearn", "scikit-learn"), ("scipy", "scipy"),
        ("threadpoolctl", "threadpoolctl"),
    ):
        try:
            importlib.import_module(module)
            try:
                installed_version = version(distribution)
            except PackageNotFoundError:
                installed_version = "version unavailable"
            lines.append(f"  {distribution}: OK ({installed_version})")
        except Exception as exc:
            lines.append(f"  {distribution}: ERROR ({type(exc).__name__}: {exc})")
            dependency_issues.append(
                f"Python dependency — {distribution}: This package is missing or could not be loaded. "
                "Install or repair the CryoID dependencies in the Python environment used to start this GUI."
            )
    from G_GUI.qt_compat import QT_API, QtCore
    lines.append(f"  {QT_API}: {QtCore.PYQT_VERSION_STR}; Qt: {QtCore.QT_VERSION_STR}")
    selected_device = device or "cpu"
    try:
        import torch
        lines.append(f"PyTorch: {torch.__version__}; CUDA build: {torch.version.cuda or 'CPU only'}")
        available = torch.cuda.is_available()
        if available:
            devices = [(f"cuda:{index}", torch.cuda.get_device_name(index))
                       for index in range(torch.cuda.device_count())]
        selected_device = device or ("cuda:0" if available else "cpu")
        target = torch.device(selected_device)
        cuda_detail = (
            "; ".join(f"{index}: {torch.cuda.get_device_name(index)}"
                       for index in range(torch.cuda.device_count()))
            if available else "not available"
        )
        lines.append(f"CUDA: {cuda_detail}")
        if target.type == "cuda":
            index = target.index if target.index is not None else 0
            if not available or index >= torch.cuda.device_count():
                compute = Check("GPU unavailable", f"The selected GPU ({selected_device}) is unavailable. "
                                "Use CPU in the CryoID configuration, or install a CUDA-enabled PyTorch "
                                "version and a compatible GPU driver.", False)
            else:
                compute = Check("GPU / CUDA", f"U-Net: {selected_device} ({torch.cuda.get_device_name(index)})", True)
        elif target.type == "cpu":
            compute = Check("CPU", f"U-Net uses CPU. CUDA: {cuda_detail}", True)
        else:
            compute = Check(str(target), f"U-Net device: {target}; availability is checked at runtime.", True)
    except Exception as exc:
        compute = Check("Unavailable", "PyTorch or the selected computing device could not be loaded. "
                        "Install or repair PyTorch in the Python environment used to start CryoID, "
                        "and select CPU or an available GPU in the CryoID configuration.", False,
                        f"{type(exc).__name__}: {exc}")
        lines.append(f"PyTorch/device check: ERROR ({compute.technical_detail})")
    lines.append(f"U-Net device: {selected_device}; {compute.detail}")

    blast = check_blast_tools()
    if blast.ok:
        directory = Path(blast.detail)
        versions, failures, technical = [], [], []
        for program in BLAST_PROGRAMS:
            try:
                result = subprocess.run(
                    [blast_executable(program, directory), "-version"],
                    capture_output=True, text=True, errors="replace", timeout=5, check=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
                )
                output = (result.stdout or result.stderr).strip()
                if not output:
                    raise ValueError("No version information returned")
                versions.append(output.splitlines()[0])
            except subprocess.TimeoutExpired as exc:
                failures.append(f"{program} did not respond. Check the installation, then use Settings > Check setup again.")
                technical.append(str(exc))
            except Exception as exc:
                failures.append(f"{program} could not run. Reinstall CryoID to restore its bundled search tools.")
                technical.append(f"{type(exc).__name__}: {exc}")
        blast = Check("Needs attention" if failures else "Bundled / CPU",
                      "\n".join(failures) + f"\nFolder: {directory}" if failures else f"{directory}\n" + "\n".join(versions),
                      not failures, "\n".join(technical))
    lines.append(f"BLAST+: {blast.summary}; {blast.detail}")
    lines.append("Acceleration: BLAST+ runs on CPU; CUDA accelerates U-Net inference only.")
    weights = check_weights(weights_path)
    lines.append(f"U-Net weights: {weights.summary}; {weights.detail}")
    return Diagnostics(blast, compute, weights, selected_device, lines, dependency_issues, devices)
