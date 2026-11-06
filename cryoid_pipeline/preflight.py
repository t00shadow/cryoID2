"""Environment validation shared by GUI setup and the headless setup command.

Keep scientific imports inside checks so a broken installation can be diagnosed.
"""
from __future__ import annotations

import argparse
import importlib
from importlib import metadata
import platform
from pathlib import Path
import struct
import sys


IMPORTS = {
    "biopython": "Bio.SeqIO", "gemmi": "gemmi", "huggingface-hub": "huggingface_hub",
    "matplotlib": "matplotlib.pyplot", "mrcfile": "mrcfile", "numpy": "numpy",
    "pandas": "pandas", "parasail": "parasail", "requests": "requests",
    "scikit-learn": "sklearn.cluster", "scipy": "scipy.ndimage", "torch": "torch",
    "threadpoolctl": "threadpoolctl", "tomli": "tomli", "packaging": "packaging",
    "pillow": "PIL.Image", "pyqt6": "PyQt6.QtWidgets", "pyqt5": "PyQt5.QtWidgets",
}


def project_requirements() -> tuple[str, list[str], dict[str, list[str]]]:
    """Use current source requirements for editable installs, wheel metadata otherwise."""
    project_file = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if project_file.is_file():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        project = tomllib.loads(project_file.read_text(encoding="utf-8"))["project"]
        if project.get("name") == "cryoid-v2-combination":
            return (project["requires-python"], project["dependencies"],
                    project["optional-dependencies"])
    dist = metadata.distribution("cryoid-v2-combination")
    # Markers on optional dependencies are evaluated with the selected GUI extra.
    return dist.metadata["Requires-Python"], dist.requires or [], {}


def check_environment(qt_api: str | None = None) -> tuple[list[str], list[str]]:
    """Validate Python, declared versions, dependency closure, and native imports."""
    lines = [f"Python architecture: {struct.calcsize('P') * 8}-bit; {platform.machine()}"]
    issues = []
    if struct.calcsize("P") != 8:
        issues.append("Python architecture: CryoID requires 64-bit Python. Create a new environment with 64-bit Python.")
    if platform.python_implementation() != "CPython":
        issues.append("Python implementation: CryoID requires CPython with compatible native package builds.")
    from D_BLAST_searching.blast_tools import bundled_platform_key
    try:
        bundled_platform_key()
    except RuntimeError as exc:
        issues.append(str(exc))
    if sys.platform == "darwin":
        try:
            if int(platform.mac_ver()[0].split(".")[0]) < 14:
                issues.append("Operating system: CryoID requires macOS 14 or newer.")
        except ValueError:
            issues.append("Operating system: could not determine the macOS version.")
    try:
        from packaging.requirements import Requirement
        from packaging.specifiers import SpecifierSet
        from packaging.utils import canonicalize_name
        python_spec, required, extras = project_requirements()
        if not SpecifierSet(python_spec).contains(platform.python_version()):
            issues.append(f"Python version: {platform.python_version()} is unsupported; requires {python_spec}.")
        selected_extra = "gui-pyqt5" if qt_api == "PyQt5" else "gui-pyqt6" if qt_api else ""
        pending = [(text, "CryoID", {selected_extra}) for text in required]
        pending.extend((text, "CryoID GUI", {""}) for text in extras.get(selected_extra, []))
        visited = set()
        imports = set()
        while pending:
            text, parent, active_extras = pending.pop(0)
            req = Requirement(text)
            if req.marker and not any(req.marker.evaluate({"extra": extra}) for extra in active_extras | {""}):
                continue
            name = canonicalize_name(req.name)
            try:
                dist = metadata.distribution(req.name)
                installed = dist.version
                if not req.specifier.contains(installed, prereleases=True):
                    raise ValueError(f"installed {installed}; requires {req.specifier}")
                if dist.metadata.get("Requires-Python") and not SpecifierSet(dist.metadata["Requires-Python"]).contains(platform.python_version()):
                    raise ValueError(f"{installed} requires Python {dist.metadata['Requires-Python']}")
            except (metadata.PackageNotFoundError, ValueError) as exc:
                problem = f"{req.name}: {exc} (required by {parent}). Repair the active CryoID environment."
                if problem not in issues:
                    issues.append(problem)
                    lines.append(f"  {req.name}: ERROR ({exc})")
                continue
            if name in IMPORTS:
                imports.add((name, installed))
            key = (name, frozenset(req.extras))
            if key in visited:
                continue
            visited.add(key)
            pending.extend((child, req.name, set(req.extras)) for child in dist.requires or [])
        for name, installed in sorted(imports):
            try:
                module = importlib.import_module(IMPORTS[name])
                if name == "parasail":
                    # This import can succeed even when its shared library cannot run.
                    if module.sw_trace_striped_16("ACDE", "ACDE", 10, 1, module.blosum62).score <= 0:
                        raise RuntimeError("Parasail native alignment failed")
                elif name == "pillow":
                    module.new("RGB", (1, 1)).close()
                lines.append(f"  {name}: OK ({installed})")
            except Exception as exc:
                lines.append(f"  {name}: ERROR ({type(exc).__name__}: {exc})")
                issues.append(f"Python dependency — {name}: {exc}. Install or repair this package in the environment used to start CryoID.")
    except Exception as exc:
        issues.append(f"Dependency validation could not finish: {type(exc).__name__}: {exc}. Reinstall CryoID and its dependencies.")
    return lines, issues


def exercise_torch(device: str) -> None:
    """Exercise NumPy interchange and the 3-D convolution used by U-Net."""
    import numpy as np
    import torch
    target = torch.device(device)
    with torch.inference_mode():
        source = np.ones((1, 1, 4, 4, 4), dtype=np.float32)
        tensor = torch.from_numpy(source).to(target)
        kernel = torch.zeros((1, 1, 3, 3, 3), device=target)
        kernel[:, :, 1, 1, 1] = 1
        output = torch.nn.functional.conv3d(tensor, kernel, padding=1).cpu().numpy()
        if not np.array_equal(output, source):
            raise RuntimeError("PyTorch/NumPy convolution check returned incorrect results")
        if target.type == "mps":
            exercise_unet(device)
        if target.type == "cuda":
            torch.cuda.synchronize(target)


def exercise_unet(device: str) -> None:
    """Run a small U-Net through pooling, normalization, and transposed convolution."""
    import numpy as np
    import torch
    from C_Unet_model_predicting.model import UNET

    with torch.inference_mode():
        model = UNET(1, 2, features=(2, 4)).eval().to(device)
        source = torch.ones((1, 1, 8, 8, 8), device=device, dtype=torch.float32)
        output = torch.softmax(model(source), dim=1).cpu().numpy()
        if output.shape != (1, 2, 8, 8, 8) or not np.isfinite(output).all():
            raise RuntimeError("U-Net execution check returned invalid predictions")
        if not np.allclose(output.sum(axis=1), 1.0):
            raise RuntimeError("U-Net execution check returned invalid probabilities")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check CryoID dependencies without loading weights or starting inference.")
    parser.add_argument("--gui", action="store_true", help="Also check the installed Qt binding")
    parser.add_argument("--device", default="cpu", help="Device to exercise (default: cpu)")
    args = parser.parse_args(argv)
    qt_api = None
    issues = []
    if args.gui:
        try:
            from G_GUI.qt_compat import QT_API
            qt_api = QT_API
        except (ImportError, OSError) as exc:
            issues.append(f"GUI: install a working GUI extra with python -m pip install -e '.[gui]'. {exc}")
    lines, dependency_issues = check_environment(qt_api)
    issues.extend(dependency_issues)
    try:
        exercise_torch(args.device)
        lines.append(f"PyTorch execution: OK ({args.device})")
    except Exception as exc:
        issues.append(f"PyTorch execution: {type(exc).__name__}: {exc}")
    from G_GUI.diagnostics import check_blast_tools
    blast = check_blast_tools()
    lines.append(f"BLAST+: {blast.detail}")
    if not blast.ok:
        issues.append(blast.detail + "\n" + blast.technical_detail)
    print("\n".join(lines))
    print("\n".join(f"ERROR: {issue}" for issue in issues) if issues else "Setup checks passed.")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
