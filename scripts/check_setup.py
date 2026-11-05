"""Exercise an installed CryoID, including native libraries and bundled tools.

Run from any directory: python /path/to/repo/scripts/check_setup.py --gui
No external weights, scientific datasets, network access, or system BLAST needed.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from importlib.metadata import version


def check_setup(*, gui: bool = False, expected_site: Path | None = None) -> dict:
    report = {"platform": platform.platform(), "machine": platform.machine(),
              "python": platform.python_version(), "executable": sys.executable, "checks": {}}
    checks = report["checks"]
    for module, distribution in (
        ("Bio", "biopython"), ("gemmi", "gemmi"), ("matplotlib", "matplotlib"),
        ("mrcfile", "mrcfile"), ("numpy", "numpy"), ("pandas", "pandas"),
        ("parasail", "parasail"), ("requests", "requests"), ("sklearn", "scikit-learn"),
        ("scipy", "scipy"), ("threadpoolctl", "threadpoolctl"), ("torch", "torch"),
    ):
        importlib.import_module(module)
        checks[distribution] = version(distribution)

    import numpy as np
    import torch
    import parasail
    import mrcfile
    from C_Unet_model_predicting.model import UNET
    from D_BLAST_searching.blast_tools import BLAST_VERSION, BLAST_PROGRAMS, blast_executable, bundled_platform_key
    from cryoid_pipeline.user_config import save_config, load_config
    import cryoid_pipeline
    report["package_root"] = str(Path(cryoid_pipeline.__file__).resolve().parent)
    if expected_site is not None:
        assert Path(cryoid_pipeline.__file__).resolve().is_relative_to(expected_site.resolve())

    # Import every production stage, including both structure-fitting paths.
    for module in ("C_Unet_model_predicting.stage", "H_Backbone_tracing.stage",
                   "D_BLAST_searching.stage", "I_Structure_fitting.stage",
                   "I_Structure_fitting.protein.core.fetch", "I_Structure_fitting.rna.structures"):
        imported = importlib.import_module(module)
        if expected_site is not None:
            assert Path(imported.__file__).resolve().is_relative_to(expected_site.resolve()), module
    checks["stage_imports"] = True
    # Import alone misses the NumPy 2 / Intel PyTorch ABI incompatibility.
    array = np.arange(8, dtype=np.float32)
    np.testing.assert_array_equal(torch.from_numpy(array).numpy(), array)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(2, previous_threads))
        # The real model's operations with small feature counts keep this check
        # fast. Scientific validation with pretrained checkpoints is separate.
        model = UNET(1, 2, features=[2, 4]).eval()
        with torch.no_grad():
            output = model(torch.ones(1, 1, 8, 8, 8))
        assert output.shape == (1, 2, 8, 8, 8) and torch.isfinite(output).all()
    finally:
        torch.set_num_threads(previous_threads)
    checks["cpu_unet_and_numpy_bridge"] = True
    for bits in (16, 32):
        result = getattr(parasail, f"sw_striped_{bits}")("ACDEFGHIK", "ACDEFGHIK", 11, 1, parasail.blosum62)
        assert result.score == 53 and not result.saturated, result.score
    checks["parasail_native_alignment"] = True

    with tempfile.TemporaryDirectory(prefix="cryoid-setup-") as tmp:
        root = Path(tmp)
        config = root / "config.toml"
        save_config({"weights_dir": str(root / "weights with spaces"), "device": "cpu"}, config)
        assert load_config(config)["weights_dir"] == str(root / "weights with spaces")
        density = root / "density.mrc"
        with mrcfile.new(density) as handle:
            handle.set_data(np.ones((4, 4, 4), dtype=np.float32))
        with mrcfile.open(density) as handle:
            assert handle.data.shape == (4, 4, 4)
        checks["config_and_map_io"] = True

        environment = dict(os.environ, PATH="", CRYOID_BLAST_BIN_DIR="invalid-legacy-override")

        def run(program, *args):
            result = subprocess.run([blast_executable(program), *map(str, args)], env=environment,
                                    capture_output=True, text=True, timeout=60)
            if result.returncode:
                raise RuntimeError(f"{program} failed ({result.returncode}): {result.stderr or result.stdout}")
            return result.stdout

        checks["blast_platform"] = bundled_platform_key()
        for program in BLAST_PROGRAMS:
            output = run(program, "-version")
            assert f"{program}: {BLAST_VERSION}+" in output, output
        for dbtype, program, sequence in (
            ("prot", "blastp", "MTEYKLVVVGAGGVGKSALTIQLIQNHFVDEYDPTIEDSYRKQVVIDGETCLLDILDTAG"),
            ("nucl", "blastn", "ACGTGTCAGTACGATCGTAGCTAGTCGATCGATGCATGCTAGCTGACTGATCGTACGTAGCTAGTC"),
        ):
            fasta = root / f"{dbtype}.fasta"
            fasta.write_text(f">setup_target\n{sequence}\n", encoding="ascii")
            database = root / f"{dbtype}_database"
            run("makeblastdb", "-in", fasta, "-dbtype", dbtype, "-out", database)
            hits = run(program, "-query", fasta, "-db", database, "-outfmt", "6 sseqid pident",
                       "-num_threads", "1")
            assert "setup_target\t100.000" in hits, hits
        checks["blast_protein_and_nucleotide_search_empty_path"] = True

        if gui:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            from G_GUI.qt_compat import QtWidgets, QT_API
            from G_GUI.settings_dialogs import OptionsWindow, ConfigurationsWindow
            from G_GUI.diagnostics import collect_diagnostics
            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            for dialog in (OptionsWindow(), ConfigurationsWindow()):
                dialog.show()
                app.processEvents()
                dialog.close()
            diagnostics = collect_diagnostics("", "cpu")
            assert diagnostics.blast.ok, diagnostics.blast
            assert diagnostics.compute.ok, diagnostics.compute
            assert not diagnostics.dependency_issues, diagnostics.dependency_issues
            checks["gui_offscreen_and_diagnostics"] = QT_API
            # Verify the installed CLI entry point through the current interpreter.
        result = subprocess.run([sys.executable, "-m", "cryoid_pipeline", "--help"],
                                capture_output=True, text=True, check=True, timeout=60)
        assert "run" in result.stdout
        checks["cli"] = True
    report["ok"] = True
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--expected-site", type=Path, help="Require production imports to come from this wheel installation.")
    args = parser.parse_args()
    report = check_setup(gui=args.gui, expected_site=args.expected_site)
    output = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
