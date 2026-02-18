#!/usr/bin/env python3
import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

DEFAULT_MAP_PATH = "Alphafold_fetch/test_files/emd_29659_CDABP0047.map"
DEFAULT_MODEL_PATH = "Alphafold_fetch/fitted_models_v6/CDABP0047__PDB_8G17_fitted_v5.pdb"

CHIMERAX_SCRIPT = r"""
import contextlib
import os
import pathlib
import sys

from chimerax.core.commands import run
from chimerax.qscore.cmd import qscore as qscore_cmd


def _write_out(path, text):
    pathlib.Path(path).write_text(text, encoding="utf-8")


def main():
    if len(sys.argv) < 4:
        _write_out(sys.argv[-1] if sys.argv else "qscore_out.txt", "ERROR: Missing args")
        return

    model_path = sys.argv[1]
    map_path = sys.argv[2]
    out_path = sys.argv[3]

    try:
        run(session, f'open "{model_path}"', log=False)
        run(session, f'open "{map_path}"', log=False)

        from chimerax.atomic import AtomicStructure
        from chimerax.map import Volume

        model = next((m for m in session.models.list() if isinstance(m, AtomicStructure)), None)
        volume = next((m for m in session.models.list() if isinstance(m, Volume)), None)
        if model is None:
            _write_out(out_path, "ERROR: No AtomicStructure loaded")
            return
        if volume is None:
            _write_out(out_path, "ERROR: No map Volume loaded")
            return

        model_spec = "#" + ".".join(str(i) for i in model.id)
        volume_spec = "#" + ".".join(str(i) for i in volume.id)

        # qscore.py may emit print() warnings; silence stdout/stderr during computation.
        with open(os.devnull, "w", encoding="utf-8") as devnull, \
                contextlib.redirect_stdout(devnull), \
                contextlib.redirect_stderr(devnull):
            run(session, f"fitmap {model_spec} inMap {volume_spec}", log=False)
            _, (_, atom_scores) = qscore_cmd(
                session,
                model.residues,
                to_volume=volume,
                use_gui=False,
                log_details=False,
            )

        score = float(atom_scores.mean())
        _write_out(out_path, f"{score:.3f}")
    except Exception as exc:
        _write_out(out_path, f"ERROR: {exc}")


main()
"""


def _find_chimerax(explicit_path: Optional[str]) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"ChimeraX not found at: {path}")

    env_path = os.environ.get("CHIMERAX_PATH")
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            return path

    for root in (Path("C:/Program Files"), Path("C:/Program Files (x86)")):
        for cand in root.glob("ChimeraX*/bin/ChimeraX.exe"):
            if cand.exists():
                return cand

    raise FileNotFoundError(
        "ChimeraX.exe not found. Set CHIMERAX_PATH or pass --chimerax."
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute ChimeraX Q-score for a model against a map."
    )
    parser.add_argument(
        "--map",
        help=f"Path to .map/.mrc file (default: {DEFAULT_MAP_PATH})",
    )
    parser.add_argument(
        "--model",
        help=f"Path to .pdb/.cif file (default: {DEFAULT_MODEL_PATH})",
    )
    parser.add_argument(
        "--chimerax",
        help="Optional path to ChimeraX.exe (or set CHIMERAX_PATH).",
    )
    args = parser.parse_args()

    map_arg = args.map or DEFAULT_MAP_PATH
    model_arg = args.model or DEFAULT_MODEL_PATH
    if not map_arg or not model_arg:
        raise SystemExit("Map/model paths are required (args or defaults).")

    map_path = Path(map_arg).expanduser().resolve()
    model_path = Path(model_arg).expanduser().resolve()
    if not map_path.exists():
        raise SystemExit(f"Map not found: {map_path}")
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    chimerax_path = _find_chimerax(args.chimerax)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        script_path = tmpdir_path / "qscore_runner.py"
        out_path = tmpdir_path / "qscore_out.txt"
        script_path.write_text(CHIMERAX_SCRIPT, encoding="utf-8")

        script_arg = f'"{script_path}" "{model_path}" "{map_path}" "{out_path}"'
        result = subprocess.run(
            [str(chimerax_path), "--nogui", "--silent", "--exit", "--script", script_arg],
            capture_output=True,
            text=True,
        )

        if not out_path.exists():
            details = (result.stdout or "") + (result.stderr or "")
            raise SystemExit(f"ChimeraX did not produce output.\n{details}".strip())

        content = out_path.read_text(encoding="utf-8").strip()
        if content.startswith("ERROR:"):
            details = (result.stdout or "") + (result.stderr or "")
            raise SystemExit(f"{content}\n{details}".strip())

        print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
