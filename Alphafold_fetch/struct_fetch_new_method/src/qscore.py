"""
Headless ChimeraX Q-score wrapper.

Spawns ChimeraX in nogui mode with an embedded Python script that loads the
model and map, runs ``fitmap`` for a final local rigid refinement, and then
invokes the ChimeraX ``qscore`` command on all residues. Returns the mean
per-atom Q-score back to the caller.

Two entry points:

- :func:`compute_model_qscore` scores one (model, map) pair.
- :func:`compute_qscore_pair` scores both a GT model and an identified model
  against the same map in one ChimeraX session.

ChimeraX is auto-detected under ``C:/Program Files/ChimeraX*/`` and via the
``CHIMERAX_PATH`` environment variable; an explicit path can be passed too.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Optional


# Embedded ChimeraX script: receives mode + paths via argv, writes a JSON
# result file. Stdout/stderr are redirected to /dev/null inside the session
# so noisy ChimeraX banners don't pollute the wrapper's output.
CHIMERAX_SCRIPT = r"""
import contextlib
import json
import os
import pathlib
import sys

from chimerax.core.commands import run
from chimerax.qscore.cmd import qscore as qscore_cmd


def _write_out(path, payload):
    pathlib.Path(path).write_text(json.dumps(payload), encoding="utf-8")


def _score_model(model_path, map_path):
    run(session, f'open "{model_path}"', log=False)
    run(session, f'open "{map_path}"', log=False)

    from chimerax.atomic import AtomicStructure
    from chimerax.map import Volume

    model = next((m for m in session.models.list() if isinstance(m, AtomicStructure)), None)
    volume = next((m for m in session.models.list() if isinstance(m, Volume)), None)
    if model is None:
        raise RuntimeError(f"No AtomicStructure loaded for {model_path}")
    if volume is None:
        raise RuntimeError(f"No map Volume loaded for {map_path}")

    model_spec = "#" + ".".join(str(i) for i in model.id)
    volume_spec = "#" + ".".join(str(i) for i in volume.id)

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
        run(session, "close all", log=False)
    return score


def main():
    if len(sys.argv) < 4:
        _write_out(sys.argv[-1] if sys.argv else "qscore_out.json", {"error": f"Missing args: {sys.argv!r}"})
        return

    mode = sys.argv[1]
    out_path = sys.argv[-1]
    try:
        if mode == "single":
            payload = {"qscore": _score_model(sys.argv[2], sys.argv[3])}
        elif mode == "pair":
            payload = {
                "gt_qscore": _score_model(sys.argv[2], sys.argv[4]),
                "identified_qscore": _score_model(sys.argv[3], sys.argv[4]),
            }
        else:
            payload = {"error": f"Unknown mode: {mode}"}
    except Exception as exc:
        payload = {"error": repr(exc)}
    _write_out(out_path, payload)


main()
"""


def find_chimerax(explicit_path: Optional[str | Path] = None) -> Path:
    """Locate the ChimeraX executable.

    Lookup order: ``explicit_path`` arg → ``CHIMERAX_PATH`` env var →
    auto-detection under ``C:/Program Files{,(x86)}/ChimeraX*/bin/``. Raises
    ``FileNotFoundError`` if no candidate exists.
    """
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.exists():
            return path.resolve()
        raise FileNotFoundError(f"ChimeraX not found at: {path}")

    env_path = os.environ.get("CHIMERAX_PATH")
    if env_path:
        path = Path(env_path).expanduser()
        if path.exists():
            return path.resolve()

    for root in (Path("C:/Program Files"), Path("C:/Program Files (x86)")):
        for pattern in ("ChimeraX*/bin/ChimeraX-console.exe", "ChimeraX*/bin/ChimeraX.exe"):
            for candidate in root.glob(pattern):
                if candidate.exists():
                    return candidate.resolve()

    raise FileNotFoundError("ChimeraX executable not found. Set CHIMERAX_PATH or pass --chimerax.")


def _run_chimerax(
    *,
    args: list[str],
    chimerax_path: Path,
    work_root: Path,
    timeout_sec: int,
) -> tuple[dict[str, Any], str, float]:
    """Spawn a one-shot ChimeraX session and return ``(payload, error, elapsed_s)``.

    Always cleans up the per-call working directory. ``error`` is empty on
    success, otherwise an opaque diagnostic string.
    """
    work_dir = work_root / uuid.uuid4().hex
    work_dir.mkdir(parents=True, exist_ok=True)
    script_path = work_dir / "qscore_runner.py"
    out_path = work_dir / "qscore_out.json"
    script_path.write_text(CHIMERAX_SCRIPT, encoding="utf-8")
    script_arg = " ".join([f'"{script_path}"', *[f'"{arg}"' for arg in args], f'"{out_path}"'])

    started = time.perf_counter()
    try:
        try:
            result = subprocess.run(
                [str(chimerax_path), "--nogui", "--silent", "--exit", "--script", script_arg],
                capture_output=True,
                text=True,
                timeout=int(timeout_sec),
            )
        except subprocess.TimeoutExpired:
            return {}, "qscore_timeout", float(time.perf_counter() - started)

        elapsed = float(time.perf_counter() - started)
        if not out_path.exists():
            details = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
            return {}, f"qscore_no_output:{details}", elapsed
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("error"):
            details = ((result.stdout or "") + "\n" + (result.stderr or "")).strip()
            return {}, f"{payload['error']} | {details}".strip(), elapsed
        return dict(payload), "", elapsed
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def compute_model_qscore(
    *,
    model_path: Path,
    map_path: Path,
    chimerax_path: Path,
    work_root: Path,
    timeout_sec: int,
) -> tuple[float | None, str, float]:
    """Run ChimeraX ``fitmap`` + ``qscore`` for one (model, map) pair.

    Returns ``(mean_qscore_or_None, error_string, elapsed_seconds)``.
    """
    for label, path in (("model_path", model_path), ("map_path", map_path), ("chimerax_path", chimerax_path)):
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    payload, error, elapsed = _run_chimerax(
        args=["single", str(model_path), str(map_path)],
        chimerax_path=chimerax_path,
        work_root=work_root,
        timeout_sec=timeout_sec,
    )
    if error:
        return None, error, elapsed
    value = payload.get("qscore")
    return (float(value) if value is not None else None), "", elapsed


def compute_qscore_pair(
    *,
    gt_model_path: Path,
    identified_model_path: Path,
    map_path: Path,
    chimerax_path: Path,
    work_root: Path,
    timeout_sec: int,
) -> tuple[float | None, float | None, str, float]:
    """Score both a GT model and an identified model against the same map in one ChimeraX session.

    Returns ``(gt_qscore_or_None, identified_qscore_or_None, error, elapsed_seconds)``.
    """
    for label, path in (
        ("gt_model_path", gt_model_path),
        ("identified_model_path", identified_model_path),
        ("map_path", map_path),
        ("chimerax_path", chimerax_path),
    ):
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    payload, error, elapsed = _run_chimerax(
        args=["pair", str(gt_model_path), str(identified_model_path), str(map_path)],
        chimerax_path=chimerax_path,
        work_root=work_root,
        timeout_sec=timeout_sec,
    )
    if error:
        return None, None, error, elapsed
    gt = payload.get("gt_qscore")
    identified = payload.get("identified_qscore")
    return (
        float(gt) if gt is not None else None,
        float(identified) if identified is not None else None,
        "",
        elapsed,
    )
