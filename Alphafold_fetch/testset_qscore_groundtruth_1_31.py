#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import importlib
import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, TextIO, Tuple

import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

STRUCT_FETCH_PATH = SCRIPT_DIR / "struct_fetch_v4.py"

# ------------------- Hardcoded parameters -------------------
CSV_PATH = SCRIPT_DIR / "ready_to_test_1_29.csv"
QSCORE_SCRIPT = SCRIPT_DIR / "qscore_chimerax.py"
# Output directory for all maps/models/logs produced during testing.
OUTPUT_DIR = SCRIPT_DIR / "qscore_groundtruth_run_2_23"
LOG_PATH = OUTPUT_DIR / "qscore_groundtruth_results.csv"
# Set to an integer to limit rows, or None for all rows.
MAX_ROWS = None

QSCORE_TIMEOUT_SEC = 1800
EMDB_DOWNLOAD_RETRIES = 6
EMDB_DOWNLOAD_BACKOFF_SEC = 2.0
EMDB_DOWNLOAD_CONNECT_TIMEOUT_SEC = 20
EMDB_DOWNLOAD_READ_TIMEOUT_SEC = 120

# Test-only randomized start pose for every fetched structure (GT + candidates).
RANDOMIZE_FETCHED_STRUCTURES = True
RANDOMIZE_FETCHED_STRUCTURES_ATTEMPTS = 128
RANDOMIZE_FETCHED_STRUCTURES_SEED: Optional[int] = None

REQUIRED_COLUMNS = (
    "gene_id",
    "sequence",
    "fitted_pdbs",
    "emdb_id",
)

EXTRA_LOG_COLUMNS = (
    "map_file",
    "ground_truth_model",
    "selected_pdb_id",
    "selected_fitted_model",
    "qscore_ground_truth",
    "qscore_identified",
    "qscore_diff",
    "status",
    "error",
)

TERMINAL_LOG_BASENAME = "terminal_session"

_FALLBACK_NOTICE_SEEN: set[str] = set()

def _fallback_notice(tag: str, message: str, once: bool = False) -> None:
    if once and tag in _FALLBACK_NOTICE_SEEN:
        return
    _FALLBACK_NOTICE_SEEN.add(tag)
    print(f"[testset_qscore_gt][fallback] {message}")


class _TeeStream:
    def __init__(self, *streams: TextIO):
        self.streams = streams

    def write(self, data: str) -> int:
        for s in self.streams:
            s.write(data)
        return len(data)

    def flush(self) -> None:
        for s in self.streams:
            s.flush()

    def isatty(self) -> bool:
        for s in self.streams:
            if hasattr(s, "isatty") and s.isatty():
                return True
        return False

    @property
    def encoding(self) -> str:
        for s in self.streams:
            enc = getattr(s, "encoding", None)
            if enc:
                return enc
        return "utf-8"


def _enable_terminal_log(output_dir: Path) -> tuple[Path, TextIO, TextIO, TextIO]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"{TERMINAL_LOG_BASENAME}_{stamp}.log"
    fh = log_path.open("a", encoding="utf-8", buffering=1)
    orig_out = sys.stdout
    orig_err = sys.stderr
    sys.stdout = _TeeStream(orig_out, fh)  # type: ignore[assignment]
    sys.stderr = _TeeStream(orig_err, fh)  # type: ignore[assignment]
    return log_path, fh, orig_out, orig_err


def _disable_terminal_log(
    fh: TextIO,
    orig_out: TextIO,
    orig_err: TextIO,
) -> None:
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    sys.stdout = orig_out  # type: ignore[assignment]
    sys.stderr = orig_err  # type: ignore[assignment]
    try:
        fh.flush()
        fh.close()
    except Exception:
        pass


def _sanitize_name(text: str) -> str:
    import re as _re
    return _re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_") or "target"


def _make_session() -> requests.Session:
    retries = Retry(total=4, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "cryoid-qscore-groundtruth/1.0"})
    return session


def _normalize_emdb_id(emdb_id: str) -> str:
    raw = str(emdb_id or "").strip().upper()
    if not raw:
        _fallback_notice("normalize_emdb_empty", "EMDB id is empty after normalization.")
        return ""
    if raw.startswith("EMD-"):
        return raw
    if raw.startswith("EMD"):
        raw = raw[3:]
        if raw.startswith("-"):
            raw = raw[1:]
    if raw.isdigit():
        return f"EMD-{raw}"
    return raw


def _emdb_numeric_id(emdb_id: str) -> Optional[str]:
    emdb_id = _normalize_emdb_id(emdb_id)
    if not emdb_id:
        _fallback_notice("emdb_numeric_missing", "Could not extract numeric EMDB id from empty input.")
        return None
    if emdb_id.startswith("EMD-"):
        suffix = emdb_id[4:]
        if suffix.isdigit():
            return suffix
        _fallback_notice("emdb_numeric_invalid_suffix", f"Invalid EMDB numeric suffix in '{emdb_id}'.")
        return None
    return None


def _download_emdb_map(
    session: requests.Session, emdb_id: str, out_path: Path
) -> Optional[Path]:
    num = _emdb_numeric_id(emdb_id)
    if not num:
        _fallback_notice("download_emdb_bad_id", f"Skipping map download because EMDB id '{emdb_id}' is invalid.")
        return None

    base = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{num}/map"
    candidates = [
        f"{base}/emd_{num}.map.gz",
        f"{base}/emd_{num}.map",
    ]

    def _is_retryable_download_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        if "ftp.ebi.ac.uk" not in msg:
            return False
        retry_tokens = (
            "read timed out",
            "connection broken",
            "incompleteread",
            "chunkedencodingerror",
            "protocolerror",
            "remote end closed connection",
            "connection aborted",
            "connection reset",
        )
        return any(tok in msg for tok in retry_tokens)

    def _download_bytes(url: str) -> Optional[bytes]:
        for attempt in range(1, EMDB_DOWNLOAD_RETRIES + 1):
            try:
                with session.get(
                    url,
                    timeout=(EMDB_DOWNLOAD_CONNECT_TIMEOUT_SEC, EMDB_DOWNLOAD_READ_TIMEOUT_SEC),
                    stream=True,
                ) as resp:
                    if not resp.ok:
                        _fallback_notice(
                            "download_emdb_non_ok",
                            f"EMDB download returned HTTP {resp.status_code} for {url}; trying fallback source.",
                        )
                        return None
                    chunks: list[bytes] = []
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            chunks.append(chunk)
                    if not chunks:
                        _fallback_notice(
                            "download_emdb_empty_body",
                            f"EMDB download body was empty for {url}; trying fallback source.",
                        )
                        return None
                    data = b"".join(chunks)
                    # EMDB FTP can return HTML error pages.
                    if data.lstrip().startswith(b"<!DOCTYPE") or data.lstrip().startswith(b"<html"):
                        _fallback_notice(
                            "download_emdb_html_body",
                            f"EMDB returned HTML instead of map content for {url}; trying fallback source.",
                        )
                        return None
                    return data
            except requests.RequestException as exc:
                if _is_retryable_download_error(exc):
                    if attempt < EMDB_DOWNLOAD_RETRIES:
                        wait_s = EMDB_DOWNLOAD_BACKOFF_SEC * (2 ** (attempt - 1))
                        print(
                            f"[emdb] transient download error for {url}; "
                            f"retrying in {wait_s:.1f}s ({attempt}/{EMDB_DOWNLOAD_RETRIES})"
                        )
                        time.sleep(wait_s)
                        continue
                    print(f"[emdb] transient download error for {url}; retries exhausted")
                    return None
                raise

    for url in candidates:
        data = _download_bytes(url)
        if not data:
            _fallback_notice(
                "download_emdb_try_next_candidate",
                f"EMDB candidate URL failed ({url}); trying next candidate.",
            )
            continue
        if url.endswith(".gz"):
            try:
                data = gzip.decompress(data)
            except OSError:
                _fallback_notice(
                    "download_emdb_gzip_failed",
                    f"Failed to decompress EMDB gzip payload from {url}; trying next candidate.",
                )
                continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        if out_path.stat().st_size > 0:
            return out_path
    _fallback_notice("download_emdb_all_failed", f"All EMDB download candidates failed for {emdb_id}.")
    return None


def _parse_pdb_id(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    for sep in (";", ",", "|"):
        if sep in raw:
            raw = raw.split(sep, 1)[0].strip()
    raw = raw.split()[0] if raw.split() else raw
    raw = raw.upper()
    return raw


def _run_qscore(map_path: Path, model_path: Path) -> Tuple[Optional[float], str]:
    cmd = [
        sys.executable,
        str(QSCORE_SCRIPT),
        "--map",
        str(map_path),
        "--model",
        str(model_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=QSCORE_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        _fallback_notice(
            "qscore_timeout",
            f"Q-score process timed out for model '{model_path.name}'.",
        )
        return None, "qscore_timeout"

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        _fallback_notice(
            "qscore_nonzero",
            f"Q-score process returned non-zero exit code ({result.returncode}) for '{model_path.name}'.",
        )
        return None, stdout or stderr or "qscore_failed"
    if not stdout:
        _fallback_notice(
            "qscore_no_output",
            f"Q-score process produced no stdout for '{model_path.name}'.",
        )
        return None, stderr or "qscore_no_output"
    last_line = stdout.splitlines()[-1].strip()
    try:
        return float(last_line), ""
    except ValueError:
        _fallback_notice(
            "qscore_parse_failed",
            f"Could not parse Q-score output for '{model_path.name}': {last_line!r}",
        )
        return None, f"qscore_parse_failed: {last_line}"


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        _fallback_notice(
            "safe_float_none",
            f"Could not parse float value {value!r}; returning None.",
        )
        return None


def _select_best_experimental(log_path: Path) -> Optional[Dict[str, str]]:
    if not log_path.exists():
        _fallback_notice(
            "select_best_log_missing",
            f"Per-target fit log not found: {log_path}",
        )
        return None

    rows: list[Dict[str, str]] = []
    with log_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("type", "")).strip().upper() != "PDB":
                continue
            score = _safe_float(row.get("score", row.get("vof", "nan")))
            row["_score"] = score if score is not None else float("-inf")
            row["_kept"] = str(row.get("kept", "")).strip().lower() in {"true", "1", "yes"}
            rows.append(row)

    if not rows:
        _fallback_notice(
            "select_best_no_pdb_rows",
            f"No experimental PDB rows found in log: {log_path}",
        )
        return None

    def _score(row: Dict[str, str]) -> float:
        try:
            return float(row.get("_score", float("-inf")))
        except Exception:
            return float("-inf")

    def _has_files(row: Dict[str, str]) -> bool:
        source_text = str(row.get("source_model", "")).strip()
        fitted_text = str(row.get("fitted_path", "")).strip()
        if not source_text or not fitted_text:
            return False
        source = Path(source_text)
        fitted = Path(fitted_text)
        return source.is_file() and fitted.is_file()

    kept = [r for r in rows if r.get("_kept")]
    if not kept:
        _fallback_notice(
            "select_best_kept_fallback",
            "No rows marked kept in log; falling back to best available PDB row.",
        )
    candidates = kept if kept else rows
    candidates = [r for r in candidates if _has_files(r)]
    if not candidates:
        _fallback_notice(
            "select_best_missing_files",
            "No candidate rows had both source and fitted files on disk.",
        )
        return None
    candidates.sort(key=_score, reverse=True)
    return candidates[0]


def _load_struct_fetch(tag: str):
    spec = importlib.util.spec_from_file_location(f"struct_fetch_v4_{tag}", STRUCT_FETCH_PATH)
    if not spec or not spec.loader:
        raise RuntimeError("Unable to load struct_fetch_v4 module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _fit_v5_module():
    return importlib.import_module("fit_v5")


def _map_world_bounds_from_fit_v5(map_path: Path) -> tuple[np.ndarray, np.ndarray]:
    fit_v5 = _fit_v5_module()
    # Use the same map parser/coordinate system as fitting.
    mapinfo = fit_v5._get_cached_mapinfo(Path(map_path), denoise_sigma=0.0)
    mins = np.array([mapinfo.eff_x0, mapinfo.eff_y0, mapinfo.eff_z0], dtype=np.float64)
    maxs = mins + np.array(
        [
            mapinfo.sx * (mapinfo.cx - 1),
            mapinfo.sy * (mapinfo.cy - 1),
            mapinfo.sz * (mapinfo.cz - 1),
        ],
        dtype=np.float64,
    )
    lo = np.minimum(mins, maxs)
    hi = np.maximum(mins, maxs)
    return lo, hi


def _collect_all_atom_coords(st) -> np.ndarray:
    coords: list[tuple[float, float, float]] = []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    p = atom.pos
                    coords.append((float(p.x), float(p.y), float(p.z)))
    return np.asarray(coords, dtype=np.float64) if coords else np.empty((0, 3), dtype=np.float64)


def _write_structure_same_format(st, model_path: Path) -> None:
    suffix = model_path.suffix.lower()
    if suffix in (".cif", ".mmcif"):
        doc = st.make_mmcif_document()
        doc.write_file(str(model_path))
    else:
        st.write_pdb(str(model_path))


def _randomize_structure_pose_in_map_bounds(
    map_path: Path,
    model_path: Path,
    rng: np.random.Generator,
    *,
    label: str = "",
) -> bool:
    fit_v5 = _fit_v5_module()
    model_path = Path(model_path)
    map_lo, map_hi = _map_world_bounds_from_fit_v5(Path(map_path))
    eps = 1e-3
    map_lo = map_lo + eps
    map_hi = map_hi - eps
    if np.any(map_hi <= map_lo):
        _fallback_notice(
            "randomize_bad_map_bounds",
            f"Map bounds are invalid for randomization ({map_path}); keeping original pose for {model_path.name}.",
        )
        return False

    st = fit_v5.gemmi.read_structure(str(model_path))
    coords = _collect_all_atom_coords(st)
    if coords.size == 0:
        _fallback_notice(
            "randomize_no_atoms",
            f"No atoms found for randomization in {model_path.name}; keeping original pose.",
        )
        return False

    center = coords.mean(axis=0)
    rel = coords - center

    attempts = max(1, int(RANDOMIZE_FETCHED_STRUCTURES_ATTEMPTS))
    for attempt in range(1, attempts + 1):
        Rm = fit_v5.R.random(random_state=rng).as_matrix()
        rel_rot = (Rm @ rel.T).T
        rel_min = rel_rot.min(axis=0)
        rel_max = rel_rot.max(axis=0)
        center_lo = map_lo - rel_min
        center_hi = map_hi - rel_max
        if np.any(center_hi < center_lo):
            continue

        c_new = rng.uniform(center_lo, center_hi)
        t = c_new - (Rm @ center)
        fit_v5.apply_transform_to_structure(st, Rm, t)
        _write_structure_same_format(st, model_path)
        lbl = f"{label}: " if label else ""
        print(f"[randomize] {lbl}{model_path.name} randomized in-bounds (attempt {attempt}/{attempts})")
        return True

    _fallback_notice(
        "randomize_in_bounds_failed",
        f"Could not find an in-bounds random pose for {model_path.name} after {attempts} attempts; keeping original pose.",
    )
    return False


def _install_randomized_fetch_wrappers(module, map_path: Path) -> None:
    if not RANDOMIZE_FETCHED_STRUCTURES:
        return
    if getattr(module, "_TESTSET_RANDOMIZE_FETCH_WRAPPED", False):
        return

    seed = RANDOMIZE_FETCHED_STRUCTURES_SEED
    rng = np.random.default_rng(seed)

    orig_download_rcsb = module.download_rcsb_model_any
    orig_download_af = module.download_alphafold_latest

    def _randomize_path_if_present(p, kind: str):
        if not p:
            return p
        path_obj = Path(p)
        try:
            _randomize_structure_pose_in_map_bounds(map_path, path_obj, rng, label=kind)
        except Exception as exc:
            _fallback_notice(
                "randomize_wrapper_exception",
                f"Randomization failed for {kind} file '{path_obj.name}': {exc}. Keeping original pose.",
            )
        return p

    def _wrapped_download_rcsb_model_any(*args, **kwargs):
        p = orig_download_rcsb(*args, **kwargs)
        return _randomize_path_if_present(p, "RCSB")

    def _wrapped_download_alphafold_latest(*args, **kwargs):
        p = orig_download_af(*args, **kwargs)
        return _randomize_path_if_present(p, "AlphaFold")

    module.download_rcsb_model_any = _wrapped_download_rcsb_model_any
    module.download_alphafold_latest = _wrapped_download_alphafold_latest
    module._TESTSET_RANDOMIZE_FETCH_WRAPPED = True
    print("[testset_qscore_gt] Randomized fetched-structure starts enabled for this run.")


def _configure_struct_fetch(map_path: Path, work_dir: Path, module) -> None:
    module.MAP_PATH = map_path
    module.OUT_DIR = work_dir / "fetched_structures"
    module.FITTED_DIR = work_dir / "fitted_models"
    module.LOGS_DIR = module.FITTED_DIR / "logs"


def _rewrite_log(log_path: Path, input_cols: list[str], rows: list[Dict[str, str]]) -> None:
    with log_path.open("w", newline="") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=list(input_cols) + list(EXTRA_LOG_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_existing_log_rows(log_path: Path, input_cols: list[str]) -> list[Dict[str, str]]:
    """Load existing log rows so reruns append instead of overwrite."""
    if not log_path.exists():
        return []
    expected_cols = list(input_cols) + list(EXTRA_LOG_COLUMNS)
    rows: list[Dict[str, str]] = []
    try:
        with log_path.open("r", newline="") as log_file:
            reader = csv.DictReader(log_file)
            for row in reader:
                norm_row = {col: row.get(col, "") for col in expected_cols}
                rows.append(norm_row)
    except Exception as exc:
        _fallback_notice(
            "load_existing_log_failed",
            f"Failed to parse existing log '{log_path}'; starting with empty log rows.",
        )
        print(f"Warning: failed to read existing log {log_path}: {exc}")
        return []
    return rows


def _cleanup_workdir(work_dir: Path) -> None:
    if not work_dir.exists():
        return
    for path in work_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in {".map", ".mrc", ".map.gz", ".pdb", ".cif", ".mmcif"}:
            try:
                path.unlink()
            except Exception:
                _fallback_notice(
                    "cleanup_file_unlink_failed",
                    f"Failed to delete temporary file '{path}'.",
                )
                pass
    # Best-effort removal of empty dirs
    for path in sorted(work_dir.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except Exception:
                _fallback_notice(
                    "cleanup_dir_rmdir_failed",
                    f"Failed to remove temporary directory '{path}'.",
                )
                pass


def _next_run_index(out_dir: Path) -> int:
    """Return next numeric prefix for run subdirectories like '001_gene'."""
    if not out_dir.exists():
        _fallback_notice(
            "next_run_index_default",
            f"Output directory '{out_dir}' does not exist; starting run index at 1.",
        )
        return 1
    max_idx = 0
    for child in out_dir.iterdir():
        if not child.is_dir():
            continue
        prefix = child.name.split("_", 1)[0]
        if prefix.isdigit():
            max_idx = max(max_idx, int(prefix))
    return max_idx + 1 if max_idx > 0 else 1


def _process_row(
    run_idx: int,
    progress_idx: int,
    row: Dict[str, str],
    input_cols: list[str],
    total: int,
) -> Dict[str, str]:
    gene_id = str(row.get("gene_id", "")).strip()
    sequence = str(row.get("sequence", "")).strip()
    emdb_id = str(row.get("emdb_id", "")).strip()
    pdb_raw = str(row.get("fitted_pdbs", "")).strip()

    status = "ok"
    error = ""
    map_file = ""
    ground_truth_model = ""
    selected_pdb_id = ""
    selected_fitted_model = ""
    qscore_ground_truth = "nan"
    qscore_identified = "nan"
    qscore_diff = "nan"

    if not gene_id or not sequence or not emdb_id or not pdb_raw:
        status = "missing_input"
        missing = []
        if not gene_id:
            missing.append("gene_id")
        if not sequence:
            missing.append("sequence")
        if not emdb_id:
            missing.append("emdb_id")
        if not pdb_raw:
            missing.append("fitted_pdbs")
        error = "missing_" + "_".join(missing)
        _fallback_notice(
            "process_row_missing_input",
            f"Skipping row due to missing required fields: {error}",
        )
        print(f"\n[{progress_idx}/{total}] Skipping row: {error}")
        out_row = {col: row.get(col, "") for col in input_cols}
        out_row.update(
            {
                "map_file": map_file,
                "ground_truth_model": ground_truth_model,
                "selected_pdb_id": selected_pdb_id,
                "selected_fitted_model": selected_fitted_model,
                "qscore_ground_truth": qscore_ground_truth,
                "qscore_identified": qscore_identified,
                "qscore_diff": qscore_diff,
                "status": status,
                "error": error,
            }
        )
        return out_row

    emdb_norm = _normalize_emdb_id(emdb_id)
    pdb_id = _parse_pdb_id(pdb_raw)
    if not emdb_norm or not pdb_id:
        status = "bad_ids"
        error = "invalid_emdb_or_pdb"
        _fallback_notice(
            "process_row_bad_ids",
            f"Skipping row due to invalid identifiers: emdb_id={emdb_id!r}, fitted_pdbs={pdb_raw!r}",
        )
        print(f"\n[{progress_idx}/{total}] Skipping row: {error}")
        out_row = {col: row.get(col, "") for col in input_cols}
        out_row.update(
            {
                "map_file": map_file,
                "ground_truth_model": ground_truth_model,
                "selected_pdb_id": selected_pdb_id,
                "selected_fitted_model": selected_fitted_model,
                "qscore_ground_truth": qscore_ground_truth,
                "qscore_identified": qscore_identified,
                "qscore_diff": qscore_diff,
                "status": status,
                "error": error,
            }
        )
        return out_row

    print(f"\n[{progress_idx}/{total}] {gene_id} | {emdb_norm} | ground truth {pdb_id}")

    work_dir = OUTPUT_DIR / f"{run_idx:03d}_{_sanitize_name(gene_id)}"
    work_dir.mkdir(parents=True, exist_ok=True)
    map_path = work_dir / f"emd_{_emdb_numeric_id(emdb_norm) or emdb_norm}.map"

    session = _make_session()
    downloaded_map = _download_emdb_map(session, emdb_norm, map_path)
    if not downloaded_map:
        status = "download_failed"
        error = "emdb_map_download_failed"
        _fallback_notice(
            "process_row_map_download_failed",
            f"Map download failed for {emdb_norm}; row marked as download_failed.",
        )
    else:
        map_file = downloaded_map.name
        print(f"[{gene_id}] Map downloaded -> {downloaded_map}")

        sf_tag = f"{run_idx}"
        sf = _load_struct_fetch(sf_tag)
        _configure_struct_fetch(downloaded_map, work_dir, sf)

        gt_model = sf.download_rcsb_model_any(pdb_id, work_dir)
        if not gt_model:
            status = "download_failed"
            error = "ground_truth_model_download_failed"
            _fallback_notice(
                "process_row_gt_model_download_failed",
                f"Ground-truth model download failed for PDB {pdb_id}.",
            )
        else:
            ground_truth_model = gt_model.name
            print(f"[{gene_id}] Ground truth model -> {gt_model}")

            print(f"[{gene_id}] Running struct_fetch_v4...")
            _install_randomized_fetch_wrappers(sf, downloaded_map)
            sf.process_target(gene_id, sequence)

            safe_q = sf.sanitize_name(gene_id)
            struct_log_path = sf.LOGS_DIR / f"{safe_q}__fit_scores.csv"
            best = _select_best_experimental(struct_log_path)
            if not best:
                status = "no_experimental_model"
                error = "no_experimental_model"
                _fallback_notice(
                    "process_row_no_experimental",
                    f"No experimental model selected from log: {struct_log_path}",
                )
                print(f"[{gene_id}] No experimental model selected.")
            else:
                selected_pdb_id = str(best.get("id", ""))
                fitted_path = Path(best.get("fitted_path", ""))
                selected_fitted_model = fitted_path.name

                if not fitted_path.is_file():
                    status = "missing_fitted_model"
                    error = "selected_fitted_model_missing"
                    _fallback_notice(
                        "process_row_selected_fitted_missing",
                        f"Selected fitted model is missing on disk: {fitted_path}",
                    )
                else:
                    print(f"[{gene_id}] Q-score (ground truth)...")
                    gt_score, gt_err = _run_qscore(downloaded_map, gt_model)
                    if gt_score is None:
                        status = "partial"
                        error = gt_err or "qscore_ground_truth_failed"
                        _fallback_notice(
                            "process_row_gt_qscore_failed",
                            f"Ground-truth Q-score failed for {gene_id}: {error}",
                        )
                        print(f"[{gene_id}] Ground truth Q-score failed: {error}")
                    else:
                        qscore_ground_truth = f"{gt_score:.3f}"

                    print(f"[{gene_id}] Q-score (identified)...")
                    id_score, id_err = _run_qscore(downloaded_map, fitted_path)
                    if id_score is None:
                        status = "partial" if status == "ok" else status
                        err_msg = id_err or "qscore_identified_failed"
                        error = f"{error}; {err_msg}".strip("; ")
                        _fallback_notice(
                            "process_row_identified_qscore_failed",
                            f"Identified-model Q-score failed for {gene_id}: {err_msg}",
                        )
                        print(f"[{gene_id}] Identified Q-score failed: {err_msg}")
                    else:
                        qscore_identified = f"{id_score:.3f}"

                    if gt_score is not None and id_score is not None:
                        qscore_diff = f"{(id_score - gt_score):.3f}"

                    print(
                        f"[{gene_id}] Q-scores -> ground_truth={qscore_ground_truth}, "
                        f"identified={qscore_identified}, diff={qscore_diff}"
                    )

    out_row = {col: row.get(col, "") for col in input_cols}
    out_row.update(
        {
            "map_file": map_file,
            "ground_truth_model": ground_truth_model,
            "selected_pdb_id": selected_pdb_id,
            "selected_fitted_model": selected_fitted_model,
            "qscore_ground_truth": qscore_ground_truth,
            "qscore_identified": qscore_identified,
            "qscore_diff": qscore_diff,
            "status": status,
            "error": error,
        }
    )
    _cleanup_workdir(work_dir)
    return out_row


def main() -> int:
    csv_path = Path(CSV_PATH).expanduser().resolve()
    if not csv_path.exists():
        raise SystemExit(f"CSV not found: {csv_path}")
    if not QSCORE_SCRIPT.exists():
        raise SystemExit(f"qscore script not found: {QSCORE_SCRIPT}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    terminal_log_path, terminal_log_fh, orig_stdout, orig_stderr = _enable_terminal_log(OUTPUT_DIR)
    try:
        print(f"Terminal log capture enabled -> {terminal_log_path}")
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                raise SystemExit("CSV is missing a header row.")
            input_cols = reader.fieldnames
            for required in REQUIRED_COLUMNS:
                if required not in input_cols:
                    raise SystemExit(f"CSV missing required column: {required}")
            rows = list(reader)

        if MAX_ROWS is not None:
            _fallback_notice(
                "main_max_rows",
                f"MAX_ROWS is set to {MAX_ROWS}; truncating input row list accordingly.",
            )
            rows = rows[: max(MAX_ROWS, 0)]

        log_path = Path(LOG_PATH).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        start_run_idx = _next_run_index(OUTPUT_DIR)
        print(f"Starting run folder index at {start_run_idx:03d}")

        total = len(rows)
        log_rows = _load_existing_log_rows(log_path, input_cols)
        if log_rows:
            print(f"Found existing log with {len(log_rows)} row(s); new results will be appended.")
        for progress_idx, row in enumerate(rows, start=1):
            run_idx = start_run_idx + progress_idx - 1
            try:
                out_row = _process_row(run_idx, progress_idx, row, input_cols, total)
            except Exception as exc:
                _fallback_notice(
                    "main_row_processing_exception",
                    f"Unhandled exception in row {progress_idx}/{total}; logging error row and continuing.",
                )
                out_row = {col: row.get(col, "") for col in input_cols}
                out_row.update(
                    {
                        "map_file": "",
                        "ground_truth_model": "",
                        "selected_pdb_id": "",
                        "selected_fitted_model": "",
                        "qscore_ground_truth": "nan",
                        "qscore_identified": "nan",
                        "qscore_diff": "nan",
                        "status": "error",
                        "error": str(exc),
                    }
                )
            log_rows.append(out_row)
            _rewrite_log(log_path, input_cols, log_rows)

        print(f"\nLog written: {log_path}")
        return 0
    finally:
        _disable_terminal_log(terminal_log_fh, orig_stdout, orig_stderr)
        print(f"Terminal session log written: {terminal_log_path}")


if __name__ == "__main__":
    raise SystemExit(main())
