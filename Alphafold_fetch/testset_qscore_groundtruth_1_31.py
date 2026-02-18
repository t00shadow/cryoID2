#!/usr/bin/env python3
from __future__ import annotations

import csv
import gzip
import importlib.util
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter, Retry

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

STRUCT_FETCH_PATH = SCRIPT_DIR / "struct_fetch_v4.py"

# ------------------- Hardcoded parameters -------------------
CSV_PATH = SCRIPT_DIR / "ready_to_test_new_1_29_ok_only_copy.csv"
QSCORE_SCRIPT = SCRIPT_DIR / "qscore_chimerax.py"
# Output directory for all maps/models/logs produced during testing.
OUTPUT_DIR = SCRIPT_DIR / "qscore_groundtruth_run_2_11"
LOG_PATH = OUTPUT_DIR / "qscore_groundtruth_results.csv"
# Set to an integer to limit rows, or None for all rows.
MAX_ROWS = None

QSCORE_TIMEOUT_SEC = 1800
EMDB_DOWNLOAD_RETRIES = 6
EMDB_DOWNLOAD_BACKOFF_SEC = 2.0
EMDB_DOWNLOAD_CONNECT_TIMEOUT_SEC = 20
EMDB_DOWNLOAD_READ_TIMEOUT_SEC = 120

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
        return None
    if emdb_id.startswith("EMD-"):
        suffix = emdb_id[4:]
        return suffix if suffix.isdigit() else None
    return None


def _download_emdb_map(
    session: requests.Session, emdb_id: str, out_path: Path
) -> Optional[Path]:
    num = _emdb_numeric_id(emdb_id)
    if not num:
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
                        return None
                    chunks: list[bytes] = []
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            chunks.append(chunk)
                    if not chunks:
                        return None
                    data = b"".join(chunks)
                    # EMDB FTP can return HTML error pages.
                    if data.lstrip().startswith(b"<!DOCTYPE") or data.lstrip().startswith(b"<html"):
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
            continue
        if url.endswith(".gz"):
            try:
                data = gzip.decompress(data)
            except OSError:
                continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(data)
        if out_path.stat().st_size > 0:
            return out_path
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
        return None, "qscore_timeout"

    stdout = (result.stdout or "").strip()
    stderr = (result.stderr or "").strip()
    if result.returncode != 0:
        return None, stdout or stderr or "qscore_failed"
    if not stdout:
        return None, stderr or "qscore_no_output"
    last_line = stdout.splitlines()[-1].strip()
    try:
        return float(last_line), ""
    except ValueError:
        return None, f"qscore_parse_failed: {last_line}"


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _select_best_experimental(log_path: Path) -> Optional[Dict[str, str]]:
    if not log_path.exists():
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
    candidates = kept if kept else rows
    candidates = [r for r in candidates if _has_files(r)]
    if not candidates:
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
                pass
    # Best-effort removal of empty dirs
    for path in sorted(work_dir.rglob("*"), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except Exception:
                pass


def _next_run_index(out_dir: Path) -> int:
    """Return next numeric prefix for run subdirectories like '001_gene'."""
    if not out_dir.exists():
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
        else:
            ground_truth_model = gt_model.name
            print(f"[{gene_id}] Ground truth model -> {gt_model}")

            print(f"[{gene_id}] Running struct_fetch_v4...")
            sf.process_target(gene_id, sequence)

            safe_q = sf.sanitize_name(gene_id)
            struct_log_path = sf.LOGS_DIR / f"{safe_q}__fit_scores.csv"
            best = _select_best_experimental(struct_log_path)
            if not best:
                status = "no_experimental_model"
                error = "no_experimental_model"
                print(f"[{gene_id}] No experimental model selected.")
            else:
                selected_pdb_id = str(best.get("id", ""))
                fitted_path = Path(best.get("fitted_path", ""))
                selected_fitted_model = fitted_path.name

                if not fitted_path.is_file():
                    status = "missing_fitted_model"
                    error = "selected_fitted_model_missing"
                else:
                    print(f"[{gene_id}] Q-score (ground truth)...")
                    gt_score, gt_err = _run_qscore(downloaded_map, gt_model)
                    if gt_score is None:
                        status = "partial"
                        error = gt_err or "qscore_ground_truth_failed"
                        print(f"[{gene_id}] Ground truth Q-score failed: {error}")
                    else:
                        qscore_ground_truth = f"{gt_score:.3f}"

                    print(f"[{gene_id}] Q-score (identified)...")
                    id_score, id_err = _run_qscore(downloaded_map, fitted_path)
                    if id_score is None:
                        status = "partial" if status == "ok" else status
                        err_msg = id_err or "qscore_identified_failed"
                        error = f"{error}; {err_msg}".strip("; ")
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
        rows = rows[: max(MAX_ROWS, 0)]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
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


if __name__ == "__main__":
    raise SystemExit(main())
