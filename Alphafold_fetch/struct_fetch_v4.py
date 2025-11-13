# =============================
# File: struct_fetch_v4.py
# =============================
#!/usr/bin/env python3
import pathlib, sys
from typing import Optional, List, Dict, Any
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
from fit_v5 import fit_model_and_score, estimate_resolution_single_map

# ----------- REQUIRED PATHS -----------
CSV_PATH = pathlib.Path("Alphafold_fetch/identifier/all_hits_v2.csv")
OUT_DIR  = pathlib.Path("Alphafold_fetch/fetched_structures_v3")
MAP_PATH = pathlib.Path("Alphafold_fetch/test_files/emd_29659.map")

# ----------- FIT/EVAL SETTINGS -----------
FITTED_DIR = pathlib.Path("Alphafold_fetch/fitted_models_v6")
MAX_EXPERIMENTAL_TRIES = 20              # cap for experimental structures (excludes AF)
EVAL_ALL_MODE = False                     # True → score all, then keep best experimental; AF always kept

# Logging
LOGS_DIR = FITTED_DIR / "logs"

# HTTP session with retries
retries = Retry(total=4, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))
session.headers.update({"User-Agent": "structure-fetcher/vof-qscore-resaware/2.1"})

# ------------------- UniProt / PDB / AF helpers -------------------
def first_uniprot_accession(query: str, timeout: int = 15) -> Optional[str]:
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {"query": query, "fields": "accession", "format": "json", "size": 1}
    try:
        r = session.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        results = r.json().get("results", [])
        return results[0]["primaryAccession"] if results else None
    except requests.RequestException:
        return None

def pdb_ids_from_pdbe_mapping(uniprot: str, timeout: int = 15) -> List[str]:
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{uniprot}"
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code == 404: return []
        r.raise_for_status()
        pdb_map = r.json().get(uniprot, {}).get("PDB", {})
        return sorted({pid.upper() for pid in pdb_map.keys()})
    except requests.RequestException:
        return []

def pdb_ids_from_rcsb_search(uniprot: str, timeout: int = 15) -> List[str]:
    url = "https://search.rcsb.org/rcsbsearch/v2/query"
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                "operator": "exact_match",
                "value": uniprot,
            },
        },
        "return_type": "entry",
        "request_options": {"results_content_type": ["experimental"], "paginate": {"start": 0, "rows": 1000}},
    }
    try:
        r = session.post(url, json=query, timeout=timeout)
        r.raise_for_status()
        result_set = r.json().get("result_set", [])
        return sorted({row["identifier"].upper() for row in result_set if "identifier" in row})
    except requests.RequestException:
        return []

def get_entry_metadata(pdb_id: str, timeout: int = 15) -> Optional[Dict[str, Any]]:
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code == 404: return None
        r.raise_for_status()
        j = r.json()
        info = j.get("rcsb_entry_info", {})
        methods = info.get("experimental_method", []) or info.get("experimental_method_class", [])
        res_list = info.get("resolution_combined", [])
        resolution = None
        if isinstance(res_list, list) and res_list:
            try: resolution = float(res_list[0])
            except Exception: resolution = None
        return {"id": pdb_id.upper(), "methods": [m.upper() for m in (methods or [])], "resolution": resolution}
    except requests.RequestException:
        return None

def _method_rank(methods: List[str]) -> int:
    joined = "".join(methods) if methods else ""
    if "EM" in joined: return 0
    if "X-RAY" in joined: return 1
    return 2

def candidate_list_with_meta(pdb_ids: List[str]) -> List[Dict[str, Any]]:
    metas: List[Dict[str, Any]] = []

    for pid in pdb_ids:
        m = get_entry_metadata(pid)
        if m: metas.append(m)
    metas.sort(key=lambda d: (_method_rank(d.get("methods", [])),
                              d.get("resolution") if d.get("resolution") is not None else 1e9))
    '''
    if not metas:
        print("[debug] no metadata fetched for any PDB id", flush=True)
    else:
        for m in metas[:10]:
            print(f"[debug] {m['id']} methods={m.get('methods')} res={m.get('resolution')}", flush=True)
    '''

    return metas

def download_rcsb_model_any(pdb_id: str, out_dir: pathlib.Path, timeout: int = 25) -> Optional[pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = [
        f"https://files.rcsb.org/download/{pdb_id}-assembly1.cif",
        f"https://files.rcsb.org/download/{pdb_id}.cif",
        f"https://files.rcsb.org/download/{pdb_id}.pdb",
    ]
    for url in candidates:
        try:
            r = session.get(url, timeout=timeout)
            txt = r.text.strip() if r.text else ""
            if r.ok and r.content and not (txt.startswith("<!DOCTYPE") or txt.startswith("<?xml")):
                ext = ".cif" if url.endswith(".cif") else ".pdb"
                path = out_dir / f"{pdb_id}{ext}"
                path.write_bytes(r.content)
                if path.stat().st_size > 0: return path
        except requests.RequestException:
            continue
    return None

def _alphafold_prediction_json(accession: str, timeout: int = 20) -> List[Dict[str, Any]]:
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{accession}"
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code == 404: return []
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except requests.RequestException:
        return []

def download_alphafold_latest(accession: str, out_dir: pathlib.Path, timeout: int = 30) -> Optional[pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = _alphafold_prediction_json(accession, timeout=timeout)
    if not preds: return None
    preds.sort(key=lambda d: d.get("modelVersion", 0), reverse=True)
    best = preds[0]
    file_url = best.get("pdbUrl") or best.get("cifUrl")
    ext = ".pdb" if best.get("pdbUrl") else ".cif"
    if not file_url: return None
    try:
        r = session.get(file_url, timeout=timeout)
        r.raise_for_status()
        txt = r.text.strip() if r.text else ""
        if r.content and not txt.startswith("<!DOCTYPE"):
            path = out_dir / f"{accession}{ext}"
            path.write_bytes(r.content)
            return path if path.stat().st_size > 0 else None
    except requests.RequestException:
        return None
    return None

# ------------------- utils -------------------
def sanitize_name(s: str) -> str:
    import re as _re
    return _re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "target"

def thresholds_from_resolution(map_res_A: Optional[float]) -> tuple[float, float]:
    """
    Dynamic thresholds for threshold-mode early accept.
    Target behavior:
      ~3 Å  → VOF≈0.55, Q≈0.50
      ~6 Å  → VOF≈0.45, Q≈0.40
      ≥8 Å  → clamp near VOF≈0.38, Q≈0.35
    """
    if not map_res_A or map_res_A <= 0:
        return 0.45, 0.40
    slope = -0.033333333333  # per Å
    vof = 0.55 + slope * (map_res_A - 3.0)
    q   = 0.50 + slope * (map_res_A - 3.0)
    vof = float(min(max(vof, 0.38), 0.62))
    q   = float(min(max(q,   0.35), 0.57))
    return vof, q

def evaluate_and_record(label: str,
                        kind: str,
                        id_or_acc: str,
                        map_path: pathlib.Path,
                        model_path: pathlib.Path,
                        map_resolution_A: Optional[float],
                        model_resolution_A: Optional[float]) -> Dict[str, Any]:
    """Run resolution-aware fitting+evaluation; return a row dict for logging."""
    FITTED_DIR.mkdir(parents=True, exist_ok=True)
    out_pdb = FITTED_DIR / f"{label}_fitted_v5.pdb"
    vof, qscore, fitted = fit_model_and_score(
        map_path, model_path, out_pdb_path=out_pdb,
        model_resolution_A=model_resolution_A,
        map_resolution_A=map_resolution_A,
    )
    row = {
        "label": label,
        "type": kind,                 # "AF" or "PDB"
        "id": id_or_acc,              # PDB ID or UniProt accession
        "map_resolution_A": map_resolution_A,
        "model_resolution_A": model_resolution_A,
        "vof": vof if vof is not None else float("nan"),
        "qscore": qscore if qscore is not None else float("nan"),
        "fitted_path": str(fitted) if fitted else str(out_pdb),
        "source_model": str(model_path),
        "kept": False,
    }
    # If the fit failed (vof/q NaN), proactively remove any leftover file
    if vof is None or qscore is None or not fitted:
        try: pathlib.Path(out_pdb).unlink(missing_ok=True)
        except Exception: pass
    return row

def _passes_both(row: Dict[str, Any], vof_thr: float, q_thr: float) -> bool:
    v = row["vof"]; q = row["qscore"]
    return (pd.notna(v) and pd.notna(q) and (v >= vof_thr) and (q >= q_thr))

def _combo_score(row: Dict[str, Any]) -> float:
    # Mean with graceful NaN handling
    v = row.get("vof", float("nan"))
    q = row.get("qscore", float("nan"))
    if pd.notna(v) and pd.notna(q): return 0.5 * (float(v) + float(q))
    if pd.notna(v): return 0.5 * float(v)
    if pd.notna(q): return 0.5 * float(q)
    return float("-inf")

def _write_log(safe_q: str, rows: List[Dict[str, Any]]) -> pathlib.Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{safe_q}__fit_scores.csv"
    pd.DataFrame(rows).to_csv(log_path, index=False)
    return log_path

def _best_pdb_pid_from_log(safe_q: str) -> Optional[str]:
    """Read the per-target log and return PDB id with highest combo score (VOF/Q)."""
    log_path = LOGS_DIR / f"{safe_q}__fit_scores.csv"
    if not log_path.exists():
        return None
    try:
        df = pd.read_csv(log_path)
    except Exception:
        return None
    if "type" not in df.columns or "id" not in df.columns:
        return None
    df = df[df["type"] == "PDB"].copy()
    if df.empty:
        return None

    def _row_combo(r):
        v = r.get("vof", float("nan"))
        q = r.get("qscore", float("nan"))
        try:
            v = float(v)
        except Exception:
            v = float("nan")
        try:
            q = float(q)
        except Exception:
            q = float("nan")
        if pd.notna(v) and pd.notna(q): return 0.5 * (v + q)
        if pd.notna(v): return 0.5 * v
        if pd.notna(q): return 0.5 * q
        return float("-inf")

    df["__combo__"] = df.apply(_row_combo, axis=1)
    df.sort_values("__combo__", ascending=False, inplace=True)
    top = df.iloc[0]
    if top["__combo__"] == float("-inf"):
        return None
    return str(top["id"])

# ------------------- main per-target flow -------------------
def process_target(q: str, map_resolution_A: Optional[float], vof_thr: float, q_thr: float):
    safe_q = sanitize_name(q)
    acc = first_uniprot_accession(q)
    if acc is None:
        print(f"{q}: no UniProt accession found")
        return

    print(f"\n=== {q} ({acc}) ===")

    rows: List[Dict[str, Any]] = []

    # ----- AlphaFold first -----
    af_model = download_alphafold_latest(acc, OUT_DIR)
    af_row: Optional[Dict[str, Any]] = None
    if af_model:
        label = f"{safe_q}__AF_{acc}"
        af_row = evaluate_and_record(label, "AF", acc, MAP_PATH, af_model, map_resolution_A, model_resolution_A=None)
        rows.append(af_row)
        vtxt = f"{af_row['vof']:.4f}" if pd.notna(af_row["vof"]) else "N/A"
        qtxt = f"{af_row['qscore']:.4f}" if pd.notna(af_row["qscore"]) else "N/A"
        print(f"AF fit: VOF={vtxt}, Q={qtxt} → {af_row['fitted_path'] if pd.notna(af_row['vof']) and pd.notna(af_row['qscore']) else '—'}")
    else:
        print(f"{q}: AlphaFold model not available")

    # ----- Experimental candidates -----
    ids = pdb_ids_from_pdbe_mapping(acc) or pdb_ids_from_rcsb_search(acc)
    meta = candidate_list_with_meta(ids)[:MAX_EXPERIMENTAL_TRIES] if ids else []

    if not EVAL_ALL_MODE:
        # Early-accept mode with early deletion of non-passing attempts
        accepted_row: Optional[Dict[str, Any]] = None

        for m in meta:
            pid = m["id"]; reso = m.get("resolution")
            model_path = download_rcsb_model_any(pid, OUT_DIR)
            if not model_path:
                print(f"{q}: could not download PDB {pid}")
                continue
            label = f"{safe_q}__PDB_{pid}"
            row = evaluate_and_record(label, "PDB", pid, MAP_PATH, model_path, map_resolution_A, reso)
            rows.append(row)
            vtxt = f"{row['vof']:.4f}" if pd.notna(row["vof"]) else "N/A"
            qtxt = f"{row['qscore']:.4f}" if pd.notna(row["qscore"]) else "N/A"
            rez  = f", {reso:.2f}Å" if isinstance(reso,(int,float)) else ""
            print(f"EXP fit [PDB {pid}{rez}]: VOF={vtxt}, Q={qtxt} → {row['fitted_path'] if pd.notna(row['vof']) and pd.notna(row['qscore']) else '—'}")

            if _passes_both(row, vof_thr, q_thr):
                row["kept"] = True
                accepted_row = row
                print(f"✔ Accepted experimental model {pid} (VOF ≥ {vof_thr:.2f} AND Q ≥ {q_thr:.2f}).")
                # delete earlier experimental attempts
                for r in rows:
                    if r is row or r["type"] != "PDB": continue
                    try: pathlib.Path(r["source_model"]).unlink(missing_ok=True)
                    except Exception: pass
                    try: pathlib.Path(r["fitted_path"]).unlink(missing_ok=True)
                    except Exception: pass
                break
            else:
                # EARLY DELETE this experimental attempt (source + fitted)
                try: pathlib.Path(row["source_model"]).unlink(missing_ok=True)
                except Exception: pass
                try: pathlib.Path(row["fitted_path"]).unlink(missing_ok=True)
                except Exception: pass

        # AF is always kept
        if af_row: af_row["kept"] = True

        if accepted_row is None:
            # No experimental passed. Write current log, then pick best from log and re-fetch/refit it.
            _write_log(safe_q, rows)
            best_pid = _best_pdb_pid_from_log(safe_q)

            if best_pid:
                # Re-download and re-fit the best-scoring experimental (files were deleted during attempts)
                model_path = download_rcsb_model_any(best_pid, OUT_DIR)
                if model_path:
                    best_label = f"{safe_q}__PDB_{best_pid}"
                    refit_row = evaluate_and_record(best_label, "PDB", best_pid, MAP_PATH, model_path,
                                                    map_resolution_A, next((m.get("resolution") for m in meta if m["id"] == best_pid), None))
                    refit_row["kept"] = True

                    # Update the original in-memory row for this PID (so the final log reflects the kept copy)
                    orig = next((r for r in rows if r["type"]=="PDB" and r["id"]==best_pid), None)
                    if orig:
                        orig.update({
                            "vof": refit_row["vof"],
                            "qscore": refit_row["qscore"],
                            "fitted_path": refit_row["fitted_path"],
                            "source_model": refit_row["source_model"],
                            "model_resolution_A": refit_row["model_resolution_A"],
                            "kept": True,
                        })
                    else:
                        rows.append(refit_row)

                    print(f"✔ No experimental met thresholds; refetched best by log score: {best_label} "
                          f"(VOF={refit_row['vof'] if pd.notna(refit_row['vof']) else 'N/A'}, "
                          f"Q={refit_row['qscore'] if pd.notna(refit_row['qscore']) else 'N/A'}).")
                else:
                    print(f"✔ Best-by-log experimental {best_pid} could not be re-downloaded; AlphaFold retained only.")
            else:
                print("✔ No experimental candidates in log; AlphaFold retained only.")

        # Final cleanup: ensure only kept experimental files remain
        for r in rows:
            if r["type"] == "PDB" and not r.get("kept", False):
                try: pathlib.Path(r["source_model"]).unlink(missing_ok=True)
                except Exception: pass
                try: pathlib.Path(r["fitted_path"]).unlink(missing_ok=True)
                except Exception: pass

    else:
        # Score up to MAX_EXPERIMENTAL_TRIES, then choose best meeting thresholds (or best overall if none).
        for m in meta:
            pid = m["id"]; reso = m.get("resolution")
            model_path = download_rcsb_model_any(pid, OUT_DIR)
            if not model_path:
                print(f"{q}: could not download PDB {pid}")
                continue
            label = f"{safe_q}__PDB_{pid}"
            row = evaluate_and_record(label, "PDB", pid, MAP_PATH, model_path, map_resolution_A, reso)
            rows.append(row)
            vtxt = f"{row['vof']:.4f}" if pd.notna(row["vof"]) else "N/A"
            qtxt = f"{row['qscore']:.4f}" if pd.notna(row["qscore"]) else "N/A"
            rez  = f", {reso:.2f}Å" if isinstance(reso,(int,float)) else ""
            print(f"EXP fit [PDB {pid}{rez}]: VOF={vtxt}, Q={qtxt} → {row['fitted_path'] if pd.notna(row['vof']) and pd.notna(row['qscore']) else '—'}")

        candidates = [r for r in rows if r["type"]=="PDB" and _passes_both(r, vof_thr, q_thr)]
        keep_exp: Optional[Dict[str, Any]] = None
        if candidates:
            candidates.sort(key=lambda r: (0.5*(r["vof"]+r["qscore"]), r["vof"]), reverse=True)
            keep_exp = candidates[0]; keep_exp["kept"] = True
            print(f"✔ Kept best experimental meeting thresholds: {keep_exp['label']} (VOF={keep_exp['vof']:.4f}, Q={keep_exp['qscore']:.4f}).")
        else:
            # None met thresholds → keep best by combined score if any usable fit exists
            all_pdb = [r for r in rows if r["type"] == "PDB"]
            if all_pdb:
                best_any = max(all_pdb, key=_combo_score)
                if _combo_score(best_any) != float("-inf"):
                    best_any["kept"] = True
                    vtxt = f"{best_any['vof']:.4f}" if pd.notna(best_any["vof"]) else "N/A"
                    qtxt = f"{best_any['qscore']:.4f}" if pd.notna(best_any["qscore"]) else "N/A"
                    print(f"✔ No experimental met both thresholds; kept best experimental by score: {best_any['label']} (VOF={vtxt}, Q={qtxt}).")
                else:
                    print("✔ No experimental met thresholds and no usable experimental fit; AlphaFold retained only.")
            else:
                print("✔ No experimental structures available; AlphaFold retained only.")

        # AF is always kept
        if af_row: af_row["kept"] = True

        # Clean unused experimental files
        for r in rows:
            if r["type"] == "PDB" and not r.get("kept", False):
                try: pathlib.Path(r["source_model"]).unlink(missing_ok=True)
                except Exception: pass
                try: pathlib.Path(r["fitted_path"]).unlink(missing_ok=True)
                except Exception: pass

    # ----- Per-target log -----
    _write_log(safe_q, rows)
    print(f"Log written: {(LOGS_DIR / f'{safe_q}__fit_scores.csv')}")

# ------------------- CLI main -------------------
def main():
    if not MAP_PATH.exists():
        sys.exit(f"Map not found: {MAP_PATH}")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        sys.exit(f"Input file not found: {CSV_PATH}")
    if "target_name" not in df.columns:
        sys.exit("CSV must contain a 'target_name' column.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FITTED_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # --- NEW: estimate map resolution once ---
    est_res = estimate_resolution_single_map(MAP_PATH, tau=1.0, max_dim=256)
    if est_res:
        print(f"[Map] Estimated resolution: {est_res:.2f} Å (single-map SNR crossing)")
    else:
        print("[Map] Estimated resolution: not determined (falling back to defaults)")
    vof_thr, q_thr = thresholds_from_resolution(est_res)
    print(f"[Thresholds] VOF ≥ {vof_thr:.2f}, Q ≥ {q_thr:.2f} (scaled for map resolution)")

    for q in df["target_name"].dropna().unique():
        process_target(str(q), est_res, vof_thr, q_thr)

    print(f"\nFitted PDBs → {FITTED_DIR.resolve().as_posix()}")
    print(f"Logs       → {LOGS_DIR.resolve().as_posix()}")

if __name__ == "__main__":
    main()
