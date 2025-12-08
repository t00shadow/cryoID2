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
from urllib.parse import urlparse

# ----------- REQUIRED PATHS -----------
CSV_PATH = pathlib.Path("Alphafold_fetch/identifier/all_hits_v2.csv")
OUT_DIR  = pathlib.Path("Alphafold_fetch/fetched_structures_v3")
MAP_PATH = pathlib.Path("Alphafold_fetch/test_files/emd_29659.map")

# ----------- OPTIONAL OVERRIDES -----------
# If set to a positive float (Angstroms), use this instead of estimating map resolution.
HARDCODED_MAP_RES_A: Optional[float] = None

# ----------- FIT/EVAL SETTINGS -----------
FITTED_DIR = pathlib.Path("Alphafold_fetch/fitted_models_v6")
MAX_EXPERIMENTAL_TRIES = 10           # cap for experimental structures (excludes AF)
EVAL_ALL_MODE = False                     # True → score all, then keep best experimental; AF always kept

# Logging
LOGS_DIR = FITTED_DIR / "logs"

# HTTP session with retries
retries = Retry(total=4, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))
session.headers.update({"User-Agent": "structure-fetcher/vof-only/2.2"})

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

def top_uniprot_accessions(query: str, timeout: int = 15, size: int = 5) -> List[str]:
    """Return all UniProt accessions tied for the top search score."""
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {"query": query, "fields": "accession", "format": "json", "size": size}
    try:
        r = session.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return []
        # Results come sorted by score desc; gather any tied with the top score.
        top_score = results[0].get("score", None)
        accs: List[str] = []
        for res in results:
            acc = res.get("primaryAccession")
            score = res.get("score", None)
            if acc is None:
                continue
            if top_score is None or score is None:
                accs.append(acc)
                continue
            if abs(score - top_score) < 1e-9:
                accs.append(acc)
            else:
                break
        return accs
    except requests.RequestException:
        return []

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
    file_name = pathlib.Path(urlparse(file_url).path).name or f"{accession}{ext}"
    if not pathlib.Path(file_name).suffix:
        file_name = f"{file_name}{ext}"
    try:
        r = session.get(file_url, timeout=timeout)
        r.raise_for_status()
        txt = r.text.strip() if r.text else ""
        if r.content and not txt.startswith("<!DOCTYPE"):
            path = out_dir / file_name
            path.write_bytes(r.content)
            return path if path.stat().st_size > 0 else None
    except requests.RequestException:
        return None
    return None

# ------------------- utils -------------------
def sanitize_name(s: str) -> str:
    import re as _re
    return _re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "target"

def vof_threshold_from_resolution(map_res_A: Optional[float]) -> float:
    """
    Dynamic VOF threshold for threshold-mode early accept.
    Target behavior (single-metric):
      ~3 Å  → VOF≈0.55
      ~6 Å  → VOF≈0.45
      ≥8 Å  → clamp near VOF≈0.38
    """
    if not map_res_A or map_res_A <= 0:
        return 0.45
    slope = -0.033333333333  # per Å
    vof = 0.55 + slope * (map_res_A - 3.0)
    vof = float(min(max(vof, 0.38), 0.62))
    return vof

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
    vof, fitted = fit_model_and_score(
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
        "fitted_path": str(fitted) if fitted else str(out_pdb),
        "source_model": str(model_path),
        "kept": False,
    }
    # If the fit failed (VOF NaN), proactively remove any leftover file
    if vof is None or not fitted:
        try: pathlib.Path(out_pdb).unlink(missing_ok=True)
        except Exception: pass
    return row

def _passes_vof(row: Dict[str, Any], vof_thr: float) -> bool:
    v = row.get("vof", float("nan"))
    try:
        v = float(v)
    except Exception:
        v = float("nan")
    return pd.notna(v) and (v >= vof_thr)

def _vof_score(row: Dict[str, Any]) -> float:
    v = row.get("vof", float("nan"))
    try:
        v = float(v)
    except Exception:
        v = float("nan")
    return v if pd.notna(v) else float("-inf")

def _write_log(safe_q: str, rows: List[Dict[str, Any]]) -> pathlib.Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{safe_q}__fit_scores.csv"
    pd.DataFrame(rows).to_csv(log_path, index=False)
    return log_path

def _best_pdb_pid_from_log(safe_q: str) -> Optional[str]:
    """Read the per-target log and return PDB id with highest VOF."""
    log_path = LOGS_DIR / f"{safe_q}__fit_scores.csv"
    if not log_path.exists():
        return None
    try:
        df = pd.read_csv(log_path)
    except Exception:
        return None
    if "type" not in df.columns or "id" not in df.columns or "vof" not in df.columns:
        return None
    df = df[df["type"] == "PDB"].copy()
    if df.empty:
        return None

    def _row_vof(r):
        try:
            return float(r.get("vof", float("nan")))
        except Exception:
            return float("nan")

    df["__vof__"] = df.apply(_row_vof, axis=1)
    df.sort_values("__vof__", ascending=False, inplace=True)
    top = df.iloc[0]
    if not pd.notna(top["__vof__"]):
        return None
    return str(top["id"])

def _select_best(rows: List[Dict[str, Any]],
                 kind: str,
                 vof_thr: Optional[float] = None,
                 allow_negative_inf: bool = False) -> Optional[Dict[str, Any]]:
    """Pick best row of a given type; prefer threshold-passers when provided."""
    candidates = [r for r in rows if r.get("type") == kind]
    if not candidates:
        return None
    if kind == "PDB" and vof_thr is not None:
        passing = [r for r in candidates if _passes_vof(r, vof_thr)]
        if passing:
            candidates = passing
    candidates.sort(key=_vof_score, reverse=True)
    best = candidates[0]
    if _vof_score(best) == float("-inf") and not allow_negative_inf:
        return None
    return best

# ------------------- main per-target flow -------------------


def process_target(q: str, map_resolution_A: Optional[float], vof_thr: float):
    safe_q = sanitize_name(q)
    acc_candidates = top_uniprot_accessions(q)
    if not acc_candidates:
        print(f"{q}: no UniProt accession found")
        return

    print(f"\n=== {q}: {len(acc_candidates)} UniProt top-hit candidate(s) (tied score) ===")
    print(f"{q}: candidates -> {', '.join(acc_candidates)}")

    all_rows: List[Dict[str, Any]] = []
    accepted_row: Optional[Dict[str, Any]] = None

    for idx, acc in enumerate(acc_candidates, start=1):
        print(f"[{q}] Trying UniProt {acc} ({idx}/{len(acc_candidates)})")

        rows: List[Dict[str, Any]] = []
        passing_af_row: Optional[Dict[str, Any]] = None

        # ----- AlphaFold first -----
        af_model = download_alphafold_latest(acc, OUT_DIR)
        if af_model:
            label = f"{safe_q}__AF_{acc}"
            af_row = evaluate_and_record(label, "AF", acc, MAP_PATH, af_model, map_resolution_A, model_resolution_A=None)
            rows.append(af_row)
            vtxt = f"{af_row['vof']:.4f}" if pd.notna(af_row["vof"]) else "N/A"
            print(f"[{q}] AF fit ({acc}): VOF={vtxt} -> {af_row['fitted_path'] if pd.notna(af_row['vof']) else 'N/A'}")
            if _passes_vof(af_row, vof_thr):
                passing_af_row = af_row
            else:
                try: pathlib.Path(af_row["source_model"]).unlink(missing_ok=True)
                except Exception: pass
                try: pathlib.Path(af_row["fitted_path"]).unlink(missing_ok=True)
                except Exception: pass
        else:
            print(f"[{q}] AlphaFold model not available for {acc}")

        # ----- Experimental candidates -----
        ids = pdb_ids_from_pdbe_mapping(acc) or pdb_ids_from_rcsb_search(acc)
        meta = candidate_list_with_meta(ids)[:MAX_EXPERIMENTAL_TRIES] if ids else []
        print(f"[{q}] UniProt {acc}: fetched {len(meta)} experimental candidate(s)")

        for m in meta:
            pid = m["id"]
            reso = m.get("resolution")
            model_path = download_rcsb_model_any(pid, OUT_DIR)
            if not model_path:
                print(f"[{q}] could not download PDB {pid}")
                continue
            label = f"{safe_q}__PDB_{pid}"
            row = evaluate_and_record(label, "PDB", pid, MAP_PATH, model_path, map_resolution_A, reso)
            rows.append(row)
            vtxt = f"{row['vof']:.4f}" if pd.notna(row["vof"]) else "N/A"
            rez = f", {reso:.2f}A." if isinstance(reso, (int, float)) else ""
            print(f"[{q}] EXP fit [PDB {pid}{rez}]: VOF={vtxt} -> {row['fitted_path'] if pd.notna(row['vof']) else 'N/A'}")
            if _passes_vof(row, vof_thr):
                accepted_row = row
                accepted_row["kept"] = True
                vtxt_keep = f"{accepted_row['vof']:.4f}" if pd.notna(accepted_row["vof"]) else "N/A"
                rez_keep = f", {accepted_row.get('model_resolution_A'):.2f}A." if isinstance(accepted_row.get("model_resolution_A"), (int, float)) else ""
                print(f"[{q}] Threshold met; keeping PDB {accepted_row['id']}{rez_keep} (VOF={vtxt_keep}) and stopping further UniProt ties.")
                break
            else:
                try: pathlib.Path(row["source_model"]).unlink(missing_ok=True)
                except Exception: pass
                try: pathlib.Path(row["fitted_path"]).unlink(missing_ok=True)
                except Exception: pass

        all_rows.extend(rows)

        # After evaluating AF + experimental for this UniProt, decide whether to stop
        if accepted_row:
            break
        elif passing_af_row:
            # AF met threshold but no experimental did; accept AF and stop UniProt ties
            passing_af_row["kept"] = True
            accepted_row = passing_af_row
            vtxt = f"{accepted_row['vof']:.4f}" if pd.notna(accepted_row["vof"]) else "N/A"
            print(f"[{q}] Threshold met by AlphaFold ({accepted_row['id']}): VOF={vtxt}; stopping further UniProt ties.")
            break

    kept_rows: List[Dict[str, Any]] = []

    if accepted_row:
        kept_rows.append(accepted_row)

    # Always keep the best AF and best PDB if they exist (even if below threshold), unless already kept.
    best_af_row = _select_best(all_rows, "AF", allow_negative_inf=False)
    best_pdb_row = _select_best(all_rows, "PDB", allow_negative_inf=False)

    if best_af_row and all(id(best_af_row) != id(k) for k in kept_rows):
        best_af_row["kept"] = True
        kept_rows.append(best_af_row)
        vtxt = f"{best_af_row['vof']:.4f}" if pd.notna(best_af_row["vof"]) else "N/A"
        print(f"[{q}] Keeping best AlphaFold overall ({best_af_row['id']}): VOF={vtxt} -> {best_af_row['fitted_path']}")

    if best_pdb_row and all(id(best_pdb_row) != id(k) for k in kept_rows):
        best_pdb_row["kept"] = True
        kept_rows.append(best_pdb_row)
        vtxt = f"{best_pdb_row['vof']:.4f}" if pd.notna(best_pdb_row["vof"]) else "N/A"
        rez = f", {best_pdb_row['model_resolution_A']:.2f}A." if isinstance(best_pdb_row.get("model_resolution_A"), (int, float)) else ""
        print(f"[{q}] Keeping best experimental PDB overall {best_pdb_row['id']}{rez}: VOF={vtxt} -> {best_pdb_row['fitted_path']}")

    # Delete any files not marked kept
    kept_ids = {id(r) for r in kept_rows}
    for r in all_rows:
        if id(r) in kept_ids:
            continue
        try: pathlib.Path(r["source_model"]).unlink(missing_ok=True)
        except Exception: pass
        try: pathlib.Path(r["fitted_path"]).unlink(missing_ok=True)
        except Exception: pass

    # ----- Per-target log -----
    _write_log(safe_q, all_rows)
    print(f"[{q}] Log written: {(LOGS_DIR / f'{safe_q}__fit_scores.csv')}")
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

    # --- Map resolution: use hardcoded override if provided, else estimate once ---
    if HARDCODED_MAP_RES_A and HARDCODED_MAP_RES_A > 0:
        est_res = float(HARDCODED_MAP_RES_A)
        print(f"[Map] Using hardcoded resolution: {est_res:.2f} Å.")
    else:
        est_res = estimate_resolution_single_map(MAP_PATH, tau=1.0, max_dim=256)
        if est_res:
            print(f"[Map] Estimated resolution: {est_res:.2f} Å (single-map SNR crossing)")
        else:
            print("[Map] Estimated resolution: not determined (falling back to defaults)")
    vof_thr = vof_threshold_from_resolution(est_res)
    print(f"[Thresholds] VOF ≥ {vof_thr:.2f} (scaled for map resolution)")

    for q in df["target_name"].dropna().unique():
        process_target(str(q), est_res, vof_thr)

    print(f"\nFitted PDBs → {FITTED_DIR.resolve().as_posix()}")
    print(f"Logs       → {LOGS_DIR.resolve().as_posix()}")

if __name__ == "__main__":
    main()
