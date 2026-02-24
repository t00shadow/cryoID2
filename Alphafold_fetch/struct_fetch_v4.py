# =============================
# File: struct_fetch_v4.py
# =============================
#!/usr/bin/env python3
import pathlib, sys
import random
import hashlib
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, List, Dict, Any, Tuple
import pandas as pd
import requests
from requests.adapters import HTTPAdapter, Retry
import parasail
from fit_v5 import fit_model_and_score
from cluster_utils import (
    set_session as _set_cluster_session,
    _fetch_json,
    _normalize_sequence,
    _resolution_rank,
    _method_rank,
    build_pdb_clusters,
    build_af_clusters,
)
from urllib.parse import urlparse, parse_qs
try:
    from Bio.Blast import NCBIWWW, NCBIXML
except Exception:  # pragma: no cover - optional dependency at runtime
    NCBIWWW = None
    NCBIXML = None

# ----------- REQUIRED PATHS -----------
CSV_PATH = pathlib.Path("Alphafold_fetch/sorted_search_pool_less_than_400aa_struct_fetch_input.csv")
OUT_DIR  = pathlib.Path("Alphafold_fetch/fetched_structures_v3")
MAP_PATH = pathlib.Path("Alphafold_fetch/cryosparc_P164_J1425_003_volume_map_sharp.mrc")

# ----------- PROCESSING LIMIT -----------
# Number of CSV rows to process (with non-empty gene_id + sequence).
# Set to 1 to process only the first valid row; set to total row count to mimic full run.
ENTRIES_TO_PROCESS = 1

# ----------- FIT/EVAL SETTINGS -----------
FITTED_DIR = pathlib.Path("Alphafold_fetch/fitted_models_v4")
MAX_EXPERIMENTAL_TRIES: Optional[int] = None # cap for experimental clusters (None => no cap)
RECOMMENDED_CONTOUR_LEVEL: Optional[float] = None  # set to None to disable
# Parallel candidate alignment+scoring workers (within one target).
# Set to 1 for strict one-at-a-time candidate evaluation.
CANDIDATE_FIT_MAX_WORKERS = 8

# ----------- Clustering settings -----------
# Toggle experimental clustering.
# True => use clustering to pick representatives.
# False => evaluate every experimental candidate PDB for all top UniProt candidates.
USE_PDB_CLUSTERING = False
# Sequence identity thresholds for clustering
AF_SEQ_CLUSTER_IDENTITY = 0.98
PDB_SEQ_CLUSTER_IDENTITY = 0.98
# Assembly id to use when extracting stoichiometry features for clustering.
ASSEMBLY_ID_FOR_CLUSTERING = "1"
# Seed material for deterministic cluster sampling when capped.
CLUSTER_SAMPLE_SEED = "cluster-v1"

# ----------- UniProt gene->sequence pool settings -----------
UNIPROT_SEARCH_URL = "https://rest.uniprot.org/uniprotkb/search"
UNIPROT_PAGE_SIZE = 200
UNIPROT_MAX_RESULTS = 10000  # cap to avoid extreme gene-wide pools
# Options for UNIPROT_REVIEWED_ONLY:
#   True / "reviewed"  -> reviewed entries only
#   False / "all"      -> include unreviewed entries
#   "fallback"         -> try reviewed only, then include unreviewed if none found
UNIPROT_REVIEWED_ONLY = "fallback"
SIMILARITY_TOL = 1e-9

# Logging
LOGS_DIR = FITTED_DIR / "logs"

# HTTP session with retries
retries = Retry(total=4, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=retries))
session.headers.update({"User-Agent": "structure-fetcher/vof-x-meanz/2.3"})
_set_cluster_session(session)

_FALLBACK_NOTICE_SEEN: set[str] = set()

def _fallback_notice(tag: str, message: str, once: bool = False) -> None:
    if once and tag in _FALLBACK_NOTICE_SEEN:
        return
    _FALLBACK_NOTICE_SEEN.add(tag)
    print(f"[struct_fetch_v4][fallback] {message}")

# Simple in-memory caches to limit repeated API calls
UNIPROT_SEQ_CACHE: Dict[str, str] = {}
AF_META_CACHE: Dict[str, Dict[str, Any]] = {}
_PARASAIL_ALPHABET = "ACDEFGHIKLMNPQRSTVWYBXZJUO*"
_PARASAIL_ALLOWED = set(_PARASAIL_ALPHABET)
# Match current SW behavior as closely as possible:
# match=1, mismatch=0, gap-open=1, gap-extend=1.
_PARASAIL_MATRIX = parasail.matrix_create(_PARASAIL_ALPHABET, 1, 0)

# ------------------- UniProt / PDB / AF helpers -------------------
def _next_cursor(link_header: Optional[str]) -> Optional[str]:
    if not link_header:
        return None
    match = re.search(r'<([^>]*)>\s*;\s*rel="next"', link_header)
    if not match:
        return None
    query = parse_qs(urlparse(match.group(1)).query)
    return query.get("cursor", [None])[0]


def blast_top_uniprot_accession(seq: str) -> Optional[str]:
    if NCBIWWW is None or NCBIXML is None:
        _fallback_notice(
            "blast_dependency_missing",
            "Biopython BLAST modules unavailable; BLAST fallback cannot run.",
            once=True,
        )
        print("Biopython not available; install biopython to enable BLAST fallback.")
        return None

    seq = str(seq or "").strip()
    if not seq:
        return None
    try:
        handle = NCBIWWW.qblast("blastp", "swissprot", seq)
    except Exception as exc:
        print(f"BLAST submission failed: {exc}")
        return None

    try:
        blast_record = NCBIXML.read(handle)
    except Exception as exc:
        print(f"Failed to parse BLAST XML: {exc}")
        return None

    if not blast_record.alignments:
        return None
    top = blast_record.alignments[0]
    hit_id = top.hit_id or ""
    parts = hit_id.split("|")
    if len(parts) >= 2 and parts[1]:
        return parts[1].split(".")[0]
    if top.accession:
        return str(top.accession).split(".")[0]
    return hit_id or None


def _normalize_reviewed_mode(value: Any) -> str:
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"reviewed", "reviewed_only", "true"}:
            return "reviewed"
        if v in {"all", "false", "unreviewed"}:
            return "all"
        if v in {"fallback", "auto", "reviewed_then_all"}:
            return "fallback"
    return "reviewed" if bool(value) else "all"


def uniprot_sequences_for_gene(
    gene_id: str, timeout: int = 20
) -> Tuple[List[Dict[str, str]], bool]:
    base_query = f"gene_exact:{gene_id}"
    mode = _normalize_reviewed_mode(UNIPROT_REVIEWED_ONLY)

    def _run_query(query: str) -> Tuple[List[Dict[str, str]], bool]:
        results: List[Dict[str, str]] = []
        cursor: Optional[str] = None
        truncated = False
        while True:
            params = {
                "query": query,
                "fields": "accession,sequence",
                "format": "json",
                "size": UNIPROT_PAGE_SIZE,
            }
            if cursor:
                params["cursor"] = cursor
            try:
                r = session.get(UNIPROT_SEARCH_URL, params=params, timeout=timeout)
                r.raise_for_status()
            except requests.RequestException:
                _fallback_notice(
                    "uniprot_query_failed",
                    f"UniProt query failed for gene '{gene_id}' (query={query!r}); returning empty result set.",
                )
                return [], False
            data = r.json()
            for item in data.get("results", []) or []:
                acc = item.get("primaryAccession")
                seq = (item.get("sequence") or {}).get("value")
                if not acc or not seq:
                    continue
                results.append({"accession": str(acc), "sequence": str(seq)})
                if UNIPROT_MAX_RESULTS and len(results) >= UNIPROT_MAX_RESULTS:
                    truncated = True
                    return results, truncated
            cursor = _next_cursor(r.headers.get("Link"))
            if not cursor:
                break
        return results, truncated

    if mode == "reviewed":
        return _run_query(f"{base_query} AND reviewed:true")
    if mode == "all":
        return _run_query(base_query)
    # fallback: reviewed first, then include unreviewed if none found
    reviewed_results, reviewed_truncated = _run_query(f"{base_query} AND reviewed:true")
    if reviewed_results:
        return reviewed_results, reviewed_truncated
    _fallback_notice(
        "uniprot_reviewed_to_all",
        f"Reviewed-only UniProt query returned no entries for gene '{gene_id}'; falling back to all entries.",
    )
    return _run_query(base_query)


def _parasail_clean_sequence(seq: str) -> str:
    # Replace unexpected symbols so parasail matrix lookup always succeeds.
    return "".join(ch if ch in _PARASAIL_ALLOWED else "X" for ch in seq)

def _local_identity(seq_a: str, seq_b: str) -> float:
    seq_a = str(seq_a or "").strip().upper()
    seq_b = str(seq_b or "").strip().upper()
    if not seq_a or not seq_b:
        return 0.0
    if seq_a == seq_b:
        return 1.0

    qa = _parasail_clean_sequence(seq_a)
    qb = _parasail_clean_sequence(seq_b)
    result = parasail.sw_stats_striped_16(qa, qb, 1, 1, _PARASAIL_MATRIX)
    aln_len = int(result.length)
    if aln_len <= 0:
        return 0.0
    return float(result.matches) / float(aln_len)


def _best_uniprot_accessions_for_gene(
    gene_id: str, target_sequence: str
) -> Tuple[List[str], Optional[float], bool]:
    entries, truncated = uniprot_sequences_for_gene(gene_id)
    if not entries:
        return [], None, truncated

    target_sequence = str(target_sequence or "").strip().upper()
    if not target_sequence:
        return [], None, truncated

    best_score = -1.0
    best_accs: List[str] = []

    for entry in entries:
        seq = entry["sequence"]
        if not seq:
            continue
        score = _local_identity(target_sequence, seq)
        if score > best_score + SIMILARITY_TOL:
            best_score = score
            best_accs = [entry["accession"]]
        elif abs(score - best_score) <= SIMILARITY_TOL:
            best_accs.append(entry["accession"])

    best_accs = sorted(set(best_accs))
    return best_accs, (best_score if best_accs else None), truncated

def pdb_ids_from_pdbe_mapping(uniprot: str, timeout: int = 15) -> List[str]:
    url = f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{uniprot}"
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code == 404:
            _fallback_notice(
                "pdbe_mapping_404",
                f"PDBe mapping returned 404 for UniProt {uniprot}; returning empty list.",
            )
            return []
        r.raise_for_status()
        pdb_map = r.json().get(uniprot, {}).get("PDB", {})
        return sorted({pid.upper() for pid in pdb_map.keys()})
    except requests.RequestException:
        _fallback_notice(
            "pdbe_mapping_failed",
            f"PDBe mapping request failed for UniProt {uniprot}; returning empty list.",
        )
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
        _fallback_notice(
            "rcsb_search_failed",
            f"RCSB search request failed for UniProt {uniprot}; returning empty list.",
        )
        return []

def get_uniprot_sequence(accession: str, timeout: int = 15) -> Optional[str]:
    acc = str(accession or "").strip()
    if not acc:
        return None
    if acc in UNIPROT_SEQ_CACHE:
        return UNIPROT_SEQ_CACHE[acc]
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.json"
    data = _fetch_json(url, timeout=timeout)
    seq = (data or {}).get("sequence", {}).get("value") if data else None
    seq = _normalize_sequence(seq) if seq else None
    if seq:
        UNIPROT_SEQ_CACHE[acc] = seq
    return seq

def download_rcsb_model_any(
    pdb_id: str,
    out_dir: pathlib.Path,
    timeout: int = 25,
    assembly_id: Optional[str] = None,
) -> Optional[pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_id = str(assembly_id).strip() if assembly_id else ""
    candidates = []
    if safe_id:
        candidates.append(f"https://files.rcsb.org/download/{pdb_id}-assembly{safe_id}.cif")
    if safe_id != "1":
        candidates.append(f"https://files.rcsb.org/download/{pdb_id}-assembly1.cif")
    candidates.extend([
        f"https://files.rcsb.org/download/{pdb_id}.cif",
        f"https://files.rcsb.org/download/{pdb_id}.pdb",
    ])
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
    _fallback_notice(
        "download_rcsb_none",
        f"No downloadable RCSB model found for {pdb_id} (tried assembly+CIF/PDB fallbacks).",
    )
    return None

def _alphafold_prediction_json(accession: str, timeout: int = 20) -> List[Dict[str, Any]]:
    url = f"https://alphafold.ebi.ac.uk/api/prediction/{accession}"
    try:
        r = session.get(url, timeout=timeout)
        if r.status_code == 404:
            _fallback_notice(
                "alphafold_prediction_missing",
                f"AlphaFold prediction endpoint returned 404 for accession {accession}.",
            )
            return []
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []
    except requests.RequestException:
        _fallback_notice(
            "alphafold_prediction_request_failed",
            f"AlphaFold prediction request failed for accession {accession}; returning empty list.",
        )
        return []

def alphafold_best_prediction_meta(accession: str, timeout: int = 20) -> Optional[Dict[str, Any]]:
    acc = str(accession or "").strip()
    if not acc:
        return None
    if acc in AF_META_CACHE:
        return AF_META_CACHE[acc]
    preds = _alphafold_prediction_json(acc, timeout=timeout)
    if not preds:
        return None
    preds.sort(key=lambda d: d.get("modelVersion", 0), reverse=True)
    best = preds[0]
    meta = {
        "accession": acc,
        "modelVersion": best.get("modelVersion", 0),
        "confidence": best.get("confidenceAvgLocalScore"),
        "pdbUrl": best.get("pdbUrl"),
        "cifUrl": best.get("cifUrl"),
    }
    AF_META_CACHE[acc] = meta
    return meta

def download_alphafold_from_meta(
    meta: Dict[str, Any],
    out_dir: pathlib.Path,
    timeout: int = 30,
) -> Optional[pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    file_url = meta.get("pdbUrl") or meta.get("cifUrl")
    ext = ".pdb" if meta.get("pdbUrl") else ".cif"
    if not file_url:
        _fallback_notice(
            "download_af_missing_url",
            f"AlphaFold metadata has no downloadable URL for accession {meta.get('accession')}.",
        )
        return None
    file_name = pathlib.Path(urlparse(file_url).path).name or f"{meta.get('accession', 'AF')}{ext}"
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
        _fallback_notice(
            "download_af_request_failed",
            f"AlphaFold model download failed for URL: {file_url}",
        )
        return None
    return None

def download_alphafold_latest(accession: str, out_dir: pathlib.Path, timeout: int = 30) -> Optional[pathlib.Path]:
    meta = alphafold_best_prediction_meta(accession, timeout=timeout)
    if not meta:
        _fallback_notice(
            "download_af_latest_no_meta",
            f"No AlphaFold metadata found for accession {accession}.",
        )
        return None
    return download_alphafold_from_meta(meta, out_dir, timeout=timeout)

# ------------------- utils -------------------
def sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s)).strip("_") or "target"

def _output_suffix_for_model(model_path: pathlib.Path) -> str:
    suffix = model_path.suffix.lower()
    return suffix if suffix in (".cif", ".mmcif") else ".pdb"

def evaluate_and_record(label: str,
                        kind: str,
                        id_or_acc: str,
                        map_path: pathlib.Path,
                        model_path: pathlib.Path) -> Dict[str, Any]:
    """Run fitting+evaluation; return a row dict for logging."""
    FITTED_DIR.mkdir(parents=True, exist_ok=True)
    out_suffix = _output_suffix_for_model(model_path)
    out_model = FITTED_DIR / f"{label}_fitted_v5{out_suffix}"
    score, fitted = fit_model_and_score(
        map_path, model_path, out_pdb_path=out_model,
        recommended_contour_level=RECOMMENDED_CONTOUR_LEVEL,
    )
    row = {
        "label": label,
        "type": kind,                 # "AF" or "PDB"
        "id": id_or_acc,              # PDB ID or UniProt accession
        "score": score if score is not None else float("nan"),
        "score_vof_meanz": score if score is not None else float("nan"),
        "score_metric": "vof_x_max_meanz",
        "fitted_path": str(fitted) if fitted else str(out_model),
        "source_model": str(model_path),
        "kept": False,
    }
    # If the fit failed (score is NaN), proactively remove any leftover file.
    if score is None or not fitted:
        _fallback_notice(
            "evaluate_fit_failed_cleanup",
            f"Fit failed for {kind}:{id_or_acc}; cleaning up partial fitted output.",
        )
        try: pathlib.Path(out_model).unlink(missing_ok=True)
        except Exception: pass
    return row

def _build_fit_row(
    label: str,
    kind: str,
    id_or_acc: str,
    model_path: pathlib.Path,
    out_model: pathlib.Path,
    score: Optional[float],
    fitted: Optional[pathlib.Path],
) -> Dict[str, Any]:
    row = {
        "label": label,
        "type": kind,
        "id": id_or_acc,
        "score": score if score is not None else float("nan"),
        "score_vof_meanz": score if score is not None else float("nan"),
        "score_metric": "vof_x_max_meanz",
        "fitted_path": str(fitted) if fitted else str(out_model),
        "source_model": str(model_path),
        "kept": False,
    }
    if score is None or not fitted:
        _fallback_notice(
            "build_fit_row_failed_cleanup",
            f"Worker fit failed for {kind}:{id_or_acc}; cleaning up partial fitted output.",
        )
        try:
            pathlib.Path(out_model).unlink(missing_ok=True)
        except Exception:
            pass
    return row

def _evaluate_jobs(
    jobs: List[Dict[str, Any]],
    map_path: pathlib.Path,
    workers: int,
) -> List[Tuple[Dict[str, Any], Dict[str, Any], str]]:
    if not jobs:
        return []

    # Sequential fallback
    if int(workers) <= 1 or len(jobs) <= 1:
        _fallback_notice(
            "evaluate_jobs_sequential",
            f"Using sequential candidate evaluation (workers={int(workers)}, jobs={len(jobs)}).",
        )
        out: List[Tuple[Dict[str, Any], Dict[str, Any], str]] = []
        for job in jobs:
            row = evaluate_and_record(
                job["label"],
                job["kind"],
                job["id"],
                map_path,
                pathlib.Path(job["model_path"]),
            )
            out.append((job, row, ""))
        return out

    out: List[Tuple[Dict[str, Any], Dict[str, Any], str]] = []
    max_workers = min(max(1, int(workers)), len(jobs))

    # Use process workers so fit_v5 runs with isolated state per worker.
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        fut_to_job = {}
        for job in jobs:
            fut = executor.submit(
                fit_model_and_score,
                pathlib.Path(map_path),
                pathlib.Path(job["model_path"]),
                pathlib.Path(job["out_model"]),
                RECOMMENDED_CONTOUR_LEVEL,
            )
            fut_to_job[fut] = job

        for fut in as_completed(fut_to_job):
            job = fut_to_job[fut]
            err = ""
            score = None
            fitted = None
            try:
                score, fitted = fut.result()
            except Exception as exc:
                err = str(exc)
            row = _build_fit_row(
                job["label"],
                job["kind"],
                job["id"],
                pathlib.Path(job["model_path"]),
                pathlib.Path(job["out_model"]),
                score,
                fitted,
            )
            out.append((job, row, err))
    return out

def _chunked(items: List[Any], chunk_size: int):
    size = max(1, int(chunk_size))
    for i in range(0, len(items), size):
        yield items[i:i + size]

def _effective_fit_workers() -> int:
    try:
        w = int(CANDIDATE_FIT_MAX_WORKERS)
    except Exception:
        _fallback_notice(
            "fit_workers_default",
            f"Invalid CANDIDATE_FIT_MAX_WORKERS={CANDIDATE_FIT_MAX_WORKERS!r}; falling back to 1 worker.",
        )
        w = 1
    return max(1, w)

def _score_value(row: Dict[str, Any]) -> float:
    v = row.get("score", float("nan"))
    try:
        v = float(v)
    except Exception:
        v = float("nan")
    return v if pd.notna(v) else float("-inf")

def _is_better_row(candidate: Dict[str, Any], incumbent: Optional[Dict[str, Any]]) -> bool:
    candidate_score = _score_value(candidate)
    if candidate_score == float("-inf"):
        return False
    if incumbent is None:
        return True
    return candidate_score > _score_value(incumbent)

def _unlink_if_exists(path_value: Any) -> None:
    if not path_value:
        return
    try:
        pathlib.Path(path_value).unlink(missing_ok=True)
    except Exception:
        pass

def _cleanup_row_files(
    row: Optional[Dict[str, Any]],
    keep_fitted: Optional[str] = None,
    keep_source: Optional[str] = None,
) -> None:
    if not row:
        return
    fitted = row.get("fitted_path")
    source = row.get("source_model")
    if fitted and str(fitted) != str(keep_fitted):
        _unlink_if_exists(fitted)
    if source and str(source) != str(keep_source):
        _unlink_if_exists(source)

def _write_log(safe_q: str, rows: List[Dict[str, Any]]) -> pathlib.Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{safe_q}__fit_scores.csv"
    pd.DataFrame(rows).to_csv(log_path, index=False)
    return log_path

# ------------------- main per-target flow -------------------


def process_target(gene_id: str, sequence: str):
    gene_id = str(gene_id or "").strip()
    sequence = str(sequence or "").strip()
    if not sequence:
        print("Skipping row with empty sequence.")
        return

    if gene_id:
        acc_candidates, best_score, truncated = _best_uniprot_accessions_for_gene(
            gene_id, sequence
        )
        if not acc_candidates:
            print(f"{gene_id}: no UniProt accession found from gene_id/sequence")
            return
        tag = gene_id
        safe_q = sanitize_name(tag)
        trunc_note = " (truncated pool)" if truncated else ""
        score_txt = f"{best_score:.4f}" if best_score is not None else "N/A"
        print(
            f"\n=== {tag}: {len(acc_candidates)} UniProt candidate(s) tied at local similarity {score_txt}{trunc_note} ==="
        )
        print(f"{tag}: candidates -> {', '.join(acc_candidates)}")
    else:
        _fallback_notice(
            "gene_to_blast",
            "gene_id missing; using BLAST fallback to pick top UniProt accession.",
        )
        print("Gene id missing; running BLASTP vs Swiss-Prot to pick top UniProt...")
        acc = blast_top_uniprot_accession(sequence)
        if not acc:
            print("BLAST did not return a UniProt accession.")
            return
        acc_candidates = [acc]
        tag = f"BLAST_{acc}"
        safe_q = sanitize_name(tag)
        print(f"\n=== {tag}: 1 UniProt candidate from BLAST ===")
        print(f"{tag}: candidate -> {acc}")

    # -------------------- AlphaFold phase --------------------
    all_rows: List[Dict[str, Any]] = []
    af_kept: Optional[Dict[str, Any]] = None
    af_best_row: Optional[Dict[str, Any]] = None
    fit_workers = _effective_fit_workers()
    mode_txt = "sequential" if fit_workers <= 1 else "parallel"
    print(f"[{tag}] Candidate fit workers: {fit_workers} ({mode_txt})")

    af_clusters = build_af_clusters(
        acc_candidates,
        get_uniprot_sequence=get_uniprot_sequence,
        alphafold_best_prediction_meta=alphafold_best_prediction_meta,
        seq_cluster_identity=AF_SEQ_CLUSTER_IDENTITY,
    )
    print(f"[{tag}] AF clusters identified: {len(af_clusters)}")

    af_sources: List[Tuple[Dict[str, Any], Dict[str, Any], str]] = []
    for cluster in af_clusters:
        member = cluster.get("rep")
        if not member:
            continue
        acc = member["id"]
        af_sources.append((cluster, member, acc))

    for batch in _chunked(af_sources, fit_workers):
        af_jobs: List[Dict[str, Any]] = []
        for cluster, member, acc in batch:
            print(f"[{tag}] AF phase: UniProt {acc} (cluster {cluster['id'] + 1}/{len(af_clusters)})")
            af_model = download_alphafold_latest(acc, OUT_DIR)
            if not af_model:
                print(f"[{tag}] AlphaFold model not available for {acc}")
                continue
            label = f"{safe_q}__AF_{acc}"
            out_suffix = _output_suffix_for_model(af_model)
            out_model = FITTED_DIR / f"{label}_fitted_v5{out_suffix}"
            af_jobs.append({
                "label": label,
                "kind": "AF",
                "id": acc,
                "model_path": str(pathlib.Path(af_model).resolve()),
                "out_model": str(pathlib.Path(out_model).resolve()),
                "cluster": cluster,
                "member": member,
            })

        for job, af_row, err in _evaluate_jobs(af_jobs, MAP_PATH, fit_workers):
            cluster = job["cluster"]
            member = job["member"]
            acc = job["id"]
            if err:
                print(f"[{tag}] AF fit worker error for {acc}: {err}")
            af_row.update({
                "cluster_id": cluster["id"],
                "cluster_key": cluster["key"],
                "cluster_size": len(cluster["members"]),
                "cluster_rep": acc == (cluster.get("rep") or {}).get("id"),
                "seq_cluster_id": cluster.get("seq_cluster_id"),
                "conformation_key": cluster.get("conformation_key"),
                "assembly_key": cluster.get("assembly_key"),
                "resolution": None,
                "assembly_id": None,
                "af_confidence": member.get("confidence"),
                "af_model_version": member.get("modelVersion"),
            })
            all_rows.append(af_row)
            stxt = f"{af_row['score']:.4f}" if pd.notna(af_row["score"]) else "N/A"
            print(
                f"[{tag}] AF fit ({acc}): SCORE={stxt} -> {af_row['fitted_path'] if pd.notna(af_row['score']) else 'N/A'}"
            )

            if _is_better_row(af_row, af_best_row):
                prev_best = af_best_row
                af_best_row = af_row
                af_row["kept"] = True
                if prev_best:
                    prev_best["kept"] = False
                    _cleanup_row_files(
                        prev_best,
                        keep_fitted=af_row.get("fitted_path"),
                        keep_source=af_row.get("source_model"),
                    )
                print(f"[{tag}] AF best updated -> {acc} (SCORE={stxt})")
            else:
                _cleanup_row_files(
                    af_row,
                    keep_fitted=af_best_row.get("fitted_path") if af_best_row else None,
                    keep_source=af_best_row.get("source_model") if af_best_row else None,
                )

    if not af_kept:
        af_kept = af_best_row

    # -------------------- Experimental phase --------------------
    exp_kept: Optional[Dict[str, Any]] = None
    exp_best_row: Optional[Dict[str, Any]] = None

    print(f"[{tag}] EXP phase: collecting PDB candidates for {len(acc_candidates)} UniProt accession(s)")
    all_ids: List[str] = []
    for acc in acc_candidates:
        ids_pdbe = pdb_ids_from_pdbe_mapping(acc)
        if ids_pdbe:
            ids = ids_pdbe
        else:
            _fallback_notice(
                "pdbe_to_rcsb",
                f"PDBe mapping returned no PDB IDs for {acc}; falling back to RCSB search.",
            )
            ids = pdb_ids_from_rcsb_search(acc)
        if ids:
            print(f"[{tag}] UniProt {acc}: {len(ids)} PDB candidate(s)")
            all_ids.extend(ids)
        else:
            print(f"[{tag}] UniProt {acc}: no PDB candidates found")

    all_ids = sorted(set(all_ids))
    uniprot_set = set(acc_candidates)
    clusters_to_test: List[Dict[str, Any]] = []
    cluster_total = 0
    if USE_PDB_CLUSTERING:
        pdb_clusters = build_pdb_clusters(
            all_ids,
            uniprot_set,
            seq_cluster_identity=PDB_SEQ_CLUSTER_IDENTITY,
            assembly_id_for_clustering=ASSEMBLY_ID_FOR_CLUSTERING,
        ) if all_ids else []
        print(f"[{tag}] EXP clusters identified: {len(pdb_clusters)}")
        clusters_to_test = list(pdb_clusters)
        cluster_total = len(pdb_clusters)
        if MAX_EXPERIMENTAL_TRIES is not None:
            if MAX_EXPERIMENTAL_TRIES <= 0:
                clusters_to_test = []
                print(f"[{tag}] MAX_EXPERIMENTAL_TRIES <= 0; skipping experimental fitting.")
            elif len(pdb_clusters) > MAX_EXPERIMENTAL_TRIES:
                seed_material = f"{safe_q}|{CLUSTER_SAMPLE_SEED}" if CLUSTER_SAMPLE_SEED else safe_q
                seed = int(hashlib.md5(seed_material.encode()).hexdigest()[:8], 16)
                rng = random.Random(seed)
                indices = list(range(len(pdb_clusters)))
                rng.shuffle(indices)
                indices = indices[:MAX_EXPERIMENTAL_TRIES]
                clusters_to_test = [pdb_clusters[i] for i in indices]
                clusters_to_test.sort(
                    key=lambda c: (
                        _resolution_rank((c.get("rep") or {}).get("resolution")),
                        _method_rank((c.get("rep") or {}).get("methods", [])),
                        (c.get("rep") or {}).get("id", ""),
                    )
                )
                print(f"[{tag}] Sampling {len(clusters_to_test)} cluster(s) out of {len(pdb_clusters)} (MAX_EXPERIMENTAL_TRIES={MAX_EXPERIMENTAL_TRIES})")
    else:
        if all_ids:
            for idx, pid in enumerate(all_ids):
                member = {"id": pid, "resolution": None, "methods": [], "assembly_id": None}
                clusters_to_test.append({
                    "id": idx,
                    "key": f"raw:{pid}",
                    "seq_cluster_id": None,
                    "conformation_key": None,
                    "assembly_key": None,
                    "members": [member],
                    "rep": member,
                })
        cluster_total = len(clusters_to_test)
        print(f"[{tag}] EXP clustering disabled; evaluating {cluster_total} PDB candidate(s)")

    for batch in _chunked(clusters_to_test, fit_workers):
        exp_jobs: List[Dict[str, Any]] = []
        for cluster in batch:
            member = cluster.get("rep")
            if not member:
                continue
            pid = member["id"]
            model_path = download_rcsb_model_any(pid, OUT_DIR, assembly_id=member.get("assembly_id"))
            if not model_path:
                print(f"[{tag}] could not download PDB {pid}")
                continue
            label = f"{safe_q}__PDB_{pid}"
            out_suffix = _output_suffix_for_model(model_path)
            out_model = FITTED_DIR / f"{label}_fitted_v5{out_suffix}"
            exp_jobs.append({
                "label": label,
                "kind": "PDB",
                "id": pid,
                "model_path": str(pathlib.Path(model_path).resolve()),
                "out_model": str(pathlib.Path(out_model).resolve()),
                "cluster": cluster,
                "member": member,
            })

        for job, row, err in _evaluate_jobs(exp_jobs, MAP_PATH, fit_workers):
            cluster = job["cluster"]
            member = job["member"]
            pid = job["id"]
            if err:
                print(f"[{tag}] EXP fit worker error for {pid}: {err}")
            row.update({
                "cluster_id": cluster["id"],
                "cluster_key": cluster["key"],
                "cluster_size": len(cluster["members"]),
                "cluster_rep": pid == (cluster.get("rep") or {}).get("id"),
                "seq_cluster_id": cluster.get("seq_cluster_id"),
                "conformation_key": cluster.get("conformation_key"),
                "assembly_key": cluster.get("assembly_key"),
                "resolution": member.get("resolution"),
                "assembly_id": member.get("assembly_id"),
            })
            all_rows.append(row)
            stxt = f"{row['score']:.4f}" if pd.notna(row["score"]) else "N/A"
            res = member.get("resolution")
            res_txt = f"{res:.2f}A" if isinstance(res, (float, int)) else "N/A"
            print(
                f"[{tag}] EXP fit [PDB {pid}, res {res_txt}, cluster {cluster['id'] + 1}/{cluster_total}]: SCORE={stxt} -> {row['fitted_path'] if pd.notna(row['score']) else 'N/A'}"
            )

            if _is_better_row(row, exp_best_row):
                prev_best = exp_best_row
                exp_best_row = row
                row["kept"] = True
                if prev_best:
                    prev_best["kept"] = False
                    _cleanup_row_files(
                        prev_best,
                        keep_fitted=row.get("fitted_path"),
                        keep_source=row.get("source_model"),
                    )
                print(f"[{tag}] EXP best updated -> {pid} (SCORE={stxt})")
            else:
                _cleanup_row_files(
                    row,
                    keep_fitted=exp_best_row.get("fitted_path") if exp_best_row else None,
                    keep_source=exp_best_row.get("source_model") if exp_best_row else None,
                )

    if not exp_kept:
        exp_kept = exp_best_row

    # ----- Final selection + cleanup (keep best AF + best PDB) -----
    if af_kept:
        af_kept["kept"] = True
    if exp_kept:
        exp_kept["kept"] = True

    keep_keys = set()
    if af_kept:
        keep_keys.add((af_kept.get("label"), af_kept.get("type")))
    if exp_kept:
        keep_keys.add((exp_kept.get("label"), exp_kept.get("type")))
    for row in all_rows:
        row["kept"] = (row.get("label"), row.get("type")) in keep_keys

    keep_fitted = set()
    keep_sources = set()
    for row in (af_kept, exp_kept):
        if not row:
            continue
        fitted = row.get("fitted_path")
        if fitted:
            try:
                keep_fitted.add(pathlib.Path(fitted).resolve())
            except Exception:
                _fallback_notice(
                    "keep_fitted_absolute",
                    f"Could not resolve fitted path '{fitted}'; using absolute() fallback.",
                )
                keep_fitted.add(pathlib.Path(fitted).absolute())
        src = row.get("source_model")
        if src:
            try:
                keep_sources.add(pathlib.Path(src).resolve())
            except Exception:
                _fallback_notice(
                    "keep_source_absolute",
                    f"Could not resolve source path '{src}'; using absolute() fallback.",
                )
                keep_sources.add(pathlib.Path(src).absolute())

    # Remove extra fitted files for this target (leave best AF + best PDB).
    for candidate in FITTED_DIR.glob(f"{safe_q}__*_fitted_v5.*"):
        if not candidate.is_file():
            continue
        try:
            cand_resolved = candidate.resolve()
        except Exception:
            _fallback_notice(
                "candidate_absolute",
                f"Could not resolve candidate fitted path '{candidate}'; using absolute() fallback.",
            )
            cand_resolved = candidate.absolute()
        if cand_resolved in keep_fitted:
            continue
        try:
            candidate.unlink(missing_ok=True)
        except Exception:
            pass

    # Remove non-kept source models fetched in this run.
    for row in all_rows:
        src = row.get("source_model")
        if not src:
            continue
        try:
            src_path = pathlib.Path(src).resolve()
        except Exception:
            _fallback_notice(
                "source_absolute",
                f"Could not resolve source model path '{src}'; using absolute() fallback.",
            )
            src_path = pathlib.Path(src).absolute()
        if src_path in keep_sources:
            continue
        try:
            src_path.unlink(missing_ok=True)
        except Exception:
            pass

    # ----- Per-target log -----
    _write_log(safe_q, all_rows)
    print(f"[{tag}] Log written: {(LOGS_DIR / f'{safe_q}__fit_scores.csv')}")
# ------------------- CLI main -------------------
def main():
    if not MAP_PATH.exists():
        sys.exit(f"Map not found: {MAP_PATH}")
    try:
        df = pd.read_csv(CSV_PATH, dtype={"gene_id": str, "sequence": str})
    except FileNotFoundError:
        sys.exit(f"Input file not found: {CSV_PATH}")
    if "gene_id" not in df.columns:
        _fallback_notice(
            "main_missing_gene_id_column",
            "Input CSV missing 'gene_id' column; injecting empty gene_id values.",
        )
        df["gene_id"] = ""
    if "sequence" not in df.columns:
        sys.exit("CSV must contain a 'sequence' column.")
    total_rows = len(df)
    if not isinstance(ENTRIES_TO_PROCESS, int):
        sys.exit("ENTRIES_TO_PROCESS must be an integer.")
    if ENTRIES_TO_PROCESS < 1 or ENTRIES_TO_PROCESS > total_rows:
        sys.exit(f"ENTRIES_TO_PROCESS must be between 1 and {total_rows} (row count).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FITTED_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    processed = 0
    for _, row in df.iterrows():
        gene_id = str(row.get("gene_id", "")).strip()
        sequence = str(row.get("sequence", "")).strip()
        if not sequence:
            continue
        process_target(gene_id, sequence)
        processed += 1
        if processed >= ENTRIES_TO_PROCESS:
            break

    if processed == 0:
        sys.exit("CSV has no rows with non-empty 'sequence'.")

    print(f"\nFitted PDBs -> {FITTED_DIR.resolve().as_posix()}")
    print(f"Logs       -> {LOGS_DIR.resolve().as_posix()}")

if __name__ == "__main__":
    main()
