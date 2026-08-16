"""
Candidate generation for StructFetch.

Given an input protein sequence, this module:

1. Aligns the input directly against every protein polymer-entity chain
   sequence deposited in the PDB. The pool is built from RCSB's bundled
   ``pdb_seqres.txt.gz``, filtered to ``mol:protein`` records, and deduplicated
   to unique sequences with back-references to every (PDB ID, chain ID) pair
   that has that sequence.
2. Identifies the top Smith-Waterman score and every unique sequence tied at
   that score.
3. Returns the union of PDB IDs that contain at least one chain with one of
   those tied top sequences as the candidate set.
4. Looks up UniProt cross-references for the candidate PDBs so the downstream
   AlphaFold candidate phase in ``pipeline.py`` can fetch predictions for the
   same proteins.

Pool lifecycle is **per-run, ephemeral**:

- The first call to :func:`candidates_for_sequence` in a process auto-downloads
  ``pdb_seqres.txt.gz`` from RCSB into ``tmp/`` if it isn't already on disk,
  then parses it into the in-memory caches.
- Subsequent calls in the same process reuse the in-memory caches.
- :func:`cleanup_pool` clears all in-memory caches and unlinks the on-disk
  FASTA. ``pipeline.run_csv_with_alignment_config`` invokes it from a ``finally``
  block, so every CLI invocation starts from a fresh download.
"""
from __future__ import annotations

import gzip
import csv
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter, Retry

try:
    import parasail
except Exception as exc:  # pragma: no cover
    raise RuntimeError("parasail is required: pip install parasail") from exc

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def _default_pool_dir() -> Path:
    env_tmp = os.environ.get("CRYOID_STRUCT_FETCH_TMP_DIR")
    if env_tmp:
        return Path(env_tmp).expanduser() / "pdb_pool"
    return Path.home() / ".cryoid" / "cache" / "structure_fitting" / "protein_pool"


POOL_FASTA_GZ = _default_pool_dir() / "pdb_seqres.txt.gz"
POOL_FASTA_PLAIN = _default_pool_dir() / "pdb_seqres.txt"
POOL_SOURCE_URL = "https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"
SEQUENCES_WITH_STRUCTURES_TSV: Path | None = None

# Alphabet accepted by parasail's BLOSUM62 matrix. Any character in the query
# or pool sequences that isn't in this set is replaced by 'X' before scoring.
ALPHABET = "ACDEFGHIKLMNPQRSTVWYBXZJUO*"
_ALLOWED = set(ALPHABET)

# Smith-Waterman scoring uses the BLAST blastp defaults so that scores are
# comparable to NCBI BLAST output for high-identity matches.
_BLOSUM62 = parasail.blosum62
_GAP_OPEN = 11
_GAP_EXTEND = 1


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "cryoid-struct-fetch-new-method/0.5"})
SESSION.mount(
    "https://",
    HTTPAdapter(max_retries=Retry(total=4, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])),
)


def configure_pool_dir(pool_dir: Path | str) -> None:
    """Route the on-disk PDB sequence pool cache into a run-local temp dir."""
    global POOL_FASTA_GZ, POOL_FASTA_PLAIN
    root = Path(pool_dir).expanduser().resolve()
    POOL_FASTA_GZ = root / "pdb_seqres.txt.gz"
    POOL_FASTA_PLAIN = root / "pdb_seqres.txt"


def configure_sequences_with_structures_path(path: Path | str) -> None:
    global SEQUENCES_WITH_STRUCTURES_TSV
    SEQUENCES_WITH_STRUCTURES_TSV = Path(path).expanduser().resolve()


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StructureSequenceSelection:
    """Candidate generation result for one input sequence.

    Attributes
    ----------
    accessions
        UniProt accessions cross-referenced from the candidate PDBs. Used by
        the downstream AlphaFold candidate phase. Sorted and deduplicated.
    pdb_ids
        PDB identifiers whose polymer-entity chain sequence is among the tied
        top SW hits. Sorted and deduplicated.
    best_score
        SW score shared by every tied top sequence. ``None`` when the input
        sequence was empty or no hits could be scored.
    metadata
        Free-form dict of diagnostics (top-score, tie counts, per-PDB matched
        chains, error string if generation failed, etc.). Schema is informal
        and intended for log output.
    """

    accessions: list[str]
    pdb_ids: list[str]
    best_score: Optional[float]
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Caches (cleared by cleanup_pool)
# ---------------------------------------------------------------------------

AF_META_CACHE: dict[str, list[dict[str, Any]]] = {}
_PDB_UNIPROT_CACHE: dict[str, dict[str, list[str]]] = {}
_NOTICE_SEEN: set[str] = set()
_POOL_SEQS: list[str] | None = None
_POOL_BACKREFS: list[list[tuple[str, str]]] | None = None
_POOL_PATH: Path | None = None


def _notice(tag: str, message: str, *, once: bool = True) -> None:
    """Print a one-time warning unless ``once=False``."""
    if once and tag in _NOTICE_SEEN:
        return
    _NOTICE_SEEN.add(tag)
    print(f"[fetch] {message}", flush=True)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def sanitize_name(value: Any) -> str:
    """Return ``value`` with non-filename-safe characters replaced by ``_``."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_") or "target"


def _normalize_sequence(value: Any) -> str:
    """Upper-case and strip everything outside ``[A-Za-z*]`` from a sequence."""
    return re.sub(r"[^A-Za-z*]", "", str(value or "")).upper()


def _sanitize_for_parasail(seq: str) -> str:
    """Normalize a sequence and force any out-of-alphabet residue to 'X'."""
    return "".join(ch if ch in _ALLOWED else "X" for ch in _normalize_sequence(seq))


def _normalize_pdb_id(value: Any) -> str:
    """Strip path/extension noise from a PDB identifier and upper-case it."""
    text = str(value or "").strip().upper()
    for suffix in (".CIF", ".PDB", ".ENT", ".MMCIF"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text


def _fetch_json(url: str, *, timeout: int = 20, **kwargs: Any) -> Any:
    """GET ``url`` and decode the JSON body. Raises on non-2xx."""
    response = SESSION.get(url, timeout=timeout, **kwargs)
    response.raise_for_status()
    return response.json()


# ---------------------------------------------------------------------------
# Pool loader (ephemeral per run)
# ---------------------------------------------------------------------------


def _download_pool(dest: Path) -> None:
    """Stream ``POOL_SOURCE_URL`` to ``dest`` via a ``.tmp`` rename for atomicity."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    started = time.perf_counter()
    print(f"[fetch] downloading PDB chain pool from {POOL_SOURCE_URL}", flush=True)
    try:
        with SESSION.get(POOL_SOURCE_URL, stream=True, timeout=600) as response:
            response.raise_for_status()
            with tmp.open("wb") as h:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        h.write(chunk)
        tmp.replace(dest)
        size_mb = dest.stat().st_size / 1024 / 1024
        print(f"[fetch] downloaded pool: {size_mb:.1f} MB in {time.perf_counter() - started:.1f}s", flush=True)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _resolve_pool_path() -> Path:
    """Return an on-disk pool path, force-downloading a fresh copy from RCSB on every call.

    Pool cleanup at the end of each run is best-effort and can fail silently
    (Windows file locks, permission issues). Any on-disk pool found at the
    start of a run is therefore treated as potentially stale (could be days
    or weeks old, while RCSB updates ``pdb_seqres.txt.gz`` weekly), so we
    delete it and download fresh. Within a single process this function is
    only reached once, since ``_ensure_pool_loaded`` short-circuits on the
    in-memory cache after the first load.
    """
    for path in (POOL_FASTA_GZ, POOL_FASTA_PLAIN):
        if path.exists():
            try:
                path.unlink()
                print(f"[fetch] removed stale pool from prior run: {path.name}", flush=True)
            except OSError as exc:
                print(f"[fetch] could not remove stale pool {path}: {exc}", flush=True)
    _download_pool(POOL_FASTA_GZ)
    return POOL_FASTA_GZ


def _open_pool(path: Path):
    """Open a pool FASTA as text, transparently decompressing ``.gz``."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def cleanup_pool() -> None:
    """Drop the in-memory pool, clear caches, and delete the on-disk FASTA.

    Called from ``pipeline.run_csv_with_alignment_config`` in a ``finally`` block
    so every CLI invocation downloads a fresh copy of the pool on its next run.
    Idempotent and safe to call when nothing has been loaded.
    """
    global _POOL_SEQS, _POOL_BACKREFS, _POOL_PATH
    _POOL_SEQS = None
    _POOL_BACKREFS = None
    _POOL_PATH = None
    AF_META_CACHE.clear()
    _PDB_UNIPROT_CACHE.clear()
    _NOTICE_SEEN.clear()
    for path in (POOL_FASTA_GZ, POOL_FASTA_PLAIN):
        try:
            if path.exists():
                path.unlink()
                print(f"[fetch] removed pool file: {path.name}", flush=True)
        except OSError as exc:
            print(f"[fetch] could not remove pool file {path}: {exc}", flush=True)


def _ensure_pool_loaded() -> tuple[list[str], list[list[tuple[str, str]]]]:
    """Load and dedupe the PDB chain pool, caching results in module globals.

    Returns two parallel lists keyed by unique sequence: ``seqs[i]`` is the i-th
    unique protein chain sequence (sanitized to the parasail alphabet) and
    ``backrefs[i]`` is the list of ``(pdb_id, chain_id)`` pairs that share it.
    """
    global _POOL_SEQS, _POOL_BACKREFS, _POOL_PATH
    if _POOL_SEQS is not None and _POOL_BACKREFS is not None:
        return _POOL_SEQS, _POOL_BACKREFS

    path = _resolve_pool_path()
    started = time.perf_counter()
    seq_to_chains: dict[str, list[tuple[str, str]]] = defaultdict(list)
    cur_id = ""
    cur_is_protein = False
    buf: list[str] = []
    n_records = 0
    with _open_pool(path) as handle:
        for line in handle:
            if line.startswith(">"):
                if cur_id and cur_is_protein and buf:
                    seq = _sanitize_for_parasail("".join(buf))
                    if seq:
                        pdb, chain = cur_id.split("_", 1) if "_" in cur_id else (cur_id, "")
                        seq_to_chains[seq].append((pdb.upper(), chain))
                n_records += 1
                header = line[1:].rstrip()
                space = header.find(" ")
                cur_id = header[:space] if space > 0 else header
                cur_is_protein = "mol:protein" in header
                buf = []
            else:
                if cur_is_protein:
                    buf.append(line.rstrip())
    if cur_id and cur_is_protein and buf:
        seq = _sanitize_for_parasail("".join(buf))
        if seq:
            pdb, chain = cur_id.split("_", 1) if "_" in cur_id else (cur_id, "")
            seq_to_chains[seq].append((pdb.upper(), chain))

    seqs = list(seq_to_chains.keys())
    backrefs = [seq_to_chains[s] for s in seqs]
    if SEQUENCES_WITH_STRUCTURES_TSV is not None:
        SEQUENCES_WITH_STRUCTURES_TSV.parent.mkdir(parents=True, exist_ok=True)
        with SEQUENCES_WITH_STRUCTURES_TSV.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["pdb_id", "chain_id", "sequence"])
            for sequence, chains in seq_to_chains.items():
                for pdb_id, chain_id in chains:
                    writer.writerow([pdb_id, chain_id, sequence])
    _POOL_SEQS, _POOL_BACKREFS, _POOL_PATH = seqs, backrefs, path
    print(
        f"[fetch] loaded PDB chain pool from {path.name}: "
        f"{n_records} records -> {sum(len(b) for b in backrefs)} protein chains -> "
        f"{len(seqs)} unique sequences ({time.perf_counter()-started:.1f}s)",
        flush=True,
    )
    return seqs, backrefs


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _sw_score(query: str, seq: str) -> int:
    """Smith-Waterman score with automatic 16->32 bit fallback on saturation.

    The 16-bit striped SIMD path (max representable score 32,767) is used by
    default for speed. If parasail reports ``result.saturated``, the same
    alignment is re-run with 32-bit precision so the returned score is
    always correct. The 32-bit re-run only happens on long high-identity
    cases (none observed in the 52-case benchmark, where the max SW score
    was 26,440); a one-time notice records the first time it fires.
    """
    result = parasail.sw_striped_16(query, seq, _GAP_OPEN, _GAP_EXTEND, _BLOSUM62)
    if result.saturated:
        _notice(
            "sw_saturation",
            "16-bit SW alignment saturated; falling back to 32-bit (slower, correct).",
        )
        result = parasail.sw_striped_32(query, seq, _GAP_OPEN, _GAP_EXTEND, _BLOSUM62)
    return int(result.score)


def _top_chain_hits(query_sanitized: str) -> tuple[int, list[int]]:
    """SW-score the query against every pool entry, return ``(top_score, tied_indices)``.

    No length-ratio prefilter is applied: every one of the ~182k unique
    sequences is aligned. Each alignment uses :func:`_sw_score`, which
    automatically promotes to 32-bit precision when 16-bit saturates.
    """
    seqs, _ = _ensure_pool_loaded()
    top_score = -1
    tied: list[int] = []
    for i, seq in enumerate(seqs):
        score = _sw_score(query_sanitized, seq)
        if score > top_score:
            top_score = score
            tied = [i]
        elif score == top_score:
            tied.append(i)
    return top_score, tied


# ---------------------------------------------------------------------------
# UniProt cross-references (used by the downstream AlphaFold candidate phase)
# ---------------------------------------------------------------------------


def _pdb_chain_uniprot_map(pdb_id: str, timeout: int = 15) -> dict[str, list[str]]:
    """Return a per-chain UniProt accession map for one PDB.

    Output: ``{chain_id: [accession, ...], ...}``. Cached per PDB so multiple
    matched chains of the same PDB share a single HTTP call.

    Why per-chain rather than per-PDB: PDBe's ``/mappings/uniprot/{pdb}`` endpoint
    returns every UniProt accession observed in the PDB entry, including binding
    partners that share an asymmetric unit with the target. For AlphaFold
    candidate fetching, we only want accessions for the chain(s) whose sequence
    actually matched the query, so we need to filter by the specific chain id.
    """
    pdb_id = _normalize_pdb_id(pdb_id)
    if not pdb_id:
        return {}
    if pdb_id in _PDB_UNIPROT_CACHE:
        return _PDB_UNIPROT_CACHE[pdb_id]
    try:
        response = SESSION.get(
            f"https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/{pdb_id.lower()}",
            timeout=timeout,
        )
        if response.status_code == 404:
            _PDB_UNIPROT_CACHE[pdb_id] = {}
            return {}
        response.raise_for_status()
        block = response.json().get(pdb_id.lower(), {}).get("UniProt", {}) or {}
        chain_to_accs: dict[str, list[str]] = defaultdict(list)
        for accession, data in block.items():
            accession = str(accession or "").strip()
            if not accession:
                continue
            for mapping in (data.get("mappings") or []):
                chain = str(mapping.get("chain_id") or "").strip()
                if chain and accession not in chain_to_accs[chain]:
                    chain_to_accs[chain].append(accession)
        result = {chain: sorted(accs) for chain, accs in chain_to_accs.items()}
        _PDB_UNIPROT_CACHE[pdb_id] = result
        return result
    except requests.RequestException:
        _notice("pdbe_pdb_uniprot_failed", f"PDBe UniProt mapping failed for {pdb_id}.", once=False)
        _PDB_UNIPROT_CACHE[pdb_id] = {}
        return {}


def _uniprots_for_top_hit_chains(
    pdb_chain_pairs: list[tuple[str, str]],
    max_workers: int = 16,
) -> list[str]:
    """Return the sorted union of UniProt accessions assigned to the given (PDB, chain) pairs.

    Only chains whose sequence is a tied top SW hit should be in
    ``pdb_chain_pairs``. Any UniProt accession associated solely with a
    different chain in the same PDB (i.e. a binding partner) is excluded.
    """
    if not pdb_chain_pairs:
        return []
    unique_pdbs = sorted({pdb for pdb, _ in pdb_chain_pairs})
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pdbe-uniprot") as ex:
        per_pdb_maps = dict(zip(unique_pdbs, ex.map(_pdb_chain_uniprot_map, unique_pdbs)))
    out: set[str] = set()
    for pdb_id, chain_id in pdb_chain_pairs:
        chain_map = per_pdb_maps.get(pdb_id, {})
        out.update(chain_map.get(chain_id, []))
    return sorted(out)


# ---------------------------------------------------------------------------
# Top-level candidate generation
# ---------------------------------------------------------------------------


def candidates_for_sequence(input_sequence: str) -> StructureSequenceSelection:
    """Return candidate PDBs (and derived UniProt accessions) for an input sequence.

    Pipeline:

    1. Normalize the input and sanitize for parasail's alphabet.
    2. Run Smith-Waterman against every unique pool sequence; identify the top
       SW score and every unique sequence tied at that score.
    3. Collect every ``(pdb_id, chain_id)`` pair that has one of those tied top
       sequences; dedupe to a sorted PDB ID list.
    4. Look up UniProt cross-references for the specific (PDB, chain) pairs
       whose sequence is a tied top hit. Only accessions that map to those
       chains are returned, so the AlphaFold path in ``pipeline.py`` does not
       fetch predictions for binding partners that happen to co-occur in the
       same PDB entry.

    Returns a :class:`StructureSequenceSelection`; an empty input or a search
    that produces no hits yields an empty selection with ``metadata["error"]``
    populated.
    """
    input_sequence = _normalize_sequence(input_sequence)
    if not input_sequence:
        return StructureSequenceSelection(
            accessions=[],
            pdb_ids=[],
            best_score=None,
            metadata={"error": "empty_input_sequence"},
        )

    seqs, backrefs = _ensure_pool_loaded()
    query = _sanitize_for_parasail(input_sequence)
    top_score, tied_indices = _top_chain_hits(query)
    if not tied_indices or top_score < 0:
        return StructureSequenceSelection(
            accessions=[],
            pdb_ids=[],
            best_score=None,
            metadata={"error": "no_top_hits"},
        )

    tied_chains: list[tuple[str, str]] = []
    for idx in tied_indices:
        tied_chains.extend(backrefs[idx])

    pdb_to_matched_chains: dict[str, list[str]] = defaultdict(list)
    for pdb_id, chain_id in tied_chains:
        pdb_to_matched_chains[pdb_id].append(chain_id)

    pdb_ids_sorted = sorted(pdb_to_matched_chains.keys())
    # Restrict UniProt-accession lookup to the chain(s) whose sequence is a
    # tied top hit. Without this filter, PDBe returns every UniProt observed
    # in the candidate PDBs, which includes binding partners (e.g. calmodulin
    # and FKBP1B in RYR1 complexes) and inflates the AlphaFold-candidate pool
    # with predictions for unrelated proteins.
    uniprot_accessions = _uniprots_for_top_hit_chains(tied_chains)

    metadata: dict[str, Any] = {
        "candidate_generation_method": "pdb_chain_sw",
        "pool_path": str(_POOL_PATH) if _POOL_PATH else "",
        "top_score": int(top_score),
        "tied_top_sequence_count": len(tied_indices),
        "tied_top_chain_count": len(tied_chains),
        "tied_top_sequence_lengths": sorted({len(seqs[i]) for i in tied_indices}),
        "total_pdb_candidates": len(pdb_ids_sorted),
        "matched_chains_per_pdb": {pid: sorted(set(chains)) for pid, chains in pdb_to_matched_chains.items()},
        "uniprot_accession_count": len(uniprot_accessions),
    }

    return StructureSequenceSelection(
        accessions=uniprot_accessions,
        pdb_ids=pdb_ids_sorted,
        best_score=float(top_score),
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# Model downloaders (AlphaFold + RCSB)
# ---------------------------------------------------------------------------


def alphafold_prediction_metas(accession: str, timeout: int = 20) -> list[dict[str, Any]]:
    """Fetch AlphaFold DB prediction metadata for ``accession``, deduped by file URL."""
    accession = str(accession or "").strip()
    if not accession:
        return []
    if accession in AF_META_CACHE:
        return AF_META_CACHE[accession]
    try:
        data = _fetch_json(f"https://alphafold.ebi.ac.uk/api/prediction/{accession}", timeout=timeout)
    except requests.RequestException:
        _notice("alphafold_prediction_failed", f"AlphaFold prediction metadata failed for {accession}.")
        AF_META_CACHE[accession] = []
        return []
    predictions = data if isinstance(data, list) else []
    if not predictions:
        AF_META_CACHE[accession] = []
        return []

    def _af_version(item: dict[str, Any]) -> int:
        try:
            return int(item.get("modelVersion") or item.get("latestVersion") or 0)
        except (TypeError, ValueError):
            return 0

    predictions.sort(
        key=lambda item: (_af_version(item), str(item.get("pdbUrl") or item.get("cifUrl") or "")),
        reverse=True,
    )
    metas: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for prediction in predictions:
        file_url = str(prediction.get("pdbUrl") or prediction.get("cifUrl") or "").strip()
        if not file_url or file_url in seen_urls:
            continue
        seen_urls.add(file_url)
        metas.append(
            {
                "accession": accession,
                "modelVersion": _af_version(prediction),
                "latestVersion": prediction.get("latestVersion"),
                "confidence": prediction.get("confidenceAvgLocalScore"),
                "pdbUrl": prediction.get("pdbUrl"),
                "cifUrl": prediction.get("cifUrl"),
                "fileUrl": file_url,
            }
        )
    AF_META_CACHE[accession] = metas
    return metas


def download_alphafold_prediction(accession: str, meta: dict[str, Any], out_dir: Path, timeout: int = 30) -> Optional[Path]:
    """Download one AlphaFold model file described by ``meta`` into ``out_dir``.

    Returns the local path on success, ``None`` on any failure. Existing files
    of the same name with non-zero size are reused without re-downloading.
    """
    file_url = meta.get("fileUrl") or meta.get("pdbUrl") or meta.get("cifUrl")
    if not file_url:
        return None
    ext = ".pdb" if meta.get("pdbUrl") else ".cif"
    file_name = Path(urlparse(str(file_url)).path).name or f"AF-{accession}-F1-model_v{meta.get('modelVersion', 0)}{ext}"
    if not Path(file_name).suffix:
        file_name = f"{file_name}{ext}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / file_name
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path
    try:
        response = SESSION.get(str(file_url), timeout=timeout)
        response.raise_for_status()
        text = response.text.strip() if response.text else ""
        if response.content and not text.startswith("<!DOCTYPE"):
            out_path.write_bytes(response.content)
            if out_path.stat().st_size > 0:
                return out_path
    except requests.RequestException:
        _notice("download_af_failed", f"AlphaFold model download failed for {accession}.", once=False)
    return None


def download_rcsb_model_any(pdb_id: str, out_dir: Path, timeout: int = 25, assembly_id: Optional[str] = None) -> Optional[Path]:
    """Download a PDB model file from RCSB, trying assembly + asymmetric unit in CIF/PDB order.

    Returns the local path of the first successful download or ``None`` if no
    candidate URL yielded a non-empty model. Existing files of the same name
    with non-zero size are reused without re-downloading.
    """
    pdb_id = str(pdb_id or "").strip().upper()
    if not pdb_id:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_assembly = str(assembly_id).strip() if assembly_id else ""
    urls: list[str] = []
    if safe_assembly:
        urls.append(f"https://files.rcsb.org/download/{pdb_id}-assembly{safe_assembly}.cif")
    if safe_assembly != "1":
        urls.append(f"https://files.rcsb.org/download/{pdb_id}-assembly1.cif")
    urls.extend([f"https://files.rcsb.org/download/{pdb_id}.cif", f"https://files.rcsb.org/download/{pdb_id}.pdb"])
    for url in urls:
        ext = ".cif" if url.endswith(".cif") else ".pdb"
        out_path = out_dir / f"{pdb_id}{ext}"
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path
        try:
            response = SESSION.get(url, timeout=timeout)
            text = response.text.strip() if response.text else ""
            if response.ok and response.content and not (text.startswith("<!DOCTYPE") or text.startswith("<?xml")):
                out_path.write_bytes(response.content)
                if out_path.stat().st_size > 0:
                    return out_path
        except requests.RequestException:
            continue
    _notice("download_rcsb_none", f"No downloadable RCSB model found for {pdb_id}.")
    return None
