#!/usr/bin/env python3
"""
Generate a CSV list of random cryo-EM structures meeting strict criteria:
- has an EMDB density map
- has exactly one UniProt accession across all polymer entities
- has exactly one polymer entity (single unique sequence)
- contains only standard protein polymers (no ligands, nucleic acids, or modified residues)
- optional UniProt organism keyword filter (default: malaria/Plasmodium)
Stratify entries by resolution: <4A, 4-6A, >6A (up to TARGET_PER_BIN each).
By default, no structure/map files are downloaded; only API metadata is queried.
When downloads are disabled, testset_list.csv includes resolution, gene id,
uniprot accession, sequence (from PDB polymer entities), pdb id, emdb id,
recommended contour level. Each PDB ID must map to a single unique sequence
or the run fails.
"""
from __future__ import annotations

import csv
import gzip
import pathlib
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

import requests
from requests.adapters import HTTPAdapter, Retry

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
RCSB_ENTITY_URL = "https://data.rcsb.org/rest/v1/core/polymer_entity/{pdb_id}/{entity_id}"
UNIPROT_ENTRY_URL = "https://rest.uniprot.org/uniprotkb/{accession}.json"
EMDB_ENTRY_URL = "https://www.ebi.ac.uk/emdb/api/entry/{emdb_id}"

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_OUT_PATH = SCRIPT_DIR / "testset_list.csv"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# ----------- HARD-CODED SETTINGS -----------
TARGET_PER_BIN = 30
LOW_RESOLUTION_MAX_A = 4.0
MID_RESOLUTION_MIN_A = 4.0
MID_RESOLUTION_MAX_A = 6.0
HIGH_RESOLUTION_MIN_A = 6.0
OUT_PATH = DEFAULT_OUT_PATH
SEED = 0
INCLUDE_HEADER = True
DOWNLOAD_PDBS_AND_MAPS = False
DOWNLOAD_DIR = SCRIPT_DIR / "testset_downloads_new"
DOWNLOAD_CSV_NAME = "download_manifest.csv"
SPECIES_KEYWORD = ""  # case-insensitive substring in UniProt organism fields; set to "" to disable.
CSV_HEADER = [
    "resolution",
    "gene id",
    "uniprot accession",
    "sequence",
    "pdb id",
    "emdb id",
    "recommended contour level",
]
BIN_ORDER = ("low", "mid", "high")


def _make_session() -> requests.Session:
    retries = Retry(total=4, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "cryoid-testset-generator/1.0"})
    return session


def _rcsb_search_ids_from_query(
    session: requests.Session, query: Dict, page_size: int = 1000
) -> List[str]:
    request_options = query.setdefault("request_options", {})
    paginate = request_options.setdefault("paginate", {})
    paginate.setdefault("rows", page_size)

    ids: List[str] = []
    start = 0
    while True:
        paginate["start"] = start
        resp = session.post(RCSB_SEARCH_URL, json=query, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        result_set = data.get("result_set", []) or []
        if not result_set:
            break
        ids.extend([row["identifier"] for row in result_set if "identifier" in row])
        start += len(result_set)
        if len(result_set) < paginate["rows"]:
            break
    return ids


def _rcsb_search_ids(
    session: requests.Session,
    max_resolution: Optional[float] = None,
    species_keyword: str = "",
    page_size: int = 1000,
) -> List[str]:
    nodes = [
        {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "exptl.method",
                "operator": "exact_match",
                "value": "ELECTRON MICROSCOPY",
            },
        },
    ]
    if max_resolution is not None:
        nodes.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entry_info.resolution_combined",
                    "operator": "less",
                    "value": max_resolution,
                },
            }
        )

    species_keyword = str(species_keyword or "").strip()
    if species_keyword:
        nodes.append(
            {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_entity_source_organism.taxonomy_lineage.name",
                    "operator": "contains_words",
                    "value": species_keyword,
                },
            }
        )

    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": nodes,
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": page_size}},
    }
    return _rcsb_search_ids_from_query(session, query, page_size=page_size)


def _fetch_entry(session: requests.Session, pdb_id: str) -> Optional[Dict]:
    url = RCSB_ENTRY_URL.format(pdb_id=pdb_id)
    resp = session.get(url, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def _entry_resolution(entry: Dict) -> Optional[float]:
    info = entry.get("rcsb_entry_info", {}) or {}
    res_list = info.get("resolution_combined") or []
    if isinstance(res_list, list) and res_list:
        vals: List[float] = []
        for v in res_list:
            try:
                vals.append(float(v))
            except Exception:
                continue
        if vals:
            return min(vals)
    return None


def _resolution_bin(resolution: float) -> Optional[str]:
    if resolution < LOW_RESOLUTION_MAX_A:
        return "low"
    if MID_RESOLUTION_MIN_A <= resolution <= MID_RESOLUTION_MAX_A:
        return "mid"
    if resolution > HIGH_RESOLUTION_MIN_A:
        return "high"
    return None


def _fetch_polymer_entity(session: requests.Session, pdb_id: str, entity_id: str) -> Optional[Dict]:
    url = RCSB_ENTITY_URL.format(pdb_id=pdb_id, entity_id=entity_id)
    resp = session.get(url, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def _extract_uniprot_accessions(entity: Dict) -> Set[str]:
    ref_ids = (
        entity.get("rcsb_polymer_entity_container_identifiers", {})
        .get("reference_sequence_identifiers", [])
        or []
    )
    accs: Set[str] = set()
    for ref in ref_ids:
        db = (ref.get("database_name") or ref.get("database_id") or "").upper()
        if db in {"UNIPROT", "UNIPROTKB", "UNP"}:
            acc = ref.get("database_accession") or ref.get("database_accession_id")
            if acc:
                accs.add(str(acc))
    return accs


def _entry_has_only_protein(entry: Dict) -> bool:
    identifiers = entry.get("rcsb_entry_container_identifiers", {}) or {}
    non_poly = identifiers.get("non_polymer_entity_ids") or identifiers.get("nonpolymer_entity_ids") or []
    branched = identifiers.get("branched_entity_ids") or []
    return not non_poly and not branched


def _standard_protein_sequence(entity: Dict) -> Optional[str]:
    entity_poly = entity.get("entity_poly") or {}
    poly_type = (entity_poly.get("type") or "").lower()
    if poly_type != "polypeptide(l)":
        return None
    non_std = entity_poly.get("rcsb_non_std_monomer_count")
    if isinstance(non_std, (int, float)) and non_std > 0:
        return None
    seq = entity_poly.get("pdbx_seq_one_letter_code_can") or ""
    seq = "".join(str(seq).split()).upper()
    if not seq:
        return None
    for ch in seq:
        if ch not in STANDARD_AA:
            return None
    return seq


def _is_standard_protein_entity(entity: Dict) -> bool:
    return _standard_protein_sequence(entity) is not None


def _entry_single_accession_and_sequence(
    session: requests.Session,
    pdb_id: str,
    entry: Optional[Dict] = None,
    require_emdb: bool = True,
    require_single_entity: bool = True,
) -> Optional[Tuple[str, str]]:
    if entry is None:
        entry = _fetch_entry(session, pdb_id)
    if not entry:
        return None
    identifiers = entry.get("rcsb_entry_container_identifiers", {}) or {}
    if require_emdb:
        emdb_ids = identifiers.get("emdb_ids") or []
        if not emdb_ids:
            return None
    if not _entry_has_only_protein(entry):
        return None

    polymer_entity_ids = identifiers.get("polymer_entity_ids") or []
    if not polymer_entity_ids:
        return None
    if require_single_entity and len(polymer_entity_ids) != 1:
        return None

    unique_acc: Optional[str] = None
    unique_seq: Optional[str] = None
    for entity_id in polymer_entity_ids:
        ent = _fetch_polymer_entity(session, pdb_id, str(entity_id))
        if not ent:
            return None
        seq = _standard_protein_sequence(ent)
        if not seq:
            return None
        accs = _extract_uniprot_accessions(ent)
        if len(accs) != 1:
            return None
        acc = next(iter(accs))
        if unique_acc is None:
            unique_acc = acc
        elif unique_acc != acc:
            return None
        if unique_seq is None:
            unique_seq = seq
        elif unique_seq != seq:
            return None
    if unique_acc and unique_seq:
        return unique_acc, unique_seq
    return None


def _single_uniprot_for_entry(
    session: requests.Session, pdb_id: str, entry: Optional[Dict] = None
) -> Optional[Tuple[str, str]]:
    return _entry_single_accession_and_sequence(
        session,
        pdb_id,
        entry=entry,
        require_emdb=True,
        require_single_entity=True,
    )


def _fetch_uniprot_entry(session: requests.Session, accession: str) -> Optional[Dict]:
    url = UNIPROT_ENTRY_URL.format(accession=accession)
    resp = session.get(url, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def _get_uniprot_entry(
    session: requests.Session,
    accession: str,
    cache: Dict[str, Optional[Dict]],
) -> Optional[Dict]:
    if accession not in cache:
        cache[accession] = _fetch_uniprot_entry(session, accession)
    return cache[accession]


def _extract_gene_name(uniprot_data: Dict) -> Optional[str]:
    genes = uniprot_data.get("genes") or []
    for gene in genes:
        gene_name = gene.get("geneName")
        name = gene_name.get("value") if isinstance(gene_name, dict) else gene_name
        if name:
            return str(name)
    return None


def _extract_organism_terms(uniprot_data: Dict) -> List[str]:
    organism = uniprot_data.get("organism") or {}
    terms: List[str] = []
    if isinstance(organism, dict):
        for key in ("scientificName", "commonName"):
            value = organism.get(key)
            if value:
                terms.append(str(value))
        synonyms = organism.get("synonyms") or []
        if isinstance(synonyms, list):
            for syn in synonyms:
                if isinstance(syn, dict):
                    value = syn.get("value")
                else:
                    value = syn
                if value:
                    terms.append(str(value))
        lineage = organism.get("lineage") or []
        if isinstance(lineage, list):
            terms.extend([str(item) for item in lineage if item])
        elif isinstance(lineage, str):
            terms.append(lineage)
    return terms


def _organism_matches_keyword(uniprot_data: Dict, keyword: str) -> bool:
    keyword = str(keyword or "").strip().lower()
    if not keyword:
        return True
    return any(keyword in term.lower() for term in _extract_organism_terms(uniprot_data))


def _sanitize_name(name: str) -> str:
    import re
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return safe or "target"


def _rcsb_search_ids_for_accession(
    session: requests.Session, accession: str, page_size: int = 1000
) -> List[str]:
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                        "operator": "exact_match",
                        "value": accession,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "exptl.method",
                        "operator": "exact_match",
                        "value": "ELECTRON MICROSCOPY",
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {"paginate": {"start": 0, "rows": page_size}},
    }
    return _rcsb_search_ids_from_query(session, query, page_size=page_size)


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


def _entry_emdb_id(entry: Dict) -> Optional[str]:
    identifiers = entry.get("rcsb_entry_container_identifiers", {}) or {}
    emdb_ids = identifiers.get("emdb_ids") or []
    if not emdb_ids:
        return None
    return _normalize_emdb_id(emdb_ids[0])


def _emdb_numeric_id(emdb_id: str) -> Optional[str]:
    emdb_id = _normalize_emdb_id(emdb_id)
    if not emdb_id:
        return None
    if emdb_id.startswith("EMD-"):
        suffix = emdb_id[4:]
        return suffix if suffix.isdigit() else None
    return None


def _fetch_contour_level(session: requests.Session, emdb_id: str) -> Optional[float]:
    emdb_id = _normalize_emdb_id(emdb_id)
    if not emdb_id:
        return None
    url = EMDB_ENTRY_URL.format(emdb_id=emdb_id)
    resp = session.get(url, timeout=30)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    data = resp.json()
    contours = data.get("map", {}).get("contour_list", {}).get("contour", []) or []
    if not isinstance(contours, list):
        return None
    primary = None
    for item in contours:
        if not isinstance(item, dict):
            continue
        if item.get("primary") is True and "level" in item:
            primary = item
            break
    chosen = primary or (contours[0] if contours else None)
    if isinstance(chosen, dict) and "level" in chosen:
        try:
            return float(chosen["level"])
        except Exception:
            return None
    return None


def _download_pdb_model(
    session: requests.Session, pdb_id: str, out_dir: pathlib.Path
) -> Optional[pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdb_id = pdb_id.upper()
    candidates = [
        f"https://files.rcsb.org/download/{pdb_id}.cif",
        f"https://files.rcsb.org/download/{pdb_id}.pdb",
    ]
    for url in candidates:
        resp = session.get(url, timeout=60)
        txt = resp.text.strip() if resp.text else ""
        if resp.ok and resp.content and not (txt.startswith("<!DOCTYPE") or txt.startswith("<?xml")):
            ext = ".cif" if url.endswith(".cif") else ".pdb"
            path = out_dir / f"{pdb_id}{ext}"
            path.write_bytes(resp.content)
            if path.stat().st_size > 0:
                return path
    return None


def _download_emdb_map(
    session: requests.Session, emdb_id: str, out_path: pathlib.Path
) -> Optional[pathlib.Path]:
    num = _emdb_numeric_id(emdb_id)
    if not num:
        return None

    base = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{num}/map"
    candidates = [
        f"{base}/emd_{num}.map.gz",
        f"{base}/emd_{num}.map",
    ]
    for url in candidates:
        resp = session.get(url, timeout=60)
        txt = resp.text.strip() if resp.text else ""
        if not (resp.ok and resp.content) or txt.startswith("<!DOCTYPE"):
            continue
        data = resp.content
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


def _best_em_pdb_for_accession(
    session: requests.Session, accession: str
) -> Optional[Tuple[str, float, str]]:
    pdb_ids = _rcsb_search_ids_for_accession(session, accession)
    if not pdb_ids:
        return None

    best: Optional[Tuple[str, float, str]] = None
    for pdb_id in pdb_ids:
        entry = _fetch_entry(session, pdb_id)
        if not entry:
            continue
        acc_and_seq = _entry_single_accession_and_sequence(
            session,
            pdb_id,
            entry=entry,
            require_emdb=True,
            require_single_entity=True,
        )
        if not acc_and_seq:
            continue
        unique_acc, _ = acc_and_seq
        if unique_acc != accession:
            continue

        res = _entry_resolution(entry)
        if res is None:
            continue
        emdb_id = _entry_emdb_id(entry)
        if not emdb_id:
            continue
        candidate = (pdb_id.upper(), res, emdb_id)
        if best is None or candidate[1] < best[1]:
            best = candidate
    return best


def _write_csv(
    out_path: pathlib.Path,
    rows: Sequence[Sequence[str]],
    header: Optional[Sequence[str]],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(list(header))
        writer.writerows(rows)


def main() -> None:
    if TARGET_PER_BIN <= 0:
        raise SystemExit("TARGET_PER_BIN must be > 0")

    session = _make_session()
    pdb_ids = _rcsb_search_ids(session, None, SPECIES_KEYWORD)
    if not pdb_ids:
        raise SystemExit("No cryo-EM entries found with the requested filters.")

    pdb_ids.sort()
    rng = random.Random(SEED)
    rng.shuffle(pdb_ids)

    selected_bins: Dict[str, List[Dict[str, object]]] = {
        "low": [],
        "mid": [],
        "high": [],
    }
    seen_accessions: Set[str] = set()
    seen_gene_ids: Set[str] = set()
    sequence_by_pdb: Dict[str, str] = {}
    uniprot_cache: Dict[str, Optional[Dict]] = {}

    for pdb_id in pdb_ids:
        if all(len(selected_bins[name]) >= TARGET_PER_BIN for name in BIN_ORDER):
            break

        entry = _fetch_entry(session, pdb_id)
        if not entry:
            continue

        resolution = _entry_resolution(entry)
        if resolution is None:
            continue
        bin_name = _resolution_bin(resolution)
        if not bin_name:
            continue
        if len(selected_bins[bin_name]) >= TARGET_PER_BIN:
            continue

        emdb_id = _entry_emdb_id(entry)
        if not emdb_id:
            continue
        if not _entry_has_only_protein(entry):
            continue

        acc_and_seq = _single_uniprot_for_entry(session, pdb_id, entry=entry)
        if not acc_and_seq:
            continue
        acc, seq = acc_and_seq

        uniprot_entry = _get_uniprot_entry(session, acc, uniprot_cache)
        if SPECIES_KEYWORD and (
            not uniprot_entry or not _organism_matches_keyword(uniprot_entry, SPECIES_KEYWORD)
        ):
            continue

        gene_id = _extract_gene_name(uniprot_entry) if uniprot_entry else None
        if not gene_id:
            continue
        if gene_id in seen_gene_ids:
            continue
        if acc in seen_accessions:
            continue

        pdb_id_upper = pdb_id.upper()
        existing_pdb_seq = sequence_by_pdb.get(pdb_id_upper)
        if existing_pdb_seq is not None and existing_pdb_seq != seq:
            raise SystemExit(
                f"Multiple sequences found for PDB {pdb_id_upper}: {existing_pdb_seq} vs {seq}"
            )
        sequence_by_pdb[pdb_id_upper] = seq

        selected_bins[bin_name].append(
            {
                "resolution": resolution,
                "gene_id": gene_id,
                "accession": acc,
                "sequence": seq,
                "pdb_id": pdb_id_upper,
                "emdb_id": emdb_id,
            }
        )
        seen_gene_ids.add(gene_id)
        seen_accessions.add(acc)

    print(f"Fetched {len(selected_bins['low'])} entries with resolution < {LOW_RESOLUTION_MAX_A} A")
    print(
        "Fetched {} entries with resolution {}-{} A".format(
            len(selected_bins["mid"]), MID_RESOLUTION_MIN_A, MID_RESOLUTION_MAX_A
        )
    )
    print(f"Fetched {len(selected_bins['high'])} entries with resolution > {HIGH_RESOLUTION_MIN_A} A")

    selected_meta: List[Dict[str, object]] = []
    for name in BIN_ORDER:
        selected_meta.extend(selected_bins[name])

    contour_cache: Dict[str, Optional[float]] = {}
    download_rows: List[Dict[str, str]] = []
    for item in selected_meta:
        emdb_id = str(item["emdb_id"])
        if emdb_id not in contour_cache:
            contour_cache[emdb_id] = _fetch_contour_level(session, emdb_id)
        contour = contour_cache[emdb_id]
        contour_str = f"{contour:.3f}" if contour is not None else "nan"
        resolution = float(item["resolution"])
        download_rows.append(
            {
                "resolution": f"{resolution:.2f}",
                "gene_id": str(item["gene_id"]),
                "accession": str(item["accession"]),
                "sequence": str(item["sequence"]),
                "pdb_id": str(item["pdb_id"]),
                "emdb_id": emdb_id,
                "contour_level": contour_str,
            }
        )

    if not DOWNLOAD_PDBS_AND_MAPS:
        header = CSV_HEADER if INCLUDE_HEADER else None
        rows = [
            [
                row["resolution"],
                row["gene_id"],
                row["accession"],
                row["sequence"],
                row["pdb_id"],
                row["emdb_id"],
                row["contour_level"],
            ]
            for row in download_rows
        ]
        _write_csv(OUT_PATH, rows, header=header)
        return

    out_dir = DOWNLOAD_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / DOWNLOAD_CSV_NAME
    with manifest_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        for row in download_rows:
            gene_id = row["gene_id"]
            accession = row["accession"]
            sequence = row["sequence"]
            pdb_id = row["pdb_id"]
            resolution = row["resolution"]
            emdb_id = row["emdb_id"]
            contour_level = row["contour_level"]

            label = gene_id or accession
            target_dir = out_dir / _sanitize_name(label)
            model_path = _download_pdb_model(session, pdb_id, target_dir)
            if not model_path:
                raise SystemExit(f"Failed to download PDB for {pdb_id}.")
            map_path = _download_emdb_map(session, emdb_id, target_dir / f"{pdb_id}.map")
            if not map_path:
                raise SystemExit(f"Failed to download EMDB map for {emdb_id}.")

            writer.writerow(
                [resolution, gene_id, accession, sequence, pdb_id, emdb_id, contour_level]
            )


if __name__ == "__main__":
    main()
