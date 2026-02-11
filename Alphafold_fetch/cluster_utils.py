# =============================
# File: cluster_utils.py
# =============================
#!/usr/bin/env python3
from __future__ import annotations

import re
from typing import Optional, List, Dict, Any, Tuple, Callable

import requests
from requests.adapters import HTTPAdapter, Retry

# ------------------- Session + caches -------------------
_SESSION: Optional[requests.Session] = None

ENTRY_CACHE: Dict[str, Dict[str, Any]] = {}
ENTITY_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
ASSEMBLY_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}


def set_session(session: requests.Session) -> None:
    global _SESSION
    _SESSION = session


def _get_session() -> requests.Session:
    global _SESSION
    if _SESSION is None:
        retries = Retry(total=4, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
        sess = requests.Session()
        sess.mount("https://", HTTPAdapter(max_retries=retries))
        sess.headers.update({"User-Agent": "cluster-utils/1.0"})
        _SESSION = sess
    return _SESSION


def _fetch_json(url: str, timeout: int = 15) -> Optional[Dict[str, Any]]:
    try:
        r = _get_session().get(url, timeout=timeout)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, dict) else None
    except requests.RequestException:
        return None


def _normalize_sequence(seq: Optional[str]) -> str:
    if not seq:
        return ""
    return re.sub(r"\s+", "", str(seq).strip().upper())


def _local_identity(seq_a: str, seq_b: str) -> float:
    seq_a = str(seq_a or "").strip().upper()
    seq_b = str(seq_b or "").strip().upper()
    if not seq_a or not seq_b:
        return 0.0
    if seq_a == seq_b:
        return 1.0

    n = len(seq_a)
    m = len(seq_b)

    score = [[0] * (m + 1) for _ in range(n + 1)]
    pointer = [[3] * (m + 1) for _ in range(n + 1)]  # 0 diag, 1 up, 2 left, 3 stop

    max_i = 0
    max_j = 0
    max_score = 0
    for i in range(1, n + 1):
        a = seq_a[i - 1]
        row = score[i]
        row_prev = score[i - 1]
        ptr_row = pointer[i]
        for j in range(1, m + 1):
            b = seq_b[j - 1]
            diag = row_prev[j - 1] + (1 if a == b else 0)
            up = row_prev[j] - 1
            left = row[j - 1] - 1
            best = 0
            ptr = 3
            if diag > best:
                best = diag
                ptr = 0
            if up > best:
                best = up
                ptr = 1
            if left > best:
                best = left
                ptr = 2
            row[j] = best
            ptr_row[j] = ptr
            if best > max_score:
                max_score = best
                max_i, max_j = i, j

    if max_score <= 0:
        return 0.0

    i, j = max_i, max_j
    matches = 0
    length = 0
    while i > 0 and j > 0 and score[i][j] > 0:
        ptr = pointer[i][j]
        if ptr == 0:
            length += 1
            if seq_a[i - 1] == seq_b[j - 1]:
                matches += 1
            i -= 1
            j -= 1
        elif ptr == 1:
            length += 1
            i -= 1
        elif ptr == 2:
            length += 1
            j -= 1
        else:
            break
    return matches / length if length else 0.0


def _best_resolution_value(res_field: Any) -> Optional[float]:
    if res_field is None:
        return None
    if isinstance(res_field, (list, tuple)):
        values = []
        for v in res_field:
            try:
                fv = float(v)
            except Exception:
                continue
            if fv > 0:
                values.append(fv)
        return min(values) if values else None
    try:
        fv = float(res_field)
        return fv if fv > 0 else None
    except Exception:
        return None


def _resolution_rank(res: Optional[float]) -> float:
    try:
        return float(res)
    except Exception:
        return float("inf")


def _norm_key_part(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return re.sub(r"\s+", "_", text)


def _pick_assembly_for_clustering(
    entry_json: Dict[str, Any],
    preferred_assembly_id: str = "1",
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    assemblies = entry_json.get("assemblies") or []
    chosen = None
    prefer = str(preferred_assembly_id)
    for assembly in assemblies:
        if str(assembly.get("assembly_id")) == prefer:
            chosen = assembly
            break
    if not chosen and assemblies:
        chosen = assemblies[0]
    assembly_id = str(chosen.get("assembly_id")) if chosen and chosen.get("assembly_id") is not None else None
    return chosen, assembly_id


def _assembly_key(
    assembly: Optional[Dict[str, Any]],
    entry_info: Dict[str, Any],
    assembly_info: Optional[Dict[str, Any]] = None,
) -> str:
    rcsb_assembly = (assembly or {}).get("rcsb_struct_assembly") or {}
    rcsb_assembly_info = (assembly_info or {}).get("rcsb_assembly_info") or {}
    oligomeric_count = (assembly or {}).get("oligomeric_count") or rcsb_assembly.get("oligomeric_count")
    oligomeric_details = (assembly or {}).get("oligomeric_details") or rcsb_assembly.get("oligomeric_details")
    stoichiometry = (assembly or {}).get("stoichiometry") or rcsb_assembly.get("stoichiometry")
    details = (assembly or {}).get("details") or rcsb_assembly.get("details")
    symmetry = rcsb_assembly.get("symmetry")
    chain_count = None
    for source in (rcsb_assembly_info, entry_info):
        if not source:
            continue
        chain_count = (
            source.get("polymer_entity_instance_count")
            or source.get("polymer_entity_instance_count_asymmetric_unit")
            or source.get("polymer_entity_count")
        )
        if chain_count is not None:
            break
    parts: List[str] = []
    stoich = _norm_key_part(stoichiometry)
    if stoich:
        parts.append(f"stoich={stoich}")
    olig_details = _norm_key_part(oligomeric_details)
    if olig_details:
        parts.append(f"olig={olig_details}")
    elif oligomeric_count is not None:
        parts.append(f"olig_count={oligomeric_count}")
    sym_txt = _norm_key_part(symmetry)
    if sym_txt:
        parts.append(f"sym={sym_txt}")
    det = _norm_key_part(details)
    if det:
        parts.append(f"details={det}")
    if chain_count is not None:
        parts.append(f"chains={chain_count}")
    if rcsb_assembly_info:
        for label, key in (
            ("interfaces", "num_interfaces"),
            ("iface_entities", "num_interface_entities"),
            ("hom_iface_entities", "num_homomeric_interface_entities"),
            ("het_iface_entities", "num_heteromeric_interface_entities"),
            ("iso_iface_entities", "num_isologous_interface_entities"),
            ("prot_iface_entities", "num_protein_interface_entities"),
            ("heterolog_iface_entities", "num_heterologous_interface_entities"),
            ("poly_inst", "polymer_entity_instance_count"),
            ("poly_entities", "polymer_entity_count"),
            ("modeled_monomers", "modeled_polymer_monomer_count"),
        ):
            val = rcsb_assembly_info.get(key)
            if val is not None:
                parts.append(f"{label}={val}")
    if not parts:
        parts.append("unknown")
    assembly_id = str((assembly or {}).get("assembly_id")) if assembly else None
    key = ";".join(parts)
    return f"assembly{assembly_id}:{key}" if assembly_id else key


def _conformation_key(entry_info: Dict[str, Any]) -> str:
    poly_comp = _norm_key_part(entry_info.get("polymer_composition")) or "unknown"
    ligand_count = entry_info.get("nonpolymer_entity_count")
    lig_txt = str(ligand_count) if ligand_count is not None else "unknown"
    parts = [
        f"polycomp={poly_comp}",
        f"ligands={lig_txt}",
    ]
    for label, key in (
        ("poly_entities", "polymer_entity_count"),
        ("poly_entities_prot", "polymer_entity_count_protein"),
        ("poly_entities_rna", "polymer_entity_count_rna"),
        ("poly_entities_dna", "polymer_entity_count_dna"),
        ("poly_entities_na", "polymer_entity_count_nucleic_acid"),
        ("poly_entities_na_hybrid", "polymer_entity_count_nucleic_acid_hybrid"),
    ):
        val = entry_info.get(key)
        if val is not None:
            parts.append(f"{label}={val}")
    for label, key in (
        ("poly_len_min", "polymer_monomer_count_minimum"),
        ("poly_len_max", "polymer_monomer_count_maximum"),
        ("poly_mw_min", "polymer_molecular_weight_minimum"),
        ("poly_mw_max", "polymer_molecular_weight_maximum"),
    ):
        val = entry_info.get(key)
        if val is not None:
            parts.append(f"{label}={val}")
    sel_types = _norm_key_part(entry_info.get("selected_polymer_entity_types"))
    if sel_types:
        parts.append(f"sel_types={sel_types}")
    na_types = _norm_key_part(entry_info.get("na_polymer_entity_types"))
    if na_types:
        parts.append(f"na_types={na_types}")
    return "|".join(parts)


def _max_seq_identity(seqs_a: List[str], seqs_b: List[str]) -> float:
    if not seqs_a or not seqs_b:
        return 0.0
    set_b = set(seqs_b)
    for a in seqs_a:
        if a in set_b:
            return 1.0
    best = 0.0
    for a in seqs_a:
        for b in seqs_b:
            if not a or not b:
                continue
            min_len = min(len(a), len(b))
            max_len = max(len(a), len(b))
            if min_len == 0:
                continue
            if max_len / min_len > 1.25:
                continue
            score = _local_identity(a, b)
            if score > best:
                best = score
            if best >= 1.0:
                return 1.0
    return best


def _assign_sequence_cluster(seqs: List[str], rep_seqs: List[List[str]], threshold: float) -> int:
    if not seqs:
        rep_seqs.append([f"unknown_{len(rep_seqs)}"])
        return len(rep_seqs) - 1
    for idx, rep in enumerate(rep_seqs):
        if _max_seq_identity(seqs, rep) >= threshold:
            return idx
    rep_seqs.append(seqs)
    return len(rep_seqs) - 1


def _method_rank(methods: List[str]) -> int:
    # Method is intentionally ignored for clustering/selection.
    return 0


def _is_better_representative(candidate: Dict[str, Any], incumbent: Dict[str, Any]) -> bool:
    res_c = _resolution_rank(candidate.get("resolution"))
    res_i = _resolution_rank(incumbent.get("resolution"))
    if res_c < res_i - 1e-9:
        return True
    if abs(res_c - res_i) <= 1e-9:
        if str(candidate.get("id", "")) < str(incumbent.get("id", "")):
            return True
    return False


def _entry_json(pdb_id: str, timeout: int = 15) -> Optional[Dict[str, Any]]:
    pid = str(pdb_id or "").upper()
    if not pid:
        return None
    if pid in ENTRY_CACHE:
        return ENTRY_CACHE[pid]
    url = f"https://data.rcsb.org/rest/v1/core/entry/{pid}"
    data = _fetch_json(url, timeout=timeout)
    if data is not None:
        ENTRY_CACHE[pid] = data
    return data


def _polymer_entity_json(pdb_id: str, entity_id: str, timeout: int = 15) -> Optional[Dict[str, Any]]:
    pid = str(pdb_id or "").upper()
    eid = str(entity_id or "").strip()
    if not pid or not eid:
        return None
    key = (pid, eid)
    if key in ENTITY_CACHE:
        return ENTITY_CACHE[key]
    url = f"https://data.rcsb.org/rest/v1/core/polymer_entity/{pid}/{eid}"
    data = _fetch_json(url, timeout=timeout)
    if data is not None:
        ENTITY_CACHE[key] = data
    return data


def _assembly_json(pdb_id: str, assembly_id: str, timeout: int = 15) -> Optional[Dict[str, Any]]:
    pid = str(pdb_id or "").upper()
    aid = str(assembly_id or "").strip()
    if not pid or not aid:
        return None
    key = (pid, aid)
    if key in ASSEMBLY_CACHE:
        return ASSEMBLY_CACHE[key]
    url = f"https://data.rcsb.org/rest/v1/core/assembly/{pid}/{aid}"
    data = _fetch_json(url, timeout=timeout)
    if data is not None:
        ASSEMBLY_CACHE[key] = data
    return data


def _entity_sequences_for_uniprots(pdb_id: str, uniprot_set: set) -> List[str]:
    entry = _entry_json(pdb_id)
    if not entry:
        return []
    uniprot_set = {str(u).upper() for u in (uniprot_set or set()) if u}
    ids = (entry.get("rcsb_entry_container_identifiers") or {}).get("polymer_entity_ids") or []
    sequences: List[str] = []
    for ent_id in ids:
        ent = _polymer_entity_json(pdb_id, ent_id)
        if not ent:
            continue
        ref_ids = (
            (ent.get("rcsb_polymer_entity_container_identifiers") or {})
            .get("reference_sequence_identifiers") or []
        )
        matched = False
        for ref in ref_ids:
            if ref.get("database_name") == "UniProt":
                acc = ref.get("database_accession")
                if acc and acc in uniprot_set:
                    matched = True
                    break
        if not matched:
            continue
        seq = (ent.get("entity_poly") or {}).get("pdbx_seq_one_letter_code_can") or \
              (ent.get("entity_poly") or {}).get("pdbx_seq_one_letter_code")
        seq = _normalize_sequence(seq)
        if seq:
            sequences.append(seq)
    seen = set()
    uniq = []
    for s in sequences:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def get_entry_cluster_data(
    pdb_id: str,
    uniprot_set: set,
    assembly_id_for_clustering: str = "1",
    timeout: int = 15,
) -> Optional[Dict[str, Any]]:
    entry = _entry_json(pdb_id, timeout=timeout)
    if not entry:
        return None
    info = entry.get("rcsb_entry_info", {})
    methods = info.get("experimental_method", []) or info.get("experimental_method_class", [])
    resolution = _best_resolution_value(info.get("resolution_combined") or info.get("resolution"))
    sequences = _entity_sequences_for_uniprots(pdb_id, uniprot_set)
    assembly, assembly_id = _pick_assembly_for_clustering(entry, preferred_assembly_id=assembly_id_for_clustering)
    if not assembly_id:
        try:
            if (info.get("assembly_count") or 0) > 0:
                assembly_id = str(assembly_id_for_clustering)
        except Exception:
            pass
    assembly_info = _assembly_json(pdb_id, assembly_id, timeout=timeout) if assembly_id else None
    assembly_key = _assembly_key(assembly, info, assembly_info)
    conformation_key = _conformation_key(info)
    return {
        "id": str(pdb_id).upper(),
        "methods": [m.upper() for m in (methods or [])],
        "resolution": resolution,
        "assembly_id": assembly_id,
        "assembly_key": assembly_key,
        "conformation_key": conformation_key,
        "sequences": sequences,
    }


def build_pdb_clusters(
    pdb_ids: List[str],
    uniprot_set: set,
    seq_cluster_identity: float = 0.98,
    assembly_id_for_clustering: str = "1",
) -> List[Dict[str, Any]]:
    seq_reps: List[List[str]] = []
    clusters_by_key: Dict[str, Dict[str, Any]] = {}
    clusters: List[Dict[str, Any]] = []

    for pid in sorted({str(p).upper() for p in pdb_ids or []}):
        meta = get_entry_cluster_data(
            pid,
            uniprot_set,
            assembly_id_for_clustering=assembly_id_for_clustering,
        )
        if not meta:
            continue
        seq_cluster_id = _assign_sequence_cluster(meta["sequences"], seq_reps, seq_cluster_identity)
        cluster_key = f"seq{seq_cluster_id}|conf={meta['conformation_key']}|asm={meta['assembly_key']}"
        cluster = clusters_by_key.get(cluster_key)
        if not cluster:
            cluster = {
                "id": len(clusters),
                "key": cluster_key,
                "seq_cluster_id": seq_cluster_id,
                "conformation_key": meta["conformation_key"],
                "assembly_key": meta["assembly_key"],
                "members": [],
                "rep": None,
            }
            clusters_by_key[cluster_key] = cluster
            clusters.append(cluster)
        member = {
            "id": meta["id"],
            "resolution": meta["resolution"],
            "methods": meta["methods"],
            "assembly_id": meta["assembly_id"],
        }
        cluster["members"].append(member)
        if cluster["rep"] is None or _is_better_representative(member, cluster["rep"]):
            cluster["rep"] = member

    for cluster in clusters:
        cluster["members"].sort(
            key=lambda m: (_resolution_rank(m.get("resolution")), _method_rank(m.get("methods", [])), m.get("id", "")),
        )

    clusters.sort(
        key=lambda c: (
            _resolution_rank(c["rep"].get("resolution")) if c.get("rep") else float("inf"),
            _method_rank((c.get("rep") or {}).get("methods", [])),
            (c.get("rep") or {}).get("id", ""),
        )
    )
    return clusters


def _af_quality_score(member: Dict[str, Any]) -> float:
    conf = member.get("confidence")
    if conf is not None:
        try:
            return float(conf)
        except Exception:
            pass
    mv = member.get("modelVersion")
    try:
        return float(mv)
    except Exception:
        return float("-inf")


def build_af_clusters(
    acc_candidates: List[str],
    get_uniprot_sequence: Callable[[str], Optional[str]],
    alphafold_best_prediction_meta: Callable[[str], Optional[Dict[str, Any]]],
    seq_cluster_identity: float = 0.98,
) -> List[Dict[str, Any]]:
    seq_reps: List[List[str]] = []
    clusters_by_key: Dict[str, Dict[str, Any]] = {}
    clusters: List[Dict[str, Any]] = []

    for acc in acc_candidates:
        seq = get_uniprot_sequence(acc)
        seqs = [seq] if seq else []
        seq_cluster_id = _assign_sequence_cluster(seqs, seq_reps, seq_cluster_identity)
        meta = alphafold_best_prediction_meta(acc) or {}
        member = {
            "id": acc,
            "sequence": seq,
            "confidence": meta.get("confidence"),
            "modelVersion": meta.get("modelVersion"),
        }
        key = f"seq{seq_cluster_id}"
        cluster = clusters_by_key.get(key)
        if not cluster:
            cluster = {
                "id": len(clusters),
                "key": key,
                "seq_cluster_id": seq_cluster_id,
                "conformation_key": "alphafold",
                "assembly_key": "alphafold",
                "members": [],
                "rep": None,
            }
            clusters_by_key[key] = cluster
            clusters.append(cluster)
        cluster["members"].append(member)
        if cluster["rep"] is None or _af_quality_score(member) > _af_quality_score(cluster["rep"]):
            cluster["rep"] = member

    for cluster in clusters:
        cluster["members"].sort(key=lambda m: _af_quality_score(m), reverse=True)

    clusters.sort(key=lambda c: _af_quality_score(c.get("rep") or {}), reverse=True)
    return clusters
