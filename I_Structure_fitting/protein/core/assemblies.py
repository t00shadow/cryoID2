"""Fixed per-chain SW and biological-assembly compatibility for protein groups."""
from __future__ import annotations

import itertools
import json
import math
import re
from pathlib import Path

import parasail

from . import fetch
from .io_utils import write_json


def validate_chains(value):
    chains = json.loads(value) if isinstance(value, str) else value
    if not isinstance(chains, list) or not chains:
        raise ValueError("Candidate chain data must be a nonempty list.")
    result, records, identifiers = [], set(), set()
    for item in chains:
        record = str(item["record_id"])
        chain = str(item["chain_id"])
        sequence = str(item["sequence"]).strip()
        copies = int(item["copies"])
        support = float(item.get("trace_support", 0.))
        if (not sequence or not record or not chain or copies < 1
                or record in records or chain in identifiers
                or not math.isfinite(support) or support < 0):
            raise ValueError("Invalid or duplicate candidate chain requirements.")
        records.add(record); identifiers.add(chain)
        result.append({"record_id": record, "chain_id": chain, "sequence": sequence,
                       "copies": copies, "trace_support": support})
    return result


def sw_details(query, target):
    query = fetch._sanitize_for_parasail(query)
    target = fetch._sanitize_for_parasail(target)
    alignment = parasail.sw_trace_striped_32(query, target, fetch._GAP_OPEN, fetch._GAP_EXTEND, fetch._BLOSUM62)
    if alignment.score <= 0:
        return {"score": 0, "identity": 0., "query_coverage": 0.}
    trace = alignment.traceback
    pairs = [(a,b) for a,b in zip(trace.query, trace.ref) if a != "-" and b != "-"]
    return {"score": int(alignment.score),
            "identity": sum(a == b for a,b in pairs) / max(1, len(pairs)),
            "query_coverage": sum(a != "-" for a in trace.query) / max(1, len(query))}


def chain_hits(chains, check_cancel):
    sequences, refs = fetch._ensure_pool_loaded()
    cache, results = {}, {}
    for chain in chains:
        check_cancel()
        query = fetch._sanitize_for_parasail(chain["sequence"])
        if query not in cache:
            scores = []
            for i, sequence in enumerate(sequences):
                if i % 512 == 0: check_cancel()
                scores.append(fetch._sw_score(query, sequence))
            best = max(scores, default=0)
            matches = []
            for i, score in enumerate(scores):
                if best > 0 and score >= .9 * best:
                    check_cancel()
                    details = sw_details(query, sequences[i])
                    if details["identity"] >= .8 and details["query_coverage"] >= .8:
                        matches.append({**details, "fraction": score / best, "chains": refs[i]})
            cache[query] = {"best_score": best, "matches": matches}
        results[chain["record_id"]] = cache[query]
    return results


def load_entries(identifiers, folder, check_cancel):
    folder.mkdir(parents=True, exist_ok=True)
    result, missing = {}, []
    for identifier in sorted(identifiers):
        path = folder / (fetch.sanitize_name(identifier) + ".json")
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            if entry["rcsb_id"] != identifier: raise ValueError("Wrong cached entry")
            result[identifier] = entry
        except (OSError, ValueError, KeyError, TypeError):
            missing.append(identifier)
    for start in range(0, len(missing), 30):
        check_cancel()
        ids = missing[start:start+30]
        query = """{entries(entry_ids:IDS){rcsb_id assemblies{rcsb_id pdbx_struct_assembly_gen{asym_id_list oper_expression}} polymer_entities{entity_poly{pdbx_seq_one_letter_code_can rcsb_entity_polymer_type} rcsb_polymer_entity_container_identifiers{asym_ids uniprot_ids}}}}""".replace("IDS", json.dumps(ids))
        response = fetch.SESSION.post("https://data.rcsb.org/graphql", json={"query":query}, timeout=(10,60))
        response.raise_for_status()
        data = response.json()
        if data.get("errors"): raise RuntimeError(f"Assembly metadata lookup failed: {data['errors']}")
        for entry in data["data"]["entries"]:
            if entry is not None:
                result[entry["rcsb_id"]] = entry
                write_json(folder / (fetch.sanitize_name(entry["rcsb_id"]) + ".json"), entry)
    return result


def expand_operations(expression):
    factors = re.findall(r"\(([^)]+)\)", expression) or [expression]
    result = []
    for factor in factors:
        values = []
        for value in factor.split(","):
            value = value.strip()
            if "-" in value:
                left, right = value.split("-")
                values.extend(str(i) for i in range(int(left), int(right)+1))
            elif value: values.append(value)
        if not values: raise ValueError(f"Invalid assembly operation expression: {expression!r}")
        result.append(values)
    return list(itertools.product(*result))


def match_assembly(chains, entry, assembly, identity):
    # RNA/ligand entities cannot satisfy a supplied protein-chain requirement.
    entities = {asym: entity for entity in entry.get("polymer_entities") or []
                if entity["entity_poly"]["rcsb_entity_polymer_type"] == "Protein"
                for asym in entity["rcsb_polymer_entity_container_identifiers"]["asym_ids"]}
    slots = {}
    for generator in assembly.get("pdbx_struct_assembly_gen") or []:
        asym_ids = generator["asym_id_list"]
        if isinstance(asym_ids, str): asym_ids = asym_ids.split(",")
        for asym in asym_ids:
            if asym in entities:
                for operation in expand_operations(generator["oper_expression"]):
                    slots[(asym,operation)] = entities[asym]
    choices, requirements, detail_cache = {}, {}, {}
    for chain in chains:
        for copy in range(chain["copies"]):
            key = (chain["record_id"],copy)
            requirements[key] = chain
            choices[key] = []
            for slot, entity in slots.items():
                target = "".join(entity["entity_poly"]["pdbx_seq_one_letter_code_can"].split())
                pair = (chain["sequence"],target)
                if pair not in detail_cache: detail_cache[pair] = sw_details(*pair)
                details = detail_cache[pair]
                if details["identity"] >= identity and details["query_coverage"] >= .8:
                    choices[key].append(slot)
    # Augmenting-path matching enforces distinct instances, including copies.
    occupied = {}
    def assign(key, visited):
        for slot in choices[key]:
            if slot in visited: continue
            visited.add(slot)
            if slot not in occupied or assign(occupied[slot], visited):
                occupied[slot] = key
                return True
        return False
    if not all(assign(key,set()) for key in sorted(choices,key=lambda key:(len(choices[key]),key))):
        return None
    return [{"record_id":key[0], "chain_id":requirements[key]["chain_id"], "copy":key[1]+1,
             "asym_id":slot[0], "operation_ids":list(slot[1]),
             "uniprot_accessions":slots[slot]["rcsb_polymer_entity_container_identifiers"].get("uniprot_ids") or []}
            for slot,key in sorted(occupied.items())]


def retrieve(chains, hits, folder, check_cancel):
    attempts = []
    def search(required, fraction, identity):
        sets = [{p for h in hits[c["record_id"]]["matches"]
                 if h["fraction"] >= fraction and h["identity"] >= identity
                 for p,_ in h["chains"]} for c in required]
        pdbs = set.intersection(*sets) if sets else set()
        entries = load_entries(pdbs, folder / "assembly_metadata", check_cancel)
        candidates = {}
        for entry in entries.values():
            check_cancel()
            for assembly in entry.get("assemblies") or []:
                assignment = match_assembly(required,entry,assembly,identity)
                if assignment is not None: candidates[assembly["rcsb_id"]] = assignment
        return candidates, len(pdbs)
    for fraction,identity in ((.98,.9),(.95,.9),(.9,.8)):
        candidates, count = search(chains,fraction,identity)
        attempts.append({"score_fraction":fraction,"identity":identity,"entries":count,"assemblies":len(candidates)})
        if candidates:
            return {"match_scope":"all_supplied_chains_compatible","candidates":candidates,
                    "attempts":attempts,"anchors":[],"missing_chains":[]}
    best = max(c["trace_support"] for c in chains)
    anchors = [c for c in chains if best > 0 and c["trace_support"] == best]
    candidates, missing = {}, {}
    for anchor in anchors:
        found,_ = search([{**anchor,"copies":1}],.98,.9)
        candidates.update(found)
        missing[anchor["chain_id"]] = {c["chain_id"]: c["copies"]-int(c is anchor)
                                       for c in chains if c["copies"]-int(c is anchor)>0}
    return {"match_scope":"partial_sequence_match" if anchors else "insufficient_trace_support",
            "candidates":candidates,"attempts":attempts,"anchors":[c["chain_id"] for c in anchors],
            "anchor_ambiguity":len(anchors)>1,"missing_chains":missing}


def download_assembly(identifier, folder, check_cancel):
    """Only the validated exact biological assembly; never fall back to an ASU."""
    pdb, assembly = identifier.rsplit("-",1)
    path = folder / (fetch.sanitize_name(identifier) + ".cif")
    folder.mkdir(parents=True,exist_ok=True)
    if path.is_file() and path.stat().st_size: return path
    check_cancel()
    response = fetch.SESSION.get(f"https://files.rcsb.org/download/{pdb}-assembly{assembly}.cif", timeout=(10,60))
    response.raise_for_status()
    if not response.content or not response.text.lstrip().startswith("data_"):
        raise RuntimeError(f"Invalid mmCIF biological assembly download: {identifier}")
    path.write_bytes(response.content)
    return path


def process_group(gene_id, chains, map_path, input_meta, run_config, align_config, entry_started):
    from . import pipeline as pipe
    tag, safe = gene_id, fetch.sanitize_name(gene_id)
    hits = chain_hits(chains,run_config.check_cancel)
    selection = retrieve(chains,hits,run_config.logs_dir/safe,run_config.check_cancel)
    scope = selection["match_scope"]
    run_config.progress(f"Candidate {tag}: {scope}; {len(selection['candidates'])} complete assembly candidate(s).")
    rows, best = [], None
    removed = {"source":[],"fitted":[],"alignment_input":[],"scratch":[]}
    candidates = sorted(selection["candidates"])
    if run_config.max_experimental_tries is not None: candidates = candidates[:run_config.max_experimental_tries]
    for identifier in candidates:
        run_config.check_cancel()
        label = f"{safe}__PDB_{fetch.sanitize_name(identifier)}"
        extra = {"sequence_id":tag,"sequence_rank":input_meta.get("sequence_rank",""),
                 "case_key":pipe._candidate_case_key(tag,identifier),"match_scope":scope,
                 "candidate_generation_method":"independent_chain_sw_assembly_matching",
                 "missing_chains":json.dumps(selection["missing_chains"],sort_keys=True)}
        try:
            model = download_assembly(identifier,run_config.source_dir/safe/"pdb",run_config.check_cancel)
        except Exception as exc:
            run_config.check_cancel()
            row = {"label":label,"type":"PDB","id":identifier,"status":"download_error",
                   "error":repr(exc),"score":float("nan"),"source_model":"","fitted_path":"",**extra}
        else:
            row = pipe._fit_candidate(label=label,kind="PDB",candidate_id=identifier,map_path=map_path,
                model_path=model,run_config=run_config,align_config=align_config,
                input_meta={**input_meta,"case_key":extra["case_key"]},extra=extra)
            pipe._cleanup_candidate_scratch(row,run_config=run_config,removed_by_kind=removed)
        rows.append(row)
        best = pipe._update_best_and_cleanup(row=row,incumbent=best,run_config=run_config,removed_by_kind=removed)
        pipe._print_candidate_result(tag,"PDB",identifier,row)
    for row in rows: row["kept"] = row is best
    generation = {**selection,"per_chain_search":hits,"candidate_generation_method":"independent_chain_sw_assembly_matching"}
    log = pipe._write_target_log(run_config.logs_dir,safe,rows)
    result = {"target":tag,"safe_name":safe,"sequence_rank":input_meta.get("sequence_rank",""),
              "status":"ok" if best else "error","error":"" if best else "no_compatible_successful_assemblies",
              "match_scope":scope,"map_path":map_path,"candidate_count":len(rows),
              "ok_candidate_count":sum(r["status"]=="ok" for r in rows),"rows":rows,
              "selected_pdb_ids":candidates,"uniprot_accessions":[],"best_af":None,"best_pdb":best,"best_overall":best,
              "candidate_generation":generation,"candidate_log":log,
              "score_metric":align_config.score_metric,"selected_method_id":align_config.method_id,
              "alignment_runner":"local_rawfield.run_raw_field","entry_total_seconds":pipe._elapsed_seconds(entry_started)}
    write_json(run_config.logs_dir/(safe+"__assembly_selection.json"),generation)
    return result
