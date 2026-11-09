"""Shared protein candidate ranking. Grouping is inferred from FASTA data."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
import json
import math
import re
from .candidate_groups import read_groups, group_score, chain_payload, is_assembly_group
from .sequence_search import read_fasta, read_sequence_pool
from .sequence_ranking import protein_ranking_rows
from .reduced_alphabet.evalues import negative_log10_composite
from .protein_search import run_protein_blast_search, BlastSearchResult
from cryoid_pipeline.models import SearchHit


def run_protein_candidate_search(*, query_fasta, candidate_pool, run_dir, progress,
                                 gap_open=15, gap_extend=3, blast_bin_dir=None, cpu_threads=None):
    groups = read_groups(candidate_pool)
    by_id = {c.record_id: c for cs in groups.values() for c in cs}
    chains = [by_id[r.record_id] for r in read_fasta(Path(candidate_pool))]
    # Plain records use the exact original file and numerical search path.
    # All candidates still execute the common aggregation below.
    canonical = Path(candidate_pool)
    if any(r.description != by_id[r.record_id].description for r in read_fasta(canonical)):
        from Bio import SeqIO
        canonical = Path(run_dir) / "candidate_chains.fasta"
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_text("".join(f">{by_id[r.id].description if r.id in by_id else r.description}\n{r.seq}\n"
                                     for r in SeqIO.parse(str(candidate_pool), "fasta")), encoding="utf-8")
    result = run_protein_blast_search(query_fasta=query_fasta, candidate_pool=canonical,
        run_dir=run_dir, progress=progress, gap_open=gap_open, gap_extend=gap_extend,
        blast_bin_dir=blast_bin_dir, cpu_threads=cpu_threads)
    input_order = bool(result.ranking_strategy and result.ranking_strategy.startswith("input_order_"))
    support = {} if input_order else alignment_support(Path(run_dir), chains)
    qids = [f"fragment_{i:08d}" for i,_ in enumerate(read_fasta(Path(query_fasta)), 1)]
    legacy = {r["candidate"]: negative_log10_composite(r) for r in result.ranking}
    legacy = {c: value if isinstance(value, float) else -math.inf for c,value in legacy.items()}
    order = {r["candidate"]: i for i,r in enumerate(result.ranking)}
    evaluated = [(name, cs, group_score(qids, [c.record_id for c in cs], support, legacy)) for name,cs in groups.items()]
    evaluated.sort(key=lambda item: (-item[2]["score"], min(order[c.record_id] for c in item[1]), item[0]))
    if not any(is_assembly_group(cs) for cs in groups.values()):
        assert [cs[0].record_id for _,cs,_ in evaluated] == [r["candidate"] for r in result.ranking]
        return replace(result, ranking_rows=result.ranking_rows or protein_ranking_rows(result.ranking, read_sequence_pool(canonical)))
    originals = {r["candidate"]: r for r in result.ranking}
    rows, ranking = [], []
    for rank,(name,cs,evidence) in enumerate(evaluated,1):
        complex_group = is_assembly_group(cs)
        anchor = cs[0] if input_order else min(cs,key=lambda c:(-legacy[c.record_id],c.record_id))
        label = name if complex_group else anchor.record_id
        if not complex_group:
            source = dict(originals[anchor.record_id]);source["rank"] = str(rank)
            row = protein_ranking_rows([source], read_sequence_pool(canonical))[0]
        else:
            payload = chain_payload(cs,evidence["chain_support"])
            score = evidence["score"]
            row = {"rank":str(rank),"candidate_id":label,"members":json.dumps([c.record_id for c in cs]),
                "sequence":anchor.sequence,"length":sum(len(c.sequence)*c.copies for c in cs),
                "primary_score_name":"input_order" if input_order else "best_chain_plus_complementary_support",
                "primary_score":score if math.isfinite(score) else "", "score_direction":"higher_is_better",
                "description":f"{name}: {len(cs)} chain sequence(s), {sum(c.copies for c in cs)} copies; trace support " +
                              ", ".join(f"{c.chain_id}={evidence['chain_support'].get(c.record_id,0.):.3g}" for c in cs),
                "candidate_chains":json.dumps(payload,separators=(",",":"))}
        if input_order:
            row.update(primary_score_name="input_order", primary_score="")
        rows.append(row)
        ranking.append({"rank":str(rank),"candidate":label,"group_score":row["primary_score"]})
    first = rows[0]
    payload = json.loads(first.get("candidate_chains") or "[]")
    hit = SearchHit("generated_protein_queries", first["candidate_id"], float(first["primary_score"] or 0.),
                    first["sequence"], first["description"], chains=payload)
    progress(f"Protein identification: selected candidate group {hit.record_id!r}.")
    return BlastSearchResult(hit, result.outputs, ranking, result.gap_open, result.gap_extend,
                             ranking_rows=rows, ranking_strategy=result.ranking_strategy)


def alignment_support(run_dir, chains):
    from argparse import Namespace
    from D_BLAST_searching.reduced_alphabet.search_pool import BlastSeq, Filter
    support = defaultdict(dict)
    from Bio import SeqIO
    original_records=list(SeqIO.parse(str(Path(run_dir)/'tmp/sequence_search/work/candidate_pool.fasta'),'fasta'))
    by_id={c.record_id:c for c in chains}
    tokens = {f'cryoid_seq_{i:08d}': by_id[r.id] for i,r in enumerate(original_records,1) if r.id in by_id}
    lengths={r.id:len(r.seq) for r in original_records}
    # Search uses an explicitly fixed effective search space: pool residues
    # times query length. BLAST's bit-score fallback only matters for E=0.
    residues = sum(len(r.seq) for r in original_records)
    for path in (Path(run_dir) / 'tmp/sequence_search/work').rglob('*_blast.txt'):
        # Production QueryMap adds ordinal suffixes to forward/reverse IDs.
        # The per-query file ordinal is the stable physical-fragment key;
        # it also keeps identically named but distinct traces independent.
        ordinal=re.search(r'_(\d+)_blast\.txt$',path.name)
        if ordinal is None:raise ValueError(f'Unknown query result filename: {path}')
        origin=f'fragment_{int(ordinal.group(1)):08d}'
        fields = []
        for line in path.read_text().splitlines():
            if not line or line.startswith('#'):
                continue
            cells = line.split('\t')
            h = BlastSeq(line, dbsize=residues)
            c = tokens[h.candidate]
            h.candidate = c.record_id
            h.set_prot_len(lengths[c.record_id])
            fields.append(h)
        fields.sort(key=lambda h: (h.candidate, h.log10_evalue))
        for h in Filter(fields, Namespace(length=[1, 100000], verbose=True)):
            support[origin][h.candidate] = max(support[origin].get(h.candidate, 0.), -h.log10_evalue, 0.)
    return dict(support)
