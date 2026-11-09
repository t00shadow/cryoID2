"""FASTA candidate membership and the legacy-compatible shared group score."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import math
import re

@dataclass(frozen=True)
class Chain:
    record_id: str
    candidate: str
    chain_id: str
    sequence: str
    description: str
    copies: int


def read_groups(path):
    from D_BLAST_searching.sequence_search import read_fasta
    groups = defaultdict(list)
    ids, pairs = set(), set()
    for r in read_fasta(Path(path)):
        tokens = r.description.split()
        explicit = any(t.startswith('candidate=') for t in tokens[1:])
        fields, description = {}, [tokens[0]]
        for token in tokens[1:]:
            key, sep, value = token.partition('=')
            if explicit and sep and key in ('candidate', 'chain', 'copies'):
                if key in fields or not value:
                    raise ValueError('Invalid or repeated grouping annotation')
                fields[key] = value
            else:
                description.append(token)
        candidate = fields.get('candidate', r.record_id)
        chain_id = fields.get('chain', r.record_id)
        copies = int(fields.get('copies', '1'))
        if copies < 1 or r.record_id in ids or (candidate, chain_id) in pairs:
            raise ValueError('Invalid copies or duplicate record/chain identifier')
        ids.add(r.record_id)
        pairs.add((candidate, chain_id))
        clean=re.sub(r'\s+(?:candidate|chain|copies)=\S+','',r.description) if explicit else r.description
        groups[candidate].append(Chain(r.record_id, candidate, chain_id,
                                     r.sequence, clean, copies))
    if not groups:
        raise ValueError('Empty candidate pool')
    labels = [name if is_assembly_group(chains) else chains[0].record_id
              for name, chains in groups.items()]
    if len(labels) != len(set(labels)):
        raise ValueError('Candidate group identifiers collide with independent record identifiers')
    return dict(groups)


def group_score(query_ids, members, support, legacy_scores):
    """Strongest original chain score plus newly explained fragment evidence.

No new threshold, size penalty, shortlist width or score customization.
One query contributes only its strongest chain evidence. Repeated members
never add evidence. Unmatched queries add no *additional* complex evidence;
the original singleton length treatment is intentionally preserved.
"""
    members = list(dict.fromkeys(members))
    available = [c for c in members if math.isfinite(legacy_scores.get(c, -math.inf))]
    if not available:
        return {'score': -math.inf, 'anchor': None, 'additional_support': 0.,
                'chain_support': {}, 'assignments': {}}
    anchor = min(available, key=lambda c: (-legacy_scores[c], c))
    extra, attribution, assignments = 0., defaultdict(float), {}
    for query in dict.fromkeys(query_ids):
        values = {c: support.get(query, {}).get(c, 0.) for c in members}
        best = max(values.values(), default=0.)
        extra += max(0., best - values[anchor])
        if best > 0:
            tied = sorted(c for c, value in values.items() if value == best)
            assignments[query] = {'chains': tied, 'support': best}
            for c in tied:
                attribution[c] += best / len(tied)
    # Return the original float directly when extra=0 (no rounding/reduction).
    score = legacy_scores[anchor] if extra == 0 else legacy_scores[anchor] + extra
    evidence_anchor = min(attribution, key=lambda c: (-attribution[c], c)) if attribution else None
    return {'score': score, 'anchor': evidence_anchor,
            'baseline_chain': anchor, 'additional_support': extra,
            'chain_support': dict(attribution), 'assignments': assignments}



def canonical_records(path):
    """Keep records independent while removing parsed grouping annotations."""
    from .sequence_search import SequenceRecord, read_fasta
    return [SequenceRecord(r.record_id, r.sequence,
            re.sub(r'\s+(?:candidate|chain|copies)=\S+', '', r.description)
            if any(t.startswith('candidate=') for t in r.description.split()[1:]) else r.description)
            for r in read_fasta(Path(path))]


def chain_payload(chains, evidence):
    return [{"record_id": c.record_id, "chain_id": c.chain_id,
             "sequence": c.sequence, "copies": c.copies,
             "trace_support": evidence.get(c.record_id, 0.)} for c in chains]


def is_assembly_group(chains):
    return sum(c.copies for c in chains) > 1
