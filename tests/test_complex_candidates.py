"""Behavioral contracts for the shared singleton/complex pipeline."""
import csv
import json
import random
from pathlib import Path

import pytest
from Bio import SeqIO

from D_BLAST_searching.candidate_groups import read_groups, group_score, canonical_records
from D_BLAST_searching.protein_candidates import run_protein_candidate_search
from D_BLAST_searching.sequence_ranking import write_sequence_ranking
from I_Structure_fitting.protein.core import assemblies
from cryoid_pipeline import CryoIDPipeline
from cryoid_pipeline.models import QueryBundle, pipeline_result_to_dict, pipeline_result_from_dict


def seq(seed):
    return "".join(random.Random(seed).choices("ACDEFGHIKLMNPQRSTVWY", k=110))


def chain(name, sequence, copies=1, support=10):
    return dict(record_id=name, chain_id=name, sequence=sequence, copies=copies, trace_support=support)


def entry(sequence, operations="1", second=None):
    def entity(asym, sequence, kind="Protein"):
        return {"entity_poly":{"rcsb_entity_polymer_type":kind, "pdbx_seq_one_letter_code_can":sequence},
                "rcsb_polymer_entity_container_identifiers":{"asym_ids":[asym],"uniprot_ids":["P00001"]}}
    entities=[entity("A",sequence)]
    if second: entities.append(entity("B",second))
    return {"rcsb_id":"TEST", "polymer_entities":entities,
            "assemblies":[{"rcsb_id":"TEST-1","pdbx_struct_assembly_gen":[
                {"asym_id_list":["A","B"] if second else ["A"], "oper_expression":operations}]}]}


def test_support_is_order_independent_and_never_counts_a_fragment_twice():
    support={"q1":{"a":10,"b":3},"q2":{"a":2,"b":9}}
    expected=group_score(["q1","q2"],["a","b"],support,{"a":12.,"b":10.})
    assert expected["score"]==19.
    assert expected==group_score(["q1","q2","q1"],["b","a","b"],support,{"a":12.,"b":10.})
    assert group_score(["q1","q2"],["a"],support,{"a":12.})["score"]==12.
    assert group_score(["q1"],["a","b"],{}, {"a":float("-inf")})["score"]==float("-inf")


@pytest.mark.parametrize("annotation", ["copies=0", "copies=oops", "chain=", "copies=2 copies=3"])
def test_bad_group_annotations_fail_clearly(tmp_path, annotation):
    p=tmp_path/"pool.fasta"; p.write_text(f">a candidate=g {annotation}\nAAAA\n")
    with pytest.raises(ValueError): read_groups(p)


def test_rna_canonical_records_stay_independent_including_duplicate_ids(tmp_path):
    p=tmp_path/"rna.fasta";p.write_text(">a original candidate=g chain=X\nAGCU\n>a other candidate=g chain=Y\nACGU\n")
    assert [r.description for r in canonical_records(p)]==["a original","a other"]


def test_harmless_singleton_metadata_is_byte_identical_with_real_blast(tmp_path):
    query=tmp_path/"q.fasta";query.write_text(f">trace\n{seq(1)[20:80]}\n")
    outputs=[]
    for name,tag in [("plain",""),("tagged"," candidate=alias chain=Z copies=1")]:
        pool=tmp_path/(name+".fasta"); pool.write_text(f">a Original description{tag}\n{seq(1)}\n>b Other{tag.replace('alias','other')}\n{seq(2)}\n")
        result=run_protein_candidate_search(query_fasta=query,candidate_pool=pool,run_dir=tmp_path/name,progress=lambda _:None,cpu_threads=1)
        path=write_sequence_ranking(tmp_path/name,mode="protein",rows=result.ranking_rows)
        outputs.append((result.selected_hit,path.read_bytes(),result.ranking))
    assert outputs[0]==outputs[1]


def test_real_blast_complex_ranking_keeps_chains_separate(tmp_path):
    q=tmp_path/"q.fasta";q.write_text(f">q1\n{seq(1)[20:80]}\n>q2\n{seq(2)[20:80]}\n")
    p=tmp_path/"pool.fasta";p.write_text(f">a candidate=complex chain=A\n{seq(1)}\n>b candidate=complex chain=B\n{seq(2)}\n>decoy\n{seq(3)}\n")
    result=run_protein_candidate_search(query_fasta=q,candidate_pool=p,run_dir=tmp_path/"run",progress=lambda _:None,cpu_threads=1)
    assert result.selected_hit.record_id=="complex"
    assert {c["sequence"] for c in result.selected_hit.chains}=={seq(1),seq(2)}
    assert len(result.ranking_rows)==2
    assert len(result.selected_hit.sequence)==110
    assert all(c["trace_support"]>0 for c in result.selected_hit.chains)


def test_assembly_assignments_require_distinct_chain_instances():
    e=entry(seq(1),"1-2"); a=e["assemblies"][0]
    assert len(assemblies.match_assembly([chain("x",seq(1),2)],e,a,.9))==2
    assert assemblies.match_assembly([chain("x",seq(1),3)],e,a,.9) is None
    assert assemblies.expand_operations("(1-2)(3,4)")==[("1","3"),("1","4"),("2","3"),("2","4")]
    e["polymer_entities"][0]["entity_poly"]["rcsb_entity_polymer_type"]="RNA"
    assert assemblies.match_assembly([chain("x",seq(1))],e,a,.9) is None


def test_separate_assemblies_do_not_satisfy_two_chains(monkeypatch,tmp_path):
    e=entry(seq(1),second=seq(2))
    e["assemblies"]=[{"rcsb_id":f"TEST-{i}","pdbx_struct_assembly_gen":[{"asym_id_list":[a],"oper_expression":"1"}]} for i,a in [(1,"A"),(2,"B")]]
    monkeypatch.setattr(assemblies,"load_entries",lambda ids,*_: {"TEST":e} if ids else {})
    cs=[chain("x",seq(1),support=12),chain("y",seq(2),support=2)]
    hits={c["record_id"]:{"matches":[{"fraction":1.,"identity":1.,"chains":[("TEST","A")]}]} for c in cs}
    result=assemblies.retrieve(cs,hits,tmp_path,lambda:None)
    assert result["match_scope"]=="partial_sequence_match"
    assert result["anchors"]==["x"] and set(result["candidates"])=={"TEST-1"}
    assert result["missing_chains"]=={"x":{"y":1}}
    cs[1]["trace_support"]=12
    result=assemblies.retrieve(cs,hits,tmp_path,lambda:None)
    assert result["anchor_ambiguity"] and set(result["candidates"])=={"TEST-1","TEST-2"}
    for c in cs:c["trace_support"]=0
    assert assemblies.retrieve(cs,hits,tmp_path,lambda:None)["match_scope"]=="insufficient_trace_support"


def test_compatible_complete_assembly_wins_before_anchor_fallback(monkeypatch,tmp_path):
    e=entry(seq(1),second=seq(2)); monkeypatch.setattr(assemblies,"load_entries",lambda *_:{"TEST":e})
    cs=[chain("x",seq(1)),chain("y",seq(2))]
    hits={c["record_id"]:{"matches":[{"fraction":1.,"identity":1.,"chains":[("TEST","A")]}]} for c in cs}
    r=assemblies.retrieve(cs,hits,tmp_path,lambda:None)
    assert r["match_scope"]=="all_supplied_chains_compatible" and r["anchors"]==[]
    assert len(r["candidates"]["TEST-1"])==2


def test_download_never_substitutes_asu_or_another_assembly(monkeypatch,tmp_path):
    calls=[]
    def fail(url,**kwargs):
        calls.append(url);raise RuntimeError("unavailable")
    monkeypatch.setattr(assemblies.fetch.SESSION,"get",fail)
    with pytest.raises(RuntimeError):assemblies.download_assembly("TEST-7",tmp_path,lambda:None)
    assert calls==["https://files.rcsb.org/download/TEST-assembly7.cif"]


@pytest.mark.parametrize("scope",["all_supplied_chains_compatible","partial_sequence_match"])
def test_default_complex_pipeline_and_resume_keep_every_chain(monkeypatch,tmp_path,scope):
    from test_cryoid_pipeline import CryoIDProjectPipelineTests
    from I_Structure_fitting.protein.core import pipeline as fitting
    helper=CryoIDProjectPipelineTests(); density=helper._make_density_case(tmp_path)
    pool=helper._write(tmp_path/"pool.fasta",f">a candidate=complex chain=A\n{seq(1)}\n>b candidate=complex chain=B\n{seq(2)}\n")
    def queries(bundle,run,progress):
        q=helper._write(run/"tmp"/"q.fasta",f">a\n{seq(1)[20:80]}\n>b\n{seq(2)[20:80]}\n")
        return QueryBundle(query_fasta=q,source="fixture",output_files=[q])
    queries.checkpoint_id="complex-query-fixture-v1"
    def prediction(density,output,progress):return helper._fake_unet_runner(density,output,progress)
    prediction.checkpoint_id="complex-prediction-fixture-v1"
    monkeypatch.setattr(assemblies,"chain_hits",lambda cs,*_: {c["record_id"]:{"best_score":1,"matches":[]} for c in cs})
    monkeypatch.setattr(assemblies,"retrieve",lambda cs,*_:dict(match_scope=scope,candidates={"TEST-1":[]},missing_chains=[] if scope.startswith("all") else {"A":{"B":1}}))
    monkeypatch.setattr(assemblies,"download_assembly",lambda ident,folder,*_:helper._write(folder/(ident+".cif"),"data_test\n"))
    fits=[]
    def fit(**kw):
        fits.append(kw["candidate_id"])
        out=helper._write(kw["run_config"].fitted_dir/"complex.cif","data_aligned_complex\n")
        return dict(status="ok",label=kw["label"],type="PDB",id=kw["candidate_id"],score=7.,score_metric="rawfield",source_model=str(kw["model_path"]),fitted_path=str(out),**kw["extra"])
    monkeypatch.setattr(fitting,"_fit_candidate",fit)
    def run():return CryoIDPipeline(project_dir=tmp_path/"project",sequence_mode="protein",unet_runner=prediction,protein_query_runner=queries).run(density,pool)
    first=run()
    assert first.ok,[(s.name,s.message) for s in first.stages]
    assert len(first.selected_hit.chains)==2 and fits==["TEST-1"]
    encoded=pipeline_result_to_dict(first)
    assert pipeline_result_from_dict(encoded).selected_hit==first.selected_hit
    with (first.run_dir/"alignment_summary.csv").open() as f: rows=list(csv.DictReader(f))
    assert rows and rows[0]["match_scope"]==scope
    assert rows[0]["sequence"] in {seq(1), seq(2)}
    selected=first.run_dir/"selected_structure_sequences.fasta"
    assert {str(r.seq) for r in SeqIO.parse(selected,"fasta")}=={seq(1),seq(2)}
    assert len(list(SeqIO.parse(tmp_path/"project"/"best_overall"/"best_structure_selected_sequence.fasta","fasta")))==2
    resumed=run()
    assert resumed.ok and resumed.selected_hit==first.selected_hit and fits==["TEST-1"]
    assert (resumed.run_dir/"checkpoints"/"complete.json").exists()
