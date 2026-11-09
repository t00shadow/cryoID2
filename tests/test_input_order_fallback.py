import csv
import json
import random
from pathlib import Path

import mrcfile
import numpy as np
import pytest

from D_BLAST_searching.protein_search import run_protein_blast_search
from D_BLAST_searching.protein_candidates import run_protein_candidate_search
from D_BLAST_searching.sequence_search import read_fasta
from H_Backbone_tracing.protein.pipeline import run_protein_query_generation


@pytest.mark.parametrize('queries,reason', [
    ('','input_order_no_queries'),
    ('; no protein fragments\n','input_order_no_queries'),
    ('>one\nY\n','input_order_single_residue_queries'),
    ('>one\nY\n>two\nG\n','input_order_single_residue_queries'),
])
@pytest.mark.parametrize('runner',[run_protein_blast_search,run_protein_candidate_search])
def test_uninformative_queries_preserve_fasta_order_without_running_blast(tmp_path,monkeypatch,queries,reason,runner):
    from D_BLAST_searching.reduced_alphabet import search_pool
    def forbidden(*args):raise AssertionError('BLAST must not run for input-order fallback')
    monkeypatch.setattr(search_pool,'Search',forbidden)
    q=tmp_path/'queries.fasta';q.write_text(queries)
    pool=tmp_path/'pool.fasta';original='>z_first Description\nMKWVTFISLLLL\n>a_second\nMAGSTIVQQ\n>m_third\nWWWWWWWW\n';pool.write_text(original)
    messages=[]
    result=runner(query_fasta=q,candidate_pool=pool,run_dir=tmp_path/'run',progress=messages.append)
    assert result.ranking_strategy==reason
    assert result.selected_hit.record_id=='z_first'
    assert [r['candidate'] for r in result.ranking]==['z_first','a_second','m_third']
    assert all(r['primary_score']=='' for r in result.ranking_rows)
    assert all(r['primary_score_name']=='input_order' for r in result.ranking_rows)
    assert all(r['average_identity']=='' for r in result.ranking_rows)
    assert all(p.is_file() and p.stat().st_size>0 for p in result.outputs)
    assert 'No BLAST alignment evidence' in messages[0]
    assert pool.read_text()==original


def test_input_order_fallback_preserves_singleton_metadata_invariance(tmp_path):
    q=tmp_path/'q.fasta';q.write_text('>one\nY\n')
    results=[]
    for label,annotation in [('plain',''),('tagged',' candidate=alias chain=X copies=1')]:
        p=tmp_path/(label+'.fasta');p.write_text(f'>z_first Description{annotation}\nMKWVTFISLLLL\n>a_second\nMAGSTIVQQ\n')
        result=run_protein_candidate_search(query_fasta=q,candidate_pool=p,run_dir=tmp_path/label,progress=lambda _:None)
        results.append((result.selected_hit,result.ranking_rows))
    assert results[0]==results[1]


def test_grouped_fallback_uses_first_occurrence_of_groups(tmp_path):
    q=tmp_path/'q.fasta';q.write_text('')
    p=tmp_path/'pool.fasta';p.write_text('>z_chain candidate=z_complex chain=Z\nMKWVTFISLLLL\n>a_single\nMAGSTIVQQ\n>b_chain candidate=z_complex chain=B\nWWWWWWWW\n')
    result=run_protein_candidate_search(query_fasta=q,candidate_pool=p,run_dir=tmp_path/'run',progress=lambda _:None)
    assert result.selected_hit.record_id=='z_complex'
    assert result.selected_hit.sequence=='MKWVTFISLLLL'
    assert [r['candidate_id'] for r in result.ranking_rows]==['z_complex','a_single']
    assert len(result.selected_hit.chains)==2
    assert all(c['trace_support']==0 for c in result.selected_hit.chains)
    assert all(r['primary_score']=='' for r in result.ranking_rows)
    assert result.ranking_strategy=='input_order_no_queries'


def test_longer_queries_keep_normal_blast_ranking(tmp_path):
    sequence=''.join(random.Random(17).choices('ACDEFGHIKLMNPQRSTVWY',k=120))
    q=tmp_path/'q.fasta';q.write_text(f'>short\nY\n>long\n{sequence[20:100]}\n')
    p=tmp_path/'pool.fasta';p.write_text(f">z_first\n{'W'*120}\n>x_target\n{sequence}\n")
    result=run_protein_candidate_search(query_fasta=q,candidate_pool=p,run_dir=tmp_path/'run',progress=lambda _:None,cpu_threads=1)
    assert result.selected_hit.record_id=='x_target'
    assert result.ranking_strategy is None


@pytest.mark.parametrize('empty_stage',['clustering','masking'])
def test_empty_tracing_is_reported_and_can_continue_to_input_order_ranking(tmp_path,monkeypatch,empty_stage):
    from test_trace_length_fallback import predictions
    from H_Backbone_tracing.common import clustering
    from H_Backbone_tracing.protein import masking
    original_clustering=clustering.run_clustering
    bundle=predictions(tmp_path,[3],monkeypatch)
    if empty_stage=='clustering':
        monkeypatch.setattr(clustering,'run_clustering',original_clustering)
        with mrcfile.open(str(bundle.raw_key_atoms_1_path),mode='r+') as volume:volume.data[:]=0
    else:monkeypatch.setattr(masking,'mask_and_save_pdb',lambda **kw:None)
    messages=[]
    query=run_protein_query_generation(bundle,tmp_path/'run',messages.append)
    assert read_fasta(query.query_fasta)==[]
    assert query.source=='protein_backbone_tracing_no_fragments'
    assert any('preserve FASTA input order' in m for m in messages)
    p=tmp_path/'pool.fasta';p.write_text('>z_first\nMKWVTFISLLLL\n>a_second\nMAGSTIVQQ\n')
    result=run_protein_candidate_search(query_fasta=query.query_fasta,candidate_pool=p,run_dir=tmp_path/'run',progress=messages.append)
    assert result.selected_hit.record_id=='z_first'
    assert result.ranking_strategy=='input_order_no_queries'


def test_missing_prediction_file_still_fails_instead_of_using_input_order(tmp_path,monkeypatch):
    from test_trace_length_fallback import predictions
    bundle=predictions(tmp_path,[3],monkeypatch)
    bundle.raw_key_atoms_1_path.unlink()
    with pytest.raises(FileNotFoundError,match='prediction map'):
        run_protein_query_generation(bundle,tmp_path/'run',lambda _:None)


def test_pipeline_and_checkpoint_resume_preserve_zero_fragment_input_order(tmp_path):
    from test_cryoid_pipeline import CryoIDProjectPipelineTests
    from cryoid_pipeline import CryoIDPipeline
    from cryoid_pipeline.models import QueryBundle
    helper=CryoIDProjectPipelineTests();density=helper._make_density_case(tmp_path)
    pool=helper._write(tmp_path/'pool.fasta','>z_first\nMKWVTFISLLLL\n>a_second\nMAGSTIVQQ\n')
    counts=dict(unet=0,queries=0)
    def unet(density,output,progress):
        counts['unet']+=1;return helper._fake_unet_runner(density,output,progress)
    def queries(bundle,run,progress):
        counts['queries']+=1;q=helper._write(run/'tmp/empty_queries.fasta','')
        return QueryBundle(q,'empty_fixture',[q])
    unet.checkpoint_id='empty-query-inference';queries.checkpoint_id='empty-query-generator'
    def run():return CryoIDPipeline(project_dir=tmp_path/'project',sequence_mode='protein',unet_runner=unet,protein_query_runner=queries).run(density,pool,stop_after_identification=True)
    first=run();assert first.ok,[(s.name,s.status,s.message) for s in first.stages]
    assert first.selected_hit.record_id=='z_first'
    search=next(s for s in first.stages if s.name=='Protein candidate BLAST')
    assert search.details['alignment_evidence'] is False
    assert search.details['ranking_strategy']=='input_order_no_queries'
    assert any(s.details.get('query_count')==0 for s in first.stages)
    with first.ranking_path.open(newline='') as f:ranking=list(csv.DictReader(f,delimiter='\t'))
    assert [r['candidate_id'] for r in ranking]==['z_first','a_second']
    assert all(r['primary_score']=='' for r in ranking)
    resumed=run();assert resumed.ok and resumed.selected_hit==first.selected_hit
    assert counts==dict(unet=1,queries=1)
