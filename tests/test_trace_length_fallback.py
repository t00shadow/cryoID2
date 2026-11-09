"""Exercise protein trace filtering through graph output and query FASTA generation."""
import json
from pathlib import Path

import mrcfile
import numpy as np
import pytest
from Bio import SeqIO

from cryoid_pipeline.models import PredictionBundle
from H_Backbone_tracing.protein.pipeline import run_protein_query_generation


def predictions(tmp_path, lengths, monkeypatch):
    from H_Backbone_tracing.common import clustering
    from H_Backbone_tracing.protein import masking
    paths=[]
    for i in range(11):
        path=tmp_path/f"prediction_{i}.mrc"
        with mrcfile.new(str(path),overwrite=True) as volume:
            volume.set_data(np.ones((10,10,240),dtype=np.float32) if i<5 else np.zeros((10,10,240),dtype=np.float32))
            volume.voxel_size=1.
        paths.append(path)
    pdb=tmp_path/"clustered.pdb"
    serial=0
    lines=[]
    for chain,length in enumerate(lengths):
        for position in range(length):
            serial+=1
            x=4.+chain*70.+position*3.8
            lines.append(f"ATOM  {serial:5d}  CA  GLY A{serial:4d}    {x:8.3f}{4.:8.3f}{4.:8.3f}  1.00  0.00           C\n")
    pdb.write_text(''.join(lines),encoding='ascii')
    monkeypatch.setattr(clustering,'run_clustering',lambda cfg:dict(output_path=str(pdb)))
    monkeypatch.setattr(masking,'mask_and_save_pdb',lambda **kwargs:str(pdb))
    return PredictionBundle(layout='test',density_map=paths[0],raw_prediction_dir=tmp_path,raw_key_atoms_1_path=paths[0],secondary_structure_paths=paths[1:5],residue_type_paths=paths[5:11])


@pytest.mark.parametrize('lengths,requested,expected,fallback',[
    ([3,2,1],10,[3,2,1],True),
    ([3,2,1],100,[3,2,1],True),
    ([1],10,[1],True),
    ([12,5,1],10,[12],False),
    ([3,2,1],3,[3],False),
    ([3,2,1],1,[3,2,1],False),
])
def test_query_generation_falls_back_only_when_no_trace_meets_requested_minimum(tmp_path,monkeypatch,lengths,requested,expected,fallback):
    bundle=predictions(tmp_path,lengths,monkeypatch)
    messages=[]
    result=run_protein_query_generation(bundle,tmp_path/'run',messages.append,min_trace_length=requested)
    records=list(SeqIO.parse(str(result.query_fasta),'fasta'))
    assert sorted([len(r.seq) for r in records],reverse=True)==sorted(expected,reverse=True)
    report=tmp_path/'run/tmp/tracing/graphs/trace_length_filter.json'
    metadata=json.loads(report.read_text())
    assert metadata['requested_min_trace_length']==requested
    assert metadata['effective_min_trace_length']==(0 if fallback else requested)
    assert metadata['fallback_applied'] is fallback
    assert metadata['component_count_before_filter']==len(lengths)
    assert metadata['longest_component_length']==max(lengths)
    assert metadata['retained_component_count']==len(expected)
    assert report.resolve() in result.output_files
    assert any('using minimum length 0' in message for message in messages) is fallback
    if not fallback and expected==[12]:
        assert result.query_fasta.read_text()=='>cc_0_updated_residues\n'+'G'*12+'\n\n'
