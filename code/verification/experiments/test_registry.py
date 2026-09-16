from dataclasses import asdict
import pytest
import torch
from experiments.registry import FINAL,CHANGES,BASE,variant,verify_single_factors
from experiments.configuration import configs,overlay,seal,unseal
from experiments.execution import inventory,train
from experiments.analysis import holm,paired
from verification.smoke.synthetic.encoder_fixture import synthetic_dataset


def test_registry_counts_and_unique_identity():
    assert len(verify_single_factors())==64
    for tier,count in [('core',630),('ablation',990),('sensitivity',180)]:
        rows=inventory(tier)
        assert len(rows)==len({r['run_id'] for r in rows})==count
    for name,(key,value) in CHANGES.items():
        assert {k for k in BASE if BASE[k]!=variant(name)['options'][k]}=={key}


def test_config_lock_and_reject_unknown(tmp_path):
    for bad in ({'ACMv9':{}},{'da':{'P':64}},{'encoder':{'encoder_lr':float('nan')}},{'memory':{'gamma':2}}):
        with pytest.raises(ValueError): overlay(bad)
    p=tmp_path/'lock.json'; seal(p,{'a':1}); assert unseal(p)=={'a':1}
    with pytest.raises(FileExistsError): seal(p,{})
    p.write_text('{"a":2,"content_hash":"x"}')
    with pytest.raises(ValueError): unseal(p)


def test_statistics():
    assert holm([.01,.04,.03])==pytest.approx([.03,.06,.06])
    r=paired([[1]*5]*18,[[1]*5]*18)
    assert r['raw_p']==1 and r['rank_biserial']==0 and r['bootstrap_95_ci']==[0,0]


@pytest.mark.parametrize('name',list(FINAL))
def test_registered_short_chain(tmp_path,name,monkeypatch):
    manifest,*_=synthetic_dataset(tmp_path/'data',sizes=(256,257))
    import data.views.training as views
    from data.cache.store import read_json
    dataset=read_json(manifest)
    forbidden={dataset['domains']['B']['evaluation_labels']['sha256']}
    if name=='B0': forbidden.add(dataset['domains']['B']['graph']['sha256'])
    original=views.verified_tensor
    def guard(root,record):
        if record['sha256'] in forbidden: raise AssertionError('Forbidden Target access during training')
        return original(root,record)
    monkeypatch.setattr(views,'verified_tensor',guard)
    row=dict(run_id='smoke__'+name,tier='smoke',variant=name,source='A',target='B',label_rate=.05,seed=0,
             configs={k:asdict(v) for k,v in configs(name=name,smoke=True).items()})
    run=train(row,manifest,tmp_path/'run',tmp_path/'cache')
    assert len(run['checkpoint_lineage'])==variant(name)['stages']
    assert run['checkpoint_lineage'][-1]['name']==variant(name)['final']
    if name=='M5': assert run['checkpoint_lineage'][2]['name']=='untrained_memory' and run['checkpoint_lineage'][2]['best_epoch']==0
    if name=='B5':
        from training.checkpoints.memory import read_verified
        state=read_verified(tmp_path/'run/source/source_pl_best.pt')
        assert set(state['pseudo_label_state'])=={'source'}


def test_orchestrator_resume_and_independent_evaluation(tmp_path,monkeypatch):
    from training.stages.domain_adaptation.trainer import DomainAdaptation
    from training.checkpoints.memory import read_verified
    from experiments.evaluate import evaluate,verified_result
    manifest,*_=synthetic_dataset(tmp_path/'data',sizes=(256,257))
    row=dict(run_id='smoke__B6',tier='smoke',variant='B6',source='A',target='B',label_rate=.05,seed=0,
        configs={k:asdict(v) for k,v in configs(smoke=True).items()})
    train(row,manifest,tmp_path/'reference',tmp_path/'cache')
    fit=DomainAdaptation.fit
    def interrupt(self,*args,**kwargs):
        fit(self,until_epoch=1)
        raise RuntimeError('simulated infrastructure interruption')
    monkeypatch.setattr(DomainAdaptation,'fit',interrupt)
    with pytest.raises(RuntimeError,match='simulated'):
        train(row,manifest,tmp_path/'resumed',tmp_path/'cache')
    monkeypatch.setattr(DomainAdaptation,'fit',fit)
    latest=tmp_path/'resumed/encoder/encoder_latest.pt'
    saved=latest.read_bytes(); latest.unlink()
    with pytest.raises(ValueError,match='missing latest'):
        train(row,manifest,tmp_path/'resumed',tmp_path/'cache',resume=True)
    latest.write_bytes(saved)
    train(row,manifest,tmp_path/'resumed',tmp_path/'cache',resume=True)
    a=read_verified(tmp_path/'reference/dual/full_best.pt')
    b=read_verified(tmp_path/'resumed/dual/full_best.pt')
    for module,state in a['model_state'].items():
        for key,value in state.items(): assert torch.equal(value,b['model_state'][module][key])
    result=evaluate(manifest,tmp_path/'resumed',tmp_path/'cache')
    assert result['status']=='COMPLETED'
    verified_result(tmp_path/'resumed')
    with pytest.raises(ValueError,match='locked'):
        train(row,manifest,tmp_path/'resumed',tmp_path/'cache',resume=True)


def test_formal_engine_cannot_bypass_gate(tmp_path):
    row=dict(tier='core',configs={k:asdict(v) for k,v in configs(name='B0').items()})
    with pytest.raises(PermissionError,match='admission'):
        train(row,tmp_path/'missing',tmp_path/'run',tmp_path/'cache')


def test_linux_scripts_have_portable_line_endings():
    from verification.gate.run import REPO
    for path in (REPO/'scripts').glob('*.sh'):
        raw=path.read_bytes()
        assert b'\r' not in raw, f'Linux script must use LF: {path}'
        assert raw.startswith(b'#!/usr/bin/env bash\n')
    assert '*.sh text eol=lf' in (REPO/'.gitattributes').read_text()
