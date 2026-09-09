import copy
from dataclasses import fields
from pathlib import Path
import sys

import numpy as np
import pytest
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contracts.data import GraphData, SourceTrainView, TargetTrainView
from data.cache.store import Cache, cache_key, digest, file_hash, save_tensor, write_json
from data.features.alignment import align_normalize, validate_vocabulary
from data.loaders.mat import RawGraph, class_labels
from data.preprocessing.graph import graph_tensors
from data.splits.source import source_split
from evaluation.final_target.data import FINAL_CHECKPOINT, load_target_evaluation


def test_graph_symmetry_duplicates_loops_and_isolates():
    a = sp.coo_matrix(([1., 1., 1., 1.], ([0, 0, 1, 2], [1, 1, 0, 2])), shape=(4, 4))
    graph, stats = graph_tensors(a, np.arange(4))
    assert graph['edge_index'].tolist() == [[0, 1], [1, 0]]
    assert graph['degree'].tolist() == [1, 1, 0, 0]
    assert stats['isolated_nodes'] == 2 and stats['removed_self_loops'] == 1
    adj = torch.sparse_coo_tensor(graph['gcn_edge_index'], graph['gcn_edge_weight'], (4, 4),
                                  check_invariants=True).to_dense()
    assert torch.allclose(adj, torch.tensor([[.5,.5,0,0],[.5,.5,0,0],[0,0,1.,0],[0,0,0,1.]]))


def test_filter_reindexes_edges_and_preserves_isolates():
    a = sp.coo_matrix(([1, 1], ([0, 1], [2, 2])), shape=(4, 4))
    graph, _ = graph_tensors(a, [0, 2, 3])
    assert graph['edge_index'].tolist() == [[0, 1], [1, 0]]
    assert graph['degree'].tolist() == [1, 1, 0]


def test_alignment_by_ids_and_zero_rows():
    x = align_normalize(np.array([[2., 6.], [0., 0.]]), [5, 2], [0, 1])
    assert x.shape == (2, 6775) and x.dtype == torch.float32
    assert x[0, 5] == .25 and x[0, 2] == .75
    assert torch.equal(x.sum(1), torch.tensor([1., 0.]))


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1])
def test_invalid_features_rejected(value):
    with pytest.raises(ValueError):
        align_normalize(np.array([[value]]), [0], [0])


def test_vocabulary_mapping_hash_and_unknown_input():
    v = [f'a{i}' for i in range(6775)]
    meta = {'vocabulary': v, 'attribute_union_hash': digest(v), 'domains': {
        'a': {'raw_sha256': 'h1', 'attribute_ids': v[::-1]},
        'b': {'raw_sha256': 'h2', 'attribute_ids': v[:2]}}}
    assert validate_vocabulary(meta, {'a': 6775, 'b': 2}, {'a':'h1','b':'h2'})['a'][0] == 6774
    for change in ('hash', 'raw', 'duplicate', 'unknown'):
        broken = copy.deepcopy(meta)
        if change == 'hash': broken['attribute_union_hash'] = 'bad'
        if change == 'raw': broken['domains']['a']['raw_sha256'] = 'bad'
        if change == 'duplicate': broken['domains']['b']['attribute_ids'] = ['a0','a0']
        if change == 'unknown': broken['domains']['b']['attribute_ids'] = ['a0','unknown']
        with pytest.raises(ValueError):
            validate_vocabulary(broken, {'a':6775,'b':2}, {'a':'h1','b':'h2'})


@pytest.mark.parametrize('rate', [.01,.03,.05])
@pytest.mark.parametrize('seed', [0,1,2,3,4,2026])
def test_source_budget_coverage_and_determinism(rate, seed):
    y = np.repeat(np.arange(3), [2, 150, 777])
    split = source_split(y, rate, seed)
    again = source_split(y, rate, seed)
    assert torch.equal(sum(t.int() for t in split.values()), torch.ones(len(y), dtype=torch.int32))
    for k in split: assert torch.equal(split[k], again[k])
    for c in range(3):
        tr = int(split['source_train_mask'][y == c].sum())
        va = int(split['source_val_mask'][y == c].sum())
        assert tr >= 1 and va >= 1
        assert tr + va == max(2, round(rate * int((y == c).sum())))


def test_rare_source_class_stops():
    with pytest.raises(ValueError, match='fewer than two'):
        source_split([0,0,1], .01, 0)


def test_multilabel_policy_is_explicit():
    raw = RawGraph(None, None, np.array([[1,1], [0,1]]), np.arange(2))
    with pytest.raises(ValueError): class_labels(raw, ['a','b'], 'reject')
    y, keep, count = class_labels(raw, ['a','b'], 'upstream_argmax')
    assert y.tolist() == ['a','b'] and keep.all() and count == 1
    assert class_labels(raw, ['a','b'], 'single_label_only')[1].tolist() == [False, True]


KEYS = {
    'features': {'dataset_version':'v','domain':'A','attribute_union_hash':'u','normalization_version':'n'},
    'split': {'dataset_version':'v','source_domain':'A','label_rate':.01,'split_seed':0},
    'metis': {'dataset_version':'v','domain':'A','P':4,'metis_seed':0},
    'anchor': {'dataset_version':'v','direction':'A->B','label_rate':.01,'split_seed':0,
               'K':8,'sampler':'degree','sampler_hparams':{'power':1}},
}


@pytest.mark.parametrize('kind', KEYS)
def test_every_decisive_cache_input_invalidates(kind, tmp_path):
    inputs = KEYS[kind]
    cache = Cache(tmp_path)
    _, original_path, hit = cache.get_or_create(kind, inputs, lambda: {'sentinel':torch.tensor([7])})
    assert not hit
    _, same_path, hit = cache.get_or_create(kind, inputs, lambda: pytest.fail('Cache miss'))
    assert hit and original_path == same_path
    for field, value in inputs.items():
        altered = dict(inputs)
        altered[field] = ({'power':2} if isinstance(value,dict) else
                          value + 1 if isinstance(value,(int,float)) else value + '-changed')
        _, path, hit = cache.get_or_create(kind, altered, lambda: {'sentinel':torch.tensor([9])})
        assert not hit and path != original_path
        missing = dict(inputs)
        del missing[field]
        with pytest.raises(ValueError): cache_key(kind, **missing)
    original_path.write_bytes(b'corruption')
    with pytest.raises(ValueError, match='Corrupt cache'):
        cache.get_or_create(kind, inputs, lambda: pytest.fail('Silently rebuilt corruption'))


def test_serialization_byte_determinism(tmp_path):
    value = {'x':torch.arange(5)}
    save_tensor(tmp_path/'a.pt', value)
    save_tensor(tmp_path/'b.pt', value)
    assert file_hash(tmp_path/'a.pt') == file_hash(tmp_path/'b.pt')


def test_training_contract_has_no_full_label_or_validation_member():
    for cls in (GraphData, TargetTrainView):
        assert not {'y','labels','group','validation_y','train_y'} & {f.name for f in fields(cls)}
    assert {f.name for f in fields(SourceTrainView)} == {'graph','train_node_id','train_y','split_hash'}


def fake_locked_run(tmp_path, variant):
    root = tmp_path/'code'
    dataset_path = root/'artifacts/datasets/manifests/dataset_manifest.json'
    labels_path = root/'artifacts/datasets/processed/test/evaluation_only/labels.pt'
    save_tensor(labels_path, {'node_id':torch.arange(2), 'labels':torch.tensor([0,1])})
    dataset = {'dataset_version':'v', 'tasks':[{'source':'A','target':'B'}],
               'splits':[{'source_domain':'A','label_rate':.01,'split_seed':0,'split_hash':'s'}],
               'domains':{'B':{'evaluation_labels':{'path':labels_path.relative_to(root).as_posix(),
                                                    'sha256':file_hash(labels_path)}}}}
    write_json(dataset_path,dataset)
    run_path = root/'artifacts/runs/core/test/run_manifest.json'
    write_json(run_path, {'variant':variant,'dataset_version':'v','source':'A','target':'B',
                         'dataset_manifest_sha256':file_hash(dataset_path),
                         'label_rate':.01,'split_seed':0,'split_hash':'s'})
    cp = run_path.parent/'final.pt'
    cp.write_bytes(b'locked checkpoint sentinel')
    lock = {'status':'locked','run_manifest_sha256':file_hash(run_path),
            'final_checkpoint':{'name':FINAL_CHECKPOINT[variant],'path':'final.pt','sha256':file_hash(cp)}}
    lock['lock_hash'] = digest(lock)
    lock_path = run_path.parent/'final_lock.json'
    write_json(lock_path,lock)
    return dataset_path, run_path, lock_path, lock


@pytest.mark.parametrize('variant', FINAL_CHECKPOINT)
def test_gate_accepts_only_correct_locked_checkpoint(tmp_path, variant):
    dataset, run, lock_path, lock = fake_locked_run(tmp_path,variant)
    assert load_target_evaluation(dataset,run,lock_path).y.tolist() == [0,1]
    lock['final_checkpoint']['name'] = 'memory_best'
    lock['lock_hash'] = digest({k:v for k,v in lock.items() if k != 'lock_hash'})
    write_json(lock_path,lock)
    with pytest.raises(PermissionError): load_target_evaluation(dataset,run,lock_path)


def test_small_full_build_common_classes_node_ids_and_split_reuse(tmp_path):
    import json
    import scipy.io as sio
    from data.prepare import prepare
    from data.views.training import load_training_views
    from evaluation.source_validation.data import load_source_validation

    root = tmp_path/'code'
    config_dir = root/'configs/base'
    production = Path(__file__).resolve().parents[2]/'configs/base'
    config = json.loads((production/'dataset.json').read_text())
    vocabulary = [f'v{i}' for i in range(6775)]
    meta = {'vocabulary':vocabulary,'attribute_union_hash':digest(vocabulary),'domains':{}}
    release = {'domains':{}}
    for domain in ['ACMv9','Citationv1','DBLPv7']:
        path = root/'artifacts/datasets/raw'/domain/'tiny.mat'
        path.parent.mkdir(parents=True)
        x = np.zeros((6,6775))
        x[:,0]=2
        x[:,1]=6
        a = sp.coo_matrix(([1,1,1],([0,2,4],[1,3,5])),shape=(6,6))
        group = np.repeat(np.eye(3,dtype=np.uint8),2,axis=0)
        sio.savemat(path, {'attrb':x,'network':a,'group':group,
                          'node_id':np.array([90,80,70,60,50,40])})
        release['domains'][domain]={'filename':'tiny.mat','sha256':file_hash(path)}
        meta['domains'][domain]={'raw_sha256':file_hash(path),'attribute_ids':vocabulary,
                                'class_ids':['common-a','common-b',f'private-{domain}']}
    write_json(config_dir/'dataset.json',config)
    write_json(config_dir/'dataset_release.json',release)
    write_json(config_dir/'attribute_metadata.json',meta)
    manifest_path = prepare(root)
    manifest = json.loads(manifest_path.read_text())
    assert manifest['class_map'] == {'common-a':0,'common-b':1}
    assert len(manifest['splits']) == 54 and len(manifest['tasks']) == 6
    for d in manifest['domains'].values():
        assert d['nodes'] == 4 and d['class_counts'] == [2,2]
        node_map = json.loads((root/d['node_mapping']['path']).read_text())
        assert node_map['raw_node_id'] == [90,80,70,60]
    first,t1=load_training_views(manifest_path,'ACMv9','Citationv1',.01,0)
    second,t2=load_training_views(manifest_path,'ACMv9','DBLPv7',.01,0)
    assert first.split_hash == second.split_hash
    assert torch.equal(first.train_node_id,second.train_node_id)
    assert len(first.train_y) == 2 and not hasattr(t1,'y') and not hasattr(t2,'y')
    val=load_source_validation(manifest_path,'ACMv9',.01,0)
    assert len(val.validation_y) == 2 and not set(val.node_id.tolist()) & set(first.train_node_id.tolist())
    before=file_hash(manifest_path)
    assert file_hash(prepare(root)) == before


@pytest.mark.parametrize('mutation', ['checkpoint','run','dataset','unlocked','digest'])
def test_gate_rejects_tampering_before_opening_labels(tmp_path, monkeypatch, mutation):
    dataset, run, lock_path, lock = fake_locked_run(tmp_path,'B0')
    if mutation == 'checkpoint': (run.parent/'final.pt').write_bytes(b'changed')
    if mutation == 'run': run.write_text(run.read_text()+' ')
    if mutation == 'dataset': dataset.write_text(dataset.read_text()+' ')
    if mutation == 'unlocked':
        lock['status']='unlocked'
        lock['lock_hash']=digest({k:v for k,v in lock.items() if k!='lock_hash'})
        write_json(lock_path,lock)
    if mutation == 'digest':
        lock['lock_hash']='bad'
        write_json(lock_path,lock)
    monkeypatch.setattr(torch,'load',lambda *a,**k:pytest.fail('Target labels opened before gate'))
    with pytest.raises(PermissionError): load_target_evaluation(dataset,run,lock_path)
