"""Offline Dataset integrity audit. This is not an experiment/model selection entry point."""
import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from contracts.data import TargetTrainView
from data.cache.store import digest, file_hash, read_json, write_json
from data.prepare import tensor_hash
from data.splits.source import source_split
from data.views.training import load_manifest, load_training_views, verified_tensor


def check(manifest_path):
    root, m = load_manifest(manifest_path)
    vocabulary = read_json(root/m['attribute_vocabulary']['path'])
    assert file_hash(root/m['attribute_vocabulary']['path']) == m['attribute_vocabulary']['sha256']
    assert digest(vocabulary['vocabulary']) == m['attribute_union_hash']
    assert len(vocabulary['vocabulary']) == len(set(vocabulary['vocabulary'])) == 6775
    assert len(m['tasks']) == 6
    labels = {}
    summary = {}
    for domain, record in m['domains'].items():
        raw = root/'artifacts/datasets/raw'/domain/m['provenance']['domains'][domain]['filename']
        assert file_hash(raw) == m['raw_hashes'][domain]
        g = verified_tensor(root,record['graph'])
        labels[domain] = verified_tensor(root,record['evaluation_labels'])['labels']
        n = record['nodes']
        assert g['x'].shape == (n,6775) and g['x'].dtype == torch.float32
        assert torch.isfinite(g['x']).all() and (g['x'] >= 0).all()
        rows = g['x'].abs().sum(1)
        assert torch.allclose(rows[rows != 0], torch.ones_like(rows[rows != 0]), atol=1e-6)
        assert torch.equal(g['node_id'],torch.arange(n))
        assert g['attribute_union_hash'] == m['attribute_union_hash']
        assert g['dataset_version'] == m['dataset_version']
        assert 'y' not in g and 'group' not in g
        e = g['edge_index'].numpy()
        assert e.shape[0] == 2 and e.min() >= 0 and e.max() < n
        assert not np.any(e[0] == e[1])
        assert len(np.unique(e[0]*n+e[1])) == e.shape[1]
        a = sp.csr_matrix((np.ones(e.shape[1]),(e[0],e[1])),shape=(n,n))
        assert (a != a.T).nnz == 0
        degree = np.asarray(a.sum(1)).ravel()
        assert np.array_equal(degree,g['degree'].numpy())
        inv = sp.diags((degree+1)**-.5)
        expected = inv @ (a+sp.eye(n)) @ inv
        ge = g['gcn_edge_index'].numpy()
        actual = sp.csr_matrix((g['gcn_edge_weight'].numpy(),(ge[0],ge[1])),shape=(n,n))
        diff = actual-expected
        assert not diff.nnz or abs(diff.data).max() < 1e-6
        assert actual.nnz == a.nnz+n
        assert record['graph_tensor_hash'] == tensor_hash({k:v for k,v in g.items() if isinstance(v,torch.Tensor)})
        assert torch.equal(verified_tensor(root,record['feature_cache'])['x'],g['x'])
        assert torch.equal(torch.bincount(labels[domain]),torch.tensor(record['class_counts']))
        summary[domain] = {'nodes':n, 'directed_edges':e.shape[1], 'feature_dim':6775,
                           'class_counts':record['class_counts'], 'anomalies':record['anomalies']}
    for s in m['splits']:
        split = verified_tensor(root,s['masks'])
        y = labels[s['source_domain']]
        assert s['split_hash'] == tensor_hash(split)
        assert torch.equal(sum(v.int() for v in split.values()),torch.ones(len(y),dtype=torch.int32))
        expected = source_split(y.numpy(),s['label_rate'],s['split_seed'])
        for k in split: assert torch.equal(split[k],expected[k])
        for key,mask in [('train_labels','source_train_mask'),('validation_labels','source_val_mask')]:
            limited = verified_tensor(root,s[key])
            assert torch.equal(limited['node_id'],torch.where(split[mask])[0])
            assert torch.equal(limited['labels'],y[split[mask]])
            assert len(torch.unique(limited['labels'])) == len(m['class_map'])
        assert s['purpose'] == ('development' if s['split_seed'] == 2026 else 'formal')
    # Sentinel audit: all direction/rate/seed loaders must succeed if complete and validation
    # label files are forbidden. Target graphs and Source train-only labels are permitted.
    original_load = torch.load
    forbidden_accesses = []
    def guarded_load(path,*args,**kwargs):
        if any(x in str(path) for x in ['evaluation_only','validation_only']):
            forbidden_accesses.append(str(path))
            raise AssertionError('Training attempted to read restricted labels')
        return original_load(path,*args,**kwargs)
    torch.load = guarded_load
    checked = 0
    try:
        for task in m['tasks']:
            if task['status'] != 'ready': continue
            for rate in m['label_rates']:
                for seed in [*m['formal_seeds'],m['development_seed']]:
                    source,target = load_training_views(manifest_path,task['source'],task['target'],rate,seed)
                    assert not hasattr(target,'y') and not hasattr(target,'graph')
                    assert not {'y','group','labels'} & {f.name for f in fields(TargetTrainView)}
                    assert target.target_unlabeled_mask.all()
                    assert source.graph.target_test_mask.sum() == 0
                    assert source.graph.source_train_mask.sum() == len(source.train_y)
                    checked += 1
    finally:
        torch.load = original_load
    assert not forbidden_accesses
    assert len(m['splits']) == 54 and checked == 108
    result = {'status':'PASS','dataset_manifest_sha256':file_hash(manifest_path),
              'dataset_version':m['dataset_version'],'domains':summary,'source_splits':len(m['splits']),
              'formal_direction_configurations':90,'development_direction_configurations':18,
              'training_views_checked':checked,'forbidden_label_reads':len(forbidden_accesses)}
    report = root/'artifacts/reports/integration/dataset_integrity.json'
    write_json(report,result)
    print(f'PASS: {len(m["splits"])} splits, {checked} direction configurations. Report: {report}')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path,default=Path('artifacts/datasets/manifests/dataset_manifest.json'))
    args = parser.parse_args()
    check(args.manifest.resolve())
