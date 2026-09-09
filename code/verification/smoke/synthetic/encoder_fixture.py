"""Small graphs written using the existing Dataset artifact and view contracts."""
from dataclasses import asdict

import numpy as np
import scipy.sparse as sp
import torch

from contracts.data import GraphData
from data.cache.store import digest, file_hash, save_tensor, write_json
from data.prepare import tensor_hash
from data.preprocessing.graph import graph_tensors
from data.views.training import load_training_views
from evaluation.source_validation.data import load_source_validation


def synthetic_dataset(root, sizes=(9, 7)):
    def save(path, value):
        save_tensor(root / path, value)
        return {'path': path, 'sha256': file_hash(root / path)}

    generator = torch.Generator().manual_seed(12)
    graphs = {}
    for domain, n, domain_id in [('A', sizes[0], 0), ('B', sizes[1], 1)]:
        if sizes == (9, 7):
            a = sp.coo_matrix(([1., 1., 1., 1.], ([0, 0, 1, 3], [1, 1, 2, 3])), shape=(n, n))
        else:
            a = sp.coo_matrix((np.ones(n-2), (np.arange(n-2), np.arange(1, n-1))), shape=(n, n))
        graph, _ = graph_tensors(a, np.arange(n))
        x = torch.rand(n, 6775, generator=generator)
        x /= x.sum(1, keepdim=True)
        x[-1] = 0  # isolated node and zero feature row
        graphs[domain] = GraphData(x=x, node_id=torch.arange(n), domain_id=torch.full((n,), domain_id),
            feature_dim=6775, attribute_union_hash=digest('synthetic-vocabulary'), dataset_version='synthetic-v1',
            source_train_mask=torch.zeros(n, dtype=torch.bool), source_val_mask=torch.zeros(n, dtype=torch.bool),
            source_unlabeled_mask=torch.ones(n, dtype=torch.bool), target_unlabeled_mask=torch.ones(n, dtype=torch.bool),
            target_test_mask=torch.ones(n, dtype=torch.bool), **graph)
    masks = {name: torch.zeros(sizes[0], dtype=torch.bool) for name in
             ('source_train_mask', 'source_val_mask', 'source_unlabeled_mask')}
    masks['source_train_mask'][[0, 3, 6]] = True
    masks['source_val_mask'][[1, 4, 7]] = True
    masks['source_unlabeled_mask'] = ~(masks['source_train_mask'] | masks['source_val_mask'])
    split = {'source_domain': 'A', 'label_rate': .05, 'split_seed': 0,
             'split_hash': tensor_hash(masks), 'masks': save('artifacts/source_masks.pt', masks),
             'train_labels': save('artifacts/train_labels.pt', {'node_id': torch.tensor([0, 3, 6]), 'labels': torch.arange(3)}),
             'validation_labels': save('artifacts/validation_only/labels.pt',
                                       {'node_id': torch.tensor([1, 4, 7]), 'labels': torch.arange(3)})}
    target_labels = root / 'artifacts/evaluation_only/target.pt'
    manifest = {'dataset_version': 'synthetic-v1', 'attribute_union_hash': graphs['A'].attribute_union_hash,
                'feature_dim': 6775, 'class_map': {'a': 0, 'b': 1, 'c': 2}, 'splits': [split],
                'tasks': [{'source': 'A', 'target': 'B'}],
                'domains': {d: {'domain_id': int(g.domain_id[0]),
                                'graph': save(f'artifacts/{d}/graph.pt', asdict(g))} for d, g in graphs.items()}}
    manifest['domains']['B']['evaluation_labels'] = save('artifacts/evaluation_only/target.pt', torch.arange(sizes[1]) % 3)
    manifest_path = root / 'artifacts/datasets/manifests/dataset_manifest.json'
    write_json(manifest_path, manifest)
    source, target = load_training_views(manifest_path, 'A', 'B', .05, 0)
    validation = load_source_validation(manifest_path, 'A', .05, 0)
    return manifest_path, source, target, validation, target_labels
