"""Run with `python -m data.prepare` from code/. Dataset stage only."""
import argparse
import platform
from pathlib import Path

import numpy as np
import scipy
import torch

from data.cache.store import Cache, digest, file_hash, read_json, save_tensor, write_json
from data.features.alignment import NORMALIZATION_VERSION, align_normalize, validate_vocabulary
from data.loaders.mat import class_labels, load_mat
from data.preprocessing.graph import GRAPH_VERSION, graph_tensors
from data.splits.source import SPLIT_VERSION, source_split

DOMAINS = ('ACMv9', 'Citationv1', 'DBLPv7')


def tensor_hash(tensors):
    import hashlib
    h = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        a = tensor.cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def prepare(root, config_path=None):
    root = Path(root).resolve()
    config = read_json(config_path or root / 'configs/base/dataset.json')
    if (config['feature_dim'] != 6775 or config['label_rates'] != [0.01, 0.03, 0.05]
            or config['formal_seeds'] != [0, 1, 2, 3, 4]
            or config['development_seed'] != 2026 or config['train_fraction'] != 0.8):
        raise ValueError('Configuration violates Dataset.md fixed settings')
    metadata = read_json(root / config['attribute_metadata'])
    release = read_json(root / config['raw_release'])
    paths = {k: root / v for k, v in config['paths'].items()}
    raw, hashes, labels, initial_keep, anomalies = {}, {}, {}, {}, {}
    for domain in DOMAINS:
        path = paths['raw'] / domain / release['domains'][domain]['filename']
        hashes[domain] = file_hash(path)
        raw[domain] = load_mat(path)
        labels[domain], initial_keep[domain], multiple = class_labels(
            raw[domain], metadata['domains'][domain]['class_ids'], config['multilabel_policy'])
        anomalies[domain] = {'multilabel_nodes': multiple, 'nonfinite_features': 0,
                             'negative_features': 0, 'unlabeled_nodes': 0}
    maps = validate_vocabulary(metadata, {d: raw[d].x.shape[1] for d in DOMAINS}, hashes)
    common = sorted(set.intersection(*(set(labels[d][initial_keep[d]]) for d in DOMAINS)))
    if not common:
        raise ValueError('No shared classes')
    class_map = {c: i for i, c in enumerate(common)}
    environment = {'python': platform.python_version(), 'numpy': np.__version__,
                   'scipy': scipy.__version__, 'torch': str(torch.__version__)}
    implementation = {p.relative_to(root).as_posix(): file_hash(p)
                      for folder in ['data', 'contracts/data', 'evaluation/source_validation',
                                     'evaluation/final_target']
                      for p in sorted((root / folder).rglob('*.py'))}
    version_inputs = {'raw_hashes': hashes, 'metadata_hash': digest(metadata),
                      'config': config, 'class_map': class_map, 'environment': environment,
                      'implementation': implementation, 'graph_version': GRAPH_VERSION,
                      'split_version': SPLIT_VERSION, 'normalization_version': NORMALIZATION_VERSION}
    version = 'sgda-' + digest(version_inputs)
    destination = paths['processed'] / version
    cache = Cache(paths['cache'])
    manifest = {'schema_version': config['schema_version'], 'dataset_version': version,
                'attribute_union_hash': metadata['attribute_union_hash'], 'feature_dim': 6775,
                'class_map': class_map, 'raw_hashes': hashes, 'version_inputs': version_inputs,
                'provenance': release, 'domains': {}, 'splits': [], 'tasks': [],
                'formal_seeds': config['formal_seeds'], 'development_seed': 2026,
                'label_rates': config['label_rates'], 'label_policy': config['multilabel_policy'],
                'cache_scope': {'features': 'materialized', 'split': 'materialized',
                                'metis': 'key contract only; no P requested in Dataset.md',
                                'anchor': 'key contract only; no K/sampler requested in Dataset.md'}}

    def record(path):
        return {'path': path.relative_to(root).as_posix(), 'sha256': file_hash(path)}

    def save(path, value):
        save_tensor(path, value)
        return record(path)

    for domain_id, domain in enumerate(DOMAINS):
        print(f'Preparing {domain}', flush=True)
        keep = np.flatnonzero(initial_keep[domain] & np.isin(labels[domain], common))
        y = torch.tensor([class_map[c] for c in labels[domain][keep]], dtype=torch.long)
        n = len(keep)
        feature_inputs = {'dataset_version': version, 'domain': domain,
                          'attribute_union_hash': metadata['attribute_union_hash'],
                          'normalization_version': NORMALIZATION_VERSION}
        feature, feature_path, _ = cache.get_or_create('features', feature_inputs,
                    lambda: {'x': align_normalize(raw[domain].x, maps[domain], keep)})
        graph, stats = graph_tensors(raw[domain].network, keep)
        node_id = torch.arange(n, dtype=torch.long)
        graph.update(x=feature['x'], node_id=node_id, domain_id=torch.full((n,), domain_id),
                     feature_dim=6775, attribute_union_hash=metadata['attribute_union_hash'],
                     dataset_version=version,
                     source_train_mask=torch.zeros(n, dtype=torch.bool),
                     source_val_mask=torch.zeros(n, dtype=torch.bool),
                     source_unlabeled_mask=torch.ones(n, dtype=torch.bool),
                     target_unlabeled_mask=torch.ones(n, dtype=torch.bool),
                     target_test_mask=torch.ones(n, dtype=torch.bool))
        graph_record = save(destination / domain / 'graph.pt', graph)
        # Complete labels never appear in graph.pt, feature cache, split masks or training views.
        eval_record = save(destination / domain / 'evaluation_only' / 'labels.pt',
                           {'node_id': node_id, 'labels': y})
        node_map_path = destination / domain / 'node_mapping.json'
        write_json(node_map_path, {'node_id': node_id.tolist(),
                                  'raw_row': keep.tolist(), 'raw_node_id': raw[domain].node_ids[keep].tolist()})
        counts = torch.bincount(y, minlength=len(common)).tolist()
        anomalies[domain].update(zero_feature_rows=int((graph['x'].abs().sum(1) == 0).sum()),
                                  removed_nodes=raw[domain].x.shape[0] - n)
        manifest['domains'][domain] = {
            'domain_id': domain_id, 'nodes': n, 'classes': len(common), 'class_counts': counts,
            'features': 6775, 'nonzero_features': int(torch.count_nonzero(graph['x'])),
            'active_feature_columns': int(torch.count_nonzero(graph['x'].abs().sum(0))),
            'graph_statistics': stats, 'anomalies': anomalies[domain], 'graph': graph_record,
            'feature_cache': record(feature_path), 'node_mapping': record(node_map_path),
            'evaluation_labels': eval_record, 'graph_tensor_hash': tensor_hash({
                k: v for k, v in graph.items() if isinstance(v, torch.Tensor)})}
        if min(counts) < 2:
            manifest['domains'][domain]['source_status'] = 'stopped_class_below_two'
            continue
        manifest['domains'][domain]['source_status'] = 'ready'
        for rate in config['label_rates']:
            for seed in [*config['formal_seeds'], config['development_seed']]:
                inputs = {'dataset_version': version, 'source_domain': domain,
                          'label_rate': rate, 'split_seed': seed}
                split, split_path, _ = cache.get_or_create('split', inputs,
                                               lambda: source_split(y.numpy(), rate, seed))
                split_hash = tensor_hash(split)
                folder = destination / domain / 'source_splits' / split_path.stem
                train = split['source_train_mask']
                val = split['source_val_mask']
                manifest['splits'].append({**inputs, 'split_hash': split_hash,
                    'purpose': 'development' if seed == 2026 else 'formal',
                    'masks': record(split_path),
                    'train_labels': save(folder / 'train_labels.pt',
                                         {'node_id': node_id[train], 'labels': y[train]}),
                    'validation_labels': save(folder / 'validation_only' / 'labels.pt',
                                              {'node_id': node_id[val], 'labels': y[val]}),
                    'counts': {k: int(v.sum()) for k, v in split.items()}})
    for source in DOMAINS:
        for target in DOMAINS:
            if source != target:
                manifest['tasks'].append({'direction': f'{source}->{target}', 'source': source,
                    'target': target, 'status': manifest['domains'][source]['source_status'],
                    'source_split_hashes': [s['split_hash'] for s in manifest['splits']
                                            if s['source_domain'] == source]})
    vocab_path = paths['manifests'] / f'attribute_vocabulary_{metadata["attribute_union_hash"]}.json'
    write_json(vocab_path, {**metadata, 'original_column_to_union': maps})
    manifest['attribute_vocabulary'] = record(vocab_path)
    output = paths['manifests'] / f'dataset_manifest_{version}.json'
    write_json(output, manifest)
    # Stable discovery pointer; run manifests must pin the versioned manifest and its SHA256.
    write_json(paths['manifests'] / 'dataset_manifest.json', manifest)
    print(f'Dataset manifest: {output}', flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()
    prepare(Path(__file__).resolve().parents[1], args.config)
