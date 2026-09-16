from dataclasses import replace
from pathlib import Path

import torch

from contracts.data import GraphData, SourceTrainView, TargetTrainView
from data.cache.store import file_hash, io_path, read_json


def verified_tensor(root, record):
    root = Path(root).resolve()
    path = (root / record['path']).resolve()
    if not path.is_relative_to(root) or file_hash(path) != record['sha256']:
        raise ValueError(f'Artifact integrity failure: {path}')
    return torch.load(io_path(path), weights_only=True)


def load_manifest(path):
    path = Path(path)
    return path.parents[3], read_json(path)


def split_record(manifest, source, label_rate, seed):
    found = [s for s in manifest['splits'] if s['source_domain'] == source
             and s['label_rate'] == label_rate and s['split_seed'] == seed]
    if len(found) != 1:
        raise ValueError('Split was not prepared; downstream re-splitting is forbidden')
    return found[0]


def load_training_views(manifest_path, source, target, label_rate, seed):
    root, manifest = load_manifest(manifest_path)
    if source == target or not any(t['source'] == source and t['target'] == target
                                  for t in manifest['tasks']):
        raise ValueError('Unknown transfer direction')
    source_view = load_source_training(manifest_path, source, label_rate, seed)
    tg = verified_tensor(root, manifest["domains"][target]["graph"])
    target_view = TargetTrainView(**{key: tg[key] for key in TargetTrainView.__dataclass_fields__})
    return source_view, target_view


def load_source_training(manifest_path, source, label_rate, seed):
    root, manifest = load_manifest(manifest_path)
    split = split_record(manifest, source, label_rate, seed)
    masks = verified_tensor(root, split['masks'])
    sg = GraphData(**verified_tensor(root, manifest['domains'][source]['graph']))
    sg = replace(sg, **masks, target_unlabeled_mask=torch.zeros_like(sg.node_id, dtype=torch.bool),
                 target_test_mask=torch.zeros_like(sg.node_id, dtype=torch.bool))
    labels = verified_tensor(root, split['train_labels'])
    source_view = SourceTrainView(sg, labels['node_id'], labels['labels'], split['split_hash'])
    return source_view
