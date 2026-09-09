"""Stage-1 checkpoint format; no Target labels or later-stage state is created."""
from dataclasses import asdict
from pathlib import Path

import torch

from contracts.training.encoder import EarlyStopState
from data.cache.store import digest, file_hash, io_path, read_json, save_tensor, write_json
from data.prepare import tensor_hash
from utils.randomness.state import capture_rng, restore_rng


def build_metadata(manifest_path, source, target, label_rate, seed, source_view, target_view, config):
    manifest = read_json(manifest_path)
    if (manifest['feature_dim'] != 6775
            or manifest['dataset_version'] != source_view.graph.dataset_version
            or manifest['dataset_version'] != target_view.dataset_version
            or manifest['attribute_union_hash'] != source_view.graph.attribute_union_hash
            or manifest['attribute_union_hash'] != target_view.attribute_union_hash):
        raise ValueError('Dataset identity or feature vocabulary mismatch')
    matches = [s for s in manifest['splits'] if s['source_domain'] == source
               and s['label_rate'] == label_rate and s['split_seed'] == seed
               and s['split_hash'] == source_view.split_hash]
    if len(matches) != 1 or not any(t['source'] == source and t['target'] == target
                                   for t in manifest['tasks']):
        raise ValueError('Dataset direction/split mismatch')
    for name, graph in [(source, source_view.graph), (target, target_view)]:
        if not torch.all(graph.domain_id == manifest['domains'][name]['domain_id']):
            raise ValueError('Graph domain identity differs from manifest direction')
    return {'stage': 'encoder_warmup', 'source': source, 'target': target,
            'label_rate': label_rate, 'seed': seed,
            'raw_feature_dim': 6775, 'hidden_dim': 256, 'output_dim': 128,
            'num_classes': len(manifest['class_map']),
            'attribute_union_hash': manifest['attribute_union_hash'],
            'dataset_version': manifest['dataset_version'],
            'dataset_manifest_hash': file_hash(manifest_path),
            'split_hash': source_view.split_hash,
            'source_train_mask_hash': tensor_hash({'mask': source_view.graph.source_train_mask}),
            'source_val_mask_hash': tensor_hash({'mask': source_view.graph.source_val_mask}),
            'resolved_config': config.resolved(), 'resolved_config_hash': digest(config.resolved()),
            'parent_checkpoint': None, 'metis_hash': None, 'anchor_hash': None}


def validate_metadata(metadata, encoder, classifier):
    required = {'stage', 'source', 'target', 'label_rate', 'seed', 'raw_feature_dim',
                'hidden_dim', 'output_dim', 'num_classes', 'attribute_union_hash',
                'dataset_version', 'dataset_manifest_hash', 'split_hash',
                'source_train_mask_hash', 'source_val_mask_hash', 'resolved_config',
                'resolved_config_hash', 'parent_checkpoint', 'metis_hash', 'anchor_hash'}
    if not required <= metadata.keys():
        raise ValueError('Missing checkpoint metadata fields')
    if (metadata['stage'] != 'encoder_warmup'
            or (metadata['raw_feature_dim'], metadata['hidden_dim'], metadata['output_dim'])
            != (6775, 256, 128) or metadata['num_classes'] != classifier.num_classes
            or tuple(encoder.gcn1.weight.shape) != (6775, 256)
            or tuple(encoder.gcn2.weight.shape) != (256, 128)):
        raise ValueError('Checkpoint dimension/class/stage mismatch')
    if digest(metadata['resolved_config']) != metadata['resolved_config_hash']:
        raise ValueError('Resolved config hash mismatch')
    for key in required:
        if key.endswith('_hash') and key not in {'metis_hash', 'anchor_hash'}:
            value = metadata[key]
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError(f'Invalid {key}')
    if any(metadata[k] is not None for k in ('parent_checkpoint', 'metis_hash', 'anchor_hash')):
        raise ValueError('Stage 1 has no parent, METIS or anchor state')


def named_modules(encoder, classifier):
    return {'shared_gcn': encoder, 'classifier': classifier}


def save_checkpoint(path, encoder, classifier, optimizer, scheduler, early_stop, epoch, metadata, history):
    validate_metadata(metadata, encoder, classifier)
    modules = named_modules(encoder, classifier)
    payload = {'schema_version': 'encoder-v1', 'stage': 'encoder_warmup', 'epoch': epoch,
               'metadata': metadata, 'shared_gcn_state': encoder.state_dict(),
               'classifier_state': classifier.state_dict(),
               'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(),
               'early_stop_state': asdict(early_stop), 'rng_state': capture_rng(),
               'requires_grad': {f'{m}.{n}': p.requires_grad for m, module in modules.items()
                                 for n, p in module.named_parameters()},
               'training_modes': {f'{m}.{n}': sub.training for m, module in modules.items()
                                  for n, sub in module.named_modules()},
               'history': history}
    save_tensor(path, payload)
    record = {'name': Path(path).stem, 'path': Path(path).name, 'sha256': file_hash(path),
              'epoch': epoch, 'metadata_hash': digest(metadata)}
    write_json(Path(path).with_suffix('.json'), record)
    return record


def load_checkpoint(path, encoder, classifier, expected_metadata, *, optimizer=None,
                    scheduler=None, resume=False, expected_sha256=None):
    """Load into existing objects. DA uses resume=False and builds its own optimizer.

    Exact Stage-1 resume additionally restores optimizer/scheduler/RNG. The
    returned parent_checkpoint record can be pinned by the later DA checkpoint.
    Checkpoints are local trusted artifacts; weights_only avoids arbitrary pickle.
    """
    validate_metadata(expected_metadata, encoder, classifier)
    record = read_json(Path(path).with_suffix('.json'))
    actual_hash = file_hash(path)
    if actual_hash != record['sha256'] or (expected_sha256 is not None and actual_hash != expected_sha256):
        raise ValueError('Checkpoint file hash mismatch')
    payload = torch.load(io_path(path), map_location='cpu', weights_only=True)
    if payload['schema_version'] != 'encoder-v1' or payload['stage'] != 'encoder_warmup':
        raise ValueError('Unsupported checkpoint schema/stage')
    if payload['metadata'] != expected_metadata or record['metadata_hash'] != digest(expected_metadata):
        raise ValueError('Checkpoint metadata/hash mismatch')
    validate_metadata(payload['metadata'], encoder, classifier)
    modules = named_modules(encoder, classifier)
    params = {f'{m}.{n}': p for m, module in modules.items() for n, p in module.named_parameters()}
    modes = {f'{m}.{n}': sub for m, module in modules.items() for n, sub in module.named_modules()}
    if (params.keys() != payload['requires_grad'].keys()
            or modes.keys() != payload['training_modes'].keys()
            or any(type(v) is not bool for v in [*payload['requires_grad'].values(), *payload['training_modes'].values()])):
        raise ValueError('Checkpoint parameter/mode contract mismatch')
    for name, module in modules.items():
        state = payload[name + '_state']
        live = module.state_dict()
        if state.keys() != live.keys() or any(state[k].shape != live[k].shape
                or state[k].dtype != live[k].dtype or not torch.isfinite(state[k]).all() for k in live):
            raise ValueError('Checkpoint model tensor mismatch/nonfinite state')
    early = EarlyStopState(**payload['early_stop_state'])
    if not 1 <= early.best_epoch <= payload['epoch'] or record['epoch'] != payload['epoch']:
        raise ValueError('Checkpoint epoch/early-stop mismatch')
    if resume:
        if optimizer is None or scheduler is None or scheduler.optimizer is not optimizer:
            raise ValueError('Resume requires the matching optimizer and scheduler')
        old_groups = payload['optimizer_state']['param_groups']
        if len(old_groups) != len(optimizer.param_groups):
            raise ValueError('Optimizer groups mismatch')
        for old, new, name in zip(old_groups, optimizer.param_groups, modules):
            expected_params = [p for n, p in modules[name].named_parameters()
                               if payload['requires_grad'][f'{name}.{n}']]
            if (old.get('name') != name or new.get('name') != name
                    or len(old['params']) != len(new['params'])
                    or list(map(id, new['params'])) != list(map(id, expected_params))):
                raise ValueError('Optimizer parameter identity/order mismatch')
            for index, p in zip(old['params'], expected_params):
                state = payload['optimizer_state']['state'].get(index, {})
                if any(k not in state or state[k].shape != p.shape or not torch.isfinite(state[k]).all()
                       for k in ('exp_avg', 'exp_avg_sq')):
                    raise ValueError('Optimizer moment tensor mismatch')
        scheduled = payload['scheduler_state']
        if (scheduled.get('last_epoch') != payload['epoch']
                or scheduled.get('base_lrs') != scheduler.base_lrs
                or scheduled.get('_last_lr') != [g['lr'] for g in old_groups]):
            raise ValueError('Scheduler epoch/learning-rate mismatch')
        cuda = payload['rng_state']['cuda']
        if len(cuda) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
            raise ValueError('CUDA RNG topology differs; exact resume is unavailable')
    for name, module in modules.items():
        module.load_state_dict(payload[name + '_state'], strict=True)
    for name, p in params.items():
        p.requires_grad_(payload['requires_grad'][name])
        p.grad = None
    for name, sub in modes.items():
        sub.training = payload['training_modes'][name]
    if resume:
        optimizer.load_state_dict(payload['optimizer_state'])
        scheduler.load_state_dict(payload['scheduler_state'])
        restore_rng(payload['rng_state'])
    return {'epoch': payload['epoch'], 'early_stop': early, 'history': payload['history'],
            'parent_checkpoint': {'path': str(Path(path).resolve()), 'sha256': actual_hash,
                                  'name': Path(path).stem}}
