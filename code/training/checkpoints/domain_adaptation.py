"""Complete Stage-2 state, with verified encoder_best lineage."""
from dataclasses import asdict
from pathlib import Path

import torch

from contracts.training.encoder import EarlyStopState
from data.cache.store import digest, file_hash, io_path, read_json, save_tensor, write_json
from utils.randomness.state import capture_rng, restore_rng


def validate_parent(metadata):
    parent = metadata['parent_checkpoint']
    if parent['name'] != 'encoder_best' or Path(parent['path']).stem != 'encoder_best':
        raise ValueError('Stage 2 requires encoder_best')
    if file_hash(parent['path']) != parent['sha256']:
        raise ValueError('Parent checkpoint hash mismatch')
    record = read_json(Path(parent['path']).with_suffix('.json'))
    if record['sha256'] != parent['sha256']:
        raise ValueError('Parent sidecar hash mismatch')
    if record['metadata_hash'] != metadata['parent_metadata_hash']:
        raise ValueError('Parent metadata hash mismatch')


def validate_metadata(metadata):
    if metadata['stage'] != 'domain_adaptation' or metadata['anchor_hash'] is not None:
        raise ValueError('Invalid DA stage/anchor state')
    if digest(metadata['resolved_config']) != metadata['resolved_config_hash']:
        raise ValueError('DA config hash mismatch')
    if metadata['metis_hash'] != digest(metadata['partitions']):
        raise ValueError('DA METIS hash mismatch')
    validate_parent(metadata)


def save_checkpoint(path, modules, optimizer, scheduler, early_stop, epoch, metadata, history):
    validate_metadata(metadata)
    payload = {'schema_version': 'da-v1', 'stage': 'domain_adaptation', 'epoch': epoch,
        'metadata': metadata, 'model_state': {n: m.state_dict() for n, m in modules.items()},
        'optimizer_state': optimizer.state_dict(), 'scheduler_state': scheduler.state_dict(),
        'early_stop_state': asdict(early_stop), 'rng_state': capture_rng(), 'history': history,
        'requires_grad': {f'{n}.{k}': p.requires_grad for n, m in modules.items() for k, p in m.named_parameters()},
        'training_modes': {f'{n}.{k}': s.training for n, m in modules.items() for k, s in m.named_modules()},
        'grl_state': {'progress': history[-1]['grl_progress'], 'lambda_adv': history[-1]['lambda_adv'],
                      'coefficient': 1., 'lambda_t': 0.},
        'anchor_state': None, 'pseudo_label_state': None}
    save_tensor(path, payload)
    write_json(Path(path).with_suffix('.json'), {'name': Path(path).stem, 'sha256': file_hash(path),
               'epoch': epoch, 'metadata_hash': digest(metadata)})


def load_checkpoint(path, modules, expected_metadata, *, optimizer=None, scheduler=None, resume=False, validate_only=False):
    validate_metadata(expected_metadata)
    record = read_json(Path(path).with_suffix('.json'))
    if file_hash(path) != record['sha256']:
        raise ValueError('DA checkpoint file hash mismatch')
    payload = torch.load(io_path(path), map_location='cpu', weights_only=True)
    if (payload['schema_version'] != 'da-v1' or payload['stage'] != 'domain_adaptation'
            or payload['metadata'] != expected_metadata or record['metadata_hash'] != digest(expected_metadata)):
        raise ValueError('DA checkpoint metadata/hash mismatch')
    if payload['anchor_state'] is not None or payload['pseudo_label_state'] is not None:
        raise ValueError('Memory state must not exist in DA')
    params = {f'{n}.{k}': p for n, m in modules.items() for k, p in m.named_parameters()}
    modes = {f'{n}.{k}': s for n, m in modules.items() for k, s in m.named_modules()}
    if (params.keys() != payload['requires_grad'].keys() or modes.keys() != payload['training_modes'].keys()
            or not all(v is True for v in payload['requires_grad'].values())
            or any(type(v) is not bool for v in payload['training_modes'].values())
            or payload['model_state'].keys() != modules.keys()):
        raise ValueError('DA module/parameter/mode contract mismatch')
    for name, module in modules.items():
        state, live = payload['model_state'][name], module.state_dict()
        if state.keys() != live.keys() or any(state[k].shape != live[k].shape
                or state[k].dtype != live[k].dtype or not torch.isfinite(state[k]).all() for k in live):
            raise ValueError('DA model tensor mismatch/nonfinite state')
    early = EarlyStopState(**payload['early_stop_state'])
    epoch = payload['epoch']
    if not 1 <= early.best_epoch <= epoch or record['epoch'] != epoch or len(payload['history']) != epoch:
        raise ValueError('DA epoch/history mismatch')
    from contracts.training.domain_adaptation import DAConfig, adversarial_weight
    config = DAConfig(**{k: expected_metadata['resolved_config'][k] for k in DAConfig.__dataclass_fields__})
    progress, weight = adversarial_weight(epoch, config)
    if payload['grl_state'] != {'progress': progress, 'lambda_adv': weight, 'coefficient': 1., 'lambda_t': 0.}:
        raise ValueError('DA GRL progress mismatch')
    if resume:
        if optimizer is None or scheduler is None or scheduler.optimizer is not optimizer:
            raise ValueError('DA resume requires matching optimizer/scheduler')
        old_groups = payload['optimizer_state']['param_groups']
        if len(old_groups) != len(modules) or len(optimizer.param_groups) != len(modules):
            raise ValueError('DA optimizer groups mismatch')
        for (name, module), old, new in zip(modules.items(), old_groups, optimizer.param_groups):
            expected = list(module.parameters())
            if (old['name'] != name or new['name'] != name or len(old['params']) != len(expected)
                    or [id(p) for p in new['params']] != [id(p) for p in expected]):
                raise ValueError('DA optimizer parameter identity/order mismatch')
            for key in ('betas', 'eps', 'weight_decay', 'amsgrad', 'maximize'):
                if old[key] != new[key]:
                    raise ValueError('DA optimizer hyperparameter mismatch')
            for index, p in zip(old['params'], expected):
                state = payload['optimizer_state']['state'].get(index, {})
                if any(k not in state or state[k].shape != p.shape or not torch.isfinite(state[k]).all()
                       for k in ('exp_avg', 'exp_avg_sq')):
                    raise ValueError('DA optimizer moment mismatch')
                if ('step' not in state or state['step'].numel() != 1
                        or not torch.isfinite(state['step']).all() or state['step'].item() != epoch):
                    raise ValueError('DA optimizer step mismatch')
        scheduled = payload['scheduler_state']
        if (scheduled['last_epoch'] != epoch or scheduled['base_lrs'] != scheduler.base_lrs
                or scheduled['_last_lr'] != [g['lr'] for g in old_groups]):
            raise ValueError('DA scheduler mismatch')
        if len(payload['rng_state']['cuda']) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
            raise ValueError('CUDA RNG topology mismatch')
    result = {'epoch': epoch, 'early_stop': early, 'history': payload['history']}
    if validate_only:
        return result
    # All compatibility checks precede mutation of caller-owned modules.
    for name, module in modules.items():
        module.load_state_dict(payload['model_state'][name], strict=True)
    for name, p in params.items():
        p.requires_grad_(payload['requires_grad'][name])
        p.grad = None
    for name, mode in modes.items():
        mode.training = payload['training_modes'][name]
    if resume:
        optimizer.load_state_dict(payload['optimizer_state'])
        scheduler.load_state_dict(payload['scheduler_state'])
        restore_rng(payload['rng_state'])
    return result
