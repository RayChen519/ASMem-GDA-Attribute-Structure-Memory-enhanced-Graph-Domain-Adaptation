"""Stage 3-5 serialization and validation before caller state is mutated."""
from dataclasses import asdict
from pathlib import Path

import torch

from contracts.training.encoder import EarlyStopState
from contracts.training.memory import MemoryConfig, STAGES
from data.cache.store import digest, file_hash, io_path, read_json, save_tensor, write_json
from data.prepare import tensor_hash
from training.optimization.memory import rates
from utils.randomness.state import capture_rng
from experiments.registry import variant


def reference(path):
    path = Path(path).resolve()
    return {'path': str(path), 'name': path.stem, 'sha256': file_hash(path)}


def read_verified(path, *, best=False, ancestors=True):
    path = Path(path)
    record = read_json(path.with_suffix('.json'))
    if file_hash(path) != record['sha256']:
        raise ValueError('Checkpoint file hash mismatch')
    payload = torch.load(io_path(path), map_location='cpu', weights_only=True)
    metadata = payload['metadata']
    if (record['metadata_hash'] != digest(metadata) or record['epoch'] != payload['epoch']
            or digest(metadata['resolved_config']) != metadata['resolved_config_hash']):
        raise ValueError('Checkpoint metadata/epoch/config hash mismatch')
    untrained = metadata.get('untrained_memory') is True and path.stem == 'untrained_memory' and metadata['resolved_config']['variant'] == 'M5' and payload['epoch'] == 0
    if best and not untrained and (not path.stem.endswith('_best') or payload['epoch'] != payload['early_stop_state']['best_epoch']):
        raise ValueError('Stage inheritance requires selected best checkpoint')
    if ancestors:
        parents = metadata.get('parents', {})
        if not parents and metadata.get('parent_checkpoint'):
            parents = {'parent': metadata['parent_checkpoint']}
        for parent in parents.values():
            if Path(parent['path']).stem != parent['name'] or file_hash(parent['path']) != parent['sha256']:
                raise ValueError('Parent checkpoint hash/name mismatch')
            read_verified(parent['path'], best=True)
    return payload


def save(path, trainer):
    payload = {'schema_version': 'memory-stages-v1', 'stage': trainer.config.stage,
               'epoch': trainer.epoch, 'metadata': trainer.metadata,
               'model_state': {n: m.state_dict() for n, m in trainer.modules.items()},
               'requires_grad': {f'{n}.{k}': p.requires_grad for n, m in trainer.modules.items()
                                 for k, p in m.named_parameters()},
               'training_modes': {f'{n}.{k}': s.training for n, m in trainer.modules.items()
                                  for k, s in m.named_modules()},
               'optimizer_state': trainer.optimizer.state_dict(),
               'scheduler_state': trainer.scheduler.state_dict(),
               'early_stop_state': asdict(trainer.early_stop), 'history': trainer.history,
               'rng_state': capture_rng(), 'anchor_state': trainer.anchor_state,
               'pseudo_label_state': trainer.pseudo_state, 'refresh_history': trainer.refresh_history,
               'grl_state': trainer.history[-1]['schedule'] if trainer.history else None}
    save_tensor(path, payload)
    write_json(Path(path).with_suffix('.json'), {'name': Path(path).stem, 'sha256': file_hash(path),
               'epoch': trainer.epoch, 'metadata_hash': digest(trainer.metadata)})


def validate(payload, modules, metadata):
    if (payload['schema_version'] != 'memory-stages-v1' or payload['metadata'] != metadata
            or payload['stage'] != metadata['stage']):
        raise ValueError('Memory checkpoint metadata/stage mismatch')
    config = MemoryConfig(**{k: metadata['resolved_config'][k] for k in MemoryConfig.__dataclass_fields__})
    if config.resolved() != metadata['resolved_config']:
        raise ValueError('Memory configuration contract mismatch')
    epoch = payload['epoch']
    if metadata.get('untrained_memory'):
        if (config.variant != 'M5' or config.stage != 'memory_warmup' or epoch != 0
                or payload['history'] or payload['optimizer_state']['state']
                or payload['pseudo_label_state'] or payload['refresh_history']
                or payload['scheduler_state']['last_epoch'] != 0):
            raise ValueError('Invalid untrained Memory artifact')
        for name,module in modules.items():
            live=module.state_dict(); saved=payload['model_state'][name]
            if live.keys()!=saved.keys() or any(live[k].shape!=saved[k].shape or not torch.isfinite(saved[k]).all() for k in live):
                raise ValueError('Invalid untrained Memory tensors')
        anchor=payload['anchor_state']
        if anchor['hash']!=metadata['anchor_hash'] or tensor_hash({'ids':anchor['ids']})!=anchor['hash']:
            raise ValueError('Invalid untrained Memory anchors')
        return EarlyStopState()
    early = EarlyStopState(**payload['early_stop_state'])
    if not 1 <= early.best_epoch <= epoch <= config.max_epochs or len(payload['history']) != epoch:
        raise ValueError('Memory epoch/history mismatch')
    replay_early = EarlyStopState()
    for number, row in enumerate(payload['history'], 1):
        if row['epoch'] != number:
            raise ValueError('Memory history epoch order mismatch')
        replay_early.update(number, row['source_validation']['macro_f1'], row['source_validation']['loss'])
    if asdict(replay_early) != asdict(early):
        raise ValueError('Memory early-stop history mismatch')
    import math
    dual = config.stage == 'dual_domain_finetuning'
    progress = (epoch - 1) / max(1, config.max_epochs - 1) if dual else 0.
    schedule = {'progress': progress, 'lambda_adv': config.lambda_adv_max *
                (2 / (1 + math.exp(-10 * progress)) - 1) if dual else 0.,
                'lambda_t': config.lambda_t_max * min(1., (epoch - 1) / (config.target_ramp_epochs - 1)) if dual else 0.,
                'coefficient': 1.}
    if payload['grl_state'] != schedule or payload['history'][-1]['schedule'] != schedule:
        raise ValueError('Memory GRL/loss schedule mismatch')
    active = rates(config.stage, epoch, config.unfreeze_epoch)
    expected_flags = {f'{n}.{k}': n in active for n, m in modules.items() for k, p in m.named_parameters()}
    expected_modes = {f'{n}.{k}': n in active for n, m in modules.items() for k, s in m.named_modules()}
    if payload['requires_grad'] != expected_flags or payload['training_modes'] != expected_modes:
        raise ValueError('Memory gradient/mode whitelist mismatch')
    if payload['model_state'].keys() != modules.keys():
        raise ValueError('Memory module contract mismatch')
    for name, module in modules.items():
        live, stored = module.state_dict(), payload['model_state'][name]
        if live.keys() != stored.keys() or any(live[k].shape != stored[k].shape
                or live[k].dtype != stored[k].dtype or not torch.isfinite(stored[k]).all() for k in live):
            raise ValueError('Memory model tensor mismatch')
    if config.stage != 'memory_warmup':
        teacher = read_verified(metadata['parents']['memory_best']['path'], best=True)['model_state']['memory']
        if any(not torch.equal(t, payload['model_state']['memory'][k]) for k, t in teacher.items()):
            raise ValueError('Frozen Memory teacher differs from memory_best')
    anchor = payload['anchor_state']
    if (anchor['hash'] != metadata['anchor_hash']
            or tensor_hash({'ids': anchor['ids']}) != anchor['hash']
            or anchor['inputs'] != metadata['anchor_inputs']
            or anchor['ids'].unique().numel() != config.K):
        raise ValueError('Memory anchor identity mismatch')
    representations = anchor['representations']
    if (set(representations) != {'h_s', 'h_as', 'key', 'value'}
            or any(t.shape != (config.K, 128) or t.requires_grad or not torch.isfinite(t).all()
                   for t in representations.values())
            or tensor_hash(representations) != anchor['representation_hash']):
        raise ValueError('Anchor representation hash/shape mismatch')
    groups = payload['optimizer_state']['param_groups']
    if [g['name'] for g in groups] != list(active):
        raise ValueError('Memory optimizer group mismatch')
    all_indices = []
    for g in groups:
        parameters = list(modules[g['name']].parameters())
        all_indices.extend(g['params'])
        if len(g['params']) != len(parameters) or g['weight_decay'] != config.weight_decay:
            raise ValueError('Memory optimizer parameters/hyperparameters mismatch')
        if (g['betas'] != (.9, .999) or g['eps'] != 1e-8 or g['amsgrad'] or g['maximize']
                or g['initial_lr'] != active[g['name']]):
            raise ValueError('Memory optimizer hyperparameter mismatch')
        for index, p in zip(g['params'], parameters):
            state = payload['optimizer_state']['state'].get(index, {})
            if any(k not in state or state[k].shape != p.shape or not torch.isfinite(state[k]).all()
                   for k in ('exp_avg', 'exp_avg_sq')):
                raise ValueError('Memory optimizer moment mismatch')
            expected_step = epoch - config.unfreeze_epoch + 1 if dual and g['name'] in ('shared_gcn', 'attribute_structure') else epoch
            if ('step' not in state or state['step'].numel() != 1
                    or not torch.isfinite(state['step']).all() or state['step'].item() != expected_step):
                raise ValueError('Memory optimizer step mismatch')
    if len(set(all_indices)) != len(all_indices):
        raise ValueError('Duplicate optimizer parameters')
    scheduler = payload['scheduler_state']
    if (scheduler['last_epoch'] != epoch or scheduler['base_lrs'] != list(active.values())
            or scheduler['_last_lr'] != [g['lr'] for g in groups]):
        raise ValueError('Memory scheduler mismatch')
    # Validate RNG in independent generators before mutating live model/optimizer/RNG.
    import random
    import numpy as np
    rng = payload['rng_state']
    try:
        random.Random().setstate(rng['python'])
        n = rng['numpy']
        np.random.RandomState().set_state((n[0], np.array(n[1], dtype=np.uint32), n[2], n[3], n[4]))
        torch.Generator().set_state(rng['torch'])
        if any(t.dtype != torch.uint8 or t.ndim != 1 for t in rng['cuda']):
            raise ValueError('Invalid CUDA RNG tensor')
    except (ValueError, TypeError, RuntimeError, IndexError, KeyError) as error:
        raise ValueError('Memory RNG state mismatch') from error
    if config.stage == 'memory_warmup' and (payload['pseudo_label_state'] or payload['refresh_history']):
        raise ValueError('Warm-up cannot contain pseudo labels')
    expected_domains = set() if config.stage == 'memory_warmup' else {'source'}
    if config.stage == 'dual_domain_finetuning':
        expected_domains.add('target')
    if set(payload['pseudo_label_state']) != expected_domains:
        raise ValueError('Pseudo-label domain/stage mismatch')
    from training.pseudo_labels.selection import select
    for domain, state in payload['pseudo_label_state'].items():
        if state['node_ids'].tolist() != metadata['pseudo_node_ids'][domain]:
            raise ValueError('Pseudo-label node order/domain mismatch')
        records = [r for r in payload['refresh_history'] if r['domain'] == domain]
        if (not records or [r['refresh_index'] for r in records] != list(range(state['refresh_index'] + 1))
                or records[-1]['epoch'] != state['epoch'] or records[-1]['stage'] != config.stage
                or records[-1]['coverage'] != state['coverage']
                or state['epoch'] != 1 + ((epoch - 1) // config.refresh_interval) * config.refresh_interval
                or state['anchor_representation_hash'] != anchor['representation_hash']
                or records[-1]['anchor_representation_hash'] != anchor['representation_hash']):
            raise ValueError('Pseudo-label refresh history/hash mismatch')
        replay = select(state['probability'], state['classifier_logits'], state['node_ids'],
                        gamma=config.gamma, q=config.q, previous=state['previous_predictions'],
                        refresh_index=state['refresh_index'],
                        consistency=variant(config.variant)['options']['consistency'])
        for key, expected in replay.items():
            actual = state[key]
            if isinstance(expected, torch.Tensor):
                if actual.requires_grad or not torch.equal(actual, expected):
                    raise ValueError('Pseudo-label tensor/filter mismatch')
            elif actual != expected:
                raise ValueError('Pseudo-label coverage/round mismatch')
    return early
