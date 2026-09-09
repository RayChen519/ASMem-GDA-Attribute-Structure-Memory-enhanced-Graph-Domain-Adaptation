"""Dataset-only final evaluation gate; no training/checkpoint selection logic."""
from dataclasses import dataclass
from pathlib import Path

import torch

from data.cache.store import digest, file_hash, read_json
from data.views.training import load_manifest, verified_tensor

FINAL_CHECKPOINT = {'B0': 'encoder_best', 'B1': 'da_best', 'B2': 'da_best',
                    'B3': 'da_best', 'B4': 'da_best', 'B5': 'source_pl_best',
                    'B6': 'full_best'}


@dataclass(frozen=True, slots=True)
class TargetEvaluationView:
    node_id: torch.Tensor
    y: torch.Tensor
    target_test_mask: torch.Tensor
    run_manifest_hash: str
    checkpoint_hash: str


def load_target_evaluation(manifest_path, run_manifest_path, lock_path):
    """Lock is created by a future run-locking stage, never by this Dataset API.

    It binds the entire run manifest, dataset manifest and final checkpoint bytes.
    The digest is an integrity contract, not an OS security boundary.
    """
    root, dataset = load_manifest(manifest_path)
    run = read_json(run_manifest_path)
    lock = read_json(lock_path)
    payload = {k: v for k, v in lock.items() if k != 'lock_hash'}
    if lock.get('lock_hash') != digest(payload) or lock.get('status') != 'locked':
        raise PermissionError('Final evaluation requires a valid immutable lock')
    if lock.get('run_manifest_sha256') != file_hash(run_manifest_path):
        raise PermissionError('Run manifest is not locked or has changed')
    if run.get('dataset_manifest_sha256') != file_hash(manifest_path):
        raise PermissionError('Run references another dataset manifest')
    if run.get('dataset_version') != dataset['dataset_version']:
        raise PermissionError('Dataset version mismatch')
    expected = FINAL_CHECKPOINT.get(run.get('variant'))
    checkpoint = lock.get('final_checkpoint', {})
    if expected is None or checkpoint.get('name') != expected:
        raise PermissionError('Wrong final checkpoint for variant')
    cp = (Path(run_manifest_path).parent / checkpoint['path']).resolve()
    if not cp.is_relative_to(Path(run_manifest_path).parent.resolve()):
        raise PermissionError('Checkpoint must belong to this run directory')
    if file_hash(cp) != checkpoint.get('sha256'):
        raise PermissionError('Final checkpoint has changed')
    valid = any(t['source'] == run.get('source') and t['target'] == run.get('target')
                for t in dataset['tasks'])
    matches = [s for s in dataset['splits'] if s['source_domain'] == run.get('source')
               and s['label_rate'] == run.get('label_rate') and s['split_seed'] == run.get('split_seed')
               and s['split_hash'] == run.get('split_hash')]
    if not valid or len(matches) != 1:
        raise PermissionError('Run direction or Source split is not locked to Dataset')
    # No true Target labels are opened until every gate above succeeds.
    labels = verified_tensor(root, dataset['domains'][run['target']]['evaluation_labels'])
    return TargetEvaluationView(labels['node_id'], labels['labels'],
                                 torch.ones_like(labels['node_id'], dtype=torch.bool),
                                 lock['run_manifest_sha256'], checkpoint['sha256'])
