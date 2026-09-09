import copy
from dataclasses import replace
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contracts.data import TargetTrainView
from contracts.training.domain_adaptation import DAConfig
from contracts.training.encoder import WarmupConfig
from data.cache.store import digest, file_hash, read_json, save_tensor, write_json
from data.partitions.metis import load_partition
from data.views.training import load_training_views
from models.classifiers.shared import SharedClassifier
from models.encoders.shared_gcn import SharedGCNEncoder
from training.checkpoints.encoder import build_metadata
from training.checkpoints.domain_adaptation import load_checkpoint
from training.stages.encoder_warmup.trainer import EncoderWarmup
from training.stages.domain_adaptation.trainer import DomainAdaptation
from utils.randomness.state import capture_rng, seed_everything
from verification.smoke.synthetic.encoder_fixture import synthetic_dataset


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seed_everything(10)


@pytest.fixture
def setup(tmp_path):
    data = synthetic_dataset(tmp_path / 'data', sizes=(256, 257))
    manifest, source, target, validation, _ = data
    wc = WarmupConfig(max_epochs=1, min_epochs=1, smoke=True)
    meta = build_metadata(manifest, 'A', 'B', .05, 0, source, target, wc)
    warm = EncoderWarmup(SharedGCNEncoder(), SharedClassifier(3), source, validation,
                         meta, tmp_path / 'encoder', wc)
    warm.fit(target)
    parts = (load_partition(source.graph, 'A', tmp_path / 'cache'), load_partition(target, 'B', tmp_path / 'cache'))
    return data, meta, tmp_path / 'encoder/encoder_best.pt', parts


def make(setup, output, **kwargs):
    data, meta, path, parts = setup
    return DomainAdaptation(SharedGCNEncoder(), SharedClassifier(3), data[1], data[2], data[3],
                            path, meta, *parts, output, DAConfig(max_epochs=3, min_epochs=1, smoke=True), **kwargs)


def test_inherit_joint_gradients_optimizer_and_no_memory(setup, tmp_path):
    data, meta, path, parts = setup
    encoder, classifier = SharedGCNEncoder(), SharedClassifier(3)
    ids = [id(p) for m in (encoder, classifier) for p in m.parameters()]
    trainer = DomainAdaptation(encoder, classifier, data[1], data[2], data[3], path, meta, *parts,
                              tmp_path / 'da', DAConfig(max_epochs=3, min_epochs=1, smoke=True))
    assert ids == [id(p) for m in (trainer.encoder, trainer.classifier) for p in m.parameters()]
    parent = torch.load(path, weights_only=True)
    equal(encoder.state_dict(), parent['shared_gcn_state'])
    equal(classifier.state_dict(), parent['classifier_state'])
    assert set(trainer.modules) == {'shared_gcn', 'attribute_structure', 'classifier', 'discriminator'}
    assert not hasattr(trainer, 'memory')
    params = [p for m in trainer.modules.values() for p in m.parameters()]
    optimizer_ids = [id(p) for g in trainer.optimizer.param_groups for p in g['params']]
    assert set(optimizer_ids) == set(map(id, params)) and len(optimizer_ids) == len(params)
    assert trainer.scheduler.base_lrs == [5e-4, 1e-3, 1e-3, 1e-3]
    assert all(g['weight_decay'] == 5e-4 for g in trainer.optimizer.param_groups)
    before = {n: copy.deepcopy(m.state_dict()) for n, m in trainer.modules.items()}
    step = trainer.train_step(2)  # positive adversarial weight
    assert step['domain_loss'] > 0
    assert step['train_loss'] == pytest.approx(step['classification_loss']+step['lambda_adv']*step['domain_loss']+1e-4*step['sparsity_loss'])
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    for name, module in trainer.modules.items():
        assert any(not torch.equal(before[name][k], v) for k, v in module.state_dict().items())
    assert trainer.attribute_structure.attribute.selection.rho.grad.abs() > 0
    assert trainer.attribute_structure.structure.selection.rho.grad.abs() > 0


def test_exact_resume_latest_best_rng_and_parent(setup, tmp_path):
    seed_everything(20)
    full = make(setup, tmp_path / 'full')
    full.fit()
    reference = torch.load(tmp_path / 'full/da_latest.pt', weights_only=True)
    seed_everything(20)
    first = make(setup, tmp_path / 'resume')
    first.fit(until_epoch=1)
    random.random(), np.random.rand(), torch.rand(9)
    second = make(setup, tmp_path / 'resume')
    second.resume(tmp_path / 'resume/da_latest.pt')
    second.fit()
    equal(reference, torch.load(tmp_path / 'resume/da_latest.pt', weights_only=True))
    best = torch.load(tmp_path / 'resume/da_best.pt', weights_only=True)
    second.resume(tmp_path / 'resume/da_best.pt')
    equal(best['optimizer_state'], second.optimizer.state_dict())
    equal(best['scheduler_state'], second.scheduler.state_dict())
    equal(best['rng_state'], capture_rng())
    for name, module in second.modules.items():
        equal(best['model_state'][name], module.state_dict())
    assert best['metadata']['parent_checkpoint']['sha256'] == file_hash(setup[2])


def test_target_sentinel_and_evaluation_file_changes_do_not_affect_training(setup, tmp_path, monkeypatch):
    data, meta, parent, parts = setup
    class Sentinel(TargetTrainView):
        @property
        def y(self):
            pytest.fail('Target labels accessed')
    original_load = torch.load
    def guarded(path, *a, **kw):
        assert 'evaluation_only' not in str(path)
        return original_load(path, *a, **kw)
    monkeypatch.setattr(torch, 'load', guarded)
    states = []
    for i, labels in enumerate([torch.arange(257)%3, torch.arange(257).flip(0)%3, torch.full((257,), -999)]):
        save_tensor(data[4], labels)
        source, target = load_training_views(data[0], 'A', 'B', .05, 0)
        target = Sentinel(**{k: getattr(target, k) for k in TargetTrainView.__dataclass_fields__})
        altered = ((data[0], source, target, data[3], data[4]), meta, parent, parts)
        seed_everything(21)
        trainer = make(altered, tmp_path / str(i))
        trainer.fit(until_epoch=2)
        states.append(original_load(tmp_path / str(i) / 'da_latest.pt', weights_only=True))
    equal(states[0], states[1])
    equal(states[0], states[2])


@pytest.mark.parametrize('part', ['parent', 'config', 'metis', 'dataset', 'split', 'optimizer', 'scheduler', 'tensor', 'grl'])
def test_checkpoint_rejects_mismatch_before_mutation(setup, tmp_path, part):
    trainer = make(setup, tmp_path / 'da')
    trainer.fit(until_epoch=1)
    path = tmp_path / 'da/da_latest.pt'
    payload = torch.load(path, weights_only=True)
    expected = copy.deepcopy(trainer.metadata)
    if part in ('parent', 'config', 'metis', 'dataset', 'split'):
        if part == 'parent': expected['parent_checkpoint']['sha256'] = 'f'*64
        if part == 'config': expected['resolved_config_hash'] = 'f'*64
        if part == 'metis': expected['metis_hash'] = 'f'*64
        if part == 'dataset': expected['dataset_manifest_hash'] = 'f'*64
        if part == 'split': expected['split_hash'] = 'f'*64
    else:
        if part == 'optimizer': payload['optimizer_state']['state'][0]['exp_avg'] = torch.zeros(1)
        if part == 'scheduler': payload['scheduler_state']['last_epoch'] += 1
        if part == 'tensor': payload['model_state']['attribute_structure']['fusion.weight'] = torch.zeros(2, 2)
        if part == 'grl': payload['grl_state']['coefficient'] = .1
        save_tensor(path, payload)
        record = read_json(path.with_suffix('.json'))
        record['sha256'] = file_hash(path)
        write_json(path.with_suffix('.json'), record)
    before = copy.deepcopy(trainer.encoder.state_dict())
    with pytest.raises(ValueError):
        load_checkpoint(path, trainer.modules, expected, optimizer=trainer.optimizer, scheduler=trainer.scheduler, resume=True)
    equal(before, trainer.encoder.state_dict())


def test_oom_evidence_no_automatic_fallback(setup, tmp_path, monkeypatch):
    trainer = make(setup, tmp_path / 'da')
    def oom(epoch):
        raise RuntimeError('out of memory: synthetic injected OOM for handler test')
    monkeypatch.setattr(trainer, 'train_step', oom)
    with pytest.raises(RuntimeError, match='out of memory'):
        trainer.fit()
    evidence = read_json(tmp_path / 'da/oom_evidence.json')
    assert evidence['automatic_fallback'] is False
    assert evidence['config']['attention_scope'] == 'full_domain_full_batch'
    assert not (tmp_path / 'da/da_best.pt').exists()


def test_source_validation_labels_do_not_change_training_step(setup, tmp_path):
    data, meta, parent, parts = setup
    changed = list(data)
    changed[3] = replace(data[3], validation_y=torch.tensor([2, 0, 1]))
    seed_everything(17)
    a = make(setup, tmp_path / 'a')
    a.train_step(2)
    seed_everything(17)
    b = make((changed, meta, parent, parts), tmp_path / 'b')
    b.train_step(2)
    for name in a.modules:
        equal(a.modules[name].state_dict(), b.modules[name].state_dict())


def test_fit_early_stop_and_maximum_control_without_training(setup, tmp_path, monkeypatch):
    import training.stages.domain_adaptation.trainer as module
    trainer = make(setup, tmp_path / 'mock')
    trainer.config = DAConfig()  # only test loop decisions; no model update or real checkpoint
    calls = []
    def step(epoch):
        calls.append(epoch)
        trainer.optimizer.step()  # no gradients: no training
        return {'grl_progress': 0., 'lambda_adv': 0.}
    monkeypatch.setattr(trainer, 'train_step', step)
    metrics = iter([(.4, 1.), (.4, .9)] + [(.3, .1)]*148)
    def validate():
        f1, loss = next(metrics)
        return {'macro_f1': f1, 'loss': loss}
    monkeypatch.setattr(trainer, 'validate', validate)
    saved = []
    monkeypatch.setattr(module, 'save_checkpoint', lambda path, *args: saved.append((Path(path).stem, args[4])))
    result = trainer.fit()
    assert result['epoch'] == 30 and result['best_epoch'] == 2 and result['stopped_early']
    assert [e for name, e in saved if name == 'da_best'] == [1, 2]
    assert len(calls) == 30
    trainer.fit()
    assert len(calls) == 30
    trainer.epoch = 149
    trainer.early_stop = type(trainer.early_stop)()
    result = trainer.fit()
    assert result['epoch'] == 150 and calls[-1] == 150


def test_resume_refuses_corrupt_best_and_new_directory(setup, tmp_path):
    trainer = make(setup, tmp_path / 'original')
    trainer.fit(until_epoch=1)
    other = make(setup, tmp_path / 'other')
    with pytest.raises(ValueError, match='original run directory'):
        other.resume(tmp_path / 'original/da_latest.pt')
    with (tmp_path / 'original/da_best.pt').open('ab') as f:
        f.write(b'corrupt')
    with pytest.raises(ValueError, match='file hash'):
        trainer.resume(tmp_path / 'original/da_latest.pt')
