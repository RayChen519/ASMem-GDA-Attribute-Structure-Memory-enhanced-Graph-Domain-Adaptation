import copy
from dataclasses import replace
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contracts.training.encoder import EarlyStopState, WarmupConfig
from data.cache.store import digest, file_hash, read_json, save_tensor, write_json
from data.views.training import load_training_views
from models.classifiers.shared import SharedClassifier
from models.encoders.shared_gcn import NormalizedGCN, SharedGCNEncoder
from training.checkpoints.encoder import build_metadata, load_checkpoint, save_checkpoint
from training.optimization.encoder import build_optimizer
from training.schedules.warmup_cosine import build_scheduler
from training.stages.encoder_warmup.trainer import EncoderWarmup, inspect_target
from utils.randomness.state import capture_rng, seed_everything
from verification.smoke.synthetic.encoder_fixture import synthetic_dataset


@pytest.fixture(autouse=True)
def deterministic():
    threads = torch.get_num_threads()
    enabled = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    seed_everything(0)
    yield
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(enabled)


@pytest.fixture
def dataset(tmp_path):
    return synthetic_dataset(tmp_path / 'code')


def make_trainer(dataset, output, config=None):
    path, source, target, validation, _ = dataset
    config = config or WarmupConfig(max_epochs=4, min_epochs=1, smoke=True)
    metadata = build_metadata(path, 'A', 'B', .05, 0, source, target, config)
    return EncoderWarmup(SharedGCNEncoder(), SharedClassifier(3), source, validation,
                         metadata, output, config)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


def test_shared_parameters_shapes_initialization_and_exact_adjacency(dataset):
    _, source, target, _, _ = dataset
    encoder = SharedGCNEncoder().eval()
    seen = []
    hook = encoder.gcn1.register_forward_hook(lambda m, a, o: seen.append(id(m.weight)))
    before = source.graph.normalized_adjacency.to_dense().clone()
    s, t = encoder.forward_pair(source.graph.x, source.graph.normalized_adjacency,
                                target.x, target.normalized_adjacency)
    hook.remove()
    assert seen == [id(encoder.gcn1.weight)] * 2
    assert len([m for m in encoder.modules() if isinstance(m, NormalizedGCN)]) == 2
    assert not any(isinstance(m, nn.Linear) for m in encoder.modules())
    for output, n in [(s, 9), (t, 7)]:
        assert output.h1.shape == (n, 256) and output.z.shape == (n, 128)
        assert torch.isfinite(output.h1).all() and torch.isfinite(output.z).all()
        assert (output.z < 0).any()
    assert torch.equal(before, source.graph.normalized_adjacency.to_dense())
    # Independent dense matrix equation detects loop/normalization duplication,
    # bias placement and an accidental final ReLU/additional propagation layer.
    h1 = F.relu(F.layer_norm(before @ (source.graph.x @ encoder.gcn1.weight), (256,)))
    z = F.layer_norm(before @ (h1 @ encoder.gcn2.weight), (128,))
    torch.testing.assert_close(s.h1, h1)
    torch.testing.assert_close(s.z, z)
    for gcn in [encoder.gcn1, encoder.gcn2]:
        assert torch.count_nonzero(gcn.bias) == 0
        assert gcn.weight.abs().max() <= (6 / sum(gcn.weight.shape)) ** .5
    classifier = SharedClassifier(3)
    assert [type(m) for m in classifier.layers] == [nn.Linear, nn.ReLU, nn.Dropout, nn.Linear]
    assert classifier.layers[0].weight.shape == (64, 128)
    assert classifier.layers[2].p == encoder.dropout.p == .5
    # Deliberately non-stochastic matrix checks that supplied weights are used verbatim.
    adjacency = torch.tensor([[.2, .7], [.4, .3]]).to_sparse()
    layer = NormalizedGCN(2, 2)
    layer.bias.data.fill_(.5)
    x = torch.eye(2)
    torch.testing.assert_close(layer(x, adjacency), adjacency.to_dense() @ layer.weight + .5)


def test_classifier_accepts_any_positive_dataset_class_count():
    assert SharedClassifier(1)(torch.zeros(2, 128)).shape == (2, 1)
    for count in [0, -1, True, 2.5]:
        with pytest.raises(ValueError, match='positive integer'):
            SharedClassifier(count)


def test_source_only_gradients_and_validation_isolation(dataset, tmp_path):
    trainer = make_trainer(dataset, tmp_path / 'train')
    zs = []
    hook = trainer.encoder.register_forward_hook(lambda m, a, out: (out.z.retain_grad(), zs.append(out.z)) and None)
    trainer.train_step()
    hook.remove()
    mask = torch.ones(9, dtype=torch.bool)
    mask[trainer.train_ids] = False
    assert torch.count_nonzero(zs[0].grad[mask]) == 0
    assert torch.count_nonzero(zs[0].grad[~mask]) > 0
    for module in [trainer.encoder, trainer.classifier]:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
    assert torch.linalg.vector_norm(torch.stack([p.grad.norm() for module in
           [trainer.encoder, trainer.classifier] for p in module.parameters()])) <= 5.00001
    before = [p.grad.clone() for p in trainer.encoder.parameters()]
    trainer.validate()
    for p, g in zip(trainer.encoder.parameters(), before):
        assert torch.equal(p.grad, g)
    # Validation labels cannot influence even one training update.
    altered = list(dataset)
    altered[3] = replace(dataset[3], validation_y=torch.tensor([2, 0, 1]))
    seed_everything(42)
    a = make_trainer(dataset, tmp_path / 'a')
    a.train_step()
    seed_everything(42)
    b = make_trainer(altered, tmp_path / 'b')
    b.train_step()
    equal(a.encoder.state_dict(), b.encoder.state_dict())
    equal(a.classifier.state_dict(), b.classifier.state_dict())


def test_target_eval_no_grad_sentinel_and_modes(dataset, tmp_path):
    from contracts.data import TargetTrainView
    class SentinelTarget(TargetTrainView):
        @property
        def y(self):
            pytest.fail('Target labels accessed')
    target = SentinelTarget(**{k: getattr(dataset[2], k) for k in TargetTrainView.__dataclass_fields__})
    trainer = make_trainer(dataset, tmp_path)
    trainer.encoder.norm1.eval()  # mixed modes must be preserved exactly
    modes = [m.training for m in trainer.encoder.modules()]
    rng = capture_rng()
    trace = []
    def observe(m, a, out):
        trace.append((m.training, torch.is_grad_enabled(), out.z.requires_grad))
    hook = trainer.encoder.register_forward_hook(observe)
    report = inspect_target(trainer.encoder, trainer.classifier, target, target.attribute_union_hash)
    hook.remove()
    assert report['z_shape'] == [7, 128] and trace == [(False, False, False)]
    assert modes == [m.training for m in trainer.encoder.modules()]
    equal(rng, capture_rng())
    assert all(p.grad is None for p in trainer.encoder.parameters())
    broken = replace(target, x=target.x * float('nan'))
    with pytest.raises(ValueError, match='Nonfinite'):
        inspect_target(trainer.encoder, trainer.classifier, broken, target.attribute_union_hash)
    assert modes == [m.training for m in trainer.encoder.modules()]


def test_evaluation_file_replacements_leave_entire_training_state_identical(dataset, tmp_path, monkeypatch):
    original_load = torch.load
    def guarded_load(path, *args, **kwargs):
        assert 'evaluation_only' not in str(path), 'Target evaluation artifact was opened'
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(torch, 'load', guarded_load)
    checkpoints = []
    for i, labels in enumerate([torch.arange(7) % 3, torch.tensor([2, 1, 0, 0, 2, 1, 2]), torch.full((7,), -999)]):
        save_tensor(dataset[4], labels)  # no manifest edit: training must never verify/open this file
        source, target = load_training_views(dataset[0], 'A', 'B', .05, 0)
        seed_everything(99)
        trainer = make_trainer((dataset[0], source, target, dataset[3], dataset[4]), tmp_path / str(i))
        trainer.fit(target)
        checkpoints.append(original_load(tmp_path / str(i) / 'encoder_latest.pt', weights_only=True))
    equal(checkpoints[0], checkpoints[1])
    equal(checkpoints[0], checkpoints[2])


def test_early_stop_minimum_patience_tie_break_and_maximum():
    config = WarmupConfig()
    assert (config.max_epochs, config.min_epochs, config.patience) == (100, 20, 15)
    state = EarlyStopState()
    assert state.update(1, .4, 1.)
    assert state.update(2, .4, .9)
    assert not state.update(3, .39, .1)
    assert not state.update(4, .4, .9)
    for epoch in range(5, 20):
        state.update(epoch, .3, 1.)
        assert not state.should_stop(epoch, config)
    state.update(20, .3, 1.)
    assert state.should_stop(20, config)
    assert state.best_epoch == 2
    assert state.update(21, .5, 2.) and state.bad_epochs == 0
    with pytest.raises(ValueError):
        state.update(22, float('nan'), 1.)
    with pytest.raises(ValueError):
        WarmupConfig(max_epochs=2)


def test_optimizer_and_schedule_parameter_groups():
    config = WarmupConfig()
    encoder, classifier = SharedGCNEncoder(), SharedClassifier(3)
    encoder.norm1.bias.requires_grad_(False)
    optimizer = build_optimizer(encoder, classifier, config)
    groups = optimizer.param_groups
    ids = [id(p) for g in groups for p in g['params']]
    assert len(ids) == len(set(ids)) and id(encoder.norm1.bias) not in ids
    assert all(g['weight_decay'] == 5e-4 for g in groups)
    assert [g['lr'] for g in groups] == [5e-4, 1e-3]
    scheduler = build_scheduler(optimizer, config)
    rates = []
    for _ in range(100):
        rates.append(groups[0]['lr'] / config.encoder_lr)
        optimizer.step()
        scheduler.step()
    assert rates[:5] == pytest.approx([.2, .4, .6, .8, 1.])
    assert rates[-1] == pytest.approx(.1)
    assert all(a > b for a, b in zip(rates[4:-1], rates[5:]))


def test_resume_exact_next_step_rng_optimizer_scheduler_and_best_inheritance(dataset, tmp_path):
    seed_everything(20)
    uninterrupted = make_trainer(dataset, tmp_path / 'full')
    uninterrupted.fit(dataset[2])
    reference = torch.load(tmp_path / 'full/encoder_latest.pt', weights_only=True)
    seed_everything(20)
    first = make_trainer(dataset, tmp_path / 'resumed')
    first.fit(dataset[2], until_epoch=2)
    random.random(), np.random.rand(), torch.rand(15)
    second = make_trainer(dataset, tmp_path / 'resumed')
    second.resume(tmp_path / 'resumed/encoder_latest.pt')
    second.fit(dataset[2])
    equal(reference, torch.load(tmp_path / 'resumed/encoder_latest.pt', weights_only=True))
    # Model inheritance preserves both module and Parameter object identities.
    ids = [id(p) for module in [second.encoder, second.classifier] for p in module.parameters()]
    state = load_checkpoint(tmp_path / 'resumed/encoder_best.pt', second.encoder, second.classifier, second.metadata)
    assert ids == [id(p) for module in [second.encoder, second.classifier] for p in module.parameters()]
    assert state['parent_checkpoint']['sha256'] == file_hash(tmp_path / 'resumed/encoder_best.pt')
    assert second.classifier(torch.zeros(5, 128)).shape == (5, 3)
    best = torch.load(tmp_path / 'resumed/encoder_best.pt', weights_only=True)
    equal(second.encoder.state_dict(), best['shared_gcn_state'])
    equal(second.classifier.state_dict(), best['classifier_state'])
    # encoder_best itself also contains a complete resumable state.
    second.resume(tmp_path / 'resumed/encoder_best.pt')
    equal(second.optimizer.state_dict(), best['optimizer_state'])
    equal(second.scheduler.state_dict(), best['scheduler_state'])
    equal(capture_rng(), best['rng_state'])
    expected_draw = (random.random(), np.random.rand(), torch.rand(3))
    random.random(), np.random.rand(), torch.rand(19)
    second.resume(tmp_path / 'resumed/encoder_best.pt')
    equal(expected_draw, (random.random(), np.random.rand(), torch.rand(3)))


def test_trainer_early_stopping_selects_tie_break_checkpoint(dataset, tmp_path, monkeypatch):
    trainer = make_trainer(dataset, tmp_path, WarmupConfig())
    metrics = iter([(.4, 1.), (.4, .9)] + [(.3, .1)] * 98)
    def validation():
        f1, loss = next(metrics)
        return {'macro_f1': f1, 'loss': loss, 'accuracy': 0.}
    monkeypatch.setattr(trainer, 'validate', validation)
    report = trainer.fit(dataset[2])
    assert report['epoch'] == 20 and report['best_epoch'] == 2 and report['stopped_early']
    best = torch.load(tmp_path / 'encoder_best.pt', weights_only=True)
    latest = torch.load(tmp_path / 'encoder_latest.pt', weights_only=True)
    assert best['epoch'] == 2 and latest['epoch'] == 20
    assert best['early_stop_state']['best_loss'] == .9
    trainer.resume(tmp_path / 'encoder_latest.pt')
    assert trainer.fit()['epoch'] == 20


def test_resume_rejects_new_directory_and_target_dataset_version(dataset, tmp_path):
    trainer = make_trainer(dataset, tmp_path / 'original')
    trainer.fit(until_epoch=1)
    moved = make_trainer(dataset, tmp_path / 'new')
    with pytest.raises(ValueError, match='original run directory'):
        moved.resume(tmp_path / 'original/encoder_latest.pt')
    with pytest.raises(ValueError, match='dataset version'):
        trainer.fit(replace(dataset[2], dataset_version='other-release'))


@pytest.mark.parametrize('part', ['optimizer', 'scheduler'])
def test_rejects_incompatible_resume_state_before_mutation(dataset, tmp_path, part):
    trainer = make_trainer(dataset, tmp_path)
    trainer.fit(until_epoch=1)
    path = tmp_path / 'encoder_latest.pt'
    payload = torch.load(path, weights_only=True)
    if part == 'optimizer':
        payload['optimizer_state']['state'][0]['exp_avg'] = torch.zeros(1)
    else:
        payload['scheduler_state']['last_epoch'] += 1
    save_tensor(path, payload)
    record = read_json(path.with_suffix('.json'))
    record['sha256'] = file_hash(path)
    write_json(path.with_suffix('.json'), record)
    before = copy.deepcopy(trainer.encoder.state_dict())
    with pytest.raises(ValueError, match='Optimizer moment|Scheduler epoch'):
        trainer.resume(path)
    equal(before, trainer.encoder.state_dict())


@pytest.mark.parametrize('field', ['raw_feature_dim', 'hidden_dim', 'output_dim', 'num_classes',
    'attribute_union_hash', 'dataset_manifest_hash', 'split_hash', 'source_train_mask_hash',
    'source_val_mask_hash', 'resolved_config_hash', 'dataset_version', 'source', 'target', 'seed', 'label_rate'])
def test_checkpoint_rejects_metadata_mismatch_before_mutation(dataset, tmp_path, field):
    trainer = make_trainer(dataset, tmp_path)
    trainer.fit(until_epoch=1)
    metadata = copy.deepcopy(trainer.metadata)
    value = metadata[field]
    metadata[field] = ('f' * 64 if field.endswith('_hash') else value + '-wrong') if isinstance(value, str) else value + 1
    before = copy.deepcopy(trainer.encoder.state_dict())
    with pytest.raises(ValueError):
        load_checkpoint(tmp_path / 'encoder_best.pt', trainer.encoder, trainer.classifier, metadata)
    equal(before, trainer.encoder.state_dict())


def test_checkpoint_rejects_file_and_tensor_tampering(dataset, tmp_path):
    trainer = make_trainer(dataset, tmp_path)
    trainer.fit(until_epoch=1)
    path = tmp_path / 'encoder_best.pt'
    payload = torch.load(path, weights_only=True)
    payload['shared_gcn_state']['gcn1.weight'] = torch.zeros(10, 256)
    save_tensor(path, payload)
    with pytest.raises(ValueError, match='file hash'):
        load_checkpoint(path, trainer.encoder, trainer.classifier, trainer.metadata)
    record = read_json(path.with_suffix('.json'))
    record['sha256'] = file_hash(path)
    write_json(path.with_suffix('.json'), record)
    with pytest.raises(ValueError, match='tensor mismatch'):
        load_checkpoint(path, trainer.encoder, trainer.classifier, trainer.metadata)


def test_requires_grad_and_mixed_modes_restored(dataset, tmp_path):
    trainer = make_trainer(dataset, tmp_path)
    trainer.fit(until_epoch=1)
    trainer.encoder.norm1.bias.requires_grad_(False)
    trainer.encoder.eval()
    trainer.encoder.dropout.train()
    optimizer = build_optimizer(trainer.encoder, trainer.classifier, trainer.config)
    scheduler = build_scheduler(optimizer, trainer.config)
    save_checkpoint(tmp_path / 'frozen.pt', trainer.encoder, trainer.classifier, optimizer, scheduler,
                    trainer.early_stop, trainer.epoch, trainer.metadata, trainer.history)
    encoder, classifier = SharedGCNEncoder(), SharedClassifier(3)
    load_checkpoint(tmp_path / 'frozen.pt', encoder, classifier, trainer.metadata)
    assert not encoder.norm1.bias.requires_grad and encoder.gcn1.weight.requires_grad
    assert not encoder.training and encoder.dropout.training
    assert all(p.grad is None for p in encoder.parameters())


@pytest.mark.parametrize('problem', ['overlap', 'indices', 'vocabulary', 'split', 'nonfinite', 'width'])
def test_invalid_input_contracts_are_rejected(dataset, tmp_path, problem):
    values = list(dataset)
    source, graph = values[1], values[1].graph
    if problem == 'overlap': graph = replace(graph, source_val_mask=graph.source_train_mask)
    if problem == 'indices': source = replace(source, train_node_id=torch.tensor([1, 3, 6]))
    if problem == 'vocabulary': graph = replace(graph, attribute_union_hash='f' * 64)
    if problem == 'split': values[3] = replace(values[3], split_hash='f' * 64)
    if problem == 'nonfinite': graph = replace(graph, x=graph.x * float('inf'))
    if problem == 'width': graph = replace(graph, x=graph.x[:, :128])
    values[1] = replace(source, graph=graph)
    with pytest.raises(ValueError):
        make_trainer(values, tmp_path)
