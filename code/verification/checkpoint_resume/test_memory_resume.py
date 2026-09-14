import copy
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contracts.data import TargetTrainView
from contracts.training.memory import MemoryConfig
from data.anchors.source import sample_anchors
from data.cache.store import file_hash, read_json, save_tensor, write_json
from training.checkpoints.memory import read_verified, validate
from training.optimization.memory import build
from training.stages.memory_warmup.trainer import MemoryTraining
from utils.randomness.state import seed_everything
from verification.checkpoint_resume.test_domain_adaptation_resume import setup, make, equal, deterministic


@pytest.fixture
def chain(setup, tmp_path):
    da = make(setup, tmp_path / 'da')
    da.fit(until_epoch=1)
    return da, setup[0][1], tmp_path / 'da/da_best.pt', tmp_path


def trainer(chain, stage, output, **kwargs):
    da, source, parent, root = chain
    config = MemoryConfig(stage=stage, max_epochs=3, min_epochs=1, smoke=True, K=8, refresh_interval=1)
    return MemoryTraining(da, source, parent, output, config, root / 'cache', **kwargs)


def prepare_parents(chain):
    root = chain[-1]
    memory = trainer(chain, 'memory_warmup', root / 'memory')
    memory.fit()
    mp = root / 'memory/memory_best.pt'
    source = trainer(chain, 'source_self_training', root / 'source', memory_best=mp)
    source.fit()
    return mp, root / 'source/source_pl_best.pt'


@pytest.mark.parametrize('stage,name', [('memory_warmup', 'memory'),
    ('source_self_training', 'source_pl'), ('dual_domain_finetuning', 'full')])
def test_exact_resume_and_frozen_parameters(chain, stage, name):
    mp, sp = prepare_parents(chain)
    root = chain[-1]
    seed_everything(44)
    full = trainer(chain, stage, root / 'full', memory_best=mp, source_pl_best=sp)
    before = {n: copy.deepcopy(m.state_dict()) for n, m in full.modules.items()}
    full.fit()
    expected = read_verified(root / 'full' / (name + '_latest.pt'))
    for n, m in full.modules.items():
        if not any(p.requires_grad for p in m.parameters()):
            equal(before[n], m.state_dict())
            assert all(p.grad is None for p in m.parameters())
    seed_everything(44)
    first = trainer(chain, stage, root / 'resume', memory_best=mp, source_pl_best=sp)
    first.fit(until_epoch=1)
    torch.rand(20)
    resumed = trainer(chain, stage, root / 'resume', memory_best=mp, source_pl_best=sp)
    resumed.resume(root / 'resume' / (name + '_latest.pt'))
    resumed.fit()
    actual = read_verified(root / 'resume' / (name + '_latest.pt'))
    equal(expected, actual)


def test_anchor_cache_source_degree_shortage(chain):
    da, source, _, root = chain
    a = sample_anchors(source, da.metadata, root / 'cache', K=8)
    b = sample_anchors(source, da.metadata, root / 'cache', K=8)
    equal(a, b)
    assert source.graph.source_unlabeled_mask[a['ids']].all()
    assert a['ids'].unique().numel() == 8
    with pytest.raises(ValueError, match='Insufficient'):
        sample_anchors(source, da.metadata, root / 'cache', K=99999)
    with pytest.raises(TypeError):
        sample_anchors(object(), da.metadata, root / 'cache', K=8)
    altered = replace(source, graph=replace(source.graph, degree=source.graph.degree + 1))
    with pytest.raises(ValueError, match='degree'):
        sample_anchors(altered, da.metadata, root / 'cache', K=8)


def test_epoch20_21_gradients_refresh_and_teacher(chain):
    mp, sp = prepare_parents(chain)
    run = trainer(chain, 'dual_domain_finetuning', chain[-1] / 'boundary', memory_best=mp, source_pl_best=sp)
    # Boundary unit test performs two optimizer steps, never a 21-epoch run.
    run.config = MemoryConfig(stage='dual_domain_finetuning', min_epochs=30, patience=20)
    run.refresh(1)
    teacher = copy.deepcopy(run.memory.state_dict())
    old_ids = run.anchor_state['ids'].clone()
    run.optimizer, run.scheduler = build(run.modules, run.config, epoch=20, completed_epochs=19)
    run.train_step(20)
    assert all(not p.requires_grad and p.grad is None for p in run.da.encoder.parameters())
    run.train_step(21)
    assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all()
               for p in run.da.encoder.parameters())
    old_hash = run.anchor_state['representation_hash']
    run.refresh(31)
    assert run.anchor_state['representation_hash'] != old_hash
    assert torch.equal(old_ids, run.anchor_state['ids'])
    equal(teacher, run.memory.state_dict())
    assert not run.memory.training and all(not p.requires_grad and p.grad is None for p in run.memory.parameters())
    assert run.scheduler.base_lrs == [1e-4, 1e-4, 5e-4, 5e-4]
    assert run.schedule(1)['lambda_t'] == 0
    assert run.schedule(30)['lambda_t'] == .5


def test_compressed_transition_exact_resume(chain):
    mp, sp = prepare_parents(chain)
    root = chain[-1]
    def make_run(folder):
        config = MemoryConfig(stage='dual_domain_finetuning', max_epochs=2, min_epochs=1,
            smoke=True, K=8, refresh_interval=1, unfreeze_epoch=2, target_ramp_epochs=2)
        return MemoryTraining(chain[0], chain[1], chain[2], root/folder, config, root/'cache',
                              memory_best=mp, source_pl_best=sp)
    seed_everything(41)
    full = make_run('compressed_full')
    full.fit()
    expected = read_verified(full.output_dir/'full_latest.pt')
    seed_everything(41)
    partial = make_run('compressed_resume')
    partial.fit(until_epoch=1)
    resumed = make_run('compressed_resume')
    resumed.resume(resumed.output_dir/'full_latest.pt')
    resumed.fit()
    actual = read_verified(resumed.output_dir/'full_latest.pt')
    equal(expected, actual)
    assert resumed.schedule(2)['lambda_t'] == .5
    assert all(p.requires_grad for p in resumed.da.encoder.parameters())
    assert actual['pseudo_label_state']['target']['refresh_index'] == 1


@pytest.mark.parametrize('part', ['anchor', 'representation', 'mask', 'round', 'optimizer', 'scheduler', 'mode', 'config', 'rng', 'teacher'])
def test_reject_corruption_before_mutation(chain, part):
    mp, sp = prepare_parents(chain)
    run = trainer(chain, 'dual_domain_finetuning', chain[-1] / 'corrupt', memory_best=mp, source_pl_best=sp)
    run.fit(until_epoch=1)
    path = run.output_dir / 'full_latest.pt'
    payload = read_verified(path)
    if part == 'anchor': payload['anchor_state']['ids'][0] = 0
    if part == 'representation': payload['anchor_state']['representations']['h_s'][0, 0] += 1
    if part == 'mask': payload['pseudo_label_state']['source']['mask'].logical_not_()
    if part == 'round': payload['pseudo_label_state']['target']['refresh_index'] += 1
    if part == 'optimizer': payload['optimizer_state']['state'][0]['exp_avg'] = torch.zeros(1)
    if part == 'scheduler': payload['scheduler_state']['last_epoch'] += 1
    if part == 'mode': payload['training_modes']['memory.'] = True
    if part == 'config': payload['metadata']['resolved_config']['gamma'] = .1
    if part == 'rng': payload['rng_state']['torch'] = torch.zeros(1, dtype=torch.uint8)
    if part == 'teacher': payload['model_state']['memory']['query.weight'][0, 0] += 1
    save_tensor(path, payload)
    record = read_json(path.with_suffix('.json'))
    record['sha256'] = file_hash(path)
    write_json(path.with_suffix('.json'), record)
    before = copy.deepcopy(run.da.encoder.state_dict())
    with pytest.raises(ValueError):
        run.resume(path)
    equal(before, run.da.encoder.state_dict())


def test_weighted_sampler_matches_plan(chain, monkeypatch):
    da, source, _, root = chain
    original = torch.multinomial
    observed = []
    def spy(weights, count, replacement, generator):
        ids = source.graph.node_id[source.graph.source_unlabeled_mask]
        expected = 2. ** torch.log(source.graph.degree[ids].double() + 1) + 1.
        torch.testing.assert_close(weights, expected)
        assert count == 8 and replacement is False
        observed.append(True)
        return original(weights, count, replacement=replacement, generator=generator)
    monkeypatch.setattr(torch, 'multinomial', spy)
    sample_anchors(source, da.metadata, root / 'weight_cache', K=8)
    assert observed == [True]


def test_dual_rejects_unfinished_source(chain):
    root = chain[-1]
    memory = trainer(chain, 'memory_warmup', root / 'memory')
    memory.fit()
    mp = root / 'memory/memory_best.pt'
    source = trainer(chain, 'source_self_training', root / 'source', memory_best=mp)
    source.fit(until_epoch=1)
    with pytest.raises(ValueError, match='must finish'):
        trainer(chain, 'dual_domain_finetuning', root / 'dual', memory_best=mp,
                source_pl_best=root / 'source/source_pl_best.pt')


def test_nonempty_pseudo_labels_train_classifier_only(chain, monkeypatch):
    mp, _ = prepare_parents(chain)
    run = trainer(chain, 'source_self_training', chain[-1] / 'nonempty', memory_best=mp)
    teacher = copy.deepcopy(run.memory.state_dict())
    original = run.memory_output
    candidate = torch.tensor([[.98, .01, .01]], requires_grad=True).repeat(len(run.source_ids), 1)
    def confident(*args, **kwargs):
        return original(*args, **kwargs)._replace(probability=candidate)
    monkeypatch.setattr(run, 'memory_output', confident)
    result = run.train_step(1)
    assert run.pseudo_state['source']['mask'].any() and result['source_pl_loss'] > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in run.da.classifier.parameters())
    assert all(not p.requires_grad and p.grad is None for p in run.memory.parameters())
    assert all(not t.requires_grad for t in run.pseudo_state['source'].values() if isinstance(t, torch.Tensor))
    equal(teacher, run.memory.state_dict())


def test_target_label_sentinel_all_three_stages(chain, setup, monkeypatch):
    class Sentinel(TargetTrainView):
        @property
        def y(self):
            raise AssertionError('Target labels accessed')
    original_load = torch.load
    def guarded(path, *args, **kwargs):
        assert 'evaluation_only' not in str(path)
        return original_load(path, *args, **kwargs)
    monkeypatch.setattr(torch, 'load', guarded)
    data, meta, encoder, parts = setup
    root = chain[-1]
    runs = []
    for variant, labels in enumerate((torch.arange(257) % 3, torch.randint(3, (257,), generator=torch.Generator().manual_seed(931)),
                                      torch.full((257,), -999))):
        save_tensor(data[4], {'node_id':torch.arange(257), 'labels':labels})
        target = Sentinel(**{k: getattr(data[2], k) for k in TargetTrainView.__dataclass_fields__})
        changed = ((data[0], data[1], target, data[3], data[4]), meta, encoder, parts)
        seed_everything(101)
        da = make(changed, root / str(variant) / 'unused_da')
        variant_chain = da, data[1], chain[2], root / str(variant)
        mp, sp = prepare_parents(variant_chain)
        dual = trainer(variant_chain, 'dual_domain_finetuning', root / str(variant) / 'dual',
                       memory_best=mp, source_pl_best=sp)
        dual.fit()
        states = []
        for path in (mp, sp, dual.output_dir / 'full_best.pt'):
            payload = read_verified(path)
            states.append({k: payload[k] for k in ('model_state', 'anchor_state', 'pseudo_label_state',
                          'refresh_history', 'epoch', 'early_stop_state', 'history', 'rng_state',
                          'optimizer_state', 'scheduler_state', 'requires_grad', 'training_modes', 'grl_state')})
        states.append({k: payload['metadata'][k] for k in ('split_hash', 'anchor_hash', 'attribute_union_hash',
                      'resolved_config_hash', 'partitions', 'data_hash', 'pseudo_node_ids')})
        from training.stages.encoder_warmup.trainer import evaluation_mode
        with evaluation_mode(*dual.modules.values()):
            states.append(dual.da.classifier(dual.da.representations(source=False).h_as).clone())
        runs.append(states)
    equal(runs[0], runs[1])
    equal(runs[0], runs[2])


@pytest.mark.parametrize('stage,bounds', [('memory_warmup', (100, 20, 15)),
    ('source_self_training', (50, 10, 10)), ('dual_domain_finetuning', (100, 30, 20))])
def test_formal_early_stop_loop_without_training(chain, monkeypatch, stage, bounds):
    import training.stages.memory_warmup.trainer as implementation
    run = trainer(chain, 'memory_warmup', chain[-1] / 'early_stop')
    run.config = MemoryConfig(stage=stage, max_epochs=bounds[0], min_epochs=bounds[1], patience=bounds[2])
    calls = []
    def step(epoch):
        calls.append(epoch)
        run.optimizer.step()  # no gradient, no model update
        return {'schedule': run.schedule(epoch)}
    monkeypatch.setattr(run, 'train_step', step)
    metrics = iter([(.5, 1.), (.5, .9)] + [(.4, .1)] * bounds[0])
    def validation():
        f1, loss = next(metrics)
        return {'macro_f1': f1, 'loss': loss}
    monkeypatch.setattr(run, 'validate', validation)
    saved = []
    monkeypatch.setattr(implementation, 'save', lambda path, trainer: saved.append((Path(path).stem, trainer.epoch)))
    result = run.fit()
    assert result['epoch'] == max(bounds[1], 2 + bounds[2])
    assert result['best_epoch'] == 2
    assert [epoch for name, epoch in saved if name.endswith('_best')] == [1, 2]
    length = len(calls)
    run.fit()
    assert len(calls) == length
