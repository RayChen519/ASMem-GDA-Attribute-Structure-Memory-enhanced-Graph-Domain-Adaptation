import copy
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from contracts.training.domain_adaptation import DAConfig, adversarial_weight
from contracts.training.encoder import EarlyStopState
from data.cache.store import Cache, file_hash, read_json, save_tensor, write_json
from data.partitions.metis import load_partition
from models.adversarial.domain import DomainDiscriminator
from models.fusion.attribute_structure import AttributeStructure, AttributeStructureOutput
from models.selection.operators import PostNormAttention, Selection, soft_threshold
from models.structure.bga import BGABlock
from training.losses.domain_adaptation import classification_loss, domain_loss, sparsity_loss
from verification.smoke.synthetic.encoder_fixture import synthetic_dataset


@pytest.fixture(autouse=True)
def deterministic():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)


def dense_attention(block, y):
    attention = ((y @ block.q.weight.T) @ (y @ block.k.weight.T).T / 128**.5).softmax(1)
    a = attention @ (y @ block.v.weight.T) @ block.o.weight.T
    h = F.layer_norm(y+a, (128,), block.norm1.weight, block.norm1.bias)
    ff = F.linear(F.relu(F.linear(h, block.ffn[0].weight, block.ffn[0].bias)), block.ffn[3].weight, block.ffn[3].bias)
    return F.layer_norm(h+ff, (128,), block.norm2.weight, block.norm2.bias)


def test_selection_exact_formula_postnorm_threshold_and_gradients():
    selection = Selection().double().eval()
    y = torch.randn(11, 128, dtype=torch.double, requires_grad=True)
    score = dense_attention(selection.scorer, y)
    tau = score.sign() * (score.abs() - F.softplus(selection.rho)).clamp_min(0)
    actual, soft = selection(y)
    torch.testing.assert_close(soft, tau)
    torch.testing.assert_close(actual, tau*y)
    (actual.square().mean()+soft.abs().mean()).backward()
    assert selection.theta >= 0 and selection.rho.grad.abs() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in selection.parameters())
    assert selection.scorer.heads == 1
    assert selection.scorer.dropout.p == selection.scorer.ffn[2].p == .1
    assert selection.scorer.ffn[0].weight.shape == (512, 128)
    x = torch.tensor([-2., -.5, 0., .5, 2.], requires_grad=True)
    theta = torch.tensor(1., requires_grad=True)
    torch.testing.assert_close(soft_threshold(x, theta), torch.tensor([-1., 0., 0., 0., 1.]))
    soft_threshold(x, theta).abs().sum().backward()
    assert theta.grad == -2


def test_full_domain_attention_sees_remote_node():
    block = PostNormAttention().eval()
    y = torch.randn(31, 128)
    reference = block(y)
    changed = y.clone()
    changed[-1] += torch.linspace(-5, 5, 128)
    assert not torch.allclose(reference[0], block(changed)[0])
    torch.testing.assert_close(reference, dense_attention(block, y))


def test_metis_cache_keys_graph_integrity_and_order(tmp_path):
    _, source, target, _, _ = synthetic_dataset(tmp_path / 'data', sizes=(256, 257))
    root = tmp_path / 'cache'
    p = load_partition(source.graph, 'A', root)
    assert not p.cache_hit and len(p.counts) == 128 and (p.counts > 0).all()
    again = load_partition(source.graph, 'A', root)
    assert again.cache_hit and torch.equal(p.assignment, again.assignment)
    # Recompute in a fresh cache, same fixed METIS seed.
    fresh = load_partition(source.graph, 'A', tmp_path / 'fresh')
    assert torch.equal(p.assignment, fresh.assignment)
    assert not load_partition(target, 'B', root).cache_hit
    assert not load_partition(source.graph, 'A', root, P=64).cache_hit
    assert not load_partition(source.graph, 'A', root, metis_seed=1).cache_hit
    assert not load_partition(replace(source.graph, dataset_version='new'), 'A', root).cache_hit
    with pytest.raises(ValueError, match='graph/backend'):
        load_partition(replace(source.graph, edge_index=source.graph.edge_index[:, :2]), 'A', root)
    with pytest.raises(ValueError, match='node IDs'):
        load_partition(replace(source.graph, node_id=source.graph.node_id.flip(0)), 'A', root)
    path = Cache(root).path('metis', p.inputs)
    with path.open('ab') as stream:
        stream.write(b'corruption')
    with pytest.raises(ValueError, match='Corrupt cache'):
        load_partition(source.graph, 'A', root)


def test_bga_equations_restore_order_sharing_and_target_bypass(tmp_path):
    _, source, target, _, _ = synthetic_dataset(tmp_path / 'data', sizes=(256, 257))
    sp = load_partition(source.graph, 'A', tmp_path / 'cache')
    tp = load_partition(target, 'B', tmp_path / 'cache')
    model = AttributeStructure().eval()
    z = torch.randn(256, 128)
    seen = []
    hook = model.structure.mlp_s.register_forward_hook(lambda m, a, o: seen.append(id(m[0].weight)))
    out = model(z, sp, source=True)
    bypass = model(torch.randn(257, 128), tp, source=False)
    hook.remove()
    assert seen == [id(model.structure.mlp_s[0].weight)]*2
    assert len([m for m in model.modules() if isinstance(m, BGABlock)]) == 1
    assert model.attribute.selection.rho is not model.structure.selection.rho
    assert set(map(id, model.attribute.selection.parameters())).isdisjoint(map(id, model.structure.selection.parameters()))
    for tensors, n in [(out, 256), (bypass, 257)]:
        assert all(t.shape == (n, 128) and torch.isfinite(t).all() for t in tensors[:4])
    h0 = model.structure.mlp_s(z)
    # Reference indexes original node rows directly, without permutation/RestoreOrder.
    locals_ = [dense_attention(model.structure.bga.intra, h0[sp.assignment == i]) for i in range(128)]
    pooled = torch.stack([h.mean(0) for h in locals_])
    global_ = dense_attention(model.structure.bga.inter, pooled)
    expected = torch.empty_like(h0)
    for i, local in enumerate(locals_):
        expected[sp.assignment == i] = F.linear(torch.cat((local, global_[i].expand(len(local), -1)), 1),
                                               model.structure.bga.broadcast_fusion.weight)
    torch.testing.assert_close(out.h_lg, expected, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(out.h_as, F.linear(torch.cat((out.h_a, out.h_s), 1), model.fusion.weight, model.fusion.bias))
    def forbidden(*args):
        pytest.fail('Target scorer called')
    hooks = [m.register_forward_pre_hook(forbidden) for m in (model.attribute.selection, model.structure.selection)]
    target_z = torch.randn(257, 128)
    bypass = model(target_z, tp, source=False)
    for hook in hooks:
        hook.remove()
    assert bypass.h_a is target_z and bypass.h_s is bypass.h_lg
    assert bypass.attribute_thresholded is bypass.structure_thresholded is None


def test_losses_use_soft_threshold_and_source_train_ids_only():
    h = torch.randn(8, 128, requires_grad=True)
    a, s = torch.randn_like(h), torch.randn_like(h)
    out = AttributeStructureOutput(h*10, h*20, h*30, h, a, s)
    torch.testing.assert_close(sparsity_loss(out), (a.abs().sum()+s.abs().sum())/(8*128))
    classifier = nn.Linear(128, 3)
    ids, labels = torch.tensor([1, 4]), torch.tensor([0, 2])
    classification_loss(classifier, out, ids, labels).backward()
    assert torch.count_nonzero(h.grad[[0, 2, 3, 5, 6, 7]]) == 0
    assert h.grad[ids].abs().sum() > 0


@pytest.mark.parametrize('weight', [1., .037])
def test_single_grl_and_weight_applied_once(weight):
    d = DomainDiscriminator().double()
    h = torch.randn(13, 128, dtype=torch.double, requires_grad=True)
    labels = torch.arange(13, dtype=torch.double).remainder(2)[:, None]
    plain = F.binary_cross_entropy_with_logits(d.layers(h), labels)*weight
    plain.backward()
    fg, dg = h.grad.clone(), [p.grad.clone() for p in d.parameters()]
    h.grad = None
    d.zero_grad()
    (F.binary_cross_entropy_with_logits(d(h), labels)*weight).backward()
    torch.testing.assert_close(h.grad, -fg)
    for p, gradient in zip(d.parameters(), dg):
        torch.testing.assert_close(p.grad, gradient)


def test_formal_early_stop_and_schedule_without_formal_training():
    config = DAConfig()
    assert (config.max_epochs, config.min_epochs, config.patience) == (150, 30, 20)
    state = EarlyStopState()
    assert state.update(1, .4, 1.)
    assert state.update(2, .4, .9)
    for epoch in range(3, 30):
        assert not state.update(epoch, .3, .1)
        assert not state.should_stop(epoch, config)
    state.update(30, .3, .1)
    assert state.should_stop(30, config)
    assert state.best_epoch == 2
    assert state.update(31, .5, 3.) and state.bad_epochs == 0
    for epoch in range(32, 51):
        state.update(epoch, .4, 1.)
        assert not state.should_stop(epoch, config)
    state.update(51, .4, 1.)
    assert state.should_stop(51, config)
    assert adversarial_weight(1, config) == (0., 0.)
    assert adversarial_weight(150, config)[1] == pytest.approx(.09999092)
    with pytest.raises(ValueError):
        DAConfig(max_epochs=2)
    with pytest.raises(ValueError):
        DAConfig(max_epochs=4, min_epochs=1, smoke=True)


def test_config_files_and_full_optimizer_schedule():
    from training.optimization.domain_adaptation import build_optimizer
    from training.schedules.warmup_cosine import build_scheduler
    from models.classifiers.shared import SharedClassifier
    from models.encoders.shared_gcn import SharedGCNEncoder
    base = Path(__file__).resolve().parents[2] / 'configs'
    config = DAConfig(**read_json(base / 'base/domain_adaptation.json'))
    assert config == DAConfig()
    smoke = {**read_json(base / 'base/domain_adaptation.json'), **read_json(base / 'overrides/smoke/domain_adaptation.json')}
    assert DAConfig(**smoke).max_epochs == 2
    modules = {'shared_gcn': SharedGCNEncoder(), 'attribute_structure': AttributeStructure(),
               'classifier': SharedClassifier(3), 'discriminator': DomainDiscriminator()}
    optimizer = build_optimizer(modules, config)
    scheduler = build_scheduler(optimizer, config)
    rates = []
    for _ in range(150):
        rates.append(optimizer.param_groups[0]['lr']/config.encoder_lr)
        optimizer.step()  # no gradients; schedule-only unit test
        scheduler.step()
    assert rates[:5] == pytest.approx([.2, .4, .6, .8, 1.])
    assert rates[-1] == pytest.approx(.1)
    assert all(x > y for x, y in zip(rates[4:-1], rates[5:]))
