import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.memory.network import MemoryNetwork
from training.pseudo_labels.selection import select
from training.losses.pseudo_labels import confidence_weighted_loss


def test_formula_mask_and_gradients():
    torch.manual_seed(1)
    model = MemoryNetwork(3)
    hs, ha, ks, va = [torch.randn(n, 128, requires_grad=True) for n in (5, 5, 2, 2)]
    result = model(hs, ha, ks, va, torch.arange(5), torch.tensor([1, 3]), source=True)
    q, k = F.normalize(model.query(hs), dim=-1), F.normalize(model.key(ks), dim=-1)
    e = q @ k.T / .1
    e[1, 0], e[3, 1] = -torch.inf, -torch.inf
    a = e.softmax(-1)
    value = model.value(va)
    fused = F.layer_norm(ha + model.fusion(torch.cat((ha, a @ value), -1)), (128,))
    torch.testing.assert_close(result.logits, model.classifier(fused))
    assert result.attention[1, 0] == result.attention[3, 1] == 0
    F.cross_entropy(result.logits, torch.arange(5) % 3).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert all(t.grad is not None for t in (hs, ha, ks, va))
    assert set(dict(model.named_parameters())) == {'query.weight', 'key.weight', 'value.weight',
        'fusion.weight', 'classifier.weight', 'classifier.bias'}


def test_all_masked_and_target_id_collision():
    model = MemoryNetwork(2)
    h = torch.randn(1, 128)
    ids = torch.tensor([4])
    source = model(h, h, h, h, ids, ids, source=True)
    assert torch.equal(source.read, torch.zeros_like(h))
    source.logits.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    target = model(h, h, h, h, ids, ids, source=False)
    assert target.attention.item() == 1


def test_cap_before_consistency_and_detach():
    p = torch.tensor([[.99, .01], [.95, .05], [.91, .09], [.1, .9], [.05, .95], [.01, .99]], requires_grad=True)
    ids = torch.arange(6)
    first = select(p, p.flip(-1), ids, q=.5)
    assert first['mask'].tolist() == [True, False, False, False, False, True]
    second = select(p, p.flip(-1), ids, q=.5, previous=first['labels'], refresh_index=1)
    assert not second['mask'].any()
    third = select(p, p, ids, q=.5, previous=first['labels'].flip(0), refresh_index=2)
    assert not third['mask'].any()
    assert all(not t.requires_grad for t in first.values() if isinstance(t, torch.Tensor))
    logits = torch.randn(6, 2, requires_grad=True)
    loss = confidence_weighted_loss(logits, first)
    expected = (F.cross_entropy(logits[[0, 5]], torch.tensor([0, 1]), reduction='none') * .99).mean()
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert p.grad is None
    empty = confidence_weighted_loss(logits, second)
    assert empty.item() == 0
    with pytest.raises(ValueError):
        select(p, p, ids, refresh_index=1)
