import ast
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from contracts.training.memory import MemoryConfig
from data.cache.store import read_json
from models.encoders.shared_gcn import SharedGCNEncoder
from models.fusion.attribute_structure import AttributeStructure
from models.selection.operators import PostNormAttention
from models.structure.bga import BGABlock


def test_fixed_structure_and_smoke_defaults():
    encoder, model = SharedGCNEncoder(), AttributeStructure()
    assert encoder.gcn1.weight.shape == (6775,256)
    assert encoder.gcn2.weight.shape == (256,128)
    assert list(dict(encoder.named_children())) == ['gcn1','norm1','dropout','gcn2','norm2']
    assert sum(isinstance(m,BGABlock) for m in model.modules()) == 1
    assert sum(isinstance(m,PostNormAttention) for m in model.attribute.modules()) == 1
    for m in model.modules():
        if isinstance(m,PostNormAttention):
            assert m.heads == 1 and m.dropout.p == m.ffn[2].p == .1
            assert m.ffn[0].weight.shape == (512,128)
    for stage in ('memory_warmup','source_self_training','dual_domain_finetuning'):
        formal = MemoryConfig(**read_json(ROOT/'configs/base'/f'{stage}.json'))
        assert (formal.unfreeze_epoch,formal.target_ramp_epochs,formal.refresh_interval,formal.K) == (21,30,10,128)
    with pytest.raises(ValueError):
        MemoryConfig(unfreeze_epoch=2)
    cfg=read_json(ROOT/'configs/overrides/smoke/integration.json')
    assert cfg['unfreeze_epoch']==2 and cfg['refresh_interval']==1 and cfg['K'] < 128


def test_no_training_import_of_target_evaluation():
    for folder in ('training','models','data/views'):
        for path in (ROOT/folder).rglob('*.py'):
            tree=ast.parse(path.read_text(encoding='utf-8-sig'))
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom):
                    assert 'final_target' not in (node.module or '')
                    assert all(a.name != 'TargetEvaluationView' for a in node.names)
                if isinstance(node,ast.Name):
                    assert node.id != 'TargetEvaluationView'
