import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from data.partitions.metis import load_partition
from models.encoders.shared_gcn import SharedGCNEncoder
from models.fusion.attribute_structure import AttributeStructure
from models.classifiers.shared import SharedClassifier
from models.adversarial.domain import DomainDiscriminator
from models.memory.network import MemoryNetwork
from verification.smoke.synthetic.encoder_fixture import synthetic_dataset


def test_all_tensors_both_domains(tmp_path):
    torch.set_num_threads(2)
    _,s,t,_,_=synthetic_dataset(tmp_path/'data',sizes=(256,257))
    assert s.graph.attribute_union_hash==t.attribute_union_hash
    encoder,branches,head,disc,mem=SharedGCNEncoder(),AttributeStructure(),SharedClassifier(3),DomainDiscriminator(),MemoryNetwork(3)
    for m in (encoder,branches,head,disc,mem): m.eval()
    bank=None
    with torch.no_grad():
        for name,graph,is_source in [('A',s.graph,True),('B',t,False)]:
            n=len(graph.x)
            partition=load_partition(graph,name,tmp_path/'cache')
            assert partition.inputs['P']==128 and partition.inputs['metis_seed']==0
            e=encoder(graph.x,graph.normalized_adjacency)
            assert e.h1.shape==(n,256) and e.z.shape==(n,128)
            a=branches(e.z,partition,source=is_source)
            assert all(x.shape==(n,128) for x in a[:4])
            assert torch.cat((a.h_a,a.h_s),-1).shape==(n,256)
            assert head(a.h_as).shape==(n,3) and disc(a.h_as).shape==(n,1)
            if bank is None: bank=(a.h_s[8:16],a.h_as[8:16])
            r=mem(a.h_s,a.h_as,*bank,graph.node_id,torch.arange(8,16),source=is_source)
            for key in ('query','read','fused'): assert getattr(r,key).shape==(n,128)
            for key in ('key','value'): assert getattr(r,key).shape==(8,128)
            for key in ('similarity','attention'): assert getattr(r,key).shape==(n,8)
            assert r.logits.shape==r.probability.shape==(n,3)
            assert all(torch.isfinite(x).all() for x in (*e,*a[:4],r.query,r.key,r.value,r.attention,r.read,r.fused,r.logits))
