"""Domain-level METIS cache, independent of direction, split and training seed."""
from dataclasses import dataclass
from importlib.metadata import version

import torch

from data.cache.store import Cache
from data.prepare import tensor_hash


@dataclass(frozen=True)
class Partition:
    assignment: torch.Tensor
    permutation: torch.Tensor
    inverse: torch.Tensor
    counts: torch.Tensor
    fingerprint: str
    cache_hit: bool
    inputs: dict

    def validate(self, n):
        p = self.inputs['P']
        a, perm, inv, counts = self.assignment, self.permutation, self.inverse, self.counts
        if (any(t.dtype != torch.long for t in (a, perm, inv, counts))
                or a.shape != (n,) or perm.shape != (n,) or inv.shape != (n,)
                or counts.shape != (p,) or (counts <= 0).any()
                or not torch.equal(perm.sort().values, torch.arange(n))
                or not torch.equal(perm[inv], torch.arange(n))
                or a.min() < 0 or a.max() >= p
                or not torch.equal(torch.bincount(a, minlength=p), counts)
                or not torch.equal(a[perm], torch.arange(p).repeat_interleave(counts))):
            raise ValueError('Invalid METIS partition/order or empty cluster')
        if self.fingerprint != tensor_hash({'assignment': a, 'node_id': torch.arange(n)}):
            raise ValueError('METIS partition hash mismatch')


def load_partition(graph, domain, cache_root, *, P=128, metis_seed=0):
    n = len(graph.node_id)
    if not 1 <= P <= n or not torch.equal(graph.node_id.cpu(), torch.arange(n)):
        raise ValueError('METIS needs N >= P and original contiguous node IDs')
    inputs = dict(dataset_version=graph.dataset_version, domain=domain, P=P, metis_seed=metis_seed)
    graph_hash = tensor_hash({'node_id': graph.node_id.cpu(), 'edges': graph.edge_index.cpu()})

    def factory():
        import pymetis
        edges = graph.edge_index.cpu()
        adjacency = [set() for _ in range(n)]
        for u, v in edges.t().tolist():
            if not 0 <= u < n or not 0 <= v < n:
                raise ValueError('Invalid METIS edge')
            if u != v:
                adjacency[u].add(v)
                adjacency[v].add(u)
        result = pymetis.part_graph(P, adjacency=[sorted(row) for row in adjacency],
                                   options=pymetis.Options(seed=metis_seed))
        return {'assignment': torch.tensor(result.vertex_part, dtype=torch.long),
                'graph_hash': graph_hash, 'backend': 'pymetis', 'backend_version': version('pymetis')}

    value, _, hit = Cache(cache_root).get_or_create('metis', inputs, factory)
    if value['graph_hash'] != graph_hash or value['backend'] != 'pymetis':
        raise ValueError('METIS cache graph/backend mismatch')
    assignment = value['assignment']
    perm = torch.argsort(assignment, stable=True)
    partition = Partition(assignment, perm, torch.argsort(perm), torch.bincount(assignment, minlength=P),
        tensor_hash({'assignment': assignment, 'node_id': graph.node_id.cpu()}), hit, inputs)
    partition.validate(n)
    return partition
