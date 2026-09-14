"""Deterministic, without-replacement Source-only anchor bank IDs."""
import math

import torch

from contracts.data import SourceTrainView
from data.cache.store import Cache, digest
from data.prepare import tensor_hash


def sample_anchors(source, metadata, cache_root, *, K=128, alpha=2., beta=1.):
    if not isinstance(source, SourceTrainView):
        raise TypeError('Anchor bank requires SourceTrainView')
    graph = source.graph
    eligible = graph.node_id[graph.source_unlabeled_mask].cpu()
    if type(K) is not int or not 1 <= K <= len(eligible):
        raise ValueError('Insufficient Source unlabeled anchors; K is never silently reduced')
    if not math.isfinite(alpha) or not math.isfinite(beta) or alpha <= 0 or beta < 0:
        raise ValueError('Invalid anchor sampling hyperparameters')
    if not torch.equal(graph.node_id.cpu(), torch.arange(len(graph.node_id))):
        raise ValueError('Anchor input node order mismatch')
    edges = graph.edge_index.cpu()
    degree = torch.bincount(edges[0, edges[0] != edges[1]], minlength=len(graph.node_id))
    if not torch.equal(degree.to(graph.degree), graph.degree):
        raise ValueError('Anchor degree must be symmetric graph degree before self loops')
    inputs = dict(dataset_version=graph.dataset_version,
                  direction=[metadata['source'], metadata['target']],
                  label_rate=metadata['label_rate'], split_seed=metadata['seed'], K=K,
                  sampler='weighted_without_replacement_v1', sampler_hparams={'alpha': alpha, 'beta': beta},
                  split_hash=source.split_hash,
                  graph_hash=tensor_hash({'ids': graph.node_id, 'degree': graph.degree,
                                          'eligible': eligible}))
    def factory():
        weights = alpha ** torch.log(degree[eligible].double() + 1) + beta
        if not torch.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError('Invalid anchor weights')
        generator = torch.Generator().manual_seed(metadata['seed'])
        ids = eligible[torch.multinomial(weights, K, replacement=False, generator=generator)]
        return {'ids': ids, 'hash': tensor_hash({'ids': ids}), 'inputs_hash': digest(inputs)}
    bank, _, _ = Cache(cache_root).get_or_create('anchor', inputs, factory)
    ids = bank['ids']
    if (ids.dtype != torch.long or ids.shape != (K,) or ids.unique().numel() != K
            or not torch.isin(ids, eligible).all() or bank['hash'] != tensor_hash({'ids': ids})
            or bank['inputs_hash'] != digest(inputs)):
        raise ValueError('Anchor IDs/hash/source membership mismatch')
    return {**bank, 'inputs': inputs}
