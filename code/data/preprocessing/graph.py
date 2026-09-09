import numpy as np
import scipy.sparse as sp
import torch

GRAPH_VERSION = 'induced-simple-undirected-remove-loops-gcn-v1'


def graph_tensors(network, keep):
    original = sp.coo_matrix(network)
    nonzero = original.data != 0
    raw_pairs = original.row[nonzero].astype(np.int64) * original.shape[0] + original.col[nonzero]
    duplicates = len(raw_pairs) - len(np.unique(raw_pairs))
    a = sp.csr_matrix(network)[keep][:, keep].copy()
    if not np.isfinite(a.data).all() or (a.data < 0).any():
        raise ValueError('Invalid adjacency values')
    a.eliminate_zeros()
    stored = a.nnz
    loops = int(np.count_nonzero(a.diagonal()))
    a.sum_duplicates()
    a.data[:] = 1
    a.setdiag(0)
    a.eliminate_zeros()
    a = a.maximum(a.T).tocsr()
    a.sort_indices()
    degree = np.asarray(a.sum(1)).ravel().astype(np.int64)
    coo = a.tocoo()
    edge = torch.from_numpy(np.vstack((coo.row, coo.col)).astype(np.int64))
    with_loops = (a + sp.eye(a.shape[0], dtype=np.float32)).tocoo()
    inv = (degree + 1).astype(np.float64) ** -0.5
    weight = (inv[with_loops.row] * inv[with_loops.col]).astype(np.float32)
    adj = torch.sparse_coo_tensor(
        torch.from_numpy(np.vstack((with_loops.row, with_loops.col)).astype(np.int64)),
        torch.from_numpy(weight), a.shape, check_invariants=True).coalesce()
    return {'edge_index': edge, 'degree': torch.from_numpy(degree),
            'gcn_edge_index': adj.indices(), 'gcn_edge_weight': adj.values()}, {
        'raw_edge_entries': len(raw_pairs), 'raw_duplicate_edge_entries': duplicates,
        'input_edge_entries_retained_nodes': stored, 'removed_self_loops': loops,
        'directed_edges': a.nnz, 'undirected_edges': a.nnz // 2,
        'isolated_nodes': int((degree == 0).sum())}
