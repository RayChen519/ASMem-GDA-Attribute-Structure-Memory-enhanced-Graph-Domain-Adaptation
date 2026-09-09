from dataclasses import dataclass

import numpy as np
import scipy.io as sio
import scipy.sparse as sp


@dataclass
class RawGraph:
    x: object
    network: object
    group: np.ndarray
    node_ids: np.ndarray


def load_mat(path):
    mat = sio.loadmat(path, spmatrix=True)
    x, a, group = mat['attrb'], mat['network'], mat['group']
    group = group.toarray() if sp.issparse(group) else np.asarray(group)
    n = x.shape[0]
    if a.shape != (n, n) or group.ndim != 2 or group.shape[0] != n:
        raise ValueError('Inconsistent feature / adjacency / label dimensions')
    values = x.data if sp.issparse(x) else x
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Nonfinite or negative bag-of-words features')
    edge_values = a.data if sp.issparse(a) else a
    if not np.isfinite(edge_values).all() or (edge_values < 0).any():
        raise ValueError('Nonfinite or negative adjacency values')
    if not np.isin(group, [0, 1]).all() or (group.sum(1) == 0).any():
        raise ValueError('Invalid or missing class membership')
    ids = np.asarray(mat.get('node_id', np.arange(n))).ravel()
    if len(ids) != n or len(np.unique(ids)) != n:
        raise ValueError('Node IDs must be unique and match feature rows')
    return RawGraph(x, a, group, ids)


def class_labels(raw, class_ids, policy):
    if len(class_ids) != raw.group.shape[1] or len(set(class_ids)) != len(class_ids):
        raise ValueError('Class column mapping is incomplete or duplicated')
    multiple = raw.group.sum(1) > 1
    if policy == 'reject' and multiple.any():
        raise ValueError('Multilabel input requires an explicit conversion policy')
    if policy not in ('reject', 'upstream_argmax', 'single_label_only'):
        raise ValueError('Unknown label policy')
    keep = ~multiple if policy == 'single_label_only' else np.ones(len(multiple), bool)
    return np.asarray(class_ids)[raw.group.argmax(1)], keep, int(multiple.sum())
