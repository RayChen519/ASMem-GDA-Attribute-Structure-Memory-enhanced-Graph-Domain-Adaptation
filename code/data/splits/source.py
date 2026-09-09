import numpy as np
import torch

SPLIT_VERSION = 'stratified-pcg64-round-half-even-80-20-v1'


def source_split(y, label_rate, seed):
    """Python round (ties to even); clamp train budget to [1, labeled-1]."""
    if label_rate not in (0.01, 0.03, 0.05):
        raise ValueError('Only Dataset label rates 1%, 3%, 5% are supported')
    labels = np.asarray(y, dtype=np.int64)
    rng = np.random.Generator(np.random.PCG64(seed))
    train = np.zeros(len(labels), dtype=bool)
    val = train.copy()
    for c in np.unique(labels):
        ids = np.flatnonzero(labels == c)
        if len(ids) < 2:
            raise ValueError(f'Source class {c} has fewer than two nodes; task stopped')
        budget = max(2, round(label_rate * len(ids)))
        count = min(budget - 1, max(1, round(0.8 * budget)))
        selected = rng.permutation(ids)[:budget]
        train[selected[:count]] = True
        val[selected[count:]] = True
    return {k: torch.from_numpy(v) for k, v in {
        'source_train_mask': train, 'source_val_mask': val,
        'source_unlabeled_mask': ~(train | val)}.items()}
