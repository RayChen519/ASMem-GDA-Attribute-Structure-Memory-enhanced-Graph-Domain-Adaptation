import numpy as np
import scipy.sparse as sp
import torch

from data.cache.store import digest

NORMALIZATION_VERSION = 'l1-absolute-row-float64-to-float32-zero-preserved-v1'


def validate_vocabulary(metadata, dimensions, raw_hashes):
    vocabulary = metadata['vocabulary']
    if len(vocabulary) != 6775 or len(set(vocabulary)) != 6775:
        raise ValueError('The attribute union must contain exactly 6775 distinct IDs')
    if digest(vocabulary) != metadata['attribute_union_hash']:
        raise ValueError('Vocabulary hash mismatch')
    seen = set()
    mappings = {}
    for domain, width in dimensions.items():
        entry = metadata['domains'][domain]
        if entry['raw_sha256'] != raw_hashes[domain]:
            raise ValueError(f'{domain}: column metadata is bound to another raw file')
        ids = entry['attribute_ids']
        if len(ids) != width or len(set(ids)) != len(ids):
            raise ValueError(f'{domain}: missing or duplicate attribute IDs')
        lookup = {name: i for i, name in enumerate(vocabulary)}
        if not set(ids) <= lookup.keys():
            raise ValueError(f'{domain}: unknown attribute IDs')
        mappings[domain] = [lookup[name] for name in ids]
        seen.update(ids)
    if seen != set(vocabulary):
        raise ValueError('Unsubstantiated columns: vocabulary is not the observed ID union')
    return mappings


def align_normalize(x, mapping, keep):
    x = sp.csr_matrix(x, dtype=np.float64)[keep]
    coo = x.tocoo()
    aligned = sp.csr_matrix((coo.data, (coo.row, np.asarray(mapping)[coo.col])),
                            shape=(x.shape[0], 6775))
    if not np.isfinite(aligned.data).all() or (aligned.data < 0).any():
        raise ValueError('Invalid feature values')
    norm = np.asarray(abs(aligned).sum(1)).ravel()
    scale = np.divide(1., norm, out=np.zeros_like(norm), where=norm != 0)
    result = (sp.diags(scale) @ aligned).toarray().astype(np.float32)
    return torch.from_numpy(result)
