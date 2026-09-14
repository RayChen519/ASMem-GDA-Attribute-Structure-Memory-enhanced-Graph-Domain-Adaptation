import torch
from torch.nn import functional as F


def confidence_weighted_loss(logits, state):
    mask = state['mask'].detach().to(logits.device)
    ids = state['node_ids'].detach().to(logits.device)
    if not mask.any():
        return logits.sum() * 0.
    labels = state['labels'].detach().to(logits.device)[mask]
    confidence = state['confidence'].detach().to(logits.device)[mask]
    return (F.cross_entropy(logits[ids[mask]], labels, reduction='none') * confidence).mean()
