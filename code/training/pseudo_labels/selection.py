"""Threshold, per predicted-class cap, then round consistency (P4/P6)."""
import math

import torch


@torch.no_grad()
def select(probability, classifier_logits, node_ids, *, gamma=.9, q=.2, previous=None, refresh_index=0):
    probability = probability.detach().clone()
    classifier_logits = classifier_logits.detach()
    node_ids = node_ids.detach().clone()
    n, classes = probability.shape
    if (not 0 <= gamma <= 1 or not 0 <= q <= 1 or refresh_index < 0
            or classifier_logits.shape != probability.shape or node_ids.shape != (n,)
            or node_ids.dtype != torch.long or node_ids.unique().numel() != n
            or not torch.isfinite(classifier_logits).all()
            or not torch.isfinite(probability).all() or (probability < 0).any()
            or not torch.allclose(probability.sum(-1), probability.new_ones(n), atol=1e-6)):
        raise ValueError('Invalid pseudo-label inputs')
    if (refresh_index == 0) != (previous is None):
        raise ValueError('Refresh requires the previous round predictions')
    confidence, labels = probability.max(-1)
    accepted = torch.zeros(n, dtype=torch.bool, device=probability.device)
    predicted_counts, accepted_counts, caps = [], [], []
    for c in range(classes):
        count = int((labels == c).sum())
        cap = math.floor(q * count)
        candidates = torch.where((labels == c) & (confidence >= gamma))[0]
        # Tie-break by node ID, independent of caller row order.
        candidates = candidates[torch.argsort(node_ids[candidates], stable=True)]
        candidates = candidates[torch.argsort(confidence[candidates], descending=True, stable=True)]
        accepted[candidates[:cap]] = True
        predicted_counts.append(count)
        caps.append(cap)
    if refresh_index:
        previous = previous.detach()
        if (previous.shape != labels.shape or previous.dtype != torch.long
                or (previous < 0).any() or (previous >= classes).any()):
            raise ValueError('Previous predictions shape/dtype mismatch')
        accepted &= (labels == classifier_logits.argmax(-1)) & (labels == previous)
    for c in range(classes):
        accepted_counts.append(int((accepted & (labels == c)).sum()))
    return {'node_ids': node_ids, 'probability': probability, 'labels': labels.detach(),
            'confidence': confidence.detach(), 'mask': accepted.detach(),
            'previous_predictions': None if previous is None else previous.detach().clone(),
            'refresh_index': refresh_index, 'predicted_counts': predicted_counts, 'caps': caps,
            'accepted_counts': accepted_counts,
            'coverage': [a / b if b else 0. for a, b in zip(accepted_counts, predicted_counts)]}
