import torch
from torch.nn import functional as F


@torch.no_grad()
def classification_metrics(logits, labels, num_classes):
    if logits.shape != (labels.numel(), num_classes) or labels.numel() == 0:
        raise ValueError('Empty or incompatible validation logits/labels')
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite validation logits')
    predicted = logits.argmax(1)
    matrix = torch.bincount(labels * num_classes + predicted,
                            minlength=num_classes ** 2).reshape(num_classes, num_classes)
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = 2 * matrix.diag().double() / denominator.clamp_min(1)
    return {'macro_f1': f1.mean().item(), 'loss': F.cross_entropy(logits, labels).item(),
            'accuracy': (predicted == labels).double().mean().item()}
