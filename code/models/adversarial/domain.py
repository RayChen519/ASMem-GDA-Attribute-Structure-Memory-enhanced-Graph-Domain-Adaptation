import torch
from torch import nn


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, gradient):
        return -gradient


def reverse_gradient(x):
    """Unit GRL: the scheduled coefficient occurs only in the total loss."""
    return _Reverse.apply(x)


class DomainDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Plan fixes widths; use ReLU between the two linear layers.
        self.layers = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, h):
        return self.layers(reverse_gradient(h))
