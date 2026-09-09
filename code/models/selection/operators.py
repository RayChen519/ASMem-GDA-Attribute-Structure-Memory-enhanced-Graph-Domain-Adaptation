"""Single-head full-domain attention and the P3 soft-threshold operator."""
import torch
from torch import nn
from torch.nn import functional as F


class PostNormAttention(nn.Module):
    heads = 1

    def __init__(self):
        super().__init__()
        self.q = nn.Linear(128, 128, bias=False)
        self.k = nn.Linear(128, 128, bias=False)
        self.v = nn.Linear(128, 128, bias=False)
        self.o = nn.Linear(128, 128, bias=False)
        self.norm1, self.norm2 = nn.LayerNorm(128), nn.LayerNorm(128)
        self.dropout = nn.Dropout(.1)
        self.ffn = nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.Dropout(.1), nn.Linear(512, 128))

    def forward(self, y):
        if y.ndim != 2 or y.shape[1] != 128 or len(y) == 0:
            raise ValueError('Attention requires nonempty [N,128]')
        # SDPA computes the complete, unmasked N x N attention. Backend tiling
        # changes neither the participating nodes nor the mathematical operator.
        q, k, v = [layer(y)[None, None] for layer in (self.q, self.k, self.v)]
        attended = self.o(F.scaled_dot_product_attention(q, k, v, dropout_p=0.)[0, 0])
        h = self.norm1(y + self.dropout(attended))
        return self.norm2(h + self.dropout(self.ffn(h)))


def soft_threshold(x, theta):
    return x.sign() * F.relu(x.abs() - theta)


class Selection(nn.Module):
    def __init__(self):
        super().__init__()
        self.scorer = PostNormAttention()
        self.rho = nn.Parameter(torch.tensor(0.))

    @property
    def theta(self):
        return F.softplus(self.rho)

    def forward(self, y):
        thresholded = soft_threshold(self.scorer(y), self.theta)
        return thresholded * y, thresholded
