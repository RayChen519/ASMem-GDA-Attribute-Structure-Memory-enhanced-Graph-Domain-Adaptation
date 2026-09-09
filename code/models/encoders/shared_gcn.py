"""Two shared GCN layers consuming Dataset's already normalized adjacency."""
import torch
from torch import nn
from torch.nn import functional as F

from contracts.models import EncoderOutput


class NormalizedGCN(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, adjacency):
        # Dataset uses matrix [row, column] indices, not a message-passing
        # edge convention. Do not add loops, normalize, or cache by domain.
        return torch.sparse.mm(adjacency, x @ self.weight) + self.bias


class SharedGCNEncoder(nn.Module):
    raw_feature_dim = 6775
    hidden_dim = 256
    output_dim = 128

    def __init__(self):
        super().__init__()
        self.gcn1 = NormalizedGCN(6775, 256)
        self.norm1 = nn.LayerNorm(256)
        self.dropout = nn.Dropout(0.5)
        self.gcn2 = NormalizedGCN(256, 128)
        self.norm2 = nn.LayerNorm(128)

    def forward(self, x, normalized_adjacency):
        if x.ndim != 2 or x.shape[1] != 6775 or x.shape[0] == 0:
            raise ValueError('Encoder requires nonempty [N, 6775] features')
        if normalized_adjacency.layout != torch.sparse_coo:
            raise ValueError('Use Dataset normalized_adjacency (sparse COO)')
        if normalized_adjacency.shape != (x.shape[0], x.shape[0]):
            raise ValueError('Adjacency and feature node counts differ')
        if (normalized_adjacency.device != x.device
                or normalized_adjacency.dtype != x.dtype):
            raise ValueError('Adjacency and features must share dtype/device')
        h1 = self.dropout(F.relu(self.norm1(self.gcn1(x, normalized_adjacency))))
        return EncoderOutput(h1, self.norm2(self.gcn2(h1, normalized_adjacency)))

    def forward_pair(self, source_x, source_adjacency, target_x, target_adjacency):
        """DA interface: both calls use this exact module and Parameter objects.

        Warm-up instead uses its eval/no_grad target inspection entry point.
        """
        return self(source_x, source_adjacency), self(target_x, target_adjacency)
