import torch
from torch import nn
from models.selection.operators import PostNormAttention, Selection


class BGABlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.intra = PostNormAttention()
        self.inter = PostNormAttention()
        self.broadcast_fusion = nn.Linear(256, 128, bias=False)

    def forward(self, h0, partition):
        perm = partition.permutation.to(h0.device)
        clusters = h0[perm].split(partition.counts.tolist())
        local = [self.intra(cluster) for cluster in clusters]
        global_clusters = self.inter(torch.stack([cluster.mean(0) for cluster in local]))
        fused = [self.broadcast_fusion(torch.cat((h, g.expand(len(h), -1)), dim=1))
                 for h, g in zip(local, global_clusters)]
        return torch.cat(fused)[partition.inverse.to(h0.device)]


class StructureBranch(nn.Module):
    blocks = 1

    def __init__(self):
        super().__init__()
        self.mlp_s = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Dropout(.1), nn.Linear(128, 128))
        self.bga = BGABlock()
        self.selection = Selection()

    def forward(self, z, partition, *, source):
        h_lg = self.bga(self.mlp_s(z), partition)
        h_s, thresholded = self.selection(h_lg) if source else (h_lg, None)
        return h_lg, h_s, thresholded
